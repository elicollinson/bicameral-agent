"""Thin wrapper around the Ollama Cloud chat API (issue #43).

A drop-in alternative to ``GeminiClient`` for running episodes/benchmarks
against an open Gemma-class model. Mirrors the ``GeminiClient`` interface
exactly -- same ``generate(...)`` signature, same ``ModelResponse`` return
type, same retry/timing/callback behaviour -- so it is interchangeable at every
call site (``EpisodeRunner``, ``SimulatedUser``, ``TaskScorer``, tools).

Transport is stdlib ``urllib`` (no extra dependency): a single non-streaming
``POST`` to ``{host}/api/chat`` with a Bearer token. The native Ollama chat
endpoint accepts a JSON schema in its ``format`` field for structured output and
returns the JSON as ``message.content`` -- the same shape the scorer and
simulated user already parse with ``json.loads(response.content)``.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from bicameral_agent.model_client import (
    ChatMessage,
    ModelResponse,
    RetryingClientBase,
    default_model,
    validate_thinking_level,
)

_MODEL = default_model("ollama")
_DEFAULT_HOST = "https://ollama.com"
_TIMEOUT_S = 120.0

# Ollama Cloud does not yet honour the ``format`` schema for grammar-level
# constrained decoding (verified 2026-07-07: gemma4:31b-cloud and the other
# -cloud tags return markdown prose even with ``format`` set). The Ollama docs
# recommend grounding the model by also stating the schema in the prompt; doing
# so makes gemma return clean parseable JSON. We send both: ``format`` (correct
# per the API spec, effective for local Ollama and forward-compatible if Cloud
# adds decoder support) and this prompt grounding (what actually fixes the
# pilot's 100% judge degradation, issue #82).
_GROUNDING_TEMPLATE = (
    "\n\nRespond with a single JSON object matching this schema exactly, and "
    "nothing else -- no markdown, no code fences, no prose:\n{schema}"
)


class OllamaCloudClient(RetryingClientBase):
    """Thin wrapper around the Ollama Cloud chat API with retry, timing, callbacks.

    Thread-safe: no mutable state after __init__.
    """

    provider = "ollama"

    def __init__(
        self,
        api_key: str | None = None,
        on_completion: Callable[[int, int, float], None] | None = None,
        model: str = _MODEL,
        host: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key required: pass api_key= or set OLLAMA_API_KEY env var"
            )
        self._api_key = resolved_key
        self._on_completion = on_completion
        self._model = model
        self._host = (host or os.environ.get("OLLAMA_HOST") or _DEFAULT_HOST).rstrip("/")

    @property
    def model(self) -> str:
        """The model name used for API calls."""
        return self._model

    def generate(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        system_prompt: str | None = None,
        thinking_level: str = "medium",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> ModelResponse:
        """Generate a response from the Ollama Cloud chat API.

        Args mirror ``GeminiClient.generate`` for drop-in compatibility:
            messages: Conversation history with 'role' and 'content' keys.
            system_prompt: Optional system instruction (prepended as a system message).
            thinking_level: 'minimal', 'low', 'medium', 'high'. 'minimal' disables
                thinking; the rest map to Ollama's ``think`` string. Requires a
                reasoning-capable model (e.g. gemma4:31b-cloud).
            temperature: Sampling temperature; None uses model default.
            max_output_tokens: Maps to Ollama ``options.num_predict``.
            tools: Function declarations (dicts with 'name', 'description',
                'parameters_json_schema' keys), translated to Ollama tool format.
            response_schema: JSON schema dict. Sent as Ollama ``format`` and,
                because Ollama Cloud ignores ``format``, also injected into the
                system prompt to ground the model (issue #82).

        Returns:
            ModelResponse with content, token counts, timing, and finish reason.
        """
        payload = self._build_payload(
            messages,
            system_prompt=system_prompt,
            thinking_level=validate_thinking_level(thinking_level),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
            response_schema=response_schema,
        )
        return self._execute_with_retry(lambda: self._attempt(payload))

    def _build_payload(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        system_prompt: str | None,
        thinking_level: str,
        temperature: float | None,
        max_output_tokens: int | None,
        tools: list[dict] | None,
        response_schema: dict | None,
    ) -> dict[str, Any]:
        api_messages: list[dict[str, str]] = []
        if response_schema is not None:
            grounding = _GROUNDING_TEMPLATE.format(schema=json.dumps(response_schema))
            system_prompt = (system_prompt or "") + grounding
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg["role"], msg["content"]
            # Gemini uses 'model' for the assistant turn; Ollama uses 'assistant'.
            if role == "model":
                role = "assistant"
            api_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "stream": False,
            # 'minimal' means "don't think"; other levels pass through.
            "think": False if thinking_level == "minimal" else thinking_level,
        }

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        if options:
            payload["options"] = options

        if response_schema is not None:
            payload["format"] = response_schema

        if tools is not None:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": decl["name"],
                        "description": decl.get("description", ""),
                        "parameters": decl.get("parameters_json_schema", {}),
                    },
                }
                for decl in tools
            ]

        return payload

    def _attempt(self, payload: dict[str, Any]) -> ModelResponse:
        """One timed request/parse round trip (retried by the base class)."""
        start_ns = time.monotonic_ns()
        data = self._post(payload)
        duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        return self._parse_response(data, duration_ms)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_response(self, data: dict[str, Any], duration_ms: float) -> ModelResponse:
        input_tokens = data.get("prompt_eval_count") or 0
        output_tokens = data.get("eval_count") or 0
        finish_reason = data.get("done_reason") or "stop"

        message = data.get("message") or {}
        content = message.get("content") or ""

        fc_parts: list[dict[str, Any]] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            fc_parts.append({
                "name": fn.get("name"),
                "args": dict(fn.get("arguments") or {}),
            })

        if self._on_completion is not None:
            self._on_completion(input_tokens, output_tokens, duration_ms)

        return ModelResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            finish_reason=finish_reason,
            function_calls=fc_parts or None,
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code == 429 or 500 <= exc.code < 600
        # Network-level failures are transient, whether they hit at connect
        # time (URLError) or mid-read: timeouts and connection resets during
        # response.read(), truncated chunked bodies (IncompleteRead), and a
        # 200 whose truncated body fails JSON decoding.
        return isinstance(
            exc,
            (
                urllib.error.URLError,
                TimeoutError,
                ConnectionResetError,
                http.client.HTTPException,
                json.JSONDecodeError,
            ),
        )
