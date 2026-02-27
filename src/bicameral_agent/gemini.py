"""Thin wrapper around the Gemini API with retry, token counting, and latency measurement.

Every LLM call in the system goes through this wrapper, ensuring uniform
latency data collection. Wraps the google-genai SDK to provide a clean
interface with automatic retry on transient errors, wall-clock latency
measurement, and an optional on_completion callback for feeding the
APILatencyModel.
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from google import genai
from google.genai import types

_MODEL = "gemini-3-flash-preview"
_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0
_MAX_JITTER = 0.5

_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A message in the conversation for the Gemini API.

    Lighter than schema.Message -- no timestamp/token_count fields,
    since those are logging concerns, not API input concerns.
    """

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class GeminiResponse:
    """Response from a Gemini API call with metadata."""

    content: str
    input_tokens: int
    output_tokens: int
    duration_ms: float
    finish_reason: str
    function_calls: list[dict[str, Any]] | None = field(default=None)


class GeminiClient:
    """Thin wrapper around the Gemini API with retry, timing, and callbacks.

    Thread-safe: no mutable state after __init__.
    """

    def __init__(
        self,
        api_key: str | None = None,
        on_completion: Callable[[int, int, float], None] | None = None,
        model: str = _MODEL,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key required: pass api_key= or set GEMINI_API_KEY env var"
            )
        self._client = genai.Client(api_key=resolved_key)
        self._on_completion = on_completion
        self._model = model

    def generate(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        system_prompt: str | None = None,
        thinking_level: str = "medium",
        max_output_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> GeminiResponse:
        """Generate a response from the Gemini API.

        Args:
            messages: Conversation history with 'role' and 'content' keys.
            system_prompt: Optional system instruction.
            thinking_level: Thinking depth: 'minimal', 'low', 'medium', 'high'.
            max_output_tokens: Maximum output tokens.
            tools: Function declarations (dicts with 'name', 'description',
                'parameters_json_schema' keys).
            response_schema: JSON schema dict for structured output.

        Returns:
            GeminiResponse with content, token counts, timing, and finish reason.
        """
        if thinking_level.lower() not in _VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking_level {thinking_level!r}; "
                f"must be one of {sorted(_VALID_THINKING_LEVELS)}"
            )

        contents = self._build_contents(messages)
        config = self._build_config(
            system_prompt=system_prompt,
            thinking_level=thinking_level.lower(),
            max_output_tokens=max_output_tokens,
            tools=tools,
            response_schema=response_schema,
        )
        return self._execute_with_retry(contents, config)

    @staticmethod
    def _build_contents(
        messages: list[ChatMessage] | list[dict[str, str]],
    ) -> list[types.Content]:
        contents = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg["role"], msg["content"]
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content)],
                )
            )
        return contents

    @staticmethod
    def _build_config(
        *,
        system_prompt: str | None,
        thinking_level: str,
        max_output_tokens: int | None,
        tools: list[dict] | None,
        response_schema: dict | None,
    ) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "thinking_config": types.ThinkingConfig(thinking_level=thinking_level),
        }

        if system_prompt is not None:
            kwargs["system_instruction"] = system_prompt

        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens

        if tools is not None:
            kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**decl) for decl in tools
                    ]
                )
            ]

        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = response_schema

        return types.GenerateContentConfig(**kwargs)

    def _execute_with_retry(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> GeminiResponse:
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _BASE_DELAY * (_BACKOFF_FACTOR ** (attempt - 1))
                jitter = random.uniform(0, _MAX_JITTER)
                time.sleep(delay + jitter)

            try:
                start_ns = time.monotonic_ns()
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000
                return self._parse_response(response, duration_ms)

            except Exception as exc:
                if self._is_retryable(exc) and attempt < _MAX_RETRIES:
                    last_exc = exc
                    continue
                raise

        raise last_exc  # type: ignore[misc]

    def _parse_response(self, response: Any, duration_ms: float) -> GeminiResponse:
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0

        candidate = response.candidates[0]
        finish_reason = (
            str(candidate.finish_reason) if candidate.finish_reason else "STOP"
        )

        text_parts: list[str] = []
        fc_parts: list[dict[str, Any]] = []

        for part in candidate.content.parts:
            if part.function_call is not None:
                fc = part.function_call
                fc_parts.append({
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                })
            elif part.text is not None and part.thought is None:
                text_parts.append(part.text)

        if self._on_completion is not None:
            self._on_completion(input_tokens, output_tokens, duration_ms)

        return GeminiResponse(
            content="".join(text_parts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            finish_reason=finish_reason,
            function_calls=fc_parts or None,
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status = getattr(exc, "code", None) or getattr(exc, "status", None)
        if isinstance(status, int):
            return status == 429 or 500 <= status < 600
        exc_str = str(exc).lower()
        if "429" in exc_str or "too many requests" in exc_str:
            return True
        return bool(re.search(r"5\d{2}", exc_str))
