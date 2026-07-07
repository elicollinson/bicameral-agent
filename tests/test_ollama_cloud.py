"""Tests for the Ollama Cloud client wrapper (Issue #43).

Runs fully offline: the HTTP transport (``urllib.request.urlopen``) is mocked,
so no live calls and no API key are required.
"""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from bicameral_agent.gemini import ChatMessage, GeminiResponse
from bicameral_agent.model_client import _MAX_RETRIES, TransportExhausted
from bicameral_agent.ollama_cloud import OllamaCloudClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response_body(
    *,
    content="Hello!",
    prompt_eval_count=10,
    eval_count=5,
    done_reason="stop",
    tool_calls=None,
) -> bytes:
    """Serialise a fake Ollama /api/chat response body."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return json.dumps({
        "model": "gemma4:31b-cloud",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }).encode("utf-8")


class _FakeHTTPResponse(io.BytesIO):
    """Minimal context-manager stand-in for an ``http.client.HTTPResponse``."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _urlopen_returning(body: bytes):
    """A urlopen replacement that records the request and returns *body*."""
    captured = {}

    def _fake(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeHTTPResponse(body)

    return _fake, captured


def _sent_payload(captured) -> dict:
    """Decode the JSON body of the captured request."""
    return json.loads(captured["request"].data.decode("utf-8"))


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://ollama.com/api/chat", code=code, msg="err", hdrs=None, fp=None
    )


# ---------------------------------------------------------------------------
# Client init / auth
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_api_key_from_param(self):
        client = OllamaCloudClient(api_key="my-key")
        assert client.model == "gemma4:31b-cloud"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
        client = OllamaCloudClient()
        assert client._api_key == "env-key"

    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            OllamaCloudClient()

    def test_custom_model(self):
        client = OllamaCloudClient(api_key="k", model="gemma3:27b-cloud")
        assert client.model == "gemma3:27b-cloud"

    def test_host_default_and_override(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert OllamaCloudClient(api_key="k")._host == "https://ollama.com"
        client = OllamaCloudClient(api_key="k", host="http://localhost:11434/")
        assert client._host == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


class TestRequestBuilding:
    def test_bearer_header_and_endpoint(self):
        fake, captured = _urlopen_returning(_make_response_body())
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="secret").generate([{"role": "user", "content": "hi"}])
        request = captured["request"]
        assert request.full_url == "https://ollama.com/api/chat"
        assert request.get_header("Authorization") == "Bearer secret"
        assert _sent_payload(captured)["stream"] is False

    def test_role_mapping_and_system_prompt(self):
        fake, captured = _urlopen_returning(_make_response_body())
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="k").generate(
                [ChatMessage(role="model", content="prev"),
                 ChatMessage(role="user", content="now")],
                system_prompt="be terse",
            )
        messages = _sent_payload(captured)["messages"]
        assert messages[0] == {"role": "system", "content": "be terse"}
        # Gemini's 'model' role becomes Ollama's 'assistant'.
        assert messages[1] == {"role": "assistant", "content": "prev"}
        assert messages[2] == {"role": "user", "content": "now"}

    def test_thinking_level_mapping(self):
        for level, expected in [("minimal", False), ("low", "low"),
                                ("medium", "medium"), ("high", "high")]:
            fake, captured = _urlopen_returning(_make_response_body())
            with patch("urllib.request.urlopen", fake):
                OllamaCloudClient(api_key="k").generate(
                    [{"role": "user", "content": "x"}], thinking_level=level
                )
            assert _sent_payload(captured)["think"] == expected

    def test_invalid_thinking_level_raises(self):
        with pytest.raises(ValueError, match="thinking_level"):
            OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}], thinking_level="bogus"
            )

    def test_options_temperature_and_max_tokens(self):
        fake, captured = _urlopen_returning(_make_response_body())
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}],
                temperature=0.0,
                max_output_tokens=100,
            )
        options = _sent_payload(captured)["options"]
        assert options == {"temperature": 0.0, "num_predict": 100}

    def test_options_omitted_when_unset(self):
        fake, captured = _urlopen_returning(_make_response_body())
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="k").generate([{"role": "user", "content": "x"}])
        assert "options" not in _sent_payload(captured)

    def test_response_schema_passthrough_to_format(self):
        schema = {"type": "object", "properties": {"q": {"type": "integer"}}}
        fake, captured = _urlopen_returning(_make_response_body())
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}], response_schema=schema
            )
        assert _sent_payload(captured)["format"] == schema

    def test_response_schema_grounds_system_prompt(self):
        # Ollama Cloud ignores ``format``; the schema is also injected into the
        # system prompt so the model returns parseable JSON (issue #82).
        import json

        schema = {"type": "object", "properties": {"q": {"type": "integer"}}}
        fake, captured = _urlopen_returning(_make_response_body())
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}],
                system_prompt="be terse",
                response_schema=schema,
            )
        system_msg = _sent_payload(captured)["messages"][0]
        assert system_msg["role"] == "system"
        assert system_msg["content"].startswith("be terse")
        assert json.dumps(schema) in system_msg["content"]

    def test_response_schema_grounds_without_system_prompt(self):
        import json

        schema = {"type": "object", "properties": {"q": {"type": "integer"}}}
        fake, captured = _urlopen_returning(_make_response_body())
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}], response_schema=schema
            )
        messages = _sent_payload(captured)["messages"]
        assert messages[0]["role"] == "system"
        assert json.dumps(schema) in messages[0]["content"]

    def test_no_system_message_without_schema_or_prompt(self):
        fake, captured = _urlopen_returning(_make_response_body())
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="k").generate([{"role": "user", "content": "x"}])
        roles = [m["role"] for m in _sent_payload(captured)["messages"]]
        assert "system" not in roles

    def test_tools_translated_to_ollama_format(self):
        tools = [{
            "name": "get_weather",
            "description": "look up weather",
            "parameters_json_schema": {"type": "object", "properties": {}},
        }]
        fake, captured = _urlopen_returning(_make_response_body())
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}], tools=tools
            )
        sent = _sent_payload(captured)["tools"]
        assert sent == [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "look up weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_content_and_token_mapping(self):
        body = _make_response_body(content="answer", prompt_eval_count=12, eval_count=7)
        fake, _ = _urlopen_returning(body)
        with patch("urllib.request.urlopen", fake):
            result = OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}]
            )
        assert isinstance(result, GeminiResponse)
        assert result.content == "answer"
        assert result.input_tokens == 12
        assert result.output_tokens == 7
        assert result.finish_reason == "stop"
        assert result.function_calls is None
        assert result.duration_ms >= 0.0

    def test_structured_output_roundtrip(self):
        body = _make_response_body(content='{"quality": 5, "accuracy": 4}')
        fake, _ = _urlopen_returning(body)
        with patch("urllib.request.urlopen", fake):
            result = OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}],
                response_schema={"type": "object"},
            )
        assert json.loads(result.content) == {"quality": 5, "accuracy": 4}

    def test_tool_calls_parsed(self):
        body = _make_response_body(
            content="",
            tool_calls=[{"function": {"name": "f", "arguments": {"a": 1}}}],
        )
        fake, _ = _urlopen_returning(body)
        with patch("urllib.request.urlopen", fake):
            result = OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}]
            )
        assert result.function_calls == [{"name": "f", "args": {"a": 1}}]

    def test_on_completion_callback(self):
        callback = MagicMock()
        body = _make_response_body(prompt_eval_count=3, eval_count=9)
        fake, _ = _urlopen_returning(body)
        with patch("urllib.request.urlopen", fake):
            OllamaCloudClient(api_key="k", on_completion=callback).generate(
                [{"role": "user", "content": "x"}]
            )
        callback.assert_called_once()
        in_toks, out_toks, duration = callback.call_args.args
        assert in_toks == 3
        assert out_toks == 9
        assert duration >= 0.0


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retries_on_429_then_succeeds(self):
        calls = {"n": 0}

        def _fake(request, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(429)
            return _FakeHTTPResponse(_make_response_body(content="ok"))

        with patch("urllib.request.urlopen", _fake), \
                patch("bicameral_agent.model_client.time.sleep"):
            result = OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}]
            )
        assert result.content == "ok"
        assert calls["n"] == 2

    def test_retries_on_503_then_succeeds(self):
        calls = {"n": 0}

        def _fake(request, timeout=None):
            calls["n"] += 1
            if calls["n"] <= 1:
                raise _http_error(503)
            return _FakeHTTPResponse(_make_response_body())

        with patch("urllib.request.urlopen", _fake), \
                patch("bicameral_agent.model_client.time.sleep"):
            OllamaCloudClient(api_key="k").generate([{"role": "user", "content": "x"}])
        assert calls["n"] == 2

    def test_non_retryable_4xx_raises_immediately(self):
        calls = {"n": 0}

        def _fake(request, timeout=None):
            calls["n"] += 1
            raise _http_error(400)

        with patch("urllib.request.urlopen", _fake), \
                patch("bicameral_agent.model_client.time.sleep"):
            with pytest.raises(urllib.error.HTTPError):
                OllamaCloudClient(api_key="k").generate([{"role": "user", "content": "x"}])
        assert calls["n"] == 1

    def test_gives_up_after_max_retries(self):
        def _fake(request, timeout=None):
            raise _http_error(500)

        with patch("urllib.request.urlopen", _fake), \
                patch("bicameral_agent.model_client.time.sleep") as sleep:
            with pytest.raises(TransportExhausted) as exc_info:
                OllamaCloudClient(api_key="k").generate([{"role": "user", "content": "x"}])
        assert sleep.call_count == _MAX_RETRIES
        assert isinstance(exc_info.value.__cause__, urllib.error.HTTPError)


# ---------------------------------------------------------------------------
# Read-phase failure retry (issue #47)
# ---------------------------------------------------------------------------


class TestReadPhaseRetry:
    class _FailingReadResponse:
        """Context-manager response whose read() raises *exc*."""

        def __init__(self, exc: Exception) -> None:
            self._exc = exc

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise self._exc

    def _generate_with_first_call_failing(self, exc: Exception):
        calls = {"n": 0}

        def _fake(request, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return self._FailingReadResponse(exc)
            return _FakeHTTPResponse(_make_response_body(content="ok"))

        with patch("urllib.request.urlopen", _fake), \
                patch("bicameral_agent.model_client.time.sleep"):
            result = OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}]
            )
        return result, calls["n"]

    def test_timeout_during_read_retried(self):
        result, n = self._generate_with_first_call_failing(TimeoutError("read timed out"))
        assert result.content == "ok"
        assert n == 2

    def test_connection_reset_during_read_retried(self):
        result, n = self._generate_with_first_call_failing(
            ConnectionResetError(54, "Connection reset by peer")
        )
        assert result.content == "ok"
        assert n == 2

    def test_incomplete_read_retried(self):
        result, n = self._generate_with_first_call_failing(
            http.client.IncompleteRead(b"partial")
        )
        assert result.content == "ok"
        assert n == 2

    def test_truncated_200_body_retried(self):
        """A 200 whose body is truncated mid-JSON is retried, not raised."""
        calls = {"n": 0}

        def _fake(request, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeHTTPResponse(_make_response_body()[:20])
            return _FakeHTTPResponse(_make_response_body(content="ok"))

        with patch("urllib.request.urlopen", _fake), \
                patch("bicameral_agent.model_client.time.sleep"):
            result = OllamaCloudClient(api_key="k").generate(
                [{"role": "user", "content": "x"}]
            )
        assert result.content == "ok"
        assert calls["n"] == 2

    def test_persistent_read_failure_raises_after_max_retries(self):
        def _fake(request, timeout=None):
            return self._FailingReadResponse(TimeoutError("read timed out"))

        with patch("urllib.request.urlopen", _fake), \
                patch("bicameral_agent.model_client.time.sleep") as sleep:
            with pytest.raises(TransportExhausted) as exc_info:
                OllamaCloudClient(api_key="k").generate(
                    [{"role": "user", "content": "x"}]
                )
        assert sleep.call_count == _MAX_RETRIES
        assert isinstance(exc_info.value.__cause__, TimeoutError)
