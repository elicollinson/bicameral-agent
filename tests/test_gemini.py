"""Tests for the Gemini API client wrapper (Issue #10)."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from bicameral_agent.gemini import (
    ChatMessage,
    GeminiClient,
    GeminiResponse,
    _BASE_DELAY,
    _BACKOFF_FACTOR,
    _MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_part(*, text=None, thought=None, function_call=None):
    """Create a mock Part with the given attributes."""
    part = MagicMock()
    part.text = text
    part.thought = thought
    part.function_call = function_call
    return part


def _make_mock_response(
    text="Hello!",
    prompt_token_count=10,
    candidates_token_count=5,
    finish_reason="STOP",
    function_calls=None,
    thinking_text=None,
):
    """Create a mock SDK response with configurable fields."""
    parts = []

    if thinking_text is not None:
        parts.append(_make_mock_part(text=thinking_text, thought=thinking_text))

    if function_calls:
        for fc in function_calls:
            fc_mock = MagicMock()
            fc_mock.name = fc["name"]
            fc_mock.args = fc.get("args", {})
            parts.append(_make_mock_part(function_call=fc_mock))
    else:
        parts.append(_make_mock_part(text=text))

    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    candidate.content.parts = parts

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata.prompt_token_count = prompt_token_count
    response.usage_metadata.candidates_token_count = candidates_token_count
    return response


def _get_config(sdk_mock):
    """Extract the GenerateContentConfig from an SDK mock's last call."""
    call_kwargs = sdk_mock.models.generate_content.call_args
    return call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """GeminiClient with mocked SDK, returns (client, mock_sdk_instance)."""
    with patch("bicameral_agent.gemini.genai.Client") as MockClient:
        instance = MockClient.return_value
        instance.models.generate_content.return_value = _make_mock_response()
        client = GeminiClient(api_key="test-key")
        yield client, instance


# ---------------------------------------------------------------------------
# TestGeminiResponse
# ---------------------------------------------------------------------------


class TestGeminiResponse:
    def test_fields(self):
        r = GeminiResponse(
            content="hi",
            input_tokens=10,
            output_tokens=5,
            duration_ms=100.0,
            finish_reason="STOP",
        )
        assert r.content == "hi"
        assert r.input_tokens == 10
        assert r.output_tokens == 5
        assert r.duration_ms == 100.0
        assert r.finish_reason == "STOP"
        assert r.function_calls is None

    def test_function_calls_field(self):
        calls = [{"name": "foo", "args": {"x": 1}}]
        r = GeminiResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            duration_ms=0.0,
            finish_reason="STOP",
            function_calls=calls,
        )
        assert r.function_calls == calls

    def test_frozen(self):
        r = GeminiResponse(
            content="hi", input_tokens=0, output_tokens=0,
            duration_ms=0.0, finish_reason="STOP",
        )
        with pytest.raises(AttributeError):
            r.content = "bye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestChatMessage
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_creation(self):
        m = ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_frozen(self):
        m = ChatMessage(role="user", content="hello")
        with pytest.raises(AttributeError):
            m.role = "model"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestClientInit
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_api_key_from_param(self):
        with patch("bicameral_agent.gemini.genai.Client"):
            client = GeminiClient(api_key="my-key")
            assert client._model == "gemini-3-flash-preview"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            GeminiClient()
            MockClient.assert_called_once_with(api_key="env-key")

    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            GeminiClient()

    def test_custom_model(self):
        with patch("bicameral_agent.gemini.genai.Client"):
            client = GeminiClient(api_key="key", model="gemini-2.5-flash")
            assert client._model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# TestRetryLogic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_retry_on_429(self, mock_client):
        client, sdk = mock_client
        exc_429 = Exception("429 Too Many Requests")
        sdk.models.generate_content.side_effect = [
            exc_429, exc_429, _make_mock_response()
        ]
        with patch("bicameral_agent.gemini.time.sleep"):
            result = client.generate([{"role": "user", "content": "hi"}])
        assert result.content == "Hello!"
        assert sdk.models.generate_content.call_count == 3

    def test_retry_on_5xx(self, mock_client):
        client, sdk = mock_client
        exc_500 = Exception("500 Internal Server Error")
        sdk.models.generate_content.side_effect = [
            exc_500, _make_mock_response()
        ]
        with patch("bicameral_agent.gemini.time.sleep"):
            result = client.generate([{"role": "user", "content": "hi"}])
        assert result.content == "Hello!"
        assert sdk.models.generate_content.call_count == 2

    def test_no_retry_on_400(self, mock_client):
        client, sdk = mock_client
        exc_400 = Exception("400 Bad Request")
        exc_400.code = 400
        sdk.models.generate_content.side_effect = exc_400
        with pytest.raises(Exception, match="400"):
            client.generate([{"role": "user", "content": "hi"}])
        assert sdk.models.generate_content.call_count == 1

    def test_max_retries_exceeded(self, mock_client):
        client, sdk = mock_client
        exc_429 = Exception("429 Too Many Requests")
        sdk.models.generate_content.side_effect = exc_429
        with patch("bicameral_agent.gemini.time.sleep"):
            with pytest.raises(Exception, match="429"):
                client.generate([{"role": "user", "content": "hi"}])
        assert sdk.models.generate_content.call_count == _MAX_RETRIES + 1

    def test_exponential_backoff_timing(self, mock_client):
        client, sdk = mock_client
        exc_429 = Exception("429 Too Many Requests")
        sdk.models.generate_content.side_effect = [
            exc_429, exc_429, _make_mock_response()
        ]
        sleep_calls = []
        with patch("bicameral_agent.gemini.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            with patch("bicameral_agent.gemini.random.uniform", return_value=0.0):
                client.generate([{"role": "user", "content": "hi"}])

        assert len(sleep_calls) == 2
        assert sleep_calls[0] == pytest.approx(_BASE_DELAY, abs=0.01)
        assert sleep_calls[1] == pytest.approx(
            _BASE_DELAY * _BACKOFF_FACTOR, abs=0.01
        )


# ---------------------------------------------------------------------------
# TestOnCompletionCallback
# ---------------------------------------------------------------------------


class TestOnCompletionCallback:
    def test_callback_fires_with_correct_values(self):
        callback = MagicMock()
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = _make_mock_response(
                prompt_token_count=42, candidates_token_count=17,
            )
            client = GeminiClient(api_key="key", on_completion=callback)
            result = client.generate([{"role": "user", "content": "hi"}])

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == 42  # input_tokens
        assert args[1] == 17  # output_tokens
        assert args[2] > 0  # duration_ms
        assert args[2] == result.duration_ms

    def test_callback_not_called_on_error(self):
        callback = MagicMock()
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            instance = MockClient.return_value
            exc = Exception("400 Bad Request")
            exc.code = 400
            instance.models.generate_content.side_effect = exc
            client = GeminiClient(api_key="key", on_completion=callback)
            with pytest.raises(Exception):
                client.generate([{"role": "user", "content": "hi"}])

        callback.assert_not_called()

    def test_no_callback_no_error(self, mock_client):
        client, _ = mock_client
        assert client._on_completion is None
        result = client.generate([{"role": "user", "content": "hi"}])
        assert result.content == "Hello!"


# ---------------------------------------------------------------------------
# TestTokenCounting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_tokens_from_usage_metadata(self):
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = _make_mock_response(
                prompt_token_count=123, candidates_token_count=456,
            )
            client = GeminiClient(api_key="key")
            result = client.generate([{"role": "user", "content": "hi"}])

        assert result.input_tokens == 123
        assert result.output_tokens == 456

    def test_none_token_counts_default_to_zero(self):
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = _make_mock_response(
                prompt_token_count=None, candidates_token_count=None,
            )
            client = GeminiClient(api_key="key")
            result = client.generate([{"role": "user", "content": "hi"}])

        assert result.input_tokens == 0
        assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# TestDurationMeasurement
# ---------------------------------------------------------------------------


class TestDurationMeasurement:
    def test_duration_calculation(self):
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = _make_mock_response()

            # Simulate 50ms elapsed
            times = [0, 50_000_000]  # nanoseconds
            with patch("bicameral_agent.gemini.time.monotonic_ns", side_effect=times):
                client = GeminiClient(api_key="key")
                result = client.generate([{"role": "user", "content": "hi"}])

        assert result.duration_ms == pytest.approx(50.0)

    def test_duration_within_tolerance_of_real_time(self):
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            instance = MockClient.return_value

            def slow_generate(**kwargs):
                time.sleep(0.05)  # 50ms
                return _make_mock_response()

            instance.models.generate_content.side_effect = slow_generate
            client = GeminiClient(api_key="key")
            result = client.generate([{"role": "user", "content": "hi"}])

        assert result.duration_ms >= 45  # within 10% of 50ms
        assert result.duration_ms <= 100  # generous upper bound


# ---------------------------------------------------------------------------
# TestFunctionCalling
# ---------------------------------------------------------------------------


class TestFunctionCalling:
    def test_function_call_response_parsed(self):
        mock_resp = _make_mock_response(
            function_calls=[
                {"name": "get_weather", "args": {"location": "Boston"}},
            ],
        )
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = mock_resp
            client = GeminiClient(api_key="key")
            result = client.generate([{"role": "user", "content": "weather?"}])

        assert result.function_calls is not None
        assert len(result.function_calls) == 1
        assert result.function_calls[0]["name"] == "get_weather"
        assert result.function_calls[0]["args"] == {"location": "Boston"}
        assert result.content == ""  # no text content

    def test_tool_declarations_passed_to_config(self, mock_client):
        client, sdk = mock_client
        tools = [{
            "name": "get_weather",
            "description": "Get weather",
            "parameters_json_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }]
        client.generate(
            [{"role": "user", "content": "hi"}],
            tools=tools,
        )
        assert _get_config(sdk).tools is not None


# ---------------------------------------------------------------------------
# TestStructuredOutput
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_response_schema_sets_config(self, mock_client):
        client, sdk = mock_client
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        client.generate(
            [{"role": "user", "content": "give me a name"}],
            response_schema=schema,
        )
        config = _get_config(sdk)
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == schema

    def test_structured_output_content_is_text(self):
        mock_resp = _make_mock_response(text='{"name": "Alice"}')
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = mock_resp
            client = GeminiClient(api_key="key")
            result = client.generate(
                [{"role": "user", "content": "name?"}],
                response_schema={"type": "object"},
            )
        assert result.content == '{"name": "Alice"}'


# ---------------------------------------------------------------------------
# TestThinkingLevel
# ---------------------------------------------------------------------------


class TestThinkingLevel:
    @pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
    def test_valid_levels(self, mock_client, level):
        client, _ = mock_client
        result = client.generate(
            [{"role": "user", "content": "hi"}],
            thinking_level=level,
        )
        assert result.content == "Hello!"

    def test_case_insensitive(self, mock_client):
        client, _ = mock_client
        result = client.generate(
            [{"role": "user", "content": "hi"}],
            thinking_level="HIGH",
        )
        assert result.content == "Hello!"

    def test_invalid_level_raises(self, mock_client):
        client, _ = mock_client
        with pytest.raises(ValueError, match="Invalid thinking_level"):
            client.generate(
                [{"role": "user", "content": "hi"}],
                thinking_level="extreme",
            )

    def test_thinking_config_passed(self, mock_client):
        client, sdk = mock_client
        client.generate(
            [{"role": "user", "content": "hi"}],
            thinking_level="high",
        )
        level = _get_config(sdk).thinking_config.thinking_level
        # SDK may store as enum or string
        assert str(level).lower() in ("high", "thinkinglevel.high")


# ---------------------------------------------------------------------------
# TestMessageConversion
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_chat_message_objects(self, mock_client):
        client, sdk = mock_client
        messages = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="model", content="hi"),
        ]
        client.generate(messages)
        call_kwargs = sdk.models.generate_content.call_args
        contents = call_kwargs.kwargs.get("contents") or call_kwargs[1].get("contents")
        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[1].role == "model"

    def test_dict_messages(self, mock_client):
        client, sdk = mock_client
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "model", "content": "hi"},
        ]
        client.generate(messages)
        call_kwargs = sdk.models.generate_content.call_args
        contents = call_kwargs.kwargs.get("contents") or call_kwargs[1].get("contents")
        assert len(contents) == 2

    def test_system_prompt_in_config(self, mock_client):
        client, sdk = mock_client
        client.generate(
            [{"role": "user", "content": "hi"}],
            system_prompt="You are helpful.",
        )
        assert _get_config(sdk).system_instruction == "You are helpful."


# ---------------------------------------------------------------------------
# TestThinkingPartsFiltered
# ---------------------------------------------------------------------------


class TestThinkingPartsFiltered:
    def test_thinking_text_excluded_from_content(self):
        mock_resp = _make_mock_response(
            text="The answer is 42.", thinking_text="Let me think..."
        )
        with patch("bicameral_agent.gemini.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = mock_resp
            client = GeminiClient(api_key="key")
            result = client.generate([{"role": "user", "content": "question"}])

        assert "think" not in result.content.lower()
        assert result.content == "The answer is 42."


# ---------------------------------------------------------------------------
# TestIntegration (skippable — requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestIntegration:
    def test_real_api_call(self):
        client = GeminiClient()
        result = client.generate(
            [{"role": "user", "content": "Say exactly: hello world"}],
            thinking_level="minimal",
            max_output_tokens=50,
        )
        assert result.content  # non-empty
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.duration_ms > 0
        assert result.finish_reason

    def test_real_structured_output(self):
        client = GeminiClient()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = client.generate(
            [{"role": "user", "content": "Give me a fictional person."}],
            response_schema=schema,
            thinking_level="minimal",
            max_output_tokens=100,
        )
        import json
        data = json.loads(result.content)
        assert "name" in data
        assert "age" in data

    def test_real_function_calling(self):
        client = GeminiClient()
        tools = [{
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state",
                    },
                },
                "required": ["location"],
            },
        }]
        result = client.generate(
            [{"role": "user", "content": "What's the weather in Boston?"}],
            tools=tools,
            thinking_level="minimal",
            max_output_tokens=100,
        )
        assert result.function_calls is not None
        assert len(result.function_calls) >= 1
        assert result.function_calls[0]["name"] == "get_weather"

    def test_real_on_completion_callback(self):
        recorded = []

        def callback(inp, out, dur):
            recorded.append((inp, out, dur))

        client = GeminiClient(on_completion=callback)
        result = client.generate(
            [{"role": "user", "content": "Say hi"}],
            thinking_level="minimal",
            max_output_tokens=20,
        )
        assert len(recorded) == 1
        assert recorded[0][0] == result.input_tokens
        assert recorded[0][1] == result.output_tokens
        assert recorded[0][2] == result.duration_ms
