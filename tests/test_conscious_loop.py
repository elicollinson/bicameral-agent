"""Tests for the ConsciousLoop and AssistantResponse."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from bicameral_agent.conscious_loop import AssistantResponse, ConsciousLoop
from bicameral_agent.gemini import ChatMessage, GeminiClient, GeminiResponse
from bicameral_agent.queue import ContextQueue, InterruptConfig, Priority, QueueItem


def _make_response(
    content: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> GeminiResponse:
    """Create a GeminiResponse for testing."""
    return GeminiResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=100.0,
        finish_reason="STOP",
    )


def _make_loop(
    generate_side_effect=None,
    system_prompt=None,
    interrupt_config=None,
    on_completion=None,
) -> tuple[ConsciousLoop, MagicMock, ContextQueue]:
    """Create a ConsciousLoop with a mocked GeminiClient."""
    client = MagicMock(spec=GeminiClient)
    if generate_side_effect is not None:
        client.generate.side_effect = generate_side_effect
    else:
        client.generate.return_value = _make_response()
    queue = ContextQueue()
    loop = ConsciousLoop(
        client,
        queue,
        system_prompt=system_prompt,
        interrupt_config=interrupt_config,
        on_completion=on_completion,
    )
    return loop, client, queue


def _enqueue_on_generate(
    queue: ContextQueue,
    responses: list[GeminiResponse],
    *,
    enqueue_on: set[int] | None = None,
    content: str = "CRITICAL",
    priority: Priority = Priority.CRITICAL,
    source_tool_id: str = "urgent",
):
    """Return a generate side_effect that enqueues items on specified call indices."""
    if enqueue_on is None:
        enqueue_on = {0}
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        idx = call_count
        call_count += 1
        if idx in enqueue_on:
            queue.enqueue(QueueItem(
                content=f"{content} {idx}" if len(enqueue_on) > 1 else content,
                priority=priority,
                source_tool_id=f"{source_tool_id}{idx}" if len(enqueue_on) > 1 else source_tool_id,
                token_count=10,
            ))
        return responses[idx]

    return side_effect


class TestBasicTurnExecution:
    """Basic run_turn, history, and turn counting."""

    def test_single_turn(self):
        loop, client, _ = _make_loop()
        result = loop.run_turn("Hi")

        assert result.content == "Hello!"
        assert result.turn_number == 1
        assert not result.interrupted
        assert not result.context_injected
        client.generate.assert_called_once()

    def test_turn_count_increments(self):
        loop, _, _ = _make_loop()
        loop.run_turn("First")
        loop.run_turn("Second")

        assert loop.turn_count == 2

    def test_history_contains_user_and_model(self):
        loop, _, _ = _make_loop()
        loop.run_turn("Hi")

        history = loop.history
        assert len(history) == 2
        assert history[0] == ChatMessage(role="user", content="Hi")
        assert history[1] == ChatMessage(role="model", content="Hello!")

    def test_history_returns_copy(self):
        loop, _, _ = _make_loop()
        loop.run_turn("Hi")

        h1 = loop.history
        h2 = loop.history
        assert h1 == h2
        assert h1 is not h2

    def test_system_prompt_passed_to_generate(self):
        loop, client, _ = _make_loop(system_prompt="Be helpful")
        loop.run_turn("Hi")

        _, kwargs = client.generate.call_args
        assert kwargs["system_prompt"] == "Be helpful"


class TestBreakpointInjection:
    """Enqueue items, verify they appear in the next generation's context."""

    def test_context_injected_at_breakpoint(self):
        loop, client, queue = _make_loop()

        queue.enqueue(QueueItem(
            content="Fact A",
            priority=Priority.HIGH,
            source_tool_id="tool1",
            token_count=5,
        ))
        queue.enqueue(QueueItem(
            content="Fact B",
            priority=Priority.MEDIUM,
            source_tool_id="tool2",
            token_count=5,
        ))

        result = loop.run_turn("Tell me something")
        assert result.context_injected

        # The user message sent to generate should contain injected context
        call_args = client.generate.call_args
        messages = call_args[0][0]
        last_msg = messages[-1]
        assert "Fact A" in last_msg.content
        assert "Fact B" in last_msg.content
        assert "Tell me something" in last_msg.content

    def test_no_injection_when_queue_empty(self):
        loop, client, _ = _make_loop()
        result = loop.run_turn("Hi")

        assert not result.context_injected
        call_args = client.generate.call_args
        messages = call_args[0][0]
        last_msg = messages[-1]
        assert last_msg.content == "Hi"

    def test_history_stores_original_message(self):
        """History should store the original user message, not the augmented one."""
        loop, _, queue = _make_loop()
        queue.enqueue(QueueItem(
            content="Context",
            priority=Priority.HIGH,
            source_tool_id="tool1",
            token_count=5,
        ))
        loop.run_turn("Hello")

        assert loop.history[0].content == "Hello"


class TestInterrupt:
    """Critical item triggers turn restart with injection."""

    def test_interrupt_triggers_regeneration(self):
        """When a critical item arrives, the turn should be interrupted and regenerated."""
        responses = [
            _make_response("first attempt", input_tokens=10, output_tokens=20),
            _make_response("second attempt", input_tokens=15, output_tokens=25),
        ]
        loop, client, queue = _make_loop(
            interrupt_config=InterruptConfig(priority_threshold=Priority.CRITICAL),
        )
        client.generate.side_effect = _enqueue_on_generate(
            queue, responses, content="CRITICAL UPDATE",
        )

        result = loop.run_turn("Hello")

        assert result.interrupted
        assert result.context_injected
        assert result.content == "second attempt"
        assert client.generate.call_count == 2

    def test_interrupt_injects_context_in_retry(self):
        """The retry generation should include the critical context."""
        responses = [_make_response("first"), _make_response("second")]
        loop, client, queue = _make_loop(
            interrupt_config=InterruptConfig(priority_threshold=Priority.CRITICAL),
        )
        client.generate.side_effect = _enqueue_on_generate(
            queue, responses, content="URGENT INFO",
        )

        loop.run_turn("Hello")

        second_call_msgs = client.generate.call_args_list[1][0][0]
        last_msg = second_call_msgs[-1]
        assert "URGENT INFO" in last_msg.content


class TestMaxOneInterrupt:
    """After 1 interrupt, freeze prevents a second."""

    def test_freeze_prevents_second_interrupt(self):
        """Only one interrupt per turn — freeze suppresses further threshold checks."""
        responses = [_make_response("first"), _make_response("second")]
        loop, client, queue = _make_loop(
            interrupt_config=InterruptConfig(priority_threshold=Priority.CRITICAL),
        )
        client.generate.side_effect = _enqueue_on_generate(
            queue, responses, enqueue_on={0, 1}, content="CRITICAL",
        )

        result = loop.run_turn("Hello")

        # Should only generate twice (one interrupt), not three times
        assert client.generate.call_count == 2
        assert result.interrupted
        assert result.content == "second"


class TestWastedTokens:
    """Wasted tokens are logged correctly on interrupt."""

    def test_wasted_tokens_reported(self):
        responses = [
            _make_response("wasted", input_tokens=50, output_tokens=30),
            _make_response("kept", input_tokens=60, output_tokens=40),
        ]
        loop, client, queue = _make_loop(
            interrupt_config=InterruptConfig(priority_threshold=Priority.CRITICAL),
        )
        client.generate.side_effect = _enqueue_on_generate(queue, responses)

        result = loop.run_turn("Hello")

        assert queue.wasted_tokens == 80  # 50 + 30
        assert result.total_tokens == 80 + 60 + 40  # wasted + final

    def test_total_tokens_without_interrupt(self):
        loop, _, _ = _make_loop()
        result = loop.run_turn("Hi")

        assert result.total_tokens == result.input_tokens + result.output_tokens


class TestTurnMetadata:
    """Turn number, tokens, and duration are accurate."""

    def test_turn_number(self):
        loop, _, _ = _make_loop()
        r1 = loop.run_turn("First")
        r2 = loop.run_turn("Second")

        assert r1.turn_number == 1
        assert r2.turn_number == 2

    def test_token_counts(self):
        loop, _, _ = _make_loop(
            generate_side_effect=[_make_response(input_tokens=100, output_tokens=200)]
        )
        result = loop.run_turn("Hi")

        assert result.input_tokens == 100
        assert result.output_tokens == 200

    def test_duration_is_positive(self):
        loop, _, _ = _make_loop()
        result = loop.run_turn("Hi")

        assert result.duration_ms > 0


class TestOnCompletionCallback:
    """Callback fires for every turn."""

    def test_callback_fires(self):
        callback = MagicMock()
        loop, _, _ = _make_loop(on_completion=callback)
        loop.run_turn("Hi")

        callback.assert_called_once()
        arg = callback.call_args[0][0]
        assert isinstance(arg, AssistantResponse)
        assert arg.turn_number == 1

    def test_callback_fires_on_every_turn(self):
        callback = MagicMock()
        loop, _, _ = _make_loop(on_completion=callback)
        loop.run_turn("First")
        loop.run_turn("Second")

        assert callback.call_count == 2

    def test_callback_receives_correct_response(self):
        responses = [
            _make_response("reply1", input_tokens=5, output_tokens=10),
            _make_response("reply2", input_tokens=15, output_tokens=20),
        ]
        callback = MagicMock()
        loop, _, _ = _make_loop(
            generate_side_effect=responses,
            on_completion=callback,
        )
        loop.run_turn("First")
        loop.run_turn("Second")

        first_arg = callback.call_args_list[0][0][0]
        assert first_arg.content == "reply1"
        assert first_arg.turn_number == 1

        second_arg = callback.call_args_list[1][0][0]
        assert second_arg.content == "reply2"
        assert second_arg.turn_number == 2


class TestRegenerateWithContext:
    """Tests for regenerate_with_context()."""

    def test_history_updated(self):
        """After regeneration, history contains the new response, not the old one."""
        responses = [_make_response("original"), _make_response("regenerated")]
        loop, client, _ = _make_loop(generate_side_effect=responses)
        loop.run_turn("Hello")

        assert loop.history[-1].content == "original"

        client.generate.return_value = _make_response("regenerated")
        result = loop.regenerate_with_context("new context info")

        assert result.content == "regenerated"
        assert result.context_injected
        assert loop.history[-1].content == "regenerated"
        # History length unchanged: still user + model
        assert len(loop.history) == 2

    def test_turn_count_unchanged(self):
        """regenerate_with_context does NOT increment turn count."""
        loop, client, _ = _make_loop()
        loop.run_turn("Hello")
        assert loop.turn_count == 1

        client.generate.return_value = _make_response("regen")
        loop.regenerate_with_context("context")
        assert loop.turn_count == 1

    def test_context_injected_flag(self):
        loop, client, _ = _make_loop()
        loop.run_turn("Hello")

        client.generate.return_value = _make_response("regen")
        result = loop.regenerate_with_context("injected context")
        assert result.context_injected is True

    def test_context_appears_in_generation(self):
        """The context string should appear in the generate call."""
        loop, client, _ = _make_loop()
        loop.run_turn("Hello")

        client.generate.return_value = _make_response("regen")
        loop.regenerate_with_context("CRITICAL_INFO")

        # Last generate call should include the context
        call_args = client.generate.call_args
        messages = call_args[0][0]
        last_msg = messages[-1]
        assert "CRITICAL_INFO" in last_msg.content

    def test_error_no_model_message(self):
        """Raises ValueError if no model message to replace."""
        loop, _, _ = _make_loop()
        with pytest.raises(ValueError, match="No model message"):
            loop.regenerate_with_context("context")

    def test_multi_turn_regeneration(self):
        """Regeneration works correctly after multiple turns."""
        responses = [_make_response("r1"), _make_response("r2"), _make_response("r2_regen")]
        loop, client, _ = _make_loop(generate_side_effect=responses)
        loop.run_turn("msg1")
        loop.run_turn("msg2")

        assert loop.turn_count == 2
        assert len(loop.history) == 4

        client.generate.return_value = _make_response("r2_regen")
        result = loop.regenerate_with_context("new info")

        assert result.content == "r2_regen"
        assert loop.turn_count == 2  # unchanged
        assert len(loop.history) == 4  # unchanged
        assert loop.history[-1].content == "r2_regen"
        assert loop.history[-2].content == "msg2"


class TestMultiTurnConversation:
    """History grows across turns."""

    def test_history_accumulates(self):
        responses = [
            _make_response("reply1"),
            _make_response("reply2"),
            _make_response("reply3"),
        ]
        loop, _, _ = _make_loop(generate_side_effect=responses)

        loop.run_turn("msg1")
        loop.run_turn("msg2")
        loop.run_turn("msg3")

        history = loop.history
        assert len(history) == 6  # 3 user + 3 model
        assert history[0].content == "msg1"
        assert history[1].content == "reply1"
        assert history[2].content == "msg2"
        assert history[3].content == "reply2"
        assert history[4].content == "msg3"
        assert history[5].content == "reply3"

    def test_prior_history_sent_to_generate(self):
        """Later turns should include full prior history in generate calls."""
        responses = [_make_response("r1"), _make_response("r2")]
        loop, client, _ = _make_loop(generate_side_effect=responses)

        loop.run_turn("msg1")
        loop.run_turn("msg2")

        # Second call should have 3 messages: user1, model1, user2
        second_call_msgs = client.generate.call_args_list[1][0][0]
        assert len(second_call_msgs) == 3
        assert second_call_msgs[0].content == "msg1"
        assert second_call_msgs[1].content == "r1"
        assert second_call_msgs[2].content == "msg2"


class TestIntegration:
    """Live integration test with actual Gemini API."""

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    )
    def test_live_multi_turn(self):
        client = GeminiClient()
        queue = ContextQueue()
        loop = ConsciousLoop(
            client,
            queue,
            system_prompt="You are a helpful assistant. Keep responses brief.",
            thinking_level="minimal",
        )

        # Turn 1
        r1 = loop.run_turn("What is 2 + 2? Reply with just the number.")
        assert r1.turn_number == 1
        assert "4" in r1.content
        assert r1.input_tokens > 0
        assert r1.output_tokens > 0

        # Turn 2 with context injection
        queue.enqueue(QueueItem(
            content="The user prefers metric units.",
            priority=Priority.MEDIUM,
            source_tool_id="prefs",
            token_count=10,
        ))
        r2 = loop.run_turn("What is the boiling point of water? Just the number and unit.")
        assert r2.turn_number == 2
        assert r2.context_injected
        assert r2.input_tokens > 0

        assert loop.turn_count == 2
        assert len(loop.history) == 4
