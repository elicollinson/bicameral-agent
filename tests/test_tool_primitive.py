"""Tests for the ToolPrimitive base class and supporting types."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from bicameral_agent.gemini import GeminiResponse
from bicameral_agent.queue import Priority, QueueItem
from bicameral_agent.schema import Message
from bicameral_agent.tool_primitive import (
    BudgetExceededError,
    StateVector,
    TokenBudget,
    ToolMetadata,
    ToolPrimitive,
    ToolResult,
    _TokenTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_BUDGET = TokenBudget(max_calls=5, max_input_tokens=1000, max_output_tokens=1000)


def _make_messages() -> list[Message]:
    return [
        Message(role="user", content="hello", timestamp_ms=1000, token_count=5),
        Message(role="assistant", content="hi there", timestamp_ms=2000, token_count=8),
    ]


def _make_state() -> StateVector:
    return np.zeros(53, dtype=np.float32)


def _fake_response(input_tokens: int = 10, output_tokens: int = 20) -> GeminiResponse:
    return GeminiResponse(
        content="response",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=50.0,
        finish_reason="STOP",
    )


class EchoTool(ToolPrimitive):
    """Minimal concrete implementation that echoes the last user message."""

    def __init__(self) -> None:
        super().__init__("echo_tool")

    def _execute(self, conversation_history, reasoning_state, client):
        last_user = ""
        for msg in reversed(conversation_history):
            if msg.role == "user":
                last_user = msg.content
                break

        resp = client.generate(
            [{"role": "user", "content": last_user}],
            system_prompt="Echo the input exactly.",
        )

        return ToolResult(
            queue_deposit=QueueItem(
                content=resp.content,
                priority=Priority.MEDIUM,
                source_tool_id=self.tool_id,
                token_count=resp.output_tokens,
            ),
            metadata=ToolMetadata(
                tool_id=self.tool_id,
                action_taken="echoed user message",
                confidence=0.95,
                items_found=1,
                estimated_relevance=0.8,
            ),
        )


class GreedyTool(ToolPrimitive):
    """A tool that makes too many LLM calls, exceeding the budget."""

    def __init__(self, calls_to_make: int = 5) -> None:
        super().__init__("greedy_tool")
        self._calls_to_make = calls_to_make

    def _execute(self, conversation_history, reasoning_state, client):
        for _ in range(self._calls_to_make):
            client.generate([{"role": "user", "content": "tick"}])
        return ToolResult(
            metadata=ToolMetadata(
                tool_id=self.tool_id,
                action_taken="greedy",
                confidence=0.5,
                items_found=0,
                estimated_relevance=0.0,
            ),
        )


def _mock_client(response: GeminiResponse | None = None) -> MagicMock:
    client = MagicMock(spec=["generate"])
    client.generate.return_value = response or _fake_response()
    return client


# ---------------------------------------------------------------------------
# Tests: Abstract base class enforces execute() method signature
# ---------------------------------------------------------------------------

class TestAbstractEnforcement:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError, match="abstract"):
            ToolPrimitive("test")  # type: ignore[abstract]

    def test_subclass_must_implement_execute(self):
        class BadTool(ToolPrimitive):
            pass

        with pytest.raises(TypeError, match="abstract"):
            BadTool("test")  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        tool = EchoTool()
        assert tool.tool_id == "echo_tool"


# ---------------------------------------------------------------------------
# Tests: Duration is automatically measured
# ---------------------------------------------------------------------------

class TestAutoMeasurement:
    def test_duration_populated(self):
        tool = EchoTool()
        result = tool.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, _mock_client())
        assert result.metadata.execution_duration_ms >= 0

    def test_duration_reflects_wall_clock(self):
        class SlowTool(ToolPrimitive):
            def __init__(self):
                super().__init__("slow_tool")

            def _execute(self, conversation_history, reasoning_state, client):
                time.sleep(0.05)
                return ToolResult(
                    metadata=ToolMetadata(
                        tool_id=self.tool_id,
                        action_taken="slept",
                        confidence=1.0,
                        items_found=0,
                        estimated_relevance=0.0,
                    ),
                )

        tool = SlowTool()
        result = tool.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, _mock_client())
        assert result.metadata.execution_duration_ms >= 40


# ---------------------------------------------------------------------------
# Tests: Token tracking correctly sums all internal LLM calls
# ---------------------------------------------------------------------------

class TestTokenTracking:
    def test_tokens_consumed_single_call(self):
        tool = EchoTool()
        client = _mock_client(_fake_response(input_tokens=10, output_tokens=20))
        result = tool.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)
        assert result.metadata.tokens_consumed == 30

    def test_tokens_consumed_multiple_calls(self):
        class MultiCallTool(ToolPrimitive):
            def __init__(self):
                super().__init__("multi_tool")

            def _execute(self, conversation_history, reasoning_state, client):
                client.generate([{"role": "user", "content": "a"}])
                client.generate([{"role": "user", "content": "b"}])
                client.generate([{"role": "user", "content": "c"}])
                return ToolResult(
                    metadata=ToolMetadata(
                        tool_id=self.tool_id,
                        action_taken="multi",
                        confidence=0.5,
                        items_found=0,
                        estimated_relevance=0.0,
                    ),
                )

        tool = MultiCallTool()
        client = _mock_client(_fake_response(input_tokens=10, output_tokens=20))
        result = tool.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)
        assert result.metadata.tokens_consumed == 90


# ---------------------------------------------------------------------------
# Tests: Budget enforcement
# ---------------------------------------------------------------------------

class TestBudgetEnforcement:
    def test_exceeding_max_calls_raises(self):
        tool = GreedyTool(calls_to_make=5)
        with pytest.raises(BudgetExceededError, match="max_calls"):
            tool.execute(
                _make_messages(), _make_state(),
                TokenBudget(max_calls=2, max_input_tokens=100000, max_output_tokens=100000),
                _mock_client(),
            )

    def test_exceeding_max_input_tokens_raises(self):
        tool = GreedyTool(calls_to_make=3)
        with pytest.raises(BudgetExceededError, match="max_input_tokens"):
            tool.execute(
                _make_messages(), _make_state(),
                TokenBudget(max_calls=10, max_input_tokens=600, max_output_tokens=100000),
                _mock_client(_fake_response(input_tokens=500, output_tokens=10)),
            )

    def test_exceeding_max_output_tokens_raises(self):
        tool = GreedyTool(calls_to_make=3)
        with pytest.raises(BudgetExceededError, match="max_output_tokens"):
            tool.execute(
                _make_messages(), _make_state(),
                TokenBudget(max_calls=10, max_input_tokens=100000, max_output_tokens=600),
                _mock_client(_fake_response(input_tokens=10, output_tokens=500)),
            )

    def test_within_budget_succeeds(self):
        tool = EchoTool()
        client = _mock_client(_fake_response(input_tokens=10, output_tokens=20))
        result = tool.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)
        assert result.metadata.tokens_consumed == 30

    def test_exact_budget_boundary_succeeds(self):
        """A tool using exactly max_calls should succeed."""
        tool = GreedyTool(calls_to_make=2)
        result = tool.execute(
            _make_messages(), _make_state(),
            TokenBudget(max_calls=2, max_input_tokens=100000, max_output_tokens=100000),
            _mock_client(),
        )
        assert result.metadata.tokens_consumed == 60  # 2 * (10 + 20)

    def test_max_calls_blocks_before_over_budget_call(self):
        """Budget check should prevent the over-budget call from executing."""
        client = _mock_client()
        tool = GreedyTool(calls_to_make=3)
        with pytest.raises(BudgetExceededError, match="max_calls"):
            tool.execute(
                _make_messages(), _make_state(),
                TokenBudget(max_calls=2, max_input_tokens=100000, max_output_tokens=100000),
                client,
            )
        # The 3rd call should have been blocked before executing
        assert client.generate.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Latency logging callback
# ---------------------------------------------------------------------------

class TestLatencyLogging:
    def test_logging_fires_per_call(self, caplog):
        class TwoCallTool(ToolPrimitive):
            def __init__(self):
                super().__init__("two_call")

            def _execute(self, conversation_history, reasoning_state, client):
                client.generate([{"role": "user", "content": "a"}])
                client.generate([{"role": "user", "content": "b"}])
                return ToolResult(
                    metadata=ToolMetadata(
                        tool_id=self.tool_id,
                        action_taken="two calls",
                        confidence=0.5,
                        items_found=0,
                        estimated_relevance=0.0,
                    ),
                )

        tool = TwoCallTool()
        client = _mock_client(_fake_response(input_tokens=15, output_tokens=25))
        with caplog.at_level(logging.INFO, logger="bicameral_agent.tool_primitive"):
            tool.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        log_lines = [r for r in caplog.records if "tool_llm_call" in r.message]
        assert len(log_lines) == 2
        assert "two_call" in log_lines[0].message
        assert "input_tokens=15" in log_lines[0].message
        assert "output_tokens=25" in log_lines[0].message
        assert "duration_ms=" in log_lines[0].message


# ---------------------------------------------------------------------------
# Tests: Echo tool end-to-end
# ---------------------------------------------------------------------------

class TestEchoTool:
    def test_echo_returns_valid_result(self):
        tool = EchoTool()
        client = _mock_client(_fake_response(input_tokens=5, output_tokens=10))
        result = tool.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, client)

        assert isinstance(result, ToolResult)
        assert result.queue_deposit is not None
        assert result.queue_deposit.source_tool_id == "echo_tool"
        assert result.queue_deposit.content == "response"

        meta = result.metadata
        assert meta.tool_id == "echo_tool"
        assert meta.action_taken == "echoed user message"
        assert meta.confidence == 0.95
        assert meta.items_found == 1
        assert meta.estimated_relevance == 0.8
        assert meta.tokens_consumed == 15
        assert meta.execution_duration_ms >= 0

    def test_no_queue_deposit(self):
        class NoDepositTool(ToolPrimitive):
            def __init__(self):
                super().__init__("no_deposit")

            def _execute(self, conversation_history, reasoning_state, client):
                return ToolResult(
                    metadata=ToolMetadata(
                        tool_id=self.tool_id,
                        action_taken="nothing",
                        confidence=0.5,
                        items_found=0,
                        estimated_relevance=0.0,
                    ),
                )

        tool = NoDepositTool()
        result = tool.execute(_make_messages(), _make_state(), _DEFAULT_BUDGET, _mock_client())
        assert result.queue_deposit is None
        assert result.metadata.tokens_consumed == 0


# ---------------------------------------------------------------------------
# Tests: TokenTracker directly
# ---------------------------------------------------------------------------

class TestTokenTracker:
    def test_tracks_cumulative_usage(self):
        tracker = _TokenTracker("t1", TokenBudget(max_calls=10, max_input_tokens=10000, max_output_tokens=10000))
        tracker.record_completion(100, 200, 50.0)
        tracker.record_completion(150, 250, 60.0)
        assert tracker.total_input_tokens == 250
        assert tracker.total_output_tokens == 450
        assert tracker.total_tokens == 700
        assert tracker.call_count == 2
