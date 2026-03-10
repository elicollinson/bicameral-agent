"""Abstract base class for intelligent tool primitives.

Defines the shared interface contract that all tool primitives must follow,
ensuring uniform input/output shapes, metadata reporting, token budget
enforcement, and latency logging.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from bicameral_agent.gemini import GeminiClient, GeminiResponse
from bicameral_agent.queue import QueueItem
from bicameral_agent.schema import Message

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Type alias: the 53-dim float32 vector produced by StateEncoder.encode()
StateVector = np.ndarray


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Resource budget constraining a tool's internal LLM usage."""

    max_calls: int
    """Maximum number of internal LLM calls allowed."""

    max_input_tokens: int
    """Maximum total input tokens across all internal calls."""

    max_output_tokens: int
    """Maximum total output tokens across all internal calls."""


class ToolMetadata(BaseModel):
    """Metadata reported by a tool after execution."""

    tool_id: str
    """Identifier of the tool that produced this metadata."""

    action_taken: str
    """Human-readable description of the action performed."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Confidence in the result, in [0.0, 1.0]."""

    items_found: int = Field(ge=0)
    """Number of items discovered or produced."""

    estimated_relevance: float = Field(ge=0.0, le=1.0)
    """Estimated relevance of the result to the current reasoning state."""

    tokens_consumed: int = Field(ge=0)
    """Total tokens consumed across all internal LLM calls."""

    execution_duration_ms: int = Field(ge=0)
    """Wall-clock execution duration in milliseconds."""


class ToolResult(BaseModel):
    """Output of a tool primitive execution."""

    queue_deposit: QueueItem | None = None
    """Optional item to deposit into the context injection queue."""

    metadata: ToolMetadata
    """Metadata about the execution."""


class BudgetExceededError(Exception):
    """Raised when a tool exceeds its token budget."""


class _TokenTracker:
    """Tracks token usage and call count for budget enforcement.

    Used as the on_completion callback for a GeminiClient scoped to a
    single tool execution.
    """

    def __init__(self, tool_id: str, budget: TokenBudget) -> None:
        self.tool_id = tool_id
        self.budget = budget
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0

    def on_completion(self, input_tokens: int, output_tokens: int, duration_ms: float) -> None:
        """Callback fired after every internal LLM call."""
        self.call_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Log per-call data for the latency model
        logger.info(
            "tool_llm_call: tool_id=%s, input_tokens=%d, output_tokens=%d, duration_ms=%.1f",
            self.tool_id,
            input_tokens,
            output_tokens,
            duration_ms,
        )

        # Budget enforcement
        if self.call_count > self.budget.max_calls:
            raise BudgetExceededError(
                f"Tool {self.tool_id!r} exceeded max_calls budget: "
                f"{self.call_count} > {self.budget.max_calls}"
            )
        if self.total_input_tokens > self.budget.max_input_tokens:
            raise BudgetExceededError(
                f"Tool {self.tool_id!r} exceeded max_input_tokens budget: "
                f"{self.total_input_tokens} > {self.budget.max_input_tokens}"
            )
        if self.total_output_tokens > self.budget.max_output_tokens:
            raise BudgetExceededError(
                f"Tool {self.tool_id!r} exceeded max_output_tokens budget: "
                f"{self.total_output_tokens} > {self.budget.max_output_tokens}"
            )

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


class ToolPrimitive(abc.ABC):
    """Abstract base class for intelligent tool primitives.

    Subclasses implement ``_execute()`` with their tool-specific logic.
    The base class wraps execution to automatically:
    - Measure wall-clock duration
    - Track token consumption via the GeminiClient callback
    - Enforce the token budget
    - Log per-call latency data
    """

    def __init__(self, tool_id: str) -> None:
        self._tool_id = tool_id

    @property
    def tool_id(self) -> str:
        return self._tool_id

    def execute(
        self,
        conversation_history: list[Message],
        reasoning_state: StateVector,
        budget: TokenBudget,
        client: GeminiClient,
    ) -> ToolResult:
        """Execute the tool with automatic timing, tracking, and budget enforcement.

        Args:
            conversation_history: The current conversation messages.
            reasoning_state: The encoded state vector from StateEncoder.
            budget: Token budget constraining internal LLM usage.
            client: A GeminiClient instance for making LLM calls.

        Returns:
            ToolResult with optional queue deposit and metadata.

        Raises:
            BudgetExceededError: If the tool exceeds its token budget.
        """
        tracker = _TokenTracker(self._tool_id, budget)

        # Create a client with the tracker callback wired in
        tracked_client = _TrackedGeminiClient(client, tracker)

        start_ns = time.monotonic_ns()
        result = self._execute(
            conversation_history=conversation_history,
            reasoning_state=reasoning_state,
            budget=budget,
            client=tracked_client,
        )
        duration_ms = int((time.monotonic_ns() - start_ns) / 1_000_000)

        # Populate auto-measured fields
        result.metadata.tokens_consumed = tracker.total_tokens
        result.metadata.execution_duration_ms = duration_ms

        return result

    @abc.abstractmethod
    def _execute(
        self,
        conversation_history: list[Message],
        reasoning_state: StateVector,
        budget: TokenBudget,
        client: GeminiClient,
    ) -> ToolResult:
        """Tool-specific execution logic. Implement in subclasses.

        The ``client`` provided here has budget tracking wired in.
        Metadata fields ``tokens_consumed`` and ``execution_duration_ms``
        will be overwritten by the base class after this method returns.
        """
        ...


class _TrackedGeminiClient(GeminiClient):
    """A GeminiClient wrapper that chains a token tracker callback.

    Delegates all calls to the underlying client while ensuring the
    tracker's on_completion callback fires for every API call.
    """

    def __init__(self, inner: GeminiClient, tracker: _TokenTracker) -> None:
        # Bypass GeminiClient.__init__ — we delegate to the inner client
        object.__init__(self)
        self._inner = inner
        self._tracker = tracker

    def generate(self, *args, **kwargs) -> GeminiResponse:
        """Delegate to inner client, then fire tracker callback."""
        response = self._inner.generate(*args, **kwargs)
        self._tracker.on_completion(
            response.input_tokens,
            response.output_tokens,
            response.duration_ms,
        )
        return response
