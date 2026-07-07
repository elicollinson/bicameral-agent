"""Cost tracking and budget enforcement for Gemini API calls.

Monitors API spend across all calls and enforces configurable session-level
and episode-level budget limits. Thread-safe via ``threading.Lock``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from bicameral_agent.gemini import GeminiClient, GeminiResponse


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-token pricing for a model (in dollars per token)."""

    input_cost_per_token: float
    output_cost_per_token: float


# Pricing in dollars per token (derived from per-million-token rates).
MODEL_PRICING: dict[str, ModelPricing] = {
    "gemini-3.1-flash-lite-preview": ModelPricing(
        input_cost_per_token=0.50 / 1_000_000,
        output_cost_per_token=3.00 / 1_000_000,
    ),
    # Ollama Cloud Gemma is subscription-flat: $0/token so calls/tokens are
    # still recorded (call_count increments) without inventing a per-token rate.
    "gemma4:31b-cloud": ModelPricing(
        input_cost_per_token=0.0,
        output_cost_per_token=0.0,
    ),
}

# Backwards-compatible alias (the dict previously held only Gemini models).
GEMINI_PRICING = MODEL_PRICING


@dataclass(frozen=True, slots=True)
class CostReport:
    """Snapshot of accumulated costs."""

    input_cost: float
    output_cost: float
    total: float
    call_count: int


class CostBudgetExceeded(Exception):
    """Raised when the accumulated cost exceeds the configured budget."""


class CostTracker:
    """Thread-safe cost tracker with session and episode accumulators.

    Session accumulators persist for the lifetime of the tracker.
    Episode accumulators can be reset between episodes via ``reset_episode()``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Session-level (never reset)
        self._session_input_cost: float = 0.0
        self._session_output_cost: float = 0.0
        self._session_call_count: int = 0
        # Episode-level (reset via reset_episode)
        self._episode_input_cost: float = 0.0
        self._episode_output_cost: float = 0.0
        self._episode_call_count: int = 0
        # Budget limits
        self._session_budget: float | None = None
        self._episode_budget: float | None = None

    def record_call(self, input_tokens: int, output_tokens: int, model: str) -> None:
        """Record a completed API call and accumulate costs.

        Raises:
            ValueError: If *model* is not in ``MODEL_PRICING``.
        """
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            raise ValueError(
                f"Unknown model {model!r}; known models: {sorted(MODEL_PRICING)}"
            )
        input_cost = input_tokens * pricing.input_cost_per_token
        output_cost = output_tokens * pricing.output_cost_per_token

        with self._lock:
            self._session_input_cost += input_cost
            self._session_output_cost += output_cost
            self._session_call_count += 1
            self._episode_input_cost += input_cost
            self._episode_output_cost += output_cost
            self._episode_call_count += 1

    def check_budget(self) -> None:
        """Raise ``CostBudgetExceeded`` if session or episode budget is exceeded."""
        if self._session_budget is None and self._episode_budget is None:
            return
        with self._lock:
            session_total = self._session_input_cost + self._session_output_cost
            episode_total = self._episode_input_cost + self._episode_output_cost

        if self._session_budget is not None and session_total >= self._session_budget:
            raise CostBudgetExceeded(
                f"Session cost ${session_total:.6f} >= budget ${self._session_budget:.6f}"
            )
        if self._episode_budget is not None and episode_total >= self._episode_budget:
            raise CostBudgetExceeded(
                f"Episode cost ${episode_total:.6f} >= budget ${self._episode_budget:.6f}"
            )

    def get_total(self) -> CostReport:
        """Return session-level cost report."""
        with self._lock:
            return CostReport(
                input_cost=self._session_input_cost,
                output_cost=self._session_output_cost,
                total=self._session_input_cost + self._session_output_cost,
                call_count=self._session_call_count,
            )

    def get_episode_cost(self) -> CostReport:
        """Return episode-level cost report."""
        with self._lock:
            return CostReport(
                input_cost=self._episode_input_cost,
                output_cost=self._episode_output_cost,
                total=self._episode_input_cost + self._episode_output_cost,
                call_count=self._episode_call_count,
            )

    def set_budget(self, max_dollars: float | None) -> None:
        """Set the session-level budget (``None`` to disable)."""
        self._session_budget = max_dollars

    def set_episode_budget(self, max_dollars: float | None) -> None:
        """Set the episode-level budget (``None`` to disable)."""
        self._episode_budget = max_dollars

    def reset_episode(self) -> CostReport:
        """Return episode report and zero the episode accumulators."""
        with self._lock:
            report = CostReport(
                input_cost=self._episode_input_cost,
                output_cost=self._episode_output_cost,
                total=self._episode_input_cost + self._episode_output_cost,
                call_count=self._episode_call_count,
            )
            self._episode_input_cost = 0.0
            self._episode_output_cost = 0.0
            self._episode_call_count = 0
            return report


class CostTrackedClient:
    """Wraps a ``GeminiClient`` with cost tracking and budget enforcement.

    Calls ``check_budget()`` before each ``generate()`` and
    ``record_call()`` after, following the same wrapper pattern as
    ``_TrackedClient`` in ``tool_primitive.py``.
    """

    def __init__(self, inner: GeminiClient, tracker: CostTracker) -> None:
        self._inner = inner
        self._tracker = tracker

    @property
    def model(self) -> str:
        """Expose the underlying client's model name."""
        return self._inner.model

    def generate(self, *args, **kwargs) -> GeminiResponse:
        """Generate with pre-call budget check and post-call cost recording."""
        self._tracker.check_budget()
        response = self._inner.generate(*args, **kwargs)
        self._tracker.record_call(
            response.input_tokens,
            response.output_tokens,
            self._inner.model,
        )
        return response
