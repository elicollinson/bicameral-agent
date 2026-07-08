"""Cost tracking and budget enforcement for model API calls.

Monitors API spend across all calls and enforces configurable session-level
and episode-level budget limits. Thread-safe via ``threading.Lock``.
"""

from __future__ import annotations

import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass

from bicameral_agent.model_client import PROVIDERS, ModelClient, ModelResponse

logger = logging.getLogger(__name__)


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

_FLAT_RATE = ModelPricing(input_cost_per_token=0.0, output_cost_per_token=0.0)

# Tags already warned about, so flat-rate fallbacks log once per process.
_warned_flat_rate: set[str] = set()


def resolve_pricing(model: str, provider: str | None = None) -> ModelPricing:
    """Resolve per-token pricing for *model*, failing fast on unknown tags.

    Tags registered in ``MODEL_PRICING`` resolve directly. Unknown tags on a
    flat-rate (subscription) provider fall back to $0/token with a one-time
    warning, so any Ollama Cloud ``--model`` override works without a
    registry edit. Unknown tags on metered providers raise, so a typo'd
    Gemini tag is rejected at client-build time -- never after a paid call.

    Raises:
        ValueError: If *model* is unpriced and *provider* is not flat-rate.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is not None:
        return pricing

    spec = PROVIDERS.get(provider) if provider is not None else None
    if spec is not None and spec.flat_rate:
        if model not in _warned_flat_rate:
            _warned_flat_rate.add(model)
            logger.warning(
                "No pricing registered for model %r; provider %r is "
                "subscription-flat, recording $0/token.",
                model,
                provider,
            )
        return _FLAT_RATE

    raise ValueError(
        f"Unknown model {model!r}; known models: {sorted(MODEL_PRICING)}"
    )


@dataclass(frozen=True, slots=True)
class CostReport:
    """Snapshot of accumulated costs."""

    input_cost: float
    output_cost: float
    total: float
    call_count: int


class CostBudgetExceeded(Exception):
    """Raised when the accumulated cost exceeds the configured budget."""


class _EpisodeCosts:
    """Mutable per-episode accumulator; mutation is guarded by the tracker lock."""

    __slots__ = ("input_cost", "output_cost", "call_count")

    def __init__(self) -> None:
        self.input_cost = 0.0
        self.output_cost = 0.0
        self.call_count = 0


class CostTracker:
    """Thread-safe cost tracker with session and episode accumulators.

    Session accumulators persist for the lifetime of the tracker.
    Episode accumulators can be reset between episodes via ``reset_episode()``.

    The episode accumulator is context-local (issue #91): concurrent
    episodes each run in their own ``contextvars`` context, so one
    episode's ``reset_episode``/``get_episode_cost`` never touches
    another's accumulator, while threads an episode spawns with a copied
    context (e.g. the scorer's worker pool) still attribute their calls to
    that episode. Session accumulators remain shared across all contexts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Session-level (never reset)
        self._session_input_cost: float = 0.0
        self._session_output_cost: float = 0.0
        self._session_call_count: int = 0
        # Episode-level (reset via reset_episode); context-local, one
        # ContextVar per tracker so independent trackers stay isolated.
        self._episode_costs: ContextVar[_EpisodeCosts | None] = ContextVar(
            f"episode_costs_{id(self)}", default=None
        )
        # Budget limits
        self._session_budget: float | None = None
        self._episode_budget: float | None = None

    def _episode(self) -> _EpisodeCosts:
        """The current context's episode accumulator, created on first use."""
        costs = self._episode_costs.get()
        if costs is None:
            costs = _EpisodeCosts()
            self._episode_costs.set(costs)
        return costs

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        pricing: ModelPricing | None = None,
    ) -> None:
        """Record a completed API call and accumulate costs.

        Args:
            input_tokens: Prompt tokens consumed by the call.
            output_tokens: Completion tokens produced by the call.
            model: Model tag the call was made against.
            pricing: Pre-resolved pricing (skips the registry lookup).

        Raises:
            ValueError: If *pricing* is omitted and *model* is not in
                ``MODEL_PRICING``.
        """
        if pricing is None:
            pricing = resolve_pricing(model)
        input_cost = input_tokens * pricing.input_cost_per_token
        output_cost = output_tokens * pricing.output_cost_per_token

        with self._lock:
            self._session_input_cost += input_cost
            self._session_output_cost += output_cost
            self._session_call_count += 1
            episode = self._episode()
            episode.input_cost += input_cost
            episode.output_cost += output_cost
            episode.call_count += 1

    def check_budget(self) -> None:
        """Raise ``CostBudgetExceeded`` if session or episode budget is exceeded."""
        if self._session_budget is None and self._episode_budget is None:
            return
        with self._lock:
            session_total = self._session_input_cost + self._session_output_cost
            episode = self._episode_costs.get()
            episode_total = (
                episode.input_cost + episode.output_cost if episode is not None else 0.0
            )

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
        """Return the current context's episode-level cost report."""
        with self._lock:
            return self._episode_report()

    def _episode_report(self) -> CostReport:
        """Snapshot the current context's episode costs (call under the lock)."""
        episode = self._episode_costs.get()
        if episode is None:
            return CostReport(input_cost=0.0, output_cost=0.0, total=0.0, call_count=0)
        return CostReport(
            input_cost=episode.input_cost,
            output_cost=episode.output_cost,
            total=episode.input_cost + episode.output_cost,
            call_count=episode.call_count,
        )

    def set_budget(self, max_dollars: float | None) -> None:
        """Set the session-level budget (``None`` to disable)."""
        self._session_budget = max_dollars

    def set_episode_budget(self, max_dollars: float | None) -> None:
        """Set the episode-level budget (``None`` to disable)."""
        self._episode_budget = max_dollars

    def reset_episode(self) -> CostReport:
        """Return the current context's episode report and start a fresh one."""
        with self._lock:
            report = self._episode_report()
            self._episode_costs.set(_EpisodeCosts())
            return report


class CostTrackedClient:
    """Wraps a model client with cost tracking and budget enforcement.

    Calls ``check_budget()`` before each ``generate()`` and
    ``record_call()`` after, following the same wrapper pattern as
    ``_TrackedClient`` in ``tool_primitive.py``.

    Pricing for the wrapped client's model is resolved once at construction,
    so an unpriced tag fails fast here -- never after a paid call.
    """

    def __init__(self, inner: ModelClient, tracker: CostTracker) -> None:
        self._inner = inner
        self._tracker = tracker
        self._pricing = resolve_pricing(
            inner.model, provider=getattr(inner, "provider", None)
        )

    @property
    def model(self) -> str:
        """Expose the underlying client's model name."""
        return self._inner.model

    def generate(self, *args, **kwargs) -> ModelResponse:
        """Generate with pre-call budget check and post-call cost recording."""
        self._tracker.check_budget()
        response = self._inner.generate(*args, **kwargs)
        self._tracker.record_call(
            response.input_tokens,
            response.output_tokens,
            self._inner.model,
            pricing=self._pricing,
        )
        return response
