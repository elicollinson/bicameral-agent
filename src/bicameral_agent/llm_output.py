"""Safe parsing and sanitization of structured LLM output (issue #47).

LLM responses routinely come back malformed: truncated at the
``max_output_tokens`` cap (``finish_reason=MAX_TOKENS``), wrapped in prose
preamble, or carrying out-of-range numerics. A single such response must not
crash an episode -- callers degrade to a no-op result or neutral score instead.

Also hosts the small text helpers shared across LLM-facing modules
(``tokenize``, ``format_conversation``), alongside ``clamp``/``coerce_int``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from bicameral_agent.schema import Message

logger = logging.getLogger(__name__)


def safe_parse_json(response: Any, *, context: str, default: dict | None = None) -> dict | None:
    """Parse a structured-output LLM response into a dict, degrading on failure.

    Tries ``json.loads`` on ``response.content``, then falls back to extracting
    the first balanced JSON object (handles prose preamble). On failure -- or if
    the parsed value is not a dict -- logs a warning that includes the
    response's ``finish_reason`` (which flags truncation at the output-token
    cap) and returns *default* instead of raising.

    Args:
        response: A ``GeminiResponse``-shaped object with ``content`` and
            ``finish_reason`` attributes.
        context: Caller name for the warning log.
        default: Value returned when no dict can be recovered.

    Returns:
        The parsed dict, or *default*.
    """
    text = getattr(response, "content", None) or ""
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start != -1:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(text, start)
            except json.JSONDecodeError:
                pass

    if isinstance(parsed, dict):
        return parsed

    logger.warning(
        "%s: unparseable structured LLM output (finish_reason=%s, %d chars); "
        "degrading to default",
        context,
        getattr(response, "finish_reason", "unknown"),
        len(text),
    )
    report_degradation(context)
    return default


def report_degradation(component: str) -> None:
    """Record one degradation for *component* with every active counter.

    Shared by ``safe_parse_json`` and other degrade-to-default sites (e.g.
    the Brave search provider's outage fallback, issue #100) so all
    degradations land in the same episode-scoped counters.
    """
    for counter in _active_counters.get():
        counter.add(component)


class DegradationCounter:
    """Tallies ``safe_parse_json`` degradations per component (issue #82).

    ``safe_parse_json`` reports each degradation (tagged with the caller's
    *context* string) to every counter active in the current ``contextvars``
    context; attach one via ``count_degradations``. Counters live in a
    ContextVar rather than on a module-level logger, so concurrently running
    episodes (``run_episode`` on a ThreadPool, issue #91) never observe each
    other's degradations -- each episode's worker task runs in its own
    context. Increments are lock-guarded because a scorer's own worker
    threads (which run in a *copy* of the episode's context) may report
    concurrently.
    """

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def add(self, component: str) -> None:
        """Record one degradation attributed to *component*."""
        with self._lock:
            self.counts[component] += 1


_active_counters: ContextVar[tuple[DegradationCounter, ...]] = ContextVar(
    "llm_output_degradation_counters", default=()
)


@contextmanager
def count_degradations() -> Iterator[DegradationCounter]:
    """Count safe-parse degradations occurring inside the ``with`` block.

    Yields a ``DegradationCounter`` whose ``counts`` maps component context
    strings (e.g. ``"TaskScorer"``) to degradation counts. Counters stack:
    nested blocks count independently, with the outer counter still seeing
    the inner block's degradations. The stack lives in a ContextVar that is
    reset on exit, so counters are scoped to their own context -- concurrent
    episodes each see only their own degradations, and threads spawned with
    a copied context (``contextvars.copy_context``) inherit their episode's
    counters.
    """
    counter = DegradationCounter()
    token = _active_counters.set(_active_counters.get() + (counter,))
    try:
        yield counter
    finally:
        _active_counters.reset(token)


def clamp(value: Any, low: float, high: float, default: float) -> float:
    """Coerce *value* to a float clamped to [low, high]; *default* if non-numeric."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(low, min(high, v))


def coerce_int(value: Any, default: int) -> int:
    """Coerce *value* to an int; *default* if non-numeric."""
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, filter empty.

    Shared lexical tokenizer used by the lexical scorers and the mock
    search provider (issue #54).
    """
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def format_conversation(history: list[Message]) -> str:
    """Format the last 10 messages as ``[role]: content`` lines.

    Shared prompt-context formatter used by the gap scanner and assumption
    auditor (issue #54). The simulated user keeps its own variant with
    different selection and labeling semantics.
    """
    return "\n".join(f"[{msg.role}]: {msg.content}" for msg in history[-10:])
