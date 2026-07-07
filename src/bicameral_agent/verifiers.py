"""Verifier registry: pick a scoring backend by metric name (Issue #56).

Formalizes the contract that ``TaskScorer`` / ``LexicalScorer`` already
duck-type -- ``score(task, answer) -> TaskScore`` -- as a plain dict registry
mirroring ``model_client.build_client``. Datasets declare a ``default_metric``
(overridable within their ``supported_metrics``); callers turn the resolved
metric name into a concrete verifier here. Future deterministic/rubric
verifiers (exact match, multiple choice, rubric coverage, ...) register the
same way.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from bicameral_agent.dataset import ResearchQATask
from bicameral_agent.scorer import LexicalScorer, TaskScore, TaskScorer


@runtime_checkable
class Verifier(Protocol):
    """Anything that scores an agent answer against a task."""

    def score(self, task: ResearchQATask, agent_answer: str) -> TaskScore: ...


# client -> verifier factories; ``client`` is any model client satisfying the
# build_client contract (only LLM-backed verifiers use it).
_VERIFIERS: dict[str, Callable[[object | None], Verifier]] = {
    "llm_judge": lambda client: TaskScorer(client=client),
    "lexical": lambda client: LexicalScorer(),
}


def verifier_names() -> list[str]:
    """Names of all registered verification metrics."""
    return sorted(_VERIFIERS)


def build_verifier(metric: str = "llm_judge", client: object | None = None) -> Verifier:
    """Construct the verifier registered under *metric*.

    Args:
        metric: Registry key (see :func:`verifier_names`).
        client: Model client for LLM-backed verifiers; None uses the
            verifier's own default. Ignored by deterministic verifiers.

    Raises:
        ValueError: If *metric* is not registered.
    """
    try:
        factory = _VERIFIERS[metric]
    except KeyError:
        raise ValueError(
            f"Unknown metric {metric!r}; known metrics: {sorted(_VERIFIERS)}"
        ) from None
    return factory(client)
