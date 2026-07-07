"""Unified evaluation report model (Issue #56).

Consolidates what ``scripts/run_baseline_benchmark.py`` previously wrote to
``summary.json`` as an ad-hoc dict: dataset/metric identity, answerer and
measurement-model provenance (Issue #53), per-condition aggregates, and
per-task results with verification detail.

The serialized shape is backward-compatible with the previous summary.json:
the ``conditions`` mapping still holds ``ConditionReport`` dicts (with
``summaries.<metric>.mean/std/ci_lower/ci_upper/n``), and the
``tasks_per_condition`` / ``max_turns`` keys are preserved, so the ui/ Review
screen keeps reading runs unchanged. New keys are purely additive.
"""

from __future__ import annotations

import dataclasses

from pydantic import BaseModel, Field

from bicameral_agent.baseline_benchmark import BenchmarkResult


class TaskResult(BaseModel):
    """One episode's scored outcome within a benchmark run."""

    task_id: str
    condition: str
    score: float | None = None
    """Normalized [0, 1] quality score -- comparable across metrics."""
    detail: str | None = None
    """Verifier-specific report (from episode.metadata["verification"])."""


class TaskFailure(BaseModel):
    """One episode that failed on a contained transport error (issue #81)."""

    task_id: str
    condition: str
    error: str


class EvalReport(BaseModel):
    """Machine-readable report for one benchmark run."""

    dataset: str
    metric: str
    answerer: dict[str, str]
    measurement: dict[str, str]
    tasks_per_condition: int
    max_turns: int
    conditions: dict[str, dict]
    """Per-condition ConditionReport dicts (summaries with mean/std/CI)."""
    results: list[TaskResult] = Field(default_factory=list)
    failures: list[TaskFailure] = Field(default_factory=list)
    """Episodes skipped after transport retries were exhausted."""

    @classmethod
    def from_benchmark(
        cls,
        result: BenchmarkResult,
        *,
        dataset: str,
        metric: str,
        answerer: dict[str, str],
        measurement: dict[str, str],
        tasks_per_condition: int,
        max_turns: int,
    ) -> EvalReport:
        """Build a report from a completed :class:`BenchmarkResult`."""
        results = [
            TaskResult(
                task_id=str(episode.metadata.get("task_id", "")),
                condition=condition,
                score=episode.outcome.quality_score,
                detail=(episode.metadata.get("verification") or {}).get("detail"),
            )
            for condition, episodes in result.episodes.items()
            for episode in episodes
        ]
        failures = [
            TaskFailure(
                task_id=failure.task_id, condition=condition, error=failure.error
            )
            for condition, condition_failures in result.failures.items()
            for failure in condition_failures
        ]
        return cls(
            dataset=dataset,
            metric=metric,
            answerer=answerer,
            measurement=measurement,
            tasks_per_condition=tasks_per_condition,
            max_turns=max_turns,
            conditions={
                condition: dataclasses.asdict(report)
                for condition, report in result.reports.items()
            },
            results=results,
            failures=failures,
        )

    def to_json(self) -> str:
        """Serialize to the summary.json payload."""
        return self.model_dump_json(indent=2)
