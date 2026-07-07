"""Baseline performance benchmark across no-subconscious / random / heuristic controllers.

Issue #23: runs each controller against a pool of evaluation tasks, extracts
per-episode metrics, aggregates them with mean / std / 95% CI, and reports
latency-prediction accuracy (mean absolute percentage error) by pairing each
controller decision's predicted latency against the actual tool duration
recorded in the episode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from bicameral_agent.ab_test import MetricSummary, compute_summary, welch_t_test_from_stats
from bicameral_agent.dataset import ResearchQATask
from bicameral_agent.episode_runner import Controller, EpisodeRunner
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.heuristic_controller import Action, DecisionLog, TOOL_IDS
from bicameral_agent.model_client import TransportExhausted
from bicameral_agent.schema import Episode, UserEventType, episode_completed

logger = logging.getLogger(__name__)

CONDITION_NAMES: tuple[str, ...] = ("no_subconscious", "random", "heuristic")

FAILURE_THRESHOLD = 0.3
"""Abort a condition once transport failures exceed this fraction of its episodes."""


def parse_conditions(spec: str) -> tuple[str, ...]:
    """Parse a comma-separated ``--conditions`` value into canonical order.

    Duplicates collapse and the result follows ``CONDITION_NAMES`` order
    regardless of input order, so a resumed run's report sections line up
    with a full run's.

    Raises:
        ValueError: On an empty spec or unknown condition names.
    """
    names = {n.strip() for n in spec.split(",") if n.strip()}
    unknown = sorted(names - set(CONDITION_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown condition(s) {unknown}; valid: {list(CONDITION_NAMES)}"
        )
    if not names:
        raise ValueError(f"No conditions given; valid: {list(CONDITION_NAMES)}")
    return tuple(c for c in CONDITION_NAMES if c in names)


@dataclass(frozen=True, slots=True)
class EpisodeFailure:
    """A per-episode transport failure contained by :func:`run_condition`."""

    condition: str
    episode_index: int
    task_id: str
    error: str


class ConditionAbortedError(RuntimeError):
    """Transport failures exceeded the threshold for one condition.

    Raised instead of silently finishing a near-empty condition when the
    network/provider is effectively down (issue #81). Resume the aborted
    conditions with ``--conditions`` once the transport recovers.
    """

    def __init__(self, condition: str, n_failures: int, n_tasks: int, threshold: float) -> None:
        self.condition = condition
        self.n_failures = n_failures
        self.n_tasks = n_tasks
        super().__init__(
            f"Aborting condition {condition!r}: {n_failures}/{n_tasks} episodes "
            f"({n_failures / n_tasks:.0%}) failed on transport errors, exceeding "
            f"the {threshold:.0%} threshold; check the network/provider and "
            f"resume with --conditions {condition}."
        )


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    """Metrics extracted from a single episode + its decision log."""

    quality_score: float | None
    total_tokens: int
    total_turns: int
    user_stops: int
    task_completed: int
    """1 if the simulated user marked the task complete, else 0."""
    wall_clock_ms: int
    tool_invocation_count: int
    tool_cost_usd: float
    avg_queue_depth: float
    interrupt_count: int
    drain_count: int
    expired_count: int
    latency_pairs: tuple[tuple[float, float], ...]
    """Per-tool-invocation pairs of (predicted_ms, actual_ms)."""


def _avg_queue_depth(decisions: list[DecisionLog]) -> float:
    if not decisions:
        return 0.0
    return sum(d.state.queue_depth for d in decisions) / len(decisions)


def _latency_pairs(
    episode: Episode, decisions: list[DecisionLog]
) -> tuple[tuple[float, float], ...]:
    """Pair each tool invocation with the predicted latency for its action.

    Decisions and tool invocations are paired in order: the nth non-DO_NOTHING
    decision corresponds to the nth ToolInvocation in the episode.
    Budget-exceeded invocations are excluded explicitly (their partial
    durations are not meaningful), as are any zero-duration invocations.
    Pairs whose invocation tool_id does not match the decision's action are
    dropped defensively: a mismatch means the positional alignment is wrong
    and the predicted/actual values belong to different tools.
    """
    tool_decisions = [d for d in decisions if d.action != Action.DO_NOTHING]
    pairs: list[tuple[float, float]] = []
    for decision, inv in zip(tool_decisions, episode.tool_invocations):
        if inv.budget_exceeded:
            continue
        actual = float(inv.completed_at_ms - inv.invoked_at_ms)
        if actual <= 0:
            continue
        tool_id = TOOL_IDS[decision.action]
        if inv.tool_id != tool_id:
            continue
        predicted = decision.state.predicted_latencies.get(tool_id)
        if predicted is None or predicted <= 0:
            continue
        pairs.append((float(predicted), actual))
    return tuple(pairs)


def extract_task_metrics(
    episode: Episode, decisions: list[DecisionLog]
) -> TaskMetrics:
    """Pull all per-episode benchmark metrics from an Episode + its decisions."""
    cost = episode.metadata.get("episode_cost") or {}
    tool_cost = float(cost.get("total", 0.0))
    user_stops = sum(
        1 for e in episode.user_events if e.event_type == UserEventType.STOP
    )
    drain_count = sum(1 for inj in episode.context_injections if inj.consumed)
    return TaskMetrics(
        quality_score=episode.outcome.quality_score,
        total_tokens=episode.outcome.total_tokens,
        total_turns=episode.outcome.total_turns,
        user_stops=user_stops,
        task_completed=int(episode_completed(episode)),
        wall_clock_ms=episode.outcome.wall_clock_ms,
        tool_invocation_count=len(episode.tool_invocations),
        tool_cost_usd=tool_cost,
        avg_queue_depth=_avg_queue_depth(decisions),
        interrupt_count=int(episode.metadata.get("interrupt_count", 0)),
        drain_count=drain_count,
        expired_count=int(episode.metadata.get("expired_queue_items", 0)),
        latency_pairs=_latency_pairs(episode, decisions),
    )


_METRIC_NAMES: tuple[str, ...] = (
    "quality_score",
    "total_tokens",
    "total_turns",
    "user_stops",
    "task_completed",
    "wall_clock_ms",
    "tool_invocation_count",
    "tool_cost_usd",
    "avg_queue_depth",
    "interrupt_count",
    "drain_count",
    "expired_count",
)


def _metric_values(metrics: list[TaskMetrics], name: str) -> list[float]:
    """Pull a metric series; ``quality_score`` drops None entries."""
    if name == "quality_score":
        return [m.quality_score for m in metrics if m.quality_score is not None]
    return [float(getattr(m, name)) for m in metrics]


def latency_mape(metrics: list[TaskMetrics]) -> tuple[float, int]:
    """Mean absolute percentage error of predicted vs actual tool durations.

    Returns (mape_percent, n_pairs). Returns (0.0, 0) if no pairs.
    """
    pairs: list[tuple[float, float]] = []
    for m in metrics:
        pairs.extend(m.latency_pairs)
    if not pairs:
        return (0.0, 0)
    errors = [abs(predicted - actual) / actual for predicted, actual in pairs]
    return (100.0 * sum(errors) / len(errors), len(pairs))


@dataclass(frozen=True, slots=True)
class ConditionReport:
    """Aggregated metrics for one controller condition."""

    condition: str
    n_episodes: int
    summaries: dict[str, MetricSummary]
    latency_mape_percent: float
    latency_n_pairs: int


def aggregate(condition: str, metrics: list[TaskMetrics]) -> ConditionReport:
    """Build a ConditionReport from per-task metrics."""
    summaries = {name: compute_summary(_metric_values(metrics, name)) for name in _METRIC_NAMES}
    mape, n = latency_mape(metrics)
    return ConditionReport(
        condition=condition,
        n_episodes=len(metrics),
        summaries=summaries,
        latency_mape_percent=mape,
        latency_n_pairs=n,
    )


@dataclass
class BenchmarkResult:
    """Full benchmark result: episodes + per-condition reports."""

    episodes: dict[str, list[Episode]] = field(default_factory=dict)
    metrics: dict[str, list[TaskMetrics]] = field(default_factory=dict)
    reports: dict[str, ConditionReport] = field(default_factory=dict)
    failures: dict[str, list[EpisodeFailure]] = field(default_factory=dict)
    """Per-condition transport failures contained during the run."""


ControllerFactory = Callable[[int], Controller]

EpisodeCallback = Callable[[str, int, Episode], None]
"""Called with (condition, episode_index, episode) as each episode completes."""


def run_condition(
    runner: EpisodeRunner,
    tasks: list[ResearchQATask],
    controller_factory: ControllerFactory,
    condition: str = "",
    on_episode: EpisodeCallback | None = None,
    failure_threshold: float = FAILURE_THRESHOLD,
) -> tuple[list[Episode], list[TaskMetrics], list[EpisodeFailure]]:
    """Run one controller condition over the task pool.

    A fresh controller is constructed per episode (via ``controller_factory(idx)``)
    so that decision logs are scoped to a single episode. ``on_episode`` (if
    given) fires after each completed episode, letting callers persist results
    incrementally so a late crash keeps prior episodes.

    A :class:`~bicameral_agent.model_client.TransportExhausted` error (a
    transport failure that outlived the client's retry budget) fails only its
    episode: the failure is recorded and the run continues with the next task.
    Once failures exceed ``failure_threshold`` of the condition's episodes,
    :class:`ConditionAbortedError` is raised instead — a dead network must not
    produce a near-empty "successful" run.
    """
    episodes: list[Episode] = []
    metrics: list[TaskMetrics] = []
    failures: list[EpisodeFailure] = []
    for idx, task in enumerate(tasks):
        controller = controller_factory(idx)
        try:
            episode = runner.run_episode(task, controller)
        except TransportExhausted as exc:
            failures.append(
                EpisodeFailure(
                    condition=condition,
                    episode_index=idx,
                    task_id=task.task_id,
                    error=str(exc),
                )
            )
            logger.warning(
                "Episode %d (task %s) in condition %r failed on transport: %s",
                idx, task.task_id, condition, exc,
            )
            if len(failures) / len(tasks) > failure_threshold:
                raise ConditionAbortedError(
                    condition, len(failures), len(tasks), failure_threshold
                ) from exc
            continue
        episodes.append(episode)
        metrics.append(extract_task_metrics(episode, list(controller.decisions)))
        if on_episode is not None:
            on_episode(condition, idx, episode)
    return episodes, metrics, failures


def _format_summary(summary: MetricSummary, fmt: str) -> str:
    return (
        f"mean={summary.mean:{fmt}} std={summary.std:{fmt}} "
        f"95% CI=[{summary.ci_lower:{fmt}}, {summary.ci_upper:{fmt}}] n={summary.n}"
    )


def run_benchmark(
    client: GeminiClient,
    tasks: list[ResearchQATask],
    conditions: dict[str, ControllerFactory],
    runner: EpisodeRunner | None = None,
    on_episode: EpisodeCallback | None = None,
    failure_threshold: float = FAILURE_THRESHOLD,
) -> BenchmarkResult:
    """Run all conditions over the task pool and aggregate.

    ``on_episode`` is forwarded to :func:`run_condition` for incremental
    persistence of completed episodes, ``failure_threshold`` for the
    per-condition transport-failure abort.
    """
    runner = runner or EpisodeRunner(client)
    result = BenchmarkResult()
    for condition, factory in conditions.items():
        logger.info("Running %d episodes for condition %r", len(tasks), condition)
        episodes, metrics, failures = run_condition(
            runner, tasks, factory, condition=condition, on_episode=on_episode,
            failure_threshold=failure_threshold,
        )
        result.episodes[condition] = episodes
        result.metrics[condition] = metrics
        result.reports[condition] = aggregate(condition, metrics)
        result.failures[condition] = failures
    return result


def heuristic_outperforms(
    reports: dict[str, ConditionReport],
    baseline: str,
    metric: str = "quality_score",
) -> bool:
    """Whether ``heuristic`` significantly exceeds ``baseline`` on ``metric``.

    Requires both a strictly higher mean and a significant Welch's t-test at
    95% confidence — a raw mean difference alone does not declare a winner.
    """
    if "heuristic" not in reports or baseline not in reports:
        return False
    h = reports["heuristic"].summaries[metric]
    b = reports[baseline].summaries[metric]
    if h.mean <= b.mean:
        return False
    _, significant = welch_t_test_from_stats(h.mean, h.std, h.n, b.mean, b.std, b.n)
    return significant


def format_report(result: BenchmarkResult) -> str:
    """Format a human-readable text report for a BenchmarkResult."""
    lines: list[str] = ["=" * 72, "Baseline Performance Benchmark", "=" * 72, ""]
    for condition, report in result.reports.items():
        lines.append(f"## Condition: {condition}  (n={report.n_episodes})")
        for name in _METRIC_NAMES:
            summary = report.summaries[name]
            fmt = (
                ".3f"
                if name in {"quality_score", "tool_cost_usd", "avg_queue_depth", "task_completed"}
                else ".1f"
            )
            lines.append(f"  {name:<24s} {_format_summary(summary, fmt)}")
        lines.append(
            f"  latency_prediction       MAPE={report.latency_mape_percent:.2f}% "
            f"n_pairs={report.latency_n_pairs}"
        )
        failures = result.failures.get(condition, [])
        if failures:
            failed_ids = ", ".join(f.task_id for f in failures)
            lines.append(
                f"  transport_failures       n={len(failures)} (tasks: {failed_ids})"
            )
        lines.append("")

    lines.append("## Comparisons")
    if "heuristic" in result.reports:
        for baseline in ("random", "no_subconscious"):
            if baseline in result.reports:
                ok = heuristic_outperforms(result.reports, baseline)
                lines.append(
                    f"  heuristic > {baseline} on quality_score "
                    f"(Welch 95%): {'YES' if ok else 'NO'}"
                )
    lines.append("")
    return "\n".join(lines)
