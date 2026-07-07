"""Comparative evaluation harness across all five controller conditions.

Issue #30: runs {no_subconscious, random, heuristic, learned_no_search,
learned_with_search} on a fixed evaluation task set with a *paired* design
(every condition sees the same tasks in the same order), aggregates the
nine comparison metrics with mean / 95% CI, runs pairwise Welch t-tests
with p-values, and breaks results down by task difficulty. Reports export
as JSON (machine-readable) and markdown (human-readable).

Metric mapping
--------------
The issue names nine metrics; each maps to what the episode metadata
actually records (proxies are documented where the named quantity is not
directly measured):

- ``task_quality``: ``episode.outcome.quality_score`` (verifier score).
- ``task_completed``: 1 if the simulated user marked the task complete.
- ``token_efficiency``: ``episode.outcome.total_tokens``. Lower is
  better; raw token consumption is the recorded quantity (no per-token
  quality attribution exists), so this is a consumption proxy for
  efficiency.
- ``user_stops``: count of STOP user events.
- ``time_to_completion_ms``: ``episode.outcome.wall_clock_ms``.
- ``tool_precision``: consumed context injections / tool invocations.
  Proxy: nothing judges whether an injection was *useful*, so
  "deposited context that the conscious loop actually consumed" is the
  recorded stand-in for a useful invocation. Undefined (skipped) for
  episodes with no tool invocations.
- ``interrupt_rate``: interrupts / total turns.
- ``queue_expiry_rate``: expired queue items / (drained + expired)
  queue items. Proxy: the episode records drain and expiry counts but
  not end-of-episode still-pending items per se, so the rate is over
  *resolved* queue items. Undefined when nothing was drained or expired.
- ``latency_mape``: per-episode mean absolute percentage error of
  predicted vs actual tool latency (from the controller decision log's
  ``predicted_latencies`` paired against ToolInvocation durations).
  Undefined for episodes with no valid prediction/actual pairs.

The issue's checklist says "9 metrics" but names eight; ``task_completed``
(completion rate) is the ninth, matching the baseline benchmark.

Determinism boundary
--------------------
Episode *collection* calls an LLM and is inherently stochastic: re-running
the harness against a live model will not reproduce episode content.
What IS deterministic for a fixed seed and fixed inputs:

- task selection and pairing (``select_tasks`` is order-based, no RNG;
  every condition receives the identical task list in identical order);
- controller-side randomness (per-condition, per-task seeds derived from
  ``base_seed`` via :func:`condition_seed`);
- everything downstream of collected episodes: metric extraction,
  summaries, CIs, Welch t-tests / p-values, difficulty breakdown, and
  both report serializations are pure functions of the episodes.

Transport failures and pairing
------------------------------
:func:`~bicameral_agent.baseline_benchmark.run_condition` contains
per-episode ``TransportExhausted`` failures (issue #81): a failed
episode is recorded, not fatal, until ``failure_threshold`` aborts the
condition. A failure breaks the pair for its task, so *all* statistics
(summary table, pairwise tests, difficulty breakdown) are computed over
the tasks that completed in every condition — dropping the broken pair
for all conditions is the smallest change that keeps the comparison
paired and unbiased (keeping asymmetric samples would let one
condition's failures shift another's means). The dropped task ids are
recorded in the report (``excluded_task_ids``) next to the raw
``failures``; per-task ``results`` rows still include every completed
episode.

Evaluation integrity: judge blinding
------------------------------------
The issue's "human eval blinded" item is N/A here: this framework has no
human-eval pathway — scoring is LLM-judged by design, with the judge
model pinned independently of the answerer (issue #53). The concern
behind blinding — the judge must not know which condition produced an
answer — holds structurally: verifiers score ``(task, agent_answer)``
only, so the judge prompt is built from the task's question / gold
answer / rubric plus the answer text, and the condition name never
reaches the runner (``run_condition`` passes it only to the persistence
callback, not to ``run_episode``). Asserted in tests: the captured
judge prompt contains no condition identity.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections import Counter
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from bicameral_agent.ab_test import MetricSummary, compute_summary, welch_t_test
from bicameral_agent.baseline_benchmark import (
    FAILURE_THRESHOLD,
    ControllerFactory,
    EpisodeCallback,
    EpisodeFailure,
    TaskMetrics,
    run_condition,
)
from bicameral_agent.dataset import ResearchQADataset, ResearchQATask, TaskDifficulty
from bicameral_agent.episode_runner import EpisodeRunner
from bicameral_agent.eval_report import EvalReport, TaskFailure, TaskResult
from bicameral_agent.no_subconscious_controller import NoSubconsciousController
from bicameral_agent.random_controller import RandomController
from bicameral_agent.schema import Episode

logger = logging.getLogger(__name__)

CONDITION_NAMES: tuple[str, ...] = (
    "no_subconscious",
    "random",
    "heuristic",
    "learned_no_search",
    "learned_with_search",
)

METRIC_NAMES: tuple[str, ...] = (
    "task_quality",
    "task_completed",
    "token_efficiency",
    "user_stops",
    "time_to_completion_ms",
    "tool_precision",
    "interrupt_rate",
    "queue_expiry_rate",
    "latency_mape",
)

_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "token_efficiency",
        "user_stops",
        "time_to_completion_ms",
        "interrupt_rate",
        "queue_expiry_rate",
        "latency_mape",
    }
)

_DIFFICULTY_ORDER: tuple[TaskDifficulty, ...] = (
    TaskDifficulty.TYPICAL,
    TaskDifficulty.HARD,
    TaskDifficulty.TRICKY,
)


def condition_seed(base_seed: int, condition: str) -> int:
    """Disjoint per-condition seed block: ``base_seed + 10_000 * index``.

    Task ``idx`` within a condition then uses ``condition_seed(...) + idx``,
    so no two (condition, task) pairs share a seed for up to 10k tasks.
    """
    return base_seed + 10_000 * CONDITION_NAMES.index(condition)


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


def parse_task_mix(spec: str) -> dict[TaskDifficulty, int]:
    """Parse a ``--tasks`` spec into per-difficulty counts.

    A plain integer ``N`` is split 50/25/25 across typical/hard/tricky
    (matching the issue's 100-task layout); explicit counts use
    ``typical=50,hard=25,tricky=25`` (omitted tiers default to 0).
    """
    spec = spec.strip()
    if spec.isdigit():
        total = int(spec)
        typical = total // 2
        hard = total // 4
        return {
            TaskDifficulty.TYPICAL: typical,
            TaskDifficulty.HARD: hard,
            TaskDifficulty.TRICKY: total - typical - hard,
        }
    mix = dict.fromkeys(_DIFFICULTY_ORDER, 0)
    for part in spec.split(","):
        key, _, value = part.partition("=")
        try:
            difficulty = TaskDifficulty(key.strip())
            count = int(value)
        except ValueError:
            raise ValueError(
                f"Invalid task mix component {part!r}; expected e.g. "
                "'typical=50,hard=25,tricky=25' or a plain integer"
            ) from None
        mix[difficulty] = count
    return mix


def select_tasks(
    dataset: ResearchQADataset, mix: dict[TaskDifficulty, int]
) -> list[ResearchQATask]:
    """Deterministic stratified pick of eval tasks per the difficulty mix.

    Takes the first N eval tasks of each tier in dataset order and
    concatenates tiers in typical/hard/tricky order — no RNG, so the
    same dataset + mix always yields the same task list (the pairing
    order shared by every condition). Raises ``ValueError`` if any tier
    is short rather than silently substituting tasks from other tiers.
    """
    eval_tasks = dataset.eval_tasks()
    selected: list[ResearchQATask] = []
    shortfalls: list[str] = []
    for difficulty in _DIFFICULTY_ORDER:
        want = mix.get(difficulty, 0)
        if want <= 0:
            continue
        available = [t for t in eval_tasks if t.difficulty == difficulty]
        if len(available) < want:
            shortfalls.append(f"{difficulty.value}: want {want}, have {len(available)}")
            continue
        selected.extend(available[:want])
    if shortfalls:
        raise ValueError(
            "Dataset cannot satisfy the requested task mix ("
            + "; ".join(shortfalls)
            + ")"
        )
    return selected


# ---------------------------------------------------------------------------
# Condition factories
# ---------------------------------------------------------------------------


def baseline_condition_factories(
    heuristic_factory: Callable[[], object],
    *,
    random_probability: float = 0.2,
    base_seed: int = 42,
) -> dict[str, ControllerFactory]:
    """Factories for the three non-learned conditions.

    ``heuristic_factory`` is a zero-arg constructor for the heuristic
    controller (e.g. ``hyper.to_heuristic_controller``).
    """
    random_base = condition_seed(base_seed, "random")
    return {
        "no_subconscious": lambda _idx: NoSubconsciousController(),
        "random": lambda idx: RandomController(
            action_probability=random_probability, seed=random_base + idx
        ),
        "heuristic": lambda _idx: heuristic_factory(),
    }


def learned_condition_factories(
    policy_checkpoint: str,
    transition_checkpoint: str,
    *,
    num_simulations: int = 50,
    max_turns: int = 25,
    policy_hidden_dim: int = 160,
    transition_hidden_dim: int = 128,
    base_seed: int = 42,
) -> dict[str, ControllerFactory]:
    """Factories for the two learned conditions, loaded from checkpoints.

    Both conditions share one loaded policy network and (for search) one
    transition model; evaluation settings are deterministic (temperature
    0, no root noise). Requires the ``torch`` extra — imports are lazy so
    the rest of this module works without it.
    """
    from bicameral_agent.learned_controller import LearnedPolicyController
    from bicameral_agent.mcts import MCTSEngine
    from bicameral_agent.policy_value_net import PolicyValueNetwork
    from bicameral_agent.training_pipeline import STATE_DIM, TrainingDataPipeline
    from bicameral_agent.transition_model import TransitionModel

    policy = PolicyValueNetwork.load(
        policy_checkpoint, input_dim=STATE_DIM, hidden_dim=policy_hidden_dim
    )
    transition = TransitionModel.load(
        transition_checkpoint, hidden_dim=transition_hidden_dim
    )
    pipeline = TrainingDataPipeline(max_turns=max_turns)
    no_search_base = condition_seed(base_seed, "learned_no_search")
    with_search_base = condition_seed(base_seed, "learned_with_search")

    def no_search(idx: int) -> LearnedPolicyController:
        return LearnedPolicyController(
            policy, pipeline=pipeline, seed=no_search_base + idx
        )

    def with_search(idx: int) -> LearnedPolicyController:
        seed = with_search_base + idx
        return LearnedPolicyController(
            policy,
            mcts_engine=MCTSEngine(policy, transition, seed=seed),
            num_simulations=num_simulations,
            pipeline=pipeline,
            seed=seed,
        )

    return {"learned_no_search": no_search, "learned_with_search": with_search}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ComparativeResult:
    """Paired episodes + extracted metrics for every condition.

    ``episodes[c]`` / ``metrics[c]`` hold only the *completed* episodes
    of condition ``c`` in task order; contained transport failures (issue
    #81) are recorded in ``failures[c]`` with their task index. Use
    :meth:`completed_indices` to map an episode position back to its
    task, and :meth:`paired_metrics` / :attr:`paired_tasks` for the
    subset of tasks completed in *every* condition (the paired design).
    """

    tasks: list[ResearchQATask]
    episodes: dict[str, list[Episode]] = dataclasses.field(default_factory=dict)
    metrics: dict[str, list[TaskMetrics]] = dataclasses.field(default_factory=dict)
    failures: dict[str, list[EpisodeFailure]] = dataclasses.field(default_factory=dict)

    @property
    def task_ids(self) -> list[str]:
        """Shared task order (identical across conditions)."""
        return [t.task_id for t in self.tasks]

    def completed_indices(self, condition: str) -> list[int]:
        """Task indices the condition completed, aligned with its episodes."""
        failed = {f.episode_index for f in self.failures.get(condition, ())}
        return [i for i in range(len(self.tasks)) if i not in failed]

    @property
    def paired_indices(self) -> list[int]:
        """Task indices completed by every condition."""
        failed = {f.episode_index for fs in self.failures.values() for f in fs}
        return [i for i in range(len(self.tasks)) if i not in failed]

    @property
    def paired_tasks(self) -> list[ResearchQATask]:
        """Tasks completed by every condition, in task order."""
        return [self.tasks[i] for i in self.paired_indices]

    @property
    def excluded_task_ids(self) -> list[str]:
        """Tasks dropped from paired analyses (failed in >= 1 condition)."""
        paired = set(self.paired_indices)
        return [t.task_id for i, t in enumerate(self.tasks) if i not in paired]

    def paired_metrics(self) -> dict[str, list[TaskMetrics]]:
        """Per-condition metrics restricted to the paired task subset."""
        paired = set(self.paired_indices)
        return {
            condition: [
                m
                for i, m in zip(self.completed_indices(condition), metrics)
                if i in paired
            ]
            for condition, metrics in self.metrics.items()
        }


class ComparativeEvaluator:
    """Runs every condition over the same task list in the same order."""

    def __init__(
        self,
        runner: EpisodeRunner,
        *,
        on_episode: EpisodeCallback | None = None,
        failure_threshold: float = FAILURE_THRESHOLD,
    ) -> None:
        self._runner = runner
        self._on_episode = on_episode
        self._failure_threshold = failure_threshold

    def run(
        self,
        tasks: Sequence[ResearchQATask],
        conditions: dict[str, ControllerFactory],
    ) -> ComparativeResult:
        """Run all conditions paired over ``tasks``.

        ``conditions`` maps condition name to a per-task controller
        factory (called with the task index). Conditions run in dict
        order; within a condition, tasks run in list order — the same
        list for every condition, which is what makes the comparison
        paired. Per-episode transport failures are contained by
        :func:`~bicameral_agent.baseline_benchmark.run_condition` (which
        raises ``ConditionAbortedError`` past ``failure_threshold``) and
        recorded on the result.
        """
        result = ComparativeResult(tasks=list(tasks))
        for condition, factory in conditions.items():
            logger.info(
                "Running %d episodes for condition %r", len(result.tasks), condition
            )
            episodes, metrics, failures = run_condition(
                self._runner,
                result.tasks,
                factory,
                condition=condition,
                on_episode=self._on_episode,
                failure_threshold=self._failure_threshold,
            )
            result.episodes[condition] = episodes
            result.metrics[condition] = metrics
            result.failures[condition] = failures
        return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_values(metrics: Sequence[TaskMetrics], name: str) -> list[float]:
    """Per-episode values for one comparison metric.

    Episodes where the metric is undefined (see module docstring) are
    skipped, so summary ``n`` reflects defined observations only.
    """
    values: list[float] = []
    for m in metrics:
        v = _metric_value(m, name)
        if v is not None:
            values.append(v)
    return values


def _metric_value(m: TaskMetrics, name: str) -> float | None:
    if name == "task_quality":
        return m.quality_score
    if name == "task_completed":
        return float(m.task_completed)
    if name == "token_efficiency":
        return float(m.total_tokens)
    if name == "user_stops":
        return float(m.user_stops)
    if name == "time_to_completion_ms":
        return float(m.wall_clock_ms)
    if name == "tool_precision":
        if m.tool_invocation_count == 0:
            return None
        return m.drain_count / m.tool_invocation_count
    if name == "interrupt_rate":
        if m.total_turns == 0:
            return None
        return m.interrupt_count / m.total_turns
    if name == "queue_expiry_rate":
        resolved = m.drain_count + m.expired_count
        if resolved == 0:
            return None
        return m.expired_count / resolved
    if name == "latency_mape":
        if not m.latency_pairs:
            return None
        errors = [abs(p - a) / a for p, a in m.latency_pairs]
        return 100.0 * sum(errors) / len(errors)
    raise ValueError(f"Unknown metric {name!r}")


def condition_summaries(metrics: Sequence[TaskMetrics]) -> dict[str, MetricSummary]:
    """Mean / std / 95% CI for every comparison metric."""
    return {name: compute_summary(metric_values(metrics, name)) for name in METRIC_NAMES}


# ---------------------------------------------------------------------------
# Statistics: Welch p-values (no scipy)
# ---------------------------------------------------------------------------


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_tailed_p(t_stat: float, df: float) -> float:
    """Two-tailed p-value for a t statistic with ``df`` degrees of freedom."""
    if df <= 0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    return _regularized_incomplete_beta(df / 2.0, 0.5, x)


def _welch_df(a: Sequence[float], b: Sequence[float]) -> float:
    """Welch-Satterthwaite degrees of freedom (fractional)."""
    na, nb = len(a), len(b)
    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    se_a = sum((x - mean_a) ** 2 for x in a) / (na - 1) / na
    se_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1) / nb
    se_sum = se_a + se_b
    if se_sum == 0.0:
        return 0.0
    return se_sum**2 / (se_a**2 / (na - 1) + se_b**2 / (nb - 1))


def welch_test_with_p(
    a: Sequence[float], b: Sequence[float]
) -> tuple[float, float, bool]:
    """Welch's t-test with a two-tailed p-value.

    Returns ``(t_stat, p_value, significant)``; ``t_stat`` and the
    significance flag come from :func:`bicameral_agent.ab_test.welch_t_test`
    (95% confidence), the p-value from the exact Welch-Satterthwaite df.
    Degenerate inputs (< 2 samples on a side, or zero variance on both)
    return ``(0.0, 1.0, False)``.
    """
    t_stat, significant = welch_t_test(list(a), list(b))
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0, False
    df = _welch_df(a, b)
    if df <= 0:
        return t_stat, 1.0, significant
    return t_stat, student_t_two_tailed_p(t_stat, df), significant


class PairwiseTestResult(BaseModel):
    """One Welch t-test between two conditions on one metric."""

    metric: str
    condition_a: str
    condition_b: str
    mean_a: float
    mean_b: float
    n_a: int
    n_b: int
    t_stat: float
    p_value: float
    significant: bool


def pairwise_tests(
    metrics_by_condition: dict[str, list[TaskMetrics]],
) -> list[PairwiseTestResult]:
    """All pairwise Welch t-tests, per metric, in condition dict order."""
    conditions = list(metrics_by_condition)
    tests: list[PairwiseTestResult] = []
    for name in METRIC_NAMES:
        values = {c: metric_values(metrics_by_condition[c], name) for c in conditions}
        for i, cond_a in enumerate(conditions):
            for cond_b in conditions[i + 1 :]:
                a, b = values[cond_a], values[cond_b]
                t_stat, p_value, significant = welch_test_with_p(a, b)
                tests.append(
                    PairwiseTestResult(
                        metric=name,
                        condition_a=cond_a,
                        condition_b=cond_b,
                        mean_a=sum(a) / len(a) if a else 0.0,
                        mean_b=sum(b) / len(b) if b else 0.0,
                        n_a=len(a),
                        n_b=len(b),
                        t_stat=t_stat,
                        p_value=p_value,
                        significant=significant,
                    )
                )
    return tests


def difficulty_breakdown(
    tasks: Sequence[ResearchQATask],
    metrics_by_condition: dict[str, list[TaskMetrics]],
) -> dict[str, dict[str, dict[str, MetricSummary]]]:
    """Per-difficulty, per-condition metric summaries.

    Relies on the paired design: ``metrics_by_condition[c][i]`` belongs
    to ``tasks[i]``, so grouping indices by task difficulty slices every
    condition identically.
    """
    breakdown: dict[str, dict[str, dict[str, MetricSummary]]] = {}
    for difficulty in _DIFFICULTY_ORDER:
        indices = [i for i, t in enumerate(tasks) if t.difficulty == difficulty]
        if not indices:
            continue
        breakdown[difficulty.value] = {
            condition: condition_summaries([metrics[i] for i in indices])
            for condition, metrics in metrics_by_condition.items()
        }
    return breakdown


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class ComparativeTaskResult(TaskResult):
    """Per-episode result with the task's difficulty tier."""

    difficulty: str = ""


class ComparativeReport(EvalReport):
    """Machine-readable report for one comparative evaluation run.

    Extends :class:`~bicameral_agent.eval_report.EvalReport` (inheriting
    the dataset/metric/provenance/conditions/results shape and
    ``to_json``) with pairwise tests, the difficulty breakdown, and the
    run's seed/task-mix identity; ``results`` rows are narrowed to
    :class:`ComparativeTaskResult` so each carries its difficulty tier.
    Here ``conditions`` holds the ``{"n_episodes", "summaries"}`` dicts
    built by :func:`build_report` (over the nine comparison metrics)
    rather than baseline ``ConditionReport`` dicts.
    """

    base_seed: int
    task_mix: dict[str, int]
    task_ids: list[str]
    pairwise: list[PairwiseTestResult]
    by_difficulty: dict[str, dict[str, dict[str, dict]]]
    results: list[ComparativeTaskResult] = Field(default_factory=list)
    excluded_task_ids: list[str] = Field(default_factory=list)
    """Tasks dropped from all paired analyses: they failed on transport
    in at least one condition, so keeping them would break the pairing.
    The raw failures are in the inherited ``failures`` field."""

    @classmethod
    def from_benchmark(cls, *args: object, **kwargs: object) -> ComparativeReport:
        """Unsupported: a comparative report needs the paired analysis.

        The inherited constructor takes an unpaired ``BenchmarkResult``
        and cannot supply the comparative-only fields (pairwise tests,
        difficulty breakdown, seed/mix identity); use
        :func:`build_report` on a :class:`ComparativeResult` instead.
        """
        msg = (
            "ComparativeReport cannot be built from a BenchmarkResult; use "
            "comparative_eval.build_report(ComparativeResult, ...) instead"
        )
        raise NotImplementedError(msg)

    def to_markdown(self) -> str:
        """Render the human-readable markdown report."""
        return _render_markdown(self)


def build_report(
    result: ComparativeResult,
    *,
    dataset: str,
    metric: str,
    answerer: dict[str, str],
    measurement: dict[str, str],
    max_turns: int,
    base_seed: int,
) -> ComparativeReport:
    """Aggregate a :class:`ComparativeResult` into a report.

    Pure function of the collected episodes/metrics: identical inputs
    produce byte-identical JSON and markdown (the deterministic side of
    the boundary documented in the module docstring).

    All statistics (summary table, pairwise tests, difficulty breakdown)
    are computed over the *paired* subset — tasks completed in every
    condition — so a transport failure in one condition cannot bias the
    comparison in either direction; the dropped tasks are recorded in
    ``excluded_task_ids`` and the raw failures in ``failures``. The
    per-task ``results`` rows keep every completed episode, mapped back
    to its task via each condition's completed indices.
    """
    paired_metrics = result.paired_metrics()
    conditions = {
        condition: {
            "n_episodes": len(result.metrics[condition]),
            "summaries": {
                name: dataclasses.asdict(summary)
                for name, summary in condition_summaries(metrics).items()
            },
        }
        for condition, metrics in paired_metrics.items()
    }
    difficulty = {
        tier: {
            condition: {
                name: dataclasses.asdict(summary) for name, summary in summaries.items()
            }
            for condition, summaries in per_condition.items()
        }
        for tier, per_condition in difficulty_breakdown(
            result.paired_tasks, paired_metrics
        ).items()
    }
    results = [
        ComparativeTaskResult(
            task_id=result.tasks[i].task_id,
            condition=condition,
            score=episode.outcome.quality_score,
            detail=(episode.metadata.get("verification") or {}).get("detail"),
            difficulty=result.tasks[i].difficulty.value,
        )
        for condition, episodes in result.episodes.items()
        for i, episode in zip(result.completed_indices(condition), episodes)
    ]
    failures = [
        TaskFailure(task_id=f.task_id, condition=condition, error=f.error)
        for condition, condition_failures in result.failures.items()
        for f in condition_failures
    ]
    task_mix = Counter(t.difficulty.value for t in result.tasks)
    return ComparativeReport(
        dataset=dataset,
        metric=metric,
        answerer=answerer,
        measurement=measurement,
        tasks_per_condition=len(result.tasks),
        max_turns=max_turns,
        base_seed=base_seed,
        task_mix=dict(task_mix),
        task_ids=result.task_ids,
        conditions=conditions,
        pairwise=pairwise_tests(paired_metrics),
        by_difficulty=difficulty,
        results=results,
        failures=failures,
        excluded_task_ids=result.excluded_task_ids,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_PRIMARY_METRIC = "task_quality"


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _summary_cell(summary: dict) -> str:
    if summary["n"] == 0:
        return "-"
    return (
        f"{_fmt(summary['mean'])} "
        f"[{_fmt(summary['ci_lower'])}, {_fmt(summary['ci_upper'])}] "
        f"(n={summary['n']})"
    )


def _summary_table(
    conditions: Sequence[str], summaries: Callable[[str], dict[str, dict]]
) -> list[str]:
    lines = [
        "| Metric | " + " | ".join(conditions) + " |",
        "|---" * (len(conditions) + 1) + "|",
    ]
    for name in METRIC_NAMES:
        direction = " (lower=better)" if name in _LOWER_IS_BETTER else ""
        cells = [_summary_cell(summaries(c)[name]) for c in conditions]
        lines.append(f"| {name}{direction} | " + " | ".join(cells) + " |")
    return lines


def _render_markdown(report: ComparativeReport) -> str:
    conditions = list(report.conditions)
    lines: list[str] = [
        "# Comparative Evaluation Report",
        "",
        f"- Dataset: `{report.dataset}` (metric: `{report.metric}`)",
        f"- Answerer: `{report.answerer.get('provider')}/{report.answerer.get('model')}`",
        (
            f"- Measurement: `{report.measurement.get('provider')}"
            f"/{report.measurement.get('model')}`"
        ),
        f"- Tasks per condition: {report.tasks_per_condition} "
        f"(mix: {report.task_mix})",
        f"- Max turns: {report.max_turns}; base seed: {report.base_seed}",
        "",
        "Cells are `mean [95% CI] (n)`; n counts episodes where the metric",
        "is defined (see module docs for the undefined cases).",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        _summary_table(conditions, lambda c: report.conditions[c]["summaries"])
    )

    lines.extend(["", f"## Pairwise Welch t-tests: {_PRIMARY_METRIC}", ""])
    lines.append("| A | B | mean A | mean B | t | p | significant |")
    lines.append("|---|---|---|---|---|---|---|")
    other_significant: list[PairwiseTestResult] = []
    for test in report.pairwise:
        if test.metric == _PRIMARY_METRIC:
            lines.append(
                f"| {test.condition_a} | {test.condition_b} "
                f"| {_fmt(test.mean_a)} | {_fmt(test.mean_b)} "
                f"| {_fmt(test.t_stat)} | {_fmt(test.p_value)} "
                f"| {'yes' if test.significant else 'no'} |"
            )
        elif test.significant:
            other_significant.append(test)

    lines.extend(["", "## Significant differences on other metrics (Welch 95%)", ""])
    if other_significant:
        for test in other_significant:
            lines.append(
                f"- {test.metric}: {test.condition_a} vs {test.condition_b} "
                f"(means {_fmt(test.mean_a)} vs {_fmt(test.mean_b)}, "
                f"t={_fmt(test.t_stat)}, p={_fmt(test.p_value)})"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Breakdown by difficulty"])
    for tier, per_condition in report.by_difficulty.items():
        tier_conditions = [c for c in conditions if c in per_condition]
        lines.extend(["", f"### {tier}", ""])
        lines.extend(
            _summary_table(tier_conditions, lambda c, _pc=per_condition: _pc[c])
        )

    lines.extend(["", "## Transport failures", ""])
    if report.failures or report.excluded_task_ids:
        for failure in report.failures:
            lines.append(
                f"- {failure.condition} / {failure.task_id}: {failure.error}"
            )
        lines.append(
            f"- Paired analyses cover {len(report.task_ids) - len(report.excluded_task_ids)}"
            f"/{len(report.task_ids)} tasks; excluded (failed in >= 1 "
            f"condition): {', '.join(report.excluded_task_ids)}"
        )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "Episode collection is LLM-stochastic; task selection/pairing,",
            "controller seeds, and everything downstream of the collected",
            "episodes (metrics, CIs, tests, this report) are deterministic",
            "for the recorded base seed.",
            "",
        ]
    )
    return "\n".join(lines)
