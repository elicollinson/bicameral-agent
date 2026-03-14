"""A/B test framework for comparing context injection strategies.

Runs each task under each condition, collects metrics, computes statistics,
and produces a structured comparison report.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from typing import Callable

from pydantic import BaseModel, Field

from bicameral_agent.coherence_judge import CoherenceJudge, CoherenceScore
from bicameral_agent.dataset import ResearchQATask
from bicameral_agent.episode_runner import (
    Controller,
    EpisodeConfig,
    EpisodeRunner,
    InjectionMode,
)
from bicameral_agent.followup_classifier import FollowUpClassifier, FollowUpType
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.schema import Episode


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Condition:
    """An A/B test condition defining an injection strategy."""

    name: str
    controller_factory: Callable[[], Controller]
    episode_config: EpisodeConfig

    @property
    def injection_mode(self) -> InjectionMode:
        """Derive injection mode from the episode config."""
        return self.episode_config.injection_mode


def default_conditions(controller_factory: Callable[[], Controller]) -> list[Condition]:
    """Return the three standard A/B test conditions."""
    return [
        Condition(
            name="synchronous",
            controller_factory=controller_factory,
            episode_config=EpisodeConfig(injection_mode=InjectionMode.SYNCHRONOUS),
        ),
        Condition(
            name="breakpoint",
            controller_factory=controller_factory,
            episode_config=EpisodeConfig(injection_mode=InjectionMode.BREAKPOINT),
        ),
        Condition(
            name="interrupt",
            controller_factory=controller_factory,
            episode_config=EpisodeConfig(injection_mode=InjectionMode.INTERRUPT),
        ),
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics extracted from a single episode under a condition."""

    task_id: str
    condition: str
    quality_score: float | None
    total_tokens: int
    coherence_score: float
    derailment_count: int
    interrupt_count: int
    total_turns: int
    wall_clock_ms: float


def count_derailments(messages: list) -> int:
    """Count REDIRECT + CORRECTION follow-ups in conversation messages."""
    count = 0
    for msg in messages:
        if msg.role == "user":
            ft = FollowUpClassifier.classify(msg.content)
            if ft in (FollowUpType.REDIRECT, FollowUpType.CORRECTION):
                count += 1
    return count


def extract_metrics(
    episode: Episode,
    condition_name: str,
    task_id: str,
    coherence: CoherenceScore,
) -> EpisodeMetrics:
    """Extract metrics from a completed episode."""
    return EpisodeMetrics(
        task_id=task_id,
        condition=condition_name,
        quality_score=episode.outcome.quality_score,
        total_tokens=episode.outcome.total_tokens,
        coherence_score=coherence.overall,
        derailment_count=count_derailments(episode.messages),
        interrupt_count=episode.metadata.get("interrupt_count", 0),
        total_turns=episode.outcome.total_turns,
        wall_clock_ms=episode.outcome.wall_clock_ms,
    )


# ---------------------------------------------------------------------------
# Statistics (no scipy)
# ---------------------------------------------------------------------------

# Two-tailed t-distribution critical values at 95% confidence (alpha=0.05)
_T_TABLE = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    60: 2.000,
    80: 1.990,
    100: 1.984,
    200: 1.972,
    500: 1.965,
    1000: 1.962,
}

_T_TABLE_KEYS = sorted(_T_TABLE.keys())


def t_critical_95(df: int) -> float:
    """Look up two-tailed t-critical value at 95% CI using interpolation.

    For df values between table entries, uses linear interpolation.
    For df > 1000, returns 1.96 (z-approximation).
    """
    if df < 1:
        raise ValueError("Degrees of freedom must be >= 1")
    if df >= 1000:
        return 1.96

    keys = _T_TABLE_KEYS
    if df in _T_TABLE:
        return _T_TABLE[df]

    # Find bracketing keys
    lower = keys[0]
    upper = keys[-1]
    for k in keys:
        if k <= df:
            lower = k
        if k >= df:
            upper = k
            break

    if lower == upper:
        return _T_TABLE[lower]

    # Linear interpolation
    t_low = _T_TABLE[lower]
    t_high = _T_TABLE[upper]
    frac = (df - lower) / (upper - lower)
    return t_low + frac * (t_high - t_low)


@dataclasses.dataclass(frozen=True)
class MetricSummary:
    """Summary statistics for a metric across episodes."""

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int


def compute_summary(values: list[float]) -> MetricSummary:
    """Compute mean, std, and 95% CI for a list of values."""
    n = len(values)
    if n == 0:
        return MetricSummary(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n=0)
    mean = sum(values) / n
    if n == 1:
        return MetricSummary(mean=mean, std=0.0, ci_lower=mean, ci_upper=mean, n=1)
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    t_crit = t_critical_95(n - 1)
    margin = t_crit * (std / math.sqrt(n))
    return MetricSummary(
        mean=mean,
        std=std,
        ci_lower=mean - margin,
        ci_upper=mean + margin,
        n=n,
    )


def welch_t_test(a: list[float], b: list[float]) -> tuple[float, bool]:
    """Welch's t-test for independent samples, returns (t_statistic, significant).

    Significant if |t| > t_critical at 95% for Welch-Satterthwaite df.
    Returns (0.0, False) if either sample has < 2 values.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, False

    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)

    se_a = var_a / na
    se_b = var_b / nb
    se_sum = se_a + se_b

    if se_sum == 0:
        return 0.0, False

    t_stat = (mean_a - mean_b) / math.sqrt(se_sum)

    # Welch-Satterthwaite degrees of freedom
    df = (se_sum ** 2) / (
        (se_a ** 2 / (na - 1)) + (se_b ** 2 / (nb - 1))
    )
    df = max(1, int(df))

    t_crit = t_critical_95(df)
    return t_stat, abs(t_stat) > t_crit


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class ConditionResult(BaseModel):
    """Results for a single condition."""

    name: str
    quality: dict = Field(default_factory=dict)
    total_tokens: dict = Field(default_factory=dict)
    coherence: dict = Field(default_factory=dict)
    derailments: dict = Field(default_factory=dict)
    interrupts: dict = Field(default_factory=dict)


class ABTestResult(BaseModel):
    """Complete A/B test results."""

    conditions: list[ConditionResult]
    episode_metrics: list[dict]
    best_condition: str
    justification: str

    def to_json(self, path: str) -> None:
        """Write results as structured JSON."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    def to_csv(self, path: str) -> None:
        """Write per-episode metrics as flat CSV."""
        if not self.episode_metrics:
            return
        fieldnames = list(self.episode_metrics[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.episode_metrics)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ABTestRunner:
    """Runs A/B tests comparing injection strategies."""

    def __init__(
        self,
        client: GeminiClient,
        score_episodes: bool = True,
        use_lexical_scorer: bool = False,
    ) -> None:
        self._client = client
        self._score_episodes = score_episodes
        self._use_lexical_scorer = use_lexical_scorer

    def run(
        self,
        tasks: list[ResearchQATask],
        conditions: list[Condition],
    ) -> ABTestResult:
        """Run all tasks under all conditions and produce results."""
        all_metrics: list[EpisodeMetrics] = []
        coherence_judge = CoherenceJudge(client=self._client)

        for condition in conditions:
            for task in tasks:
                controller = condition.controller_factory()
                config = dataclasses.replace(
                    condition.episode_config,
                    score_episode=self._score_episodes,
                    use_lexical_scorer=self._use_lexical_scorer,
                )
                runner = EpisodeRunner(self._client, config)
                episode = runner.run_episode(task, controller)

                coherence = coherence_judge.score(episode.messages)
                metrics = extract_metrics(
                    episode, condition.name, task.task_id, coherence
                )
                all_metrics.append(metrics)

        # Compute statistics per condition
        condition_results = []
        metrics_by_condition: dict[str, list[EpisodeMetrics]] = {}
        summaries_by_condition: dict[str, dict[str, MetricSummary]] = {}
        for m in all_metrics:
            metrics_by_condition.setdefault(m.condition, []).append(m)

        for cond in conditions:
            cond_metrics = metrics_by_condition.get(cond.name, [])
            quality_vals = [m.quality_score for m in cond_metrics if m.quality_score is not None]
            token_vals = [float(m.total_tokens) for m in cond_metrics]
            coherence_vals = [m.coherence_score for m in cond_metrics]
            derailment_vals = [float(m.derailment_count) for m in cond_metrics]
            interrupt_vals = [float(m.interrupt_count) for m in cond_metrics]

            summaries = {
                "quality": compute_summary(quality_vals) if quality_vals else None,
                "total_tokens": compute_summary(token_vals),
                "coherence": compute_summary(coherence_vals),
                "derailments": compute_summary(derailment_vals),
                "interrupts": compute_summary(interrupt_vals),
            }
            summaries_by_condition[cond.name] = summaries

            condition_results.append(ConditionResult(
                name=cond.name,
                quality=dataclasses.asdict(summaries["quality"]) if summaries["quality"] else {},
                total_tokens=dataclasses.asdict(summaries["total_tokens"]),
                coherence=dataclasses.asdict(summaries["coherence"]),
                derailments=dataclasses.asdict(summaries["derailments"]),
                interrupts=dataclasses.asdict(summaries["interrupts"]),
            ))

        # Determine best condition (primary: quality_score, secondary: total_tokens)
        best_condition = _select_best(conditions, summaries_by_condition)
        justification = _build_justification(
            conditions, summaries_by_condition, metrics_by_condition
        )

        episode_dicts = [dataclasses.asdict(m) for m in all_metrics]

        return ABTestResult(
            conditions=condition_results,
            episode_metrics=episode_dicts,
            best_condition=best_condition,
            justification=justification,
        )


def _select_best(
    conditions: list[Condition],
    summaries_by_condition: dict[str, dict[str, MetricSummary | None]],
) -> str:
    """Select best condition by quality_score (primary), then total_tokens (secondary)."""
    best_name = conditions[0].name
    best_quality = -1.0
    best_tokens = float("inf")

    for cond in conditions:
        summaries = summaries_by_condition.get(cond.name, {})
        q = summaries.get("quality")
        t = summaries.get("total_tokens")

        mean_quality = q.mean if q else 0.0
        mean_tokens = t.mean if t else 0.0

        if mean_quality > best_quality or (
            mean_quality == best_quality and mean_tokens < best_tokens
        ):
            best_quality = mean_quality
            best_tokens = mean_tokens
            best_name = cond.name

    return best_name


def _build_justification(
    conditions: list[Condition],
    summaries_by_condition: dict[str, dict[str, MetricSummary | None]],
    metrics_by_condition: dict[str, list[EpisodeMetrics]],
) -> str:
    """Build a justification string with statistical claims."""
    parts = []
    for cond in conditions:
        summaries = summaries_by_condition.get(cond.name, {})
        q = summaries.get("quality")
        t = summaries.get("total_tokens")
        c = summaries.get("coherence")

        quality_str = f"quality={q.mean:.3f}" if q else "quality=N/A"
        tokens_str = f"tokens={t.mean:.0f}" if t else "tokens=0"
        coherence_str = f"coherence={c.mean:.3f}" if c else "coherence=0"
        parts.append(f"{cond.name}: {quality_str}, {tokens_str}, {coherence_str}")

    # Pairwise significance for quality
    cond_names = [c.name for c in conditions]
    for i in range(len(cond_names)):
        for j in range(i + 1, len(cond_names)):
            a_metrics = metrics_by_condition.get(cond_names[i], [])
            b_metrics = metrics_by_condition.get(cond_names[j], [])
            a_vals = [m.quality_score for m in a_metrics if m.quality_score is not None]
            b_vals = [m.quality_score for m in b_metrics if m.quality_score is not None]
            if a_vals and b_vals:
                t_stat, sig = welch_t_test(a_vals, b_vals)
                sig_str = "significant" if sig else "not significant"
                parts.append(
                    f"{cond_names[i]} vs {cond_names[j]}: t={t_stat:.3f} ({sig_str})"
                )

    return "; ".join(parts)
