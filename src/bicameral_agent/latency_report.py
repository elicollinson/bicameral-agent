"""Reporting utilities for latency observations.

Produces accuracy metrics (Layer 1 token estimator and Layer 2 API latency
model) and a scatter plot of predicted vs. actual latency. Matplotlib is an
optional dependency; when unavailable the scatter export falls back to a
TSV that can be plotted later.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from bicameral_agent.latency_collection import (
    CONSCIOUS_LOOP_TOOL_ID,
    LatencyObservation,
    ToolObservation,
)

# Acceptance-criteria thresholds from issue #35.
LAYER_2_PCT = 0.25
LAYER_2_MIN_COVERAGE = 0.70
LAYER_1_PCT = 0.30
LAYER_1_MIN_COVERAGE = 0.70


def coverage_within_pct(
    predicted: Iterable[float], actual: Iterable[float], pct: float
) -> float:
    """Fraction of pairs whose ``predicted`` is within ``pct`` of ``actual``.

    A pair counts as covered iff ``abs(predicted - actual) / actual <= pct``.
    Pairs with ``actual <= 0`` are skipped from both numerator and denominator.
    Returns 0.0 when no valid pairs are provided.
    """
    pred_list = list(predicted)
    act_list = list(actual)
    if len(pred_list) != len(act_list):
        raise ValueError(
            f"predicted and actual must have equal length: {len(pred_list)} vs {len(act_list)}"
        )
    matched = 0
    valid = 0
    for p, a in zip(pred_list, act_list):
        if a <= 0:
            continue
        valid += 1
        if abs(p - a) / a <= pct:
            matched += 1
    if valid == 0:
        return 0.0
    return matched / valid


def layer2_coverage(api_obs: list[LatencyObservation], pct: float = LAYER_2_PCT) -> float:
    """Fraction of API observations whose predicted mean latency is within ``pct``."""
    return coverage_within_pct(
        (o.predicted_mean_ms for o in api_obs),
        (o.actual_duration_ms for o in api_obs),
        pct,
    )


def layer1_coverage(tool_obs: list[ToolObservation], pct: float = LAYER_1_PCT) -> float:
    """Fraction of tool observations whose predicted output tokens are within ``pct``."""
    return coverage_within_pct(
        (o.predicted_output_tokens for o in tool_obs),
        (o.actual_output_tokens for o in tool_obs),
        pct,
    )


def format_text_report(
    api_obs: list[LatencyObservation],
    tool_obs: list[ToolObservation],
) -> str:
    """Render a human-readable summary of the collected observations."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Latency Data Collection Report (Issue #35)")
    lines.append("=" * 72)
    lines.append(f"Total API observations:  {len(api_obs)}")
    lines.append(f"Total tool observations: {len(tool_obs)}")
    lines.append("")

    # ------------------------------------------------------------------
    # Layer 2 (API latency model) coverage
    # ------------------------------------------------------------------
    lines.append("Layer 2 — API latency model")
    lines.append("-" * 72)
    overall_l2 = layer2_coverage(api_obs)
    lines.append(
        f"  Within ±{LAYER_2_PCT * 100:.0f}% of actual: {overall_l2 * 100:5.1f}% "
        f"(threshold: {LAYER_2_MIN_COVERAGE * 100:.0f}%)  "
        f"[{'OK' if overall_l2 >= LAYER_2_MIN_COVERAGE else 'MISS'}]"
    )

    by_bucket = _group_by(api_obs, lambda o: o.conversation_length_bucket)
    for bucket in sorted(by_bucket):
        cov = layer2_coverage(by_bucket[bucket])
        n = len(by_bucket[bucket])
        lines.append(
            f"    bucket={bucket:>6}t  n={n:>3}  coverage={cov * 100:5.1f}%"
        )

    lines.append("")

    by_tool = _group_by(api_obs, lambda o: o.tool_id)
    for tool_id in sorted(by_tool):
        cov = layer2_coverage(by_tool[tool_id])
        n = len(by_tool[tool_id])
        lines.append(f"    tool={tool_id:<28}  n={n:>3}  coverage={cov * 100:5.1f}%")

    lines.append("")
    lines.append(f"  Mean residual:   {_mean_signed_residual(api_obs):+8.1f} ms")
    lines.append(f"  Median |residual|: {_median_abs_residual(api_obs):8.1f} ms")
    lines.append("")

    # ------------------------------------------------------------------
    # Layer 1 (token estimator) coverage
    # ------------------------------------------------------------------
    lines.append("Layer 1 — Token estimator (per-tool output token count)")
    lines.append("-" * 72)
    overall_l1 = layer1_coverage(tool_obs)
    lines.append(
        f"  Within ±{LAYER_1_PCT * 100:.0f}% of actual: {overall_l1 * 100:5.1f}% "
        f"(threshold: {LAYER_1_MIN_COVERAGE * 100:.0f}%)  "
        f"[{'OK' if overall_l1 >= LAYER_1_MIN_COVERAGE else 'MISS'}]"
    )

    tool_by_id = _group_by(tool_obs, lambda o: o.tool_id)
    for tool_id in sorted(tool_by_id):
        sub = tool_by_id[tool_id]
        cov = layer1_coverage(sub)
        n = len(sub)
        lines.append(f"    tool={tool_id:<28}  n={n:>3}  coverage={cov * 100:5.1f}%")

    lines.append("")
    lines.append("Acceptance criteria")
    lines.append("-" * 72)
    lines.append(f"  AC1 ≥90 API observations: {len(api_obs)}  "
                 f"[{'OK' if len(api_obs) >= 90 else 'MISS'}]")
    cl_buckets = sorted({
        o.conversation_length_bucket
        for o in api_obs
        if o.tool_id == CONSCIOUS_LOOP_TOOL_ID
    })
    lines.append(
        f"  AC2 conscious-loop buckets ({len(cl_buckets)}): {cl_buckets}  "
        f"[{'OK' if len(cl_buckets) >= 6 else 'MISS'}]"
    )
    lines.append(
        f"  AC3 Layer 2 ≥70% within 25%: {overall_l2 * 100:.1f}%  "
        f"[{'OK' if overall_l2 >= LAYER_2_MIN_COVERAGE else 'MISS'}]"
    )
    lines.append(
        f"  AC4 Layer 1 ≥70% within 30%: {overall_l1 * 100:.1f}%  "
        f"[{'OK' if overall_l1 >= LAYER_1_MIN_COVERAGE else 'MISS'}]"
    )
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


def save_scatter_plot(
    api_obs: list[LatencyObservation], output_path: str | os.PathLike
) -> str:
    """Save a predicted-vs-actual scatter plot.

    If ``matplotlib`` is importable the plot is saved as a PNG (or whatever
    suffix ``output_path`` carries). Otherwise the data points are written
    as a tab-separated file with the suffix swapped to ``.tsv``.

    Returns the path actually written.
    """
    pred = [o.predicted_mean_ms for o in api_obs]
    act = [o.actual_duration_ms for o in api_obs]
    target = Path(output_path)

    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        tsv_path = target.with_suffix(".tsv")
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with tsv_path.open("w") as f:
            f.write("predicted_ms\tactual_ms\ttool_id\tconv_length\n")
            for o in api_obs:
                f.write(
                    f"{o.predicted_mean_ms:.3f}\t{o.actual_duration_ms:.3f}\t"
                    f"{o.tool_id}\t{o.conversation_length_bucket}\n"
                )
        return str(tsv_path)

    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    by_tool = _group_by(api_obs, lambda o: o.tool_id)
    colors = plt.get_cmap("tab10")(range(len(by_tool)))
    for i, (tool_id, obs) in enumerate(sorted(by_tool.items())):
        ax.scatter(
            [o.predicted_mean_ms for o in obs],
            [o.actual_duration_ms for o in obs],
            label=tool_id,
            alpha=0.7,
            color=colors[i],
        )

    if pred and act:
        upper = max(max(pred), max(act))
        ax.plot([0, upper], [0, upper], "k--", linewidth=1, label="y = x")

    ax.set_xlabel("Predicted latency (ms)")
    ax.set_ylabel("Actual latency (ms)")
    ax.set_title("Latency model: predicted vs. actual")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(target, dpi=120)
    plt.close(fig)
    return str(target)


def _group_by(items, key):
    grouped: dict = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return grouped


def _mean_signed_residual(api_obs: list[LatencyObservation]) -> float:
    if not api_obs:
        return 0.0
    return sum(o.predicted_mean_ms - o.actual_duration_ms for o in api_obs) / len(api_obs)


def _median_abs_residual(api_obs: list[LatencyObservation]) -> float:
    if not api_obs:
        return 0.0
    abs_residuals = sorted(abs(o.predicted_mean_ms - o.actual_duration_ms) for o in api_obs)
    n = len(abs_residuals)
    mid = n // 2
    if n % 2 == 0:
        return (abs_residuals[mid - 1] + abs_residuals[mid]) / 2
    return abs_residuals[mid]
