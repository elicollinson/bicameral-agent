"""MVP success-criteria validation (issue #31).

Extracts each of the six MVP criteria's numbers from committed run
artifacts, determines pass/fail mechanically, prints a human-readable
report, and writes a machine-readable JSON verdict. No live LLM calls are
made, so re-running after new data lands is free and deterministic.

Inputs (all committed):

- ``data/comparative/report.json`` — #30 comparative evaluation
  (5 conditions x 100 tasks; criteria 1, 2, 5)
- ``data/mcts_training/metrics_history.json`` — #29 MCTS training run
  (criterion 4)
- ``docs/figures/emergent/emergent_stats.json`` — #32 emergent-behavior
  analysis (criterion 3)
- committed episode parquets and any A/B result JSON under ``data/``
  (criterion 6: queue-based vs synchronous injection)

Usage::

    uv run python scripts/validate_mvp.py --out data/mvp_validation.json

The narrative companion is ``docs/mvp_validation_2026-07.md``; the
determinations there are transcribed from this script's output (enforced
by ``tests/test_validate_mvp.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from bicameral_agent.ab_test import t_critical_95

# The learned condition used for the headline criteria. learned_no_search is
# the trained policy exactly as it acts at deployment (no inference-time MCTS);
# learned_with_search is reported alongside as a sensitivity check.
LEARNED_CONDITION = "learned_no_search"

# Convergence rule for criterion 4 (thresholds stated in the report):
# final policy entropy below this many nats ...
CONVERGED_ENTROPY_MAX = 0.2
# ... and final train loss at most this fraction of the iteration-0 loss.
CONVERGED_LOSS_FRACTION_MAX = 0.25
EPISODE_BUDGET = 500

# Emergent-pattern rule for criterion 3: the turn-1 vs turn-2+ invocation gap
# of the final checkpoint must be at least this wide, and the multi-tool
# episode rate must at least halve from iteration 0 to the final iteration.
INHIBITION_GAP_MIN = 0.5

_INJECTION_MODE_RE = re.compile(r'"injection_mode":\s*"([a-z_]+)"')


# ---------------------------------------------------------------------------
# Shared statistics
# ---------------------------------------------------------------------------


def welch_diff_ci(a: dict, b: dict) -> tuple[float, float, float]:
    """95% Welch CI for mean(a) - mean(b) from two MetricSummary dicts."""
    se_a = a["std"] ** 2 / a["n"]
    se_b = b["std"] ** 2 / b["n"]
    se = math.sqrt(se_a + se_b)
    diff = a["mean"] - b["mean"]
    if se == 0:
        return diff, diff, diff
    df = (se_a + se_b) ** 2 / (se_a**2 / (a["n"] - 1) + se_b**2 / (b["n"] - 1))
    margin = t_critical_95(max(1, int(df))) * se
    return diff, diff - margin, diff + margin


def _summary(report: dict, condition: str, metric: str) -> dict:
    return report["conditions"][condition]["summaries"][metric]


def _pairwise(report: dict, a: str, b: str, metric: str = "task_quality") -> dict | None:
    for row in report["pairwise"]:
        if row["metric"] != metric:
            continue
        if {row["condition_a"], row["condition_b"]} == {a, b}:
            return row
    return None


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def quality_criterion(report: dict, baseline: str, threshold_pct: float, name: str) -> dict:
    """Criteria 1 and 2: learned beats *baseline* by >= threshold_pct on quality."""
    learned = _summary(report, LEARNED_CONDITION, "task_quality")
    base = _summary(report, baseline, "task_quality")
    diff, ci_lo, ci_hi = welch_diff_ci(learned, base)
    rel_pct = 100.0 * diff / base["mean"]
    pair = _pairwise(report, LEARNED_CONDITION, baseline)
    p_value = pair["p_value"] if pair else None
    threshold_met = rel_pct >= threshold_pct
    significant = p_value is not None and p_value < 0.05
    return {
        "name": name,
        "determination": "PASS" if (threshold_met and significant) else "FAIL",
        "threshold_relative_pct": threshold_pct,
        "learned_condition": LEARNED_CONDITION,
        "baseline_condition": baseline,
        "learned_mean": learned["mean"],
        "learned_ci": [learned["ci_lower"], learned["ci_upper"]],
        "baseline_mean": base["mean"],
        "baseline_ci": [base["ci_lower"], base["ci_upper"]],
        "absolute_difference": diff,
        "difference_ci95": [ci_lo, ci_hi],
        "relative_improvement_pct": rel_pct,
        "welch_p_value": p_value,
        "threshold_met": threshold_met,
        "significant_at_95": significant,
        "n_per_condition": [learned["n"], base["n"]],
    }


def emergent_criterion(stats: dict) -> dict:
    """Criterion 3: at least one emergent timing pattern not in the heuristic."""
    per_iter = stats["per_iteration_episode_stats"]
    iterations = sorted(per_iter, key=int)
    multi_tool_rates = [per_iter[i]["multi_tool_rate"] for i in iterations]
    gap = stats["inhibition_gap"]
    p_turn1 = gap["p_invoke_turn1"]
    p_turn2plus = gap["p_invoke_turn2plus"]
    gaps = [t1 - t2 for t1, t2 in zip(p_turn1, p_turn2plus)]
    final_gap = gaps[-1]
    rate_halved = multi_tool_rates[-1] <= multi_tool_rates[0] / 2
    passed = final_gap >= INHIBITION_GAP_MIN and rate_halved
    return {
        "name": "At least one emergent timing pattern not in the heuristic",
        "determination": "PASS" if passed else "FAIL",
        "pattern": "secondary-tool inhibition gate (fire one tool early, then suppress)",
        "multi_tool_episode_rate_by_iteration": multi_tool_rates,
        "p_invoke_turn1_final": p_turn1[-1],
        "p_invoke_turn2plus_final": p_turn2plus[-1],
        "inhibition_gap_by_iteration": gaps,
        "inhibition_gap_final": final_gap,
        "inhibition_gap_min_required": INHIBITION_GAP_MIN,
        "multi_tool_rate_halved": rate_halved,
        "evidence": "docs/emergent_behavior_2026-07.md, docs/figures/emergent/",
    }


def convergence_criterion(history: list[dict]) -> dict:
    """Criterion 4: training converged within 500 episodes."""
    total_episodes = sum(it["n_episodes"] for it in history)
    entropy = [it["policy_entropy"] for it in history]
    loss = [it["train_loss"] for it in history]
    kl = [it["kl_from_heuristic"] for it in history]
    within_budget = total_episodes <= EPISODE_BUDGET
    entropy_converged = entropy[-1] <= CONVERGED_ENTROPY_MAX
    loss_converged = loss[-1] <= CONVERGED_LOSS_FRACTION_MAX * loss[0]
    passed = within_budget and entropy_converged and loss_converged
    return {
        "name": f"Training converged within {EPISODE_BUDGET} episodes",
        "determination": "PASS" if passed else "FAIL",
        "total_episodes": total_episodes,
        "episode_budget": EPISODE_BUDGET,
        "policy_entropy_by_iteration": entropy,
        "train_loss_by_iteration": loss,
        "kl_from_heuristic_by_iteration": kl,
        "entropy_threshold": CONVERGED_ENTROPY_MAX,
        "loss_fraction_threshold": CONVERGED_LOSS_FRACTION_MAX,
        "within_budget": within_budget,
        "entropy_converged": entropy_converged,
        "loss_converged": loss_converged,
        "curve": "docs/figures/emergent/training_dynamics.png",
    }


def interrupt_criterion(report: dict) -> dict:
    """Criterion 5: learned triggers fewer unintended interrupts than heuristic."""
    learned = _summary(report, LEARNED_CONDITION, "interrupt_rate")
    heuristic = _summary(report, "heuristic", "interrupt_rate")
    diff, ci_lo, ci_hi = welch_diff_ci(learned, heuristic)
    if learned["mean"] < heuristic["mean"]:
        determination = "PASS"
    elif learned["mean"] > heuristic["mean"]:
        determination = "FAIL"
    else:
        determination = "UNRESOLVABLE"
    return {
        "name": "Learned policy triggers fewer unintended interrupts than heuristic",
        "determination": determination,
        "learned_mean": learned["mean"],
        "learned_ci": [learned["ci_lower"], learned["ci_upper"]],
        "heuristic_mean": heuristic["mean"],
        "heuristic_ci": [heuristic["ci_lower"], heuristic["ci_upper"]],
        "absolute_difference": diff,
        "difference_ci95": [ci_lo, ci_hi],
        "n_per_condition": [learned["n"], heuristic["n"]],
        "note": (
            "Both conditions measured exactly zero interrupts across all "
            "episodes. The evaluation ran in BREAKPOINT injection mode, which "
            "drains the queue at every turn boundary before an interrupt "
            "threshold can be crossed, so mid-turn interrupts are structurally "
            "impossible and equality-at-zero cannot demonstrate 'fewer'."
        ),
    }


def _injection_mode_counts(data_dir: Path) -> dict[str, int]:
    """Count episodes per injection mode across committed episode parquets."""
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow is a core dependency
        return {}
    counts: dict[str, int] = {}
    for path in sorted(data_dir.rglob("*.parquet")):
        try:
            table = pq.read_table(path, columns=["payload"])
        except Exception:
            continue  # not an episode parquet (e.g. training examples)
        for payload in table["payload"]:
            match = _INJECTION_MODE_RE.search(payload.as_py())
            mode = match.group(1) if match else "unknown"
            counts[mode] = counts.get(mode, 0) + 1
    return counts


def _find_ab_result(data_dir: Path) -> tuple[Path, dict] | None:
    """Locate a committed ABTestResult JSON with a synchronous arm, if any."""
    for path in sorted(data_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict) or "conditions" not in payload:
            continue
        conditions = payload["conditions"]
        if not isinstance(conditions, list):
            continue
        names = {c.get("name") for c in conditions if isinstance(c, dict)}
        if "synchronous" in names and "breakpoint" in names:
            return path, payload
    return None


AB_TEST_COMMAND = """\
GEMINI_API_KEY=... uv run python - <<'PY'
from pathlib import Path

from bicameral_agent.ab_test import ABTestRunner, default_conditions
from bicameral_agent.eval_datasets import build_dataset
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.heuristic_controller import HeuristicController

Path("data/ab_test").mkdir(parents=True, exist_ok=True)
tasks = build_dataset("builtin").load().eval_tasks()[:50]
result = ABTestRunner(GeminiClient()).run(tasks, default_conditions(HeuristicController))
result.to_json("data/ab_test/report.json")
result.to_csv("data/ab_test/episodes.csv")
PY"""


def derailment_criterion(data_dir: Path) -> dict:
    """Criterion 6: queue-based delivery beats synchronous injection on derailments.

    Requires per-condition derailment counts from the #22 A/B framework
    (``bicameral_agent.ab_test``). All committed episode corpora were collected
    in BREAKPOINT mode, so unless a synchronous-arm A/B result JSON has been
    committed under ``data/``, the criterion is UNEVALUATED.
    """
    mode_counts = _injection_mode_counts(data_dir)
    found = _find_ab_result(data_dir)
    base = {
        "name": "Queue-based delivery produces fewer derailments than synchronous injection",
        "committed_episodes_by_injection_mode": mode_counts,
    }
    if found is None:
        return {
            **base,
            "determination": "UNEVALUATED",
            "note": (
                "No committed run contains SYNCHRONOUS-mode episodes; every "
                "committed episode ran in BREAKPOINT mode. The #22 A/B "
                "framework (bicameral_agent.ab_test) is implemented and "
                "tested but has not been run live against a synchronous arm."
            ),
            "command_to_produce_data": AB_TEST_COMMAND,
        }
    path, payload = found
    by_name = {c["name"]: c for c in payload["conditions"]}
    sync = by_name["synchronous"]["derailments"]
    queued = by_name["breakpoint"]["derailments"]
    return {
        **base,
        "determination": "PASS" if queued["mean"] < sync["mean"] else "FAIL",
        "source": str(path),
        "breakpoint_derailments_mean": queued["mean"],
        "synchronous_derailments_mean": sync["mean"],
        "breakpoint_ci": [queued["ci_lower"], queued["ci_upper"]],
        "synchronous_ci": [sync["ci_lower"], sync["ci_upper"]],
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_validation(
    comparative_report: dict,
    metrics_history: list[dict],
    emergent_stats: dict,
    data_dir: Path,
) -> dict:
    criteria = {
        "criterion_1": quality_criterion(
            comparative_report,
            baseline="random",
            threshold_pct=15.0,
            name="Learned policy outperforms random by >= 15% on task quality",
        ),
        "criterion_2": quality_criterion(
            comparative_report,
            baseline="heuristic",
            threshold_pct=8.0,
            name="Learned policy outperforms heuristic by >= 8% on task quality",
        ),
        "criterion_3": emergent_criterion(emergent_stats),
        "criterion_4": convergence_criterion(metrics_history),
        "criterion_5": interrupt_criterion(comparative_report),
        "criterion_6": derailment_criterion(data_dir),
    }
    return {
        "issue": 31,
        "determinations": {key: c["determination"] for key, c in criteria.items()},
        "criteria": criteria,
    }


def print_report(validation: dict) -> None:
    print("MVP Success Criteria Validation (issue #31)")
    print("=" * 60)
    for key, criterion in validation["criteria"].items():
        number = key.split("_")[1]
        print(f"\nCriterion {number}: {criterion['name']}")
        print(f"  Determination: {criterion['determination']}")
        for field in (
            "relative_improvement_pct",
            "threshold_relative_pct",
            "welch_p_value",
            "difference_ci95",
            "inhibition_gap_final",
            "multi_tool_episode_rate_by_iteration",
            "total_episodes",
            "policy_entropy_by_iteration",
            "train_loss_by_iteration",
            "learned_mean",
            "heuristic_mean",
            "committed_episodes_by_injection_mode",
        ):
            if field in criterion:
                value = criterion[field]
                if isinstance(value, float):
                    value = f"{value:.4g}"
                elif isinstance(value, list) and all(isinstance(v, float) for v in value):
                    value = "[" + ", ".join(f"{v:.4g}" for v in value) + "]"
                print(f"  {field}: {value}")
        if criterion.get("note"):
            print(f"  note: {criterion['note']}")
    print("\nSummary:", json.dumps(validation["determinations"], indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparative-report", default="data/comparative/report.json")
    parser.add_argument("--metrics-history", default="data/mcts_training/metrics_history.json")
    parser.add_argument("--emergent-stats", default="docs/figures/emergent/emergent_stats.json")
    parser.add_argument("--data-dir", default="data", help="Scanned for synchronous-arm A/B data")
    parser.add_argument("--out", default="data/mvp_validation.json")
    args = parser.parse_args(argv)

    validation = build_validation(
        comparative_report=json.loads(Path(args.comparative_report).read_text()),
        metrics_history=json.loads(Path(args.metrics_history).read_text()),
        emergent_stats=json.loads(Path(args.emergent_stats).read_text()),
        data_dir=Path(args.data_dir),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(validation, indent=2) + "\n")
    print_report(validation)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
