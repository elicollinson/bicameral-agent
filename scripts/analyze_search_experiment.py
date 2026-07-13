"""Paired analysis of the #101 search-vs-nosearch baseline reruns.

Compares the real-search rerun (``data/baseline_search/``, Brave backend,
#100) against the committed no-search baseline (``data/baseline/``, mock
backend) on the hard_benchmark pool. Because transport failures clustered
unevenly across arms, raw per-arm means are survivor-biased; every
conclusion here is computed on the task-id intersection completed in all
six arms (3 conditions x 2 runs), pairing episodes by task.

Computes, on that intersection:

- per-arm paired quality means;
- per-condition paired deltas (search - nosearch) with 95% CIs;
- the heuristic - no_subconscious contrast within each run, overall and
  split by tier (FRAMES hard vs CREPE tricky);
- drain / injection statistics per arm, including the number of
  URL-bearing injections (live-Brave provenance: the mock provider has no
  URLs, so any injection containing a literal URL proves real search);
- transport-failure counts per arm from each run's ``summary.json``.

Quality is ``payload -> outcome.quality_score``; the pairing key is
``payload -> metadata.task_id``. CIs use the repo's t-based
:func:`bicameral_agent.ab_test.compute_summary`.

Usage::

    uv run python scripts/analyze_search_experiment.py \\
        --nosearch-dir data/baseline \\
        --search-dir data/baseline_search \\
        --json-out data/baseline_search/paired_analysis.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from bicameral_agent.ab_test import compute_summary

CONDITIONS: tuple[str, ...] = ("no_subconscious", "random", "heuristic")

TIERS: dict[str, str] = {"frames_hard": "hard", "crepe_tricky": "tricky"}


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """The per-episode fields the paired analysis needs."""

    task_id: str
    quality: float | None
    n_injections: int
    n_consumed: int
    n_url_bearing: int


def tier_of(task_id: str) -> str:
    """Map a task id (e.g. ``frames_hard_012``) to its tier name."""
    prefix = task_id.rsplit("_", 1)[0]
    return TIERS.get(prefix, prefix)


def load_condition(path: Path) -> dict[str, EpisodeRecord]:
    """Load one arm's parquet into task_id -> record.

    Raises:
        ValueError: If the same task_id appears twice (pairing would be
            ambiguous).
    """
    table = pq.read_table(path)
    records: dict[str, EpisodeRecord] = {}
    for raw in table.column("payload"):
        payload = json.loads(raw.as_py())
        task_id = payload["metadata"]["task_id"]
        if task_id in records:
            raise ValueError(f"duplicate task_id {task_id!r} in {path}")
        injections = payload.get("context_injections", [])
        records[task_id] = EpisodeRecord(
            task_id=task_id,
            quality=payload["outcome"]["quality_score"],
            n_injections=len(injections),
            n_consumed=sum(1 for inj in injections if inj.get("consumed")),
            n_url_bearing=sum(
                1
                for inj in injections
                if "http://" in inj["content"] or "https://" in inj["content"]
            ),
        )
    return records


def load_run(run_dir: Path) -> dict[str, dict[str, EpisodeRecord]]:
    """Load all three condition parquets of one run."""
    return {
        cond: load_condition(run_dir / f"{cond}.parquet") for cond in CONDITIONS
    }


def transport_failures(run_dir: Path) -> dict[str, int]:
    """Per-condition transport-failure counts from summary.json."""
    summary = json.loads((run_dir / "summary.json").read_text())
    counts = {cond: 0 for cond in CONDITIONS}
    for failure in summary.get("failures", []):
        counts[failure["condition"]] += 1
    return counts


def paired_intersection(
    runs: dict[str, dict[str, dict[str, EpisodeRecord]]],
) -> list[str]:
    """Task ids completed (with a quality score) in all six arms."""
    common: set[str] | None = None
    for conditions in runs.values():
        for records in conditions.values():
            scored = {t for t, r in records.items() if r.quality is not None}
            common = scored if common is None else common & scored
    return sorted(common or ())


def _delta_summary(diffs: list[float]) -> dict[str, float | int]:
    s = compute_summary(diffs)
    return {
        "mean": s.mean,
        "ci_lower": s.ci_lower,
        "ci_upper": s.ci_upper,
        "n": s.n,
    }


def _by_tier(tasks: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"all": list(tasks)}
    for task in tasks:
        groups.setdefault(tier_of(task), []).append(task)
    return groups


def analyze(
    nosearch: dict[str, dict[str, EpisodeRecord]],
    search: dict[str, dict[str, EpisodeRecord]],
    nosearch_failures: dict[str, int],
    search_failures: dict[str, int],
) -> dict:
    """Full paired analysis; returns a JSON-serializable dict."""
    runs = {"nosearch": nosearch, "search": search}
    tasks = paired_intersection(runs)
    tier_groups = _by_tier(tasks)

    def quality(run: str, cond: str, task: str) -> float:
        q = runs[run][cond][task].quality
        assert q is not None  # guaranteed by paired_intersection
        return q

    paired_means = {
        run: {
            cond: statistics.mean(quality(run, cond, t) for t in tasks)
            for cond in CONDITIONS
        }
        for run in runs
    }

    search_vs_nosearch = {
        cond: {
            tier: _delta_summary(
                [
                    quality("search", cond, t) - quality("nosearch", cond, t)
                    for t in group
                ]
            )
            for tier, group in tier_groups.items()
        }
        for cond in CONDITIONS
    }

    heuristic_vs_no_subconscious = {
        run: {
            tier: _delta_summary(
                [
                    quality(run, "heuristic", t)
                    - quality(run, "no_subconscious", t)
                    for t in group
                ]
            )
            for tier, group in tier_groups.items()
        }
        for run in runs
    }

    injection_stats = {
        run: {
            cond: {
                "n_episodes": len(records),
                "injections": sum(r.n_injections for r in records.values()),
                "consumed": sum(r.n_consumed for r in records.values()),
                "url_bearing": sum(r.n_url_bearing for r in records.values()),
            }
            for cond, records in conditions.items()
        }
        for run, conditions in runs.items()
    }

    return {
        "paired_task_count": len(tasks),
        "tier_task_counts": {
            tier: len(group) for tier, group in tier_groups.items()
        },
        "paired_tasks": tasks,
        "paired_means": paired_means,
        "search_vs_nosearch": search_vs_nosearch,
        "heuristic_vs_no_subconscious": heuristic_vs_no_subconscious,
        "injection_stats": injection_stats,
        "transport_failures": {
            "nosearch": nosearch_failures,
            "search": search_failures,
        },
    }


def _fmt(delta: dict[str, float | int]) -> str:
    return (
        f"{delta['mean']:+.3f} "
        f"[{delta['ci_lower']:+.3f}, {delta['ci_upper']:+.3f}] n={delta['n']}"
    )


def format_report(result: dict) -> str:
    """Human-readable rendering of :func:`analyze` output."""
    lines: list[str] = []
    tiers = result["tier_task_counts"]
    lines.append("Paired search-vs-nosearch analysis (#101)")
    lines.append(
        "Intersection completed in all six arms: "
        f"{result['paired_task_count']} tasks "
        f"(hard {tiers.get('hard', 0)}, tricky {tiers.get('tricky', 0)})"
    )
    lines.append("")
    lines.append("Paired quality means on the intersection:")
    for run, means in result["paired_means"].items():
        for cond, mean in means.items():
            lines.append(f"  {run:<9} {cond:<16} {mean:.3f}")
    lines.append("")
    lines.append("Quality delta, search - nosearch (paired, 95% CI):")
    for cond, by_tier in result["search_vs_nosearch"].items():
        for tier, delta in by_tier.items():
            lines.append(f"  {cond:<16} {tier:<7} {_fmt(delta)}")
    lines.append("")
    lines.append("Quality delta, heuristic - no_subconscious (paired, 95% CI):")
    for run, by_tier in result["heuristic_vs_no_subconscious"].items():
        for tier, delta in by_tier.items():
            lines.append(f"  {run:<9} {tier:<7} {_fmt(delta)}")
    lines.append("")
    lines.append("Injection stats (all episodes, not just intersection):")
    for run, conds in result["injection_stats"].items():
        for cond, stats in conds.items():
            lines.append(
                f"  {run:<9} {cond:<16} episodes={stats['n_episodes']} "
                f"injections={stats['injections']} "
                f"consumed={stats['consumed']} "
                f"url_bearing={stats['url_bearing']}"
            )
    lines.append("")
    lines.append("Transport failures per arm (summary.json):")
    for run, counts in result["transport_failures"].items():
        rendered = ", ".join(f"{c}={n}" for c, n in counts.items())
        lines.append(f"  {run:<9} {rendered} (total {sum(counts.values())})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nosearch-dir",
        type=Path,
        default=Path("data/baseline"),
        help="No-search baseline run directory (mock provider).",
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=Path("data/baseline_search"),
        help="Real-search rerun directory (Brave provider).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/baseline_search/paired_analysis.json"),
        help="Where to write the analysis JSON.",
    )
    args = parser.parse_args(argv)

    result = analyze(
        nosearch=load_run(args.nosearch_dir),
        search=load_run(args.search_dir),
        nosearch_failures=transport_failures(args.nosearch_dir),
        search_failures=transport_failures(args.search_dir),
    )
    print(format_report(result))
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
