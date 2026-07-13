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
- transport-failure counts per arm from each run's ``summary.json``;
- a lexical utilization measure over each run's heuristic arm
  (consumption is not utilization): for every episode with a consumed
  injection, whether the injected content detectably surfaces in
  post-injection assistant messages — a verbatim injected-URL citation,
  or >= 2 distinctive novel tokens reused (see :func:`utilization_stats`)
  — plus the quality split between utilized and non-utilized episodes.

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
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from bicameral_agent.ab_test import compute_summary

CONDITIONS: tuple[str, ...] = ("no_subconscious", "random", "heuristic")

TIERS: dict[str, str] = {"frames_hard": "hard", "crepe_tricky": "tricky"}

# Function/discourse words ignored by the utilization tokenizer (only words
# of length >= _MIN_TOKEN_LEN matter). Standard English stopwords plus the
# generic hedging/verification vocabulary these transcripts use heavily.
_STOPWORDS: frozenset[str] = frozenset("""
    about above according additionally after again against also answer
    based because been before being below between both cannot check claim
    claims confirm confirmed confirming could does doing down during each
    ensure ensures ensuring even ever from further given have having here
    herself himself however include includes including indeed into itself
    just like made make might moreover most must myself only other ought
    ourselves over provide provided provides regarding respectively said
    same says shall should since some specifically still such than that
    their theirs them themselves then there therefore these they this
    those through thus under until upon very well were what when where
    whether which while whom will with within without would your yours
    yourself yourselves
""".split())

_MIN_TOKEN_LEN = 4
_URL_RE = re.compile(r"https?://\S+")


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """The per-episode fields the paired analysis needs."""

    task_id: str
    quality: float | None
    n_injections: int
    n_consumed: int
    n_url_bearing: int
    injections: tuple[dict, ...] = ()
    """Raw injection dicts (content / consumed / consumed_at_turn)."""
    turn_messages: tuple[tuple[int, str, str], ...] = ()
    """(turn, role, content) triples; turn N starts at the Nth user message."""


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
        turn_messages: list[tuple[int, str, str]] = []
        turn = 0
        for message in payload.get("messages", []):
            if message["role"] == "user":
                turn += 1
            turn_messages.append((turn, message["role"], message["content"]))
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
            injections=tuple(injections),
            turn_messages=tuple(turn_messages),
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


def _tokens(text: str) -> set[str]:
    """Stopword-filtered lowercase word tokens of length >= 4."""
    return {
        tok
        for tok in re.findall(r"[a-z0-9']+", text.lower())
        if len(tok) >= _MIN_TOKEN_LEN and tok not in _STOPWORDS
    }


def utilization_stats(
    records: dict[str, EpisodeRecord],
    min_distinctive: int = 2,
    df_fraction: float = 0.25,
    df_min: int = 3,
) -> dict:
    """Lexical utilization of consumed injections in one arm's answers.

    Consumption (an injection drained into context) is not utilization
    (the answerer's subsequent text engaging with it). For every episode
    with at least one consumed injection, an injection counts as utilized
    when either:

    - an injected URL is quoted verbatim in a post-injection assistant
      message, or
    - at least ``min_distinctive`` *distinctive novel* tokens from the
      injection appear in post-injection assistant messages, where
      distinctive means not scanner boilerplate (tokens occurring in
      >= max(df_min, ceil(df_fraction * n_injections)) of the arm's
      injections) and novel means absent from all messages before the
      injection's ``consumed_at_turn`` (the scanner paraphrases the
      assistant's own claims, which must not count as evidence uptake).

    This is a lenient lexical proxy — an upper bound on genuine citation;
    the verbatim-URL count is the strict lower bound.
    """
    doc_freq: Counter[str] = Counter()
    n_injections = 0
    for record in records.values():
        for inj in record.injections:
            doc_freq.update(_tokens(inj["content"]))
            n_injections += 1
    df_cutoff = max(df_min, math.ceil(df_fraction * n_injections))
    boilerplate = {tok for tok, count in doc_freq.items() if count >= df_cutoff}

    per_episode: dict[str, dict] = {}
    quality_split: dict[bool, list[float]] = {True: [], False: []}
    url_citations = 0
    for task_id, record in sorted(records.items()):
        consumed = [inj for inj in record.injections if inj.get("consumed")]
        if not consumed:
            continue
        utilized = False
        used_tokens: set[str] = set()
        for inj in consumed:
            cut = inj.get("consumed_at_turn") or 1
            pre = " ".join(
                content for t, _, content in record.turn_messages if t < cut
            )
            post = " ".join(
                content
                for t, role, content in record.turn_messages
                if t >= cut and role == "assistant"
            )
            distinctive_novel = (
                _tokens(inj["content"]) - boilerplate - _tokens(pre)
            )
            used = distinctive_novel & _tokens(post)
            url_cited = bool(
                set(_URL_RE.findall(inj["content"])) & set(_URL_RE.findall(post))
            )
            url_citations += url_cited
            if url_cited or len(used) >= min_distinctive:
                utilized = True
            used_tokens |= used
        per_episode[task_id] = {
            "utilized": utilized,
            "used_tokens": sorted(used_tokens),
            "quality": record.quality,
        }
        if record.quality is not None:
            quality_split[utilized].append(record.quality)

    def _mean(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None

    return {
        "episodes_with_consumed_injections": len(per_episode),
        "utilized_episodes": sum(
            1 for ep in per_episode.values() if ep["utilized"]
        ),
        "url_citations": url_citations,
        "quality_utilized": {
            "mean": _mean(quality_split[True]),
            "n": len(quality_split[True]),
        },
        "quality_not_utilized": {
            "mean": _mean(quality_split[False]),
            "n": len(quality_split[False]),
        },
        "per_episode": per_episode,
    }


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

    utilization = {
        run: utilization_stats(conditions["heuristic"])
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
        "utilization": utilization,
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
    lines.append(
        "Utilization of consumed injections, heuristic arm "
        "(lexical; URL verbatim or >=2 distinctive novel tokens):"
    )
    for run, stats in result["utilization"].items():
        qual_u = stats["quality_utilized"]
        qual_n = stats["quality_not_utilized"]

        def _q(split: dict) -> str:
            mean = split["mean"]
            rendered = "n/a" if mean is None else f"{mean:.3f}"
            return f"{rendered} (n={split['n']})"

        lines.append(
            f"  {run:<9} consumed-inj episodes="
            f"{stats['episodes_with_consumed_injections']} "
            f"utilized={stats['utilized_episodes']} "
            f"url_citations={stats['url_citations']} | "
            f"quality utilized={_q(qual_u)} vs not={_q(qual_n)}"
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
