"""Run the Issue #23 baseline performance benchmark.

Runs three controllers (no-subconscious, random p=0.2, heuristic) against
50 evaluation tasks (mix of typical/hard/tricky) and emits:

  - Per-episode parquet files under ``--output-dir/<condition>/``
  - ``report.txt`` with per-condition mean ± std ± 95% CI for every metric
  - ``summary.json`` with the same numbers in machine-readable form

Usage:
    GEMINI_API_KEY=… uv run python scripts/run_baseline_benchmark.py \\
        --output-dir data/baseline --tasks-per-condition 50

Real API calls are made; budget the run accordingly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from bicameral_agent.baseline_benchmark import format_report, run_benchmark
from bicameral_agent.cost_tracker import CostTracker
from bicameral_agent.dataset import ResearchQADataset, ResearchQATask, TaskDifficulty
from bicameral_agent.episode_runner import EpisodeConfig, EpisodeRunner
from bicameral_agent.gemini import GeminiClient
from bicameral_agent.heuristic_controller import HeuristicController
from bicameral_agent.no_subconscious_controller import NoSubconsciousController
from bicameral_agent.random_controller import RandomController
from bicameral_agent.serialization import episodes_to_parquet

logger = logging.getLogger(__name__)


def select_tasks(dataset: ResearchQADataset, total: int) -> list[ResearchQATask]:
    """Stratified pick of eval tasks across typical / hard / tricky.

    Splits the budget 50/25/25 across the three difficulties and falls back
    to filling from the full eval pool if any stratum is short.
    """
    eval_tasks = dataset.eval_tasks()
    by_diff = {
        d: [t for t in eval_tasks if t.difficulty == d]
        for d in (TaskDifficulty.TYPICAL, TaskDifficulty.HARD, TaskDifficulty.TRICKY)
    }
    quotas = {
        TaskDifficulty.TYPICAL: total // 2,
        TaskDifficulty.HARD: total // 4,
        TaskDifficulty.TRICKY: total - total // 2 - total // 4,
    }
    selected: list[ResearchQATask] = []
    for diff, n in quotas.items():
        selected.extend(by_diff[diff][:n])

    if len(selected) < total:
        seen = {t.task_id for t in selected}
        for t in eval_tasks:
            if t.task_id not in seen:
                selected.append(t)
                if len(selected) == total:
                    break
    return selected[:total]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/baseline")
    parser.add_argument("--tasks-per-condition", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-probability", type=float, default=0.2)
    parser.add_argument("--episode-budget", type=float, default=None,
                        help="Optional per-episode cost ceiling in USD.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ResearchQADataset()
    tasks = select_tasks(dataset, args.tasks_per_condition)
    logger.info("Selected %d tasks for benchmark", len(tasks))

    cost_tracker = CostTracker()
    if args.episode_budget is not None:
        cost_tracker.set_episode_budget(args.episode_budget)

    client = GeminiClient()
    runner = EpisodeRunner(
        client,
        config=EpisodeConfig(max_turns=args.max_turns, score_episode=True),
        cost_tracker=cost_tracker,
    )

    conditions = {
        "no_subconscious": lambda _idx: NoSubconsciousController(),
        "random": lambda idx: RandomController(
            action_probability=args.random_probability,
            seed=args.random_seed + idx,
        ),
        "heuristic": lambda _idx: HeuristicController(),
    }

    result = run_benchmark(client, tasks, conditions, runner=runner)

    for condition, episodes in result.episodes.items():
        episodes_to_parquet(episodes, str(output_dir / f"{condition}.parquet"))

    report_text = format_report(result)
    (output_dir / "report.txt").write_text(report_text)
    sys.stdout.write(report_text)

    summary = {
        "tasks_per_condition": args.tasks_per_condition,
        "max_turns": args.max_turns,
        "conditions": {
            condition: dataclasses.asdict(report)
            for condition, report in result.reports.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary.json and report.txt to %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
