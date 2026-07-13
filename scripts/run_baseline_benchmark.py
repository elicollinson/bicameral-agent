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
import logging
import sys
from pathlib import Path
from typing import Callable

from bicameral_agent.baseline_benchmark import (
    CONDITION_NAMES,
    format_report,
    parse_conditions,
    run_benchmark,
)
from bicameral_agent.config import HyperConfig
from bicameral_agent.dataset import ResearchQADataset, ResearchQATask, TaskDifficulty
from bicameral_agent.episode_runner import EpisodeRunner
from bicameral_agent.eval_datasets import build_dataset, dataset_names, resolve_metric
from bicameral_agent.eval_report import EvalReport
from bicameral_agent.no_subconscious_controller import NoSubconsciousController
from bicameral_agent.random_controller import RandomController
from bicameral_agent.runner_setup import (
    add_model_args,
    effective_hyper_config,
    resolve_parallel_episodes,
    resolve_runner_clients,
    resolve_search_provider,
)
from bicameral_agent.schema import Episode
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


def build_conditions(
    args: argparse.Namespace, hyper: HyperConfig, selected: tuple[str, ...]
) -> dict[str, Callable]:
    """Map the selected condition names to their controller factories.

    Raises:
        RuntimeError: If the factories here drift out of sync with
            ``CONDITION_NAMES`` (which ``--conditions`` is validated
            against), so a mismatch fails loudly at startup instead of
            KeyError-ing mid-run or silently dropping a condition.
    """
    factories = {
        "no_subconscious": lambda _idx: NoSubconsciousController(),
        "random": lambda idx: RandomController(
            action_probability=args.random_probability,
            seed=args.random_seed + idx,
        ),
        "heuristic": lambda _idx: hyper.to_heuristic_controller(),
    }
    if set(factories) != set(CONDITION_NAMES):
        raise RuntimeError(
            f"Controller factories {sorted(factories)} are out of sync with "
            f"CONDITION_NAMES {sorted(CONDITION_NAMES)}; update both together."
        )
    return {name: factories[name] for name in selected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/baseline")
    add_model_args(parser)
    parser.add_argument("--dataset", choices=dataset_names(), default="builtin",
                        help="Evaluation dataset to run against. External "
                             "datasets must be fetched first via "
                             "scripts/fetch_dataset.py.")
    parser.add_argument("--metric", default=None,
                        help="Verification metric (defaults to the dataset's "
                             "default_metric; must be in its supported_metrics).")
    parser.add_argument("--conditions", default=",".join(CONDITION_NAMES),
                        help="Comma-separated subset of conditions to run "
                             f"(of: {', '.join(CONDITION_NAMES)}; default all). "
                             "Lets an aborted run be resumed per-condition.")
    parser.add_argument("--tasks-per-condition", type=int, default=50)
    parser.add_argument("--parallel-episodes", type=int, default=None,
                        help="Episodes run concurrently within a condition "
                             "(bounded thread pool; 1 = sequential). Set this "
                             "to the provider plan's concurrent-request "
                             "allowance. Defaults to the config's [run] "
                             "parallel_episodes (else 1).")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-probability", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        selected_conditions = parse_conditions(args.conditions)
    except ValueError as exc:
        parser.error(str(exc))
    if args.parallel_episodes is not None and args.parallel_episodes < 1:
        parser.error(f"--parallel-episodes must be >= 1, got {args.parallel_episodes}")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hyper = (
        HyperConfig.from_toml(args.config) if args.config else HyperConfig.from_defaults()
    ).with_env_overrides()
    parallel_episodes = resolve_parallel_episodes(args, hyper)

    eval_dataset = build_dataset(args.dataset)
    metric = resolve_metric(eval_dataset, args.metric)
    dataset = eval_dataset.load()
    tasks = select_tasks(dataset, args.tasks_per_condition)
    logger.info(
        "Selected %d tasks from dataset %r (metric %r)", len(tasks), args.dataset, metric
    )

    cost_tracker = hyper.to_cost_tracker()
    if args.episode_budget is not None:
        cost_tracker.set_episode_budget(args.episode_budget)

    client, judge_client, provenance = resolve_runner_clients(args, hyper)

    runner = EpisodeRunner(
        client,
        config=hyper.to_episode_config(
            max_turns=args.max_turns,
            score_episode=True,
            metric=metric,
        ),
        hyper_config=effective_hyper_config(args, hyper, provenance),
        cost_tracker=cost_tracker,
        judge_client=judge_client,
        sim_user_client=judge_client,
        search_provider=resolve_search_provider(args, hyper),
    )

    conditions = build_conditions(args, hyper, selected_conditions)

    # Persist episodes incrementally: rewrite the condition's parquet after
    # every completed episode so a late crash keeps all prior results.
    # Episodes are keyed by task index and written sorted, so the file stays
    # in task order even when --parallel-episodes completes out of order
    # (run_condition serializes these callbacks on its coordinating thread).
    completed: dict[str, dict[int, Episode]] = {}

    def persist_episode(condition: str, idx: int, episode: Episode) -> None:
        by_index = completed.setdefault(condition, {})
        by_index[idx] = episode
        episodes_to_parquet(
            [by_index[i] for i in sorted(by_index)],
            str(output_dir / f"{condition}.parquet"),
        )

    result = run_benchmark(
        client, tasks, conditions, runner=runner, on_episode=persist_episode,
        parallel_episodes=parallel_episodes,
    )

    report_text = format_report(result)
    (output_dir / "report.txt").write_text(report_text)
    sys.stdout.write(report_text)

    report = EvalReport.from_benchmark(
        result,
        dataset=args.dataset,
        metric=metric,
        answerer=provenance["answerer"],
        measurement=provenance["measurement"],
        tasks_per_condition=args.tasks_per_condition,
        max_turns=args.max_turns,
    )
    (output_dir / "summary.json").write_text(report.to_json())
    logger.info("Wrote summary.json and report.txt to %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
