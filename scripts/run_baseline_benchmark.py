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

from bicameral_agent.baseline_benchmark import format_report, run_benchmark
from bicameral_agent.config import HyperConfig
from bicameral_agent.dataset import ResearchQADataset, ResearchQATask, TaskDifficulty
from bicameral_agent.episode_runner import EpisodeRunner
from bicameral_agent.eval_datasets import build_dataset, dataset_names, resolve_metric
from bicameral_agent.eval_report import EvalReport
from bicameral_agent.model_client import build_client, default_model, provider_names
from bicameral_agent.no_subconscious_controller import NoSubconsciousController
from bicameral_agent.random_controller import RandomController
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/baseline")
    parser.add_argument("--config", default=None,
                        help="Hyperparameter TOML file (defaults to the bundled "
                             "config; BICAMERAL_ env overrides always apply, "
                             "CLI flags win over both).")
    parser.add_argument("--provider", choices=list(provider_names()), default=None,
                        help="Model backend to run the answerer against "
                             "(overrides the config file).")
    parser.add_argument("--model", default=None,
                        help="Model id/tag (overrides the config file).")
    parser.add_argument("--judge-provider", choices=list(provider_names()),
                        default=None,
                        help="Model backend for the measurement roles (LLM "
                             "judge and simulated user), held fixed while "
                             "--provider varies. Defaults to the config's "
                             "[measurement_model], else gemini.")
    parser.add_argument("--judge-model", default=None,
                        help="Measurement model id/tag (overrides the config "
                             "file; unset uses the judge provider's default).")
    parser.add_argument("--dataset", choices=dataset_names(), default="builtin",
                        help="Evaluation dataset to run against. External "
                             "datasets must be fetched first via "
                             "scripts/fetch_dataset.py.")
    parser.add_argument("--metric", default=None,
                        help="Verification metric (defaults to the dataset's "
                             "default_metric; must be in its supported_metrics).")
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

    hyper = (
        HyperConfig.from_toml(args.config) if args.config else HyperConfig.from_defaults()
    ).with_env_overrides()

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

    provider = args.provider or hyper.model.provider
    # The configured model name only applies to the configured provider.
    model = args.model or (hyper.model.name if provider == hyper.model.provider else None)
    client = build_client(provider, model)

    # Measurement roles (LLM judge + simulated user) are pinned independently
    # of the answerer so cross-model comparisons stay on one judging scale
    # (issue #53). Precedence mirrors the answerer's: CLI > config > gemini.
    measurement = hyper.measurement_model
    judge_provider = args.judge_provider or (
        measurement.provider if measurement is not None else "gemini"
    )
    judge_model = args.judge_model or (
        measurement.name
        if measurement is not None and judge_provider == measurement.provider
        else None
    )
    resolved_judge_model = judge_model or default_model(judge_provider)
    if judge_provider == provider and resolved_judge_model == client.model:
        judge_client = client
    else:
        judge_client = build_client(judge_provider, judge_model)
    logger.info(
        "Answerer: %s/%s; measurement (judge + sim-user): %s/%s",
        provider, client.model, judge_provider, judge_client.model,
    )

    runner = EpisodeRunner(
        client,
        config=hyper.to_episode_config(
            max_turns=args.max_turns,
            score_episode=True,
            metric=metric,
        ),
        hyper_config=hyper,
        cost_tracker=cost_tracker,
        judge_client=judge_client,
        sim_user_client=judge_client,
    )

    conditions = {
        "no_subconscious": lambda _idx: NoSubconsciousController(),
        "random": lambda idx: RandomController(
            action_probability=args.random_probability,
            seed=args.random_seed + idx,
        ),
        "heuristic": lambda _idx: hyper.to_heuristic_controller(),
    }

    # Persist episodes incrementally: rewrite the condition's parquet after
    # every completed episode so a late crash keeps all prior results.
    completed: dict[str, list[Episode]] = {}

    def persist_episode(condition: str, _idx: int, episode: Episode) -> None:
        completed.setdefault(condition, []).append(episode)
        episodes_to_parquet(
            completed[condition], str(output_dir / f"{condition}.parquet")
        )

    result = run_benchmark(
        client, tasks, conditions, runner=runner, on_episode=persist_episode
    )

    report_text = format_report(result)
    (output_dir / "report.txt").write_text(report_text)
    sys.stdout.write(report_text)

    report = EvalReport.from_benchmark(
        result,
        dataset=args.dataset,
        metric=metric,
        answerer={"provider": provider, "model": client.model},
        measurement={"provider": judge_provider, "model": judge_client.model},
        tasks_per_condition=args.tasks_per_condition,
        max_turns=args.max_turns,
    )
    (output_dir / "summary.json").write_text(report.to_json())
    logger.info("Wrote summary.json and report.txt to %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
