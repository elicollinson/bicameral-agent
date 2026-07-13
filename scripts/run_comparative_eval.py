"""Run the Issue #30 comparative evaluation across all five controllers.

Runs {no_subconscious, random, heuristic, learned_no_search,
learned_with_search} paired over the same evaluation task set and emits:

  - Per-episode parquet files under ``--output-dir/<condition>.parquet``
    (rewritten after every episode so a late crash keeps prior results)
  - ``report.json`` (machine-readable: summaries, pairwise Welch t-tests
    with p-values, per-difficulty breakdown)
  - ``report.md`` (human-readable rendering of the same numbers)

``--tasks`` takes either a total (split 50/25/25 across
typical/hard/tricky) or an explicit mix, e.g. ``typical=50,hard=25,tricky=25``.
The learned conditions need the ``torch`` extra and checkpoints from
scripts/train_mcts.py (or scripts/pretrain_policy.py +
scripts/train_transition_model.py).

Usage:
    GEMINI_API_KEY=… uv run --extra torch python scripts/run_comparative_eval.py \\
        --output-dir data/comparative --dataset hard_benchmark \\
        --tasks typical=50,hard=25,tricky=25 \\
        --policy-checkpoint ckpt/policy_value.pt \\
        --transition-checkpoint ckpt/transition.pt

Real API calls are made; budget the run accordingly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bicameral_agent.baseline_benchmark import FAILURE_THRESHOLD
from bicameral_agent.comparative_eval import (
    ComparativeEvaluator,
    baseline_condition_factories,
    build_report,
    learned_condition_factories,
    parse_task_mix,
    select_tasks,
)
from bicameral_agent.config import HyperConfig
from bicameral_agent.episode_runner import EpisodeRunner
from bicameral_agent.eval_datasets import build_dataset, dataset_names, resolve_metric
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/comparative")
    add_model_args(parser)
    parser.add_argument("--dataset", choices=dataset_names(), default="builtin",
                        help="Evaluation dataset to run against. External "
                             "datasets must be fetched first via "
                             "scripts/fetch_dataset.py.")
    parser.add_argument("--metric", default=None,
                        help="Verification metric (defaults to the dataset's "
                             "default_metric; must be in its supported_metrics).")
    parser.add_argument("--tasks", default="100",
                        help="Task count (split 50/25/25 across "
                             "typical/hard/tricky) or an explicit mix, e.g. "
                             "'typical=50,hard=25,tricky=25'.")
    parser.add_argument("--parallel-episodes", type=int, default=None,
                        help="Episodes run concurrently within a condition "
                             "(bounded thread pool; 1 = sequential). Set this "
                             "to the provider plan's concurrent-request "
                             "allowance. Defaults to the config's [run] "
                             "parallel_episodes (else 1).")
    parser.add_argument("--policy-checkpoint", required=True,
                        help="PolicyValueNetwork checkpoint (.pt) for the "
                             "learned conditions.")
    parser.add_argument("--transition-checkpoint", required=True,
                        help="TransitionModel checkpoint (.pt) for MCTS search.")
    parser.add_argument("--policy-hidden-dim", type=int, default=160,
                        help="Hidden width of the policy checkpoint "
                             "(must match training).")
    parser.add_argument("--transition-hidden-dim", type=int, default=128,
                        help="Hidden width of the transition checkpoint "
                             "(must match training).")
    parser.add_argument("--num-simulations", type=int, default=50,
                        help="MCTS budget per decision for learned_with_search.")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; per-condition seeds are derived "
                             "deterministically from it.")
    parser.add_argument("--random-probability", type=float, default=0.2)
    parser.add_argument("--failure-threshold", type=float, default=FAILURE_THRESHOLD,
                        help="Abort a condition once contained transport "
                             "failures exceed this fraction of its episodes.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

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
    mix = parse_task_mix(args.tasks)
    tasks = select_tasks(dataset, mix)
    logger.info(
        "Selected %d tasks from dataset %r (metric %r, mix %s)",
        len(tasks), args.dataset, metric,
        {d.value: n for d, n in mix.items() if n},
    )

    conditions = {
        **baseline_condition_factories(
            hyper.to_heuristic_controller,
            random_probability=args.random_probability,
            base_seed=args.seed,
        ),
        **learned_condition_factories(
            args.policy_checkpoint,
            args.transition_checkpoint,
            num_simulations=args.num_simulations,
            max_turns=args.max_turns,
            policy_hidden_dim=args.policy_hidden_dim,
            transition_hidden_dim=args.transition_hidden_dim,
            base_seed=args.seed,
        ),
    }

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

    evaluator = ComparativeEvaluator(
        runner,
        on_episode=persist_episode,
        failure_threshold=args.failure_threshold,
        parallel_episodes=parallel_episodes,
    )
    result = evaluator.run(tasks, conditions)

    report = build_report(
        result,
        dataset=args.dataset,
        metric=metric,
        answerer=provenance["answerer"],
        measurement=provenance["measurement"],
        max_turns=args.max_turns,
        base_seed=args.seed,
    )
    (output_dir / "report.json").write_text(report.to_json())
    markdown = report.to_markdown()
    (output_dir / "report.md").write_text(markdown)
    sys.stdout.write(markdown)
    logger.info("Wrote report.json and report.md to %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
