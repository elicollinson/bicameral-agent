"""Run the Issue #29 MCTS training loop.

Live mode (default) collects episodes with the current learned policy via
the EpisodeRunner (real API calls; budget accordingly), generates MCTS
targets, trains the policy/value network, evaluates against the heuristic
on held-out tasks, and checkpoints every iteration:

    GEMINI_API_KEY=... uv run python scripts/train_mcts.py \\
        --output-dir data/mcts_training \\
        --iterations 10 --episodes-per-iteration 50 --simulations 50

Offline mode consumes pre-collected episode parquet files instead (no LLM
client needed; collection and held-out evaluation are skipped):

    uv run python scripts/train_mcts.py \\
        --output-dir data/mcts_training \\
        --episodes-parquet data/baseline/heuristic.parquet \\
        --iterations 3 --simulations 50

Outputs under --output-dir: iteration-NNN/{policy_value.pt,transition.pt,
metrics.json}, metrics_history.json, store/ (training examples), and
episodes/iteration-NNN.parquet for live-collected episodes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from bicameral_agent.config import HyperConfig
from bicameral_agent.dataset import (
    ResearchQADataset,
    ResearchQATask,
    TaskDifficulty,
)
from bicameral_agent.episode_runner import EpisodeRunner
from bicameral_agent.mcts_trainer import MCTSTrainer, MCTSTrainerConfig
from bicameral_agent.policy_value_net import PolicyValueNetwork
from bicameral_agent.runner_setup import add_model_args, resolve_runner_clients
from bicameral_agent.schema import Episode
from bicameral_agent.serialization import episodes_from_parquet
from bicameral_agent.training_pipeline import STATE_DIM, TrainingDataPipeline
from bicameral_agent.transition_model import TransitionModel

logger = logging.getLogger(__name__)


def select_eval_tasks(
    dataset: ResearchQADataset, total: int
) -> list[ResearchQATask]:
    """Stratified held-out pick across typical / hard / tricky (50/25/25)."""
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


def load_models(args: argparse.Namespace) -> tuple[PolicyValueNetwork, TransitionModel]:
    """Build (or load from checkpoints) the policy/value and transition nets."""
    if args.policy_checkpoint:
        policy = PolicyValueNetwork.load(
            args.policy_checkpoint,
            input_dim=STATE_DIM,
            hidden_dim=args.policy_hidden_dim,
        )
    else:
        policy = PolicyValueNetwork(
            input_dim=STATE_DIM, hidden_dim=args.policy_hidden_dim
        )
    if args.transition_checkpoint:
        transition = TransitionModel.load(
            args.transition_checkpoint, hidden_dim=args.transition_hidden_dim
        )
    else:
        transition = TransitionModel(hidden_dim=args.transition_hidden_dim)
    return policy, transition


def build_runner(args: argparse.Namespace, hyper: HyperConfig) -> tuple[EpisodeRunner, object]:
    """Build the live EpisodeRunner (answerer + pinned measurement roles)."""
    cost_tracker = hyper.to_cost_tracker()
    if args.episode_budget is not None:
        cost_tracker.set_episode_budget(args.episode_budget)

    client, judge_client, _provenance = resolve_runner_clients(args, hyper)

    runner = EpisodeRunner(
        client,
        config=hyper.to_episode_config(
            max_turns=args.max_turns,
            score_episode=True,
            metric=args.metric,
        ),
        hyper_config=hyper,
        cost_tracker=cost_tracker,
        judge_client=judge_client,
        sim_user_client=judge_client,
    )
    return runner, cost_tracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/mcts_training")
    add_model_args(parser)
    parser.add_argument("--metric", default="llm_judge",
                        help="Verification metric used to score episodes.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes-per-iteration", type=int, default=50)
    parser.add_argument("--parallel-episodes", type=int, default=1,
                        help="Collection episodes run concurrently (bounded "
                             "thread pool; 1 = sequential). Match the "
                             "provider's concurrent-request allowance "
                             "(Ollama Cloud: 3).")
    parser.add_argument("--simulations", type=int, default=50,
                        help="MCTS budget per decision point.")
    parser.add_argument("--eval-tasks", type=int, default=20,
                        help="Held-out evaluation tasks per iteration "
                             "(0 disables evaluation).")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Episode turn limit; also configures the "
                             "pipeline's completion-fraction features.")
    parser.add_argument("--episodes-parquet", nargs="*", default=None,
                        help="Pre-collected episode parquet file(s): run "
                             "offline iterations on these episodes instead "
                             "of collecting (no LLM client needed).")
    parser.add_argument("--policy-checkpoint", default=None,
                        help="Initial policy/value weights (e.g. from the "
                             "issue #26 pre-training run).")
    parser.add_argument("--policy-hidden-dim", type=int, default=160)
    parser.add_argument("--transition-checkpoint", default=None,
                        help="Initial transition-model weights (issue #27).")
    parser.add_argument("--transition-hidden-dim", type=int, default=128)
    parser.add_argument("--retrain-transition", action="store_true",
                        help="Refit the transition model on all stored "
                             "examples each iteration.")
    parser.add_argument("--no-search", action="store_true",
                        help="Collect and evaluate with the raw policy "
                             "instead of MCTS-improved actions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offline = bool(args.episodes_parquet)

    policy, transition = load_models(args)

    if args.parallel_episodes < 1:
        parser.error(f"--parallel-episodes must be >= 1, got {args.parallel_episodes}")

    trainer_config = MCTSTrainerConfig(
        collect_with_search=not args.no_search,
        eval_with_search=not args.no_search,
        retrain_transition=args.retrain_transition,
        max_turns=args.max_turns,
        seed=args.seed,
        parallel_episodes=args.parallel_episodes,
    )
    pipeline = TrainingDataPipeline(max_turns=args.max_turns)

    offline_episodes: list[Episode] | None = None
    runner = None
    cost_tracker = None
    train_tasks: list[ResearchQATask] = []
    eval_tasks: list[ResearchQATask] = []
    hyper = (
        HyperConfig.from_toml(args.config) if args.config else HyperConfig.from_defaults()
    ).with_env_overrides()

    if offline:
        offline_episodes = []
        for path in args.episodes_parquet:
            offline_episodes.extend(episodes_from_parquet(path))
        logger.info(
            "Offline mode: %d episodes from %d parquet file(s); collection "
            "and held-out evaluation are skipped",
            len(offline_episodes), len(args.episodes_parquet),
        )
    else:
        runner, cost_tracker = build_runner(args, hyper)
        dataset = ResearchQADataset()
        eval_tasks = select_eval_tasks(dataset, args.eval_tasks)
        held_out = {t.task_id for t in eval_tasks}
        train_tasks = [t for t in dataset.tasks() if t.task_id not in held_out]
        logger.info(
            "Live mode: %d collection tasks, %d held-out eval tasks",
            len(train_tasks), len(eval_tasks),
        )

    trainer = MCTSTrainer(
        policy,
        transition,
        checkpoint_dir=output_dir,
        config=trainer_config,
        runner=runner,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        heuristic_factory=hyper.to_heuristic_controller,
        pipeline=pipeline,
    )

    start = time.perf_counter()
    for _ in range(args.iterations):
        metrics = trainer.run_iteration(
            args.episodes_per_iteration,
            args.simulations,
            episodes=offline_episodes,
        )
        line = (
            f"iteration {metrics.iteration}: loss={metrics.train_loss:.4f} "
            f"entropy={metrics.policy_entropy:.3f} "
            f"kl_from_heuristic={metrics.kl_from_heuristic:.4f} "
            f"agreement={metrics.heuristic_agreement:.3f} "
            f"eval={metrics.eval_score} heuristic={metrics.heuristic_eval_score}"
        )
        sys.stdout.write(line + "\n")
    total_seconds = time.perf_counter() - start

    sys.stdout.write(
        f"completed {args.iterations} iteration(s) in {total_seconds:.1f}s; "
        f"checkpoints + metrics_history.json under {output_dir}\n"
    )
    if cost_tracker is not None:
        report = cost_tracker.get_total()
        sys.stdout.write(
            f"session API cost: ${report.total:.4f} across {report.call_count} calls\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
