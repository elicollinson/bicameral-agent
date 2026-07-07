"""Fit the MCTS transition model (issue #27) from parquet episode files.

Loads episodes, converts them to (state, action, next_state, reward)
tuples with :class:`~bicameral_agent.training_pipeline.TrainingDataPipeline`,
trains :class:`~bicameral_agent.transition_model.TransitionModel` with a
deterministic episode-level held-out split, and writes:

- ``<out-dir>/transition_model.pt``  — model checkpoint
- ``<out-dir>/metrics.json``         — config, data stats, loss curve,
  held-out evaluation and per-AC pass/fail flags

Usage (on the completed #46 baseline re-run)::

    uv run --extra torch python scripts/train_transition_model.py \\
        data/baseline_rerun/*.parquet --out-dir data/transition_model

Exit code is 0 whenever the fit itself succeeds; acceptance-criteria
outcomes are data-dependent and reported in ``metrics.json`` /  stdout
rather than via the exit code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from bicameral_agent.serialization import episodes_from_parquet
from bicameral_agent.training_pipeline import DEFAULT_MAX_TURNS, TrainingDataPipeline
from bicameral_agent.transition_model import (
    TransitionTrainingConfig,
    fit_transition_model,
)

# Issue #27 acceptance-criteria thresholds.
AC_STATE_MSE_MAX = 0.1
AC_REWARD_CORR_MIN = 0.4
AC_LATENCY_MS_MAX = 2.0
AC_TRAIN_SECONDS_MAX = 30 * 60


def acceptance_criteria(metrics: dict, train_seconds: float) -> dict[str, bool | None]:
    """Per-AC pass/fail flags (*None* = not evaluable on this data)."""
    rollout = metrics.get("rollout") or {}
    corr = metrics.get("reward_correlation")
    mse = metrics.get("state_mse_per_dim_mean")
    return {
        "state_mse_per_dim_mean_lt_0.1": None if mse is None else mse < AC_STATE_MSE_MAX,
        "reward_correlation_gt_0.4": None if corr is None else corr > AC_REWARD_CORR_MIN,
        "rollout_5_step_bounded": rollout.get("bounded"),
        "forward_latency_lt_2ms": metrics["latency_ms_median"] < AC_LATENCY_MS_MAX,
        "training_lt_30min_cpu": train_seconds < AC_TRAIN_SECONDS_MAX,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episodes", nargs="+",
        help="Parquet episode files (multi-episode format, e.g. data/baseline_rerun/*.parquet).",
    )
    parser.add_argument(
        "--out-dir", default="data/transition_model",
        help="Directory for transition_model.pt and metrics.json (default: %(default)s).",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
        help="EpisodeConfig.max_turns used when the episodes were generated "
        "(default: %(default)s).",
    )
    args = parser.parse_args(argv)

    episodes = []
    for path in args.episodes:
        loaded = episodes_from_parquet(path)
        print(f"loaded {len(loaded):3d} episodes from {path}")
        episodes.extend(loaded)
    if not episodes:
        print("error: no episodes found in the given files", file=sys.stderr)
        return 1

    pipeline = TrainingDataPipeline(max_turns=args.max_turns)
    examples = pipeline.process_episodes(episodes)
    if not examples:
        print("error: episodes produced no training examples", file=sys.stderr)
        return 1
    print(f"{len(episodes)} episodes -> {len(examples)} (state, action, next_state, reward) tuples")

    config = TransitionTrainingConfig(
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    result = fit_transition_model(examples, config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "transition_model.pt"
    result.model.save(model_path)

    acs = acceptance_criteria(result.metrics, result.train_seconds)
    payload = {
        "config": dataclasses.asdict(config),
        "data": {
            "episode_files": list(args.episodes),
            "n_episodes": len(episodes),
            "n_examples": len(examples),
            "n_train": result.n_train,
            "n_val": result.n_val,
        },
        "train": {
            "train_seconds": result.train_seconds,
            "first_epoch_loss": result.epoch_losses[0],
            "final_epoch_loss": result.epoch_losses[-1],
            "loss_curve_total": [e["total"] for e in result.epoch_losses],
        },
        "eval": result.metrics,
        "acceptance_criteria": acs,
        "model_path": str(model_path),
        "param_count": result.model.param_count,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    m = result.metrics
    print(f"\nsaved {model_path} ({result.model.param_count} params) and {metrics_path}")
    print(f"train: {result.n_train} examples, val: {result.n_val}, "
          f"{config.epochs} epochs in {result.train_seconds:.1f}s")
    print(f"train loss: {result.epoch_losses[0]['total']:.4f} -> "
          f"{result.epoch_losses[-1]['total']:.4f}")
    print(f"held-out state MSE (per-dim mean): {fmt(m['state_mse_per_dim_mean'])}  "
          f"(max dim: {fmt(m['state_mse_per_dim_max'])})")
    print(f"held-out reward MSE: {fmt(m['reward_mse'])}, "
          f"correlation r: {fmt(m['reward_correlation'])}")
    rollout = m.get("rollout") or {}
    print(f"rollout ({rollout.get('steps')}-step) max state norm: "
          f"{fmt(rollout.get('max_state_norm'))} (bound {fmt(rollout.get('norm_bound'))})")
    print(f"forward latency (median): {m['latency_ms_median']:.3f} ms")
    print("\nacceptance criteria:")
    for name, ok in acs.items():
        status = "PASS" if ok else ("FAIL" if ok is not None else "N/A ")
        print(f"  [{status}] {name}")
    return 0


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    sys.exit(main())
