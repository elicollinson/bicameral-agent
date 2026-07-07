"""Pre-train the policy/value network on heuristic episodes (issue #26).

Loads heuristic-condition episodes from parquet files (and/or training
examples from a :class:`~bicameral_agent.training_data_store.TrainingDataStore`),
converts episodes to (state, action, discounted-return) examples with
:class:`~bicameral_agent.training_pipeline.TrainingDataPipeline`, trains
:class:`~bicameral_agent.policy_value_net.PolicyValueNetwork` with a
deterministic episode-grouped 80/20 split and early stopping on the
validation loss, and writes:

- ``<out-dir>/policy_value_pretrained.pt`` — model checkpoint
- ``<out-dir>/metrics.json``               — config, data stats, per-epoch
  history, held-out evaluation and per-AC pass/fail flags
- ``<out-dir>/training_curves.png``        — loss / action accuracy /
  value correlation vs. epoch

Usage (on the completed #46 baseline re-run)::

    uv run --extra dev --extra torch python scripts/pretrain_policy.py \\
        data/baseline_rerun/heuristic.parquet --out-dir data/pretrain

Exit code is 0 whenever training itself succeeds; acceptance-criteria
outcomes are data-dependent and reported in ``metrics.json`` / stdout
rather than via the exit code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from bicameral_agent.pretrain import (
    PretrainConfig,
    PretrainResult,
    pretrain_policy_value,
)
from bicameral_agent.serialization import episodes_from_parquet
from bicameral_agent.training_pipeline import DEFAULT_MAX_TURNS, TrainingDataPipeline

# Issue #26 acceptance-criteria thresholds.
AC_ACTION_ACCURACY_MIN = 0.8
AC_VALUE_CORR_MIN = 0.3
AC_MONOTONIC_EPOCHS = 10
AC_OVERFIT_GAP_MAX = 0.2
AC_TRAIN_SECONDS_MAX = 30 * 60

# Series colors: categorical slots 1 (blue) and 2 (aqua) of the
# reference palette, light mode (PNGs render on a white surface).
_C_TRAIN = "#2a78d6"
_C_VAL = "#1baf7a"
_C_THRESHOLD = "#52514e"


def acceptance_criteria(result: PretrainResult) -> dict[str, bool | None]:
    """Per-AC pass/fail flags (*None* = not evaluable on this data)."""
    accuracy = result.metrics.get("action_accuracy")
    corr = result.metrics.get("value_correlation")
    train_losses = [e["train_loss"] for e in result.history]
    monotonic = len(train_losses) >= AC_MONOTONIC_EPOCHS and all(
        train_losses[i + 1] < train_losses[i] for i in range(AC_MONOTONIC_EPOCHS - 1)
    )
    best = result.history[result.best_epoch - 1]
    no_overfit = best["val_loss"] <= (1.0 + AC_OVERFIT_GAP_MAX) * best["train_loss"]
    return {
        "action_accuracy_ge_0.8": None if accuracy is None else accuracy >= AC_ACTION_ACCURACY_MIN,
        "value_correlation_gt_0.3": None if corr is None else corr > AC_VALUE_CORR_MIN,
        "train_loss_monotonic_10_epochs": monotonic,
        "val_loss_within_20pct_of_train": bool(no_overfit),
        "training_lt_30min_cpu": result.train_seconds < AC_TRAIN_SECONDS_MAX,
    }


def save_training_curves(result: PretrainResult, path: Path) -> None:
    """Plot loss / accuracy / correlation vs. epoch and save as one PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(result.history) + 1))
    fig, (ax_loss, ax_acc, ax_corr) = plt.subplots(1, 3, figsize=(11.0, 3.4))

    def style(ax, title: str) -> None:
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        ax.set_xlabel("epoch", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.5, alpha=0.25)
        ax.tick_params(labelsize=8)

    ax_loss.plot(
        epochs, [e["train_loss"] for e in result.history],
        color=_C_TRAIN, linewidth=2, label="train",
    )
    ax_loss.plot(
        epochs, [e["val_loss"] for e in result.history],
        color=_C_VAL, linewidth=2, label="validation",
    )
    ax_loss.axvline(
        result.best_epoch, color=_C_THRESHOLD, linewidth=1, linestyle="--", alpha=0.6
    )
    style(ax_loss, "Loss (best epoch dashed)")
    ax_loss.legend(fontsize=8, frameon=False)

    ax_acc.plot(
        epochs, [e["val_action_accuracy"] for e in result.history],
        color=_C_TRAIN, linewidth=2,
    )
    ax_acc.axhline(
        AC_ACTION_ACCURACY_MIN, color=_C_THRESHOLD, linewidth=1, linestyle="--", alpha=0.6
    )
    ax_acc.set_ylim(0.0, 1.05)  # headroom so a 1.0 line is not clipped
    style(ax_acc, "Validation action accuracy (AC: 0.8)")

    # Correlation can be None on degenerate epochs; plot those as gaps.
    corr = [
        e["val_value_correlation"] if e["val_value_correlation"] is not None else float("nan")
        for e in result.history
    ]
    ax_corr.plot(epochs, corr, color=_C_TRAIN, linewidth=2)
    ax_corr.axhline(
        AC_VALUE_CORR_MIN, color=_C_THRESHOLD, linewidth=1, linestyle="--", alpha=0.6
    )
    ax_corr.set_ylim(-1.0, 1.0)
    style(ax_corr, "Validation value correlation (AC: 0.3)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episodes", nargs="*",
        help="Parquet episode files for the heuristic condition "
        "(e.g. data/baseline_rerun/heuristic.parquet).",
    )
    parser.add_argument(
        "--store", default=None,
        help="Optional TrainingDataStore root to load training examples from, "
        "in addition to (or instead of) parquet episode files.",
    )
    parser.add_argument(
        "--out-dir", default="data/pretrain",
        help="Directory for the checkpoint, metrics.json and training_curves.png "
        "(default: %(default)s).",
    )
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
        help="EpisodeConfig.max_turns used when the episodes were generated "
        "(default: %(default)s).",
    )
    args = parser.parse_args(argv)

    if not args.episodes and args.store is None:
        parser.error("provide parquet episode files and/or --store")

    examples = []
    n_episodes = 0
    if args.episodes:
        episodes = []
        for path in args.episodes:
            loaded = episodes_from_parquet(path)
            print(f"loaded {len(loaded):3d} episodes from {path}")
            episodes.extend(loaded)
        n_episodes = len(episodes)
        pipeline = TrainingDataPipeline(max_turns=args.max_turns)
        examples.extend(pipeline.process_episodes(episodes))
    if args.store is not None:
        from bicameral_agent.training_data_store import TrainingDataStore

        stored = TrainingDataStore(args.store).load_examples()
        print(f"loaded {len(stored):3d} examples from store {args.store}")
        examples.extend(stored)
    if not examples:
        print("error: inputs produced no training examples", file=sys.stderr)
        return 1
    print(f"{n_episodes} episodes -> {len(examples)} (state, action, return) examples")

    config = PretrainConfig(
        hidden_dim=args.hidden_dim,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        value_loss_weight=args.value_loss_weight,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    result = pretrain_policy_value(examples, config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "policy_value_pretrained.pt"
    result.model.save(model_path)
    curves_path = out_dir / "training_curves.png"
    save_training_curves(result, curves_path)

    acs = acceptance_criteria(result)
    payload = {
        "config": dataclasses.asdict(config),
        "data": {
            "episode_files": list(args.episodes),
            "store": args.store,
            "n_episodes": n_episodes,
            "n_examples": len(examples),
            "n_train": result.n_train,
            "n_val": result.n_val,
        },
        "train": {
            "train_seconds": result.train_seconds,
            "epochs_run": len(result.history),
            "best_epoch": result.best_epoch,
            "first_epoch": result.history[0],
            "final_epoch": result.history[-1],
            "history": result.history,
        },
        "eval": result.metrics,
        "acceptance_criteria": acs,
        "model_path": str(model_path),
        "curves_path": str(curves_path),
        "param_count": result.model.param_count,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    m = result.metrics
    print(f"\nsaved {model_path} ({result.model.param_count} params), "
          f"{metrics_path} and {curves_path}")
    print(f"train: {result.n_train} examples, val: {result.n_val}, "
          f"{len(result.history)} epochs (best: {result.best_epoch}) "
          f"in {result.train_seconds:.1f}s")
    print(f"train loss: {result.history[0]['train_loss']:.4f} -> "
          f"{result.history[-1]['train_loss']:.4f}, "
          f"best val loss: {result.history[result.best_epoch - 1]['val_loss']:.4f}")
    print(f"val action accuracy: {fmt(m['action_accuracy'])}  "
          f"(constant-majority baseline: {fmt(m['majority_action_fraction'])})")
    print(f"val value correlation r: {fmt(m['value_correlation'])}, "
          f"value MSE: {fmt(m['value_mse'])}")
    print("\nacceptance criteria:")
    for name, ok in acs.items():
        status = "PASS" if ok else ("FAIL" if ok is not None else "N/A ")
        print(f"  [{status}] {name}")
    return 0


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    sys.exit(main())
