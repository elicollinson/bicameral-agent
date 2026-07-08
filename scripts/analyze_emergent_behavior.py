"""Emergent behavior analysis of the learned MCTS policy (issue #32).

Reproducibly characterizes the behavior the policy/value network learned
during MCTS self-improvement, contrasts it with the hand-written heuristic
baseline, and surfaces where (if anywhere) the learned policy departs from
what was explicitly programmed.

Data sources (all committed under ``data/``):

- ``--mcts-dir`` (default ``data/mcts_training``): ``metrics_history.json``
  (per-iteration training metrics), ``episodes/iteration-*.parquet`` (the
  learned-policy episodes collected each iteration) and per-iteration
  ``policy_value.pt`` checkpoints.
- ``--comparative-dir`` (default ``data/comparative``): per-condition
  episode parquet files from the #30 comparative evaluation, used for the
  empirical learned-vs-heuristic tool-usage cross-check.

Outputs (default ``docs/figures/emergent/``): a set of PNG figures plus
``emergent_stats.json`` with every computed number.

Figures that probe the trained network require the ``torch`` extra; when
torch is unavailable they are skipped and the metrics/episode-derived
figures are still produced, so the script runs on any checkout.

Usage::

    uv run --extra torch python scripts/analyze_emergent_behavior.py \\
        --mcts-dir data/mcts_training \\
        --comparative-dir data/comparative \\
        --out-dir docs/figures/emergent
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bicameral_agent.encoder import _QUEUE_DEPTH_CAP
from bicameral_agent.heuristic_controller import (
    Action,
    FullState,
    HeuristicController,
)
from bicameral_agent.replay import EpisodeReplayer
from bicameral_agent.serialization import episodes_from_parquet
from bicameral_agent.signal_classifier import SignalClassifier
from bicameral_agent.training_pipeline import (
    _ACTION_INDEX,
    _OFF_QUEUE,
    STATE_DIM,
    TrainingDataPipeline,
)

if TYPE_CHECKING:  # pragma: no cover
    from bicameral_agent.schema import Episode

logger = logging.getLogger(__name__)

# Action index order (mirrors policy_value_net.ACTION_ORDER).
ACTION_ORDER: tuple[Action, ...] = (
    Action.SCANNER,
    Action.AUDITOR,
    Action.REFRESHER,
    Action.DO_NOTHING,
)
_DO_NOTHING_IDX = 3

# Fixed categorical palette (blue/orange is a CVD-safe pair). Color follows
# the entity, never its rank.
C_LEARNED = "#2563eb"
C_HEURISTIC = "#d97706"
C_INVOKE = "#059669"
C_INHIBIT = "#7c3aed"
C_MUTED = "#6b7280"
_GRID = {"color": "#d1d5db", "linewidth": 0.6, "alpha": 0.7}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_metrics_history(mcts_dir: Path) -> list[dict]:
    return json.loads((mcts_dir / "metrics_history.json").read_text(encoding="utf-8"))


def load_iteration_episodes(mcts_dir: Path) -> dict[int, list[Episode]]:
    """Map iteration index -> learned-policy episodes collected that round."""
    out: dict[int, list[Episode]] = {}
    for path in sorted((mcts_dir / "episodes").glob("iteration-*.parquet")):
        it = int(path.stem.split("-")[-1])
        out[it] = episodes_from_parquet(str(path))
    return out


# ---------------------------------------------------------------------------
# Episode-derived (torch-free) statistics
# ---------------------------------------------------------------------------


def episode_tool_stats(episodes: list[Episode]) -> dict:
    """Per-iteration behavioral summary derived directly from episodes."""
    tool_counts: Counter[str] = Counter()
    n_inv: Counter[int] = Counter()
    per_turn_tool: Counter[int] = Counter()
    multi_tool = 0
    secondary_tools: Counter[str] = Counter()  # 2nd+ tool within an episode
    interrupts = 0
    expiries = 0
    quality = 0.0
    for ep in episodes:
        invs = ep.tool_invocations
        n_inv[len(invs)] += 1
        if len(invs) > 1:
            multi_tool += 1
            for inv in sorted(invs, key=lambda i: i.invoked_at_ms)[1:]:
                secondary_tools[inv.tool_id] += 1
        for inv in invs:
            tool_counts[inv.tool_id] += 1
            if inv.turn is not None:
                per_turn_tool[inv.turn] += 1
        interrupts += ep.metadata.get("interrupt_count", 0)
        expiries += ep.metadata.get("expired_queue_items", 0)
        quality += ep.outcome.quality_score or 0.0
    n = len(episodes)
    return {
        "n_episodes": n,
        "tool_counts": dict(tool_counts),
        "n_invocations_dist": {str(k): v for k, v in n_inv.items()},
        "multi_tool_episodes": multi_tool,
        "multi_tool_rate": multi_tool / n if n else 0.0,
        "secondary_tool_counts": dict(secondary_tools),
        "per_turn_invocations": {str(k): v for k, v in per_turn_tool.items()},
        "interrupt_count": interrupts,
        "expired_queue_items": expiries,
        "mean_quality": quality / n if n else 0.0,
    }


def compound_sequence_stats(episodes: list[Episode]) -> dict:
    """Observed rate of a 2nd *distinct* tool following the 1st vs chance.

    "Compound skill" = the policy chaining tool A then a different tool B
    inside one episode. Chance baseline: if each follow-on slot drew a tool
    uniformly from the 3-tool vocabulary, P(different from first) = 2/3.
    """
    episodes_with_2plus = 0
    distinct_pair_episodes = 0
    for ep in episodes:
        invs = sorted(ep.tool_invocations, key=lambda i: i.invoked_at_ms)
        if len(invs) < 2:
            continue
        episodes_with_2plus += 1
        if any(inv.tool_id != invs[0].tool_id for inv in invs[1:]):
            distinct_pair_episodes += 1
    observed = (
        distinct_pair_episodes / episodes_with_2plus if episodes_with_2plus else 0.0
    )
    return {
        "episodes_with_2plus_invocations": episodes_with_2plus,
        "episodes_with_distinct_second_tool": distinct_pair_episodes,
        "observed_distinct_second_rate": observed,
        "chance_distinct_second_rate": 2.0 / 3.0,
    }


def comparative_tool_usage(comparative_dir: Path) -> dict:
    """Empirical tool-invocation counts per comparative condition."""
    out: dict[str, dict] = {}
    for path in sorted(comparative_dir.glob("*.parquet")):
        cond = path.stem
        eps = episodes_from_parquet(str(path))
        tools: Counter[str] = Counter()
        n_inv: Counter[int] = Counter()
        for ep in eps:
            n_inv[len(ep.tool_invocations)] += 1
            for inv in ep.tool_invocations:
                tools[inv.tool_id] += 1
        out[cond] = {
            "n_episodes": len(eps),
            "tool_counts": dict(tools),
            "n_invocations_dist": {str(k): v for k, v in n_inv.items()},
        }
    return out


def heuristic_actions_for_episodes(
    episodes: list[Episode],
) -> list[tuple[int, int, int]]:
    """Return (turn, heuristic_action_idx, queue_depth) per decision point.

    Mirrors ``mcts_trainer._heuristic_comparison``: the heuristic decides on
    FullStates reconstructed from each episode's decision points.
    """
    out: list[tuple[int, int, int]] = []
    for ep in episodes:
        controller = HeuristicController()
        for dp in EpisodeReplayer(ep).iter_decision_points():
            signals = SignalClassifier.classify(
                list(dp.state.messages), list(dp.state.user_events)
            )
            fs = FullState(
                turn_number=dp.state.turn_number,
                stop_count=signals.stop_count.value,
                followup_type=signals.followup_type,
                queue_depth=len(dp.state.pending_injections),
                executing_tools=(),
                predicted_latencies={},
            )
            out.append(
                (
                    dp.state.turn_number,
                    _ACTION_INDEX[controller.decide(fs)],
                    len(dp.state.pending_injections),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Policy-network probes (torch-gated)
# ---------------------------------------------------------------------------


def reconstruct_states(
    episodes: list[Episode], pipeline: TrainingDataPipeline
) -> tuple[np.ndarray, np.ndarray]:
    """Build (states, turns) for every decision point in ``episodes``."""
    states: list[np.ndarray] = []
    turns: list[int] = []
    for ep in episodes:
        for dp in EpisodeReplayer(ep).iter_decision_points():
            states.append(
                pipeline.build_state_vector(dp.state, dp.action.timestamp_ms)
            )
            turns.append(dp.state.turn_number)
    return np.asarray(states, dtype=np.float32), np.asarray(turns, dtype=np.int64)


def policy_probs(net, states: np.ndarray) -> np.ndarray:
    """(n, 4) action-probability matrix for a batch of states."""
    return np.asarray([net.predict(s)[0] for s in states], dtype=np.float64)


def policy_value(net, state: np.ndarray) -> float:
    return float(net.predict(state)[1])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style(ax):
    ax.grid(True, **_GRID)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def fig_training_dynamics(plt, history: list[dict], out: Path) -> None:
    its = [m["iteration"] for m in history]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Training dynamics across MCTS iterations", fontweight="bold")

    ax = _style(axes[0, 0])
    ax.plot(its, [m["policy_entropy"] for m in history], "o-", color=C_LEARNED)
    ax.set_title("Policy entropy (nats)")
    ax.set_xlabel("iteration")

    ax = _style(axes[0, 1])
    ax.plot(its, [m["kl_from_heuristic"] for m in history], "o-", color=C_LEARNED)
    ax.set_title("KL(policy ∥ heuristic)")
    ax.set_xlabel("iteration")

    ax = _style(axes[0, 2])
    ax.plot(its, [m["heuristic_agreement"] for m in history], "o-", color=C_LEARNED)
    ax.set_ylim(0.9, 1.01)
    ax.set_title("Argmax agreement with heuristic")
    ax.set_xlabel("iteration")

    ax = _style(axes[1, 0])
    ax.plot(its, [m["eval_score"] for m in history], "o-", color=C_LEARNED, label="learned")
    ax.plot(
        its,
        [m["heuristic_eval_score"] for m in history],
        "s--",
        color=C_HEURISTIC,
        label="heuristic",
    )
    ax.set_title("Held-out eval quality")
    ax.set_xlabel("iteration")
    ax.legend(frameon=False)

    ax = _style(axes[1, 1])
    ax.plot(its, [m["train_loss"] for m in history], "o-", color=C_LEARNED, label="total")
    ax.plot(
        its, [m["train_policy_loss"] for m in history], "s--", color=C_MUTED, label="policy"
    )
    ax.plot(
        its, [m["train_value_loss"] for m in history], "^:", color=C_INVOKE, label="value"
    )
    ax.set_title("Training loss")
    ax.set_xlabel("iteration")
    ax.legend(frameon=False)

    ax = _style(axes[1, 2])
    ax.plot(its, [m["value_correlation"] for m in history], "o-", color=C_LEARNED)
    ax.set_title("Value-head correlation")
    ax.set_xlabel("iteration")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_degenerate_dynamics(plt, per_iter: dict[int, dict], out: Path) -> None:
    its = sorted(per_iter)
    interrupts = [per_iter[i]["interrupt_count"] for i in its]
    expiries = [per_iter[i]["expired_queue_items"] for i in its]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _style(ax)
    ax.plot(its, interrupts, "o-", color=C_HEURISTIC, label="queue interrupts")
    ax.plot(its, expiries, "s--", color=C_INHIBIT, label="expired queue items")
    ax.set_ylim(-0.5, 5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("count across 50 episodes")
    ax.set_title("Interrupt & queue-expiry rate over training (degenerate: flat zero)")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.5,
        0.5,
        "0 everywhere — BREAKPOINT injection mode drains the queue at\n"
        "turn boundaries, so no mid-turn interrupts or expiries ever occur.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=C_MUTED,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_emergent_inhibition(
    plt, per_iter: dict[int, dict], gaps: dict | None, out: Path
) -> None:
    its = sorted(per_iter)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle(
        "Emergent secondary-tool inhibition: fire once early, then suppress",
        fontweight="bold",
    )

    ax = _style(axes[0])
    rates = [100 * per_iter[i]["multi_tool_rate"] for i in its]
    secondary = [sum(per_iter[i]["secondary_tool_counts"].values()) for i in its]
    ax.bar(
        [i - 0.18 for i in its], rates, width=0.36, color=C_LEARNED,
        label="multi-tool episodes (%)",
    )
    ax.bar(
        [i + 0.18 for i in its], secondary, width=0.36, color=C_INHIBIT,
        label="secondary tool invocations",
    )
    for i, r in zip(its, rates):
        ax.text(i - 0.18, r + 0.4, f"{r:.0f}", ha="center", fontsize=8, color=C_LEARNED)
    ax.set_xlabel("iteration")
    ax.set_title("Redundant second-tool use collapses")
    ax.legend(frameon=False)

    ax = _style(axes[1])
    if gaps is not None:
        gi = gaps["iterations"]
        ax.plot(gi, gaps["p_invoke_turn1"], "o-", color=C_INVOKE, label="P(invoke) | turn 1")
        ax.plot(
            gi, gaps["p_invoke_turn2plus"], "s-", color=C_INHIBIT, label="P(invoke) | turn 2+"
        )
        ax.fill_between(
            gi, gaps["p_invoke_turn2plus"], gaps["p_invoke_turn1"], color=C_MUTED, alpha=0.12
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("iteration")
        ax.set_title("Policy gate sharpens (network probe)")
        ax.legend(frameon=False, loc="center right")
    else:
        ax.text(
            0.5, 0.5, "network probe skipped\n(torch extra unavailable)",
            transform=ax.transAxes, ha="center", va="center", color=C_MUTED,
        )
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_timing_by_turn(plt, learned_dist: dict, heuristic_dist: dict, out: Path) -> None:
    stages = ["turn 1", "turn 2+"]
    labels = [a.value for a in ACTION_ORDER]
    colors = [C_INVOKE, C_HEURISTIC, C_INHIBIT, C_MUTED]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.suptitle("Timing: action distribution by turn stage", fontweight="bold")
    for ax, (title, dist) in zip(
        axes, [("Learned policy (argmax)", learned_dist), ("Heuristic", heuristic_dist)]
    ):
        _style(ax)
        bottom = np.zeros(len(stages))
        for k, lab in enumerate(labels):
            vals = np.array([dist[s][k] for s in stages], dtype=float)
            ax.bar(stages, vals, bottom=bottom, color=colors[k], label=lab, width=0.55)
            bottom += vals
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
    axes[0].set_ylabel("fraction of decision points")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_learned_vs_heuristic(
    plt, learned_marginal: np.ndarray, heuristic_marginal: np.ndarray, out: Path
) -> None:
    labels = [a.value for a in ACTION_ORDER]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _style(ax)
    ax.bar(x - 0.2, learned_marginal, width=0.4, color=C_LEARNED, label="learned")
    ax.bar(x + 0.2, heuristic_marginal, width=0.4, color=C_HEURISTIC, label="heuristic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("fraction of decision points")
    ax.set_title("Marginal action distribution: learned vs heuristic (final iteration)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_queue_counterfactual(plt, cf: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _style(ax)
    ax.plot(cf["depths"], cf["p_invoke"], "o-", color=C_LEARNED)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("synthetic queue depth (feature override on turn-1 states)")
    ax.set_ylabel("mean P(invoke)")
    ax.set_title("Queue-aware inhibition probe (negative result: flat)")
    ax.text(
        0.5,
        0.35,
        "P(invoke) is invariant to queue depth — the policy never learned\n"
        "queue-aware inhibition because queue depth was ~0 in all training data.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=C_MUTED,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_compound_sequences(plt, per_iter_compound: dict[int, dict], out: Path) -> None:
    its = sorted(per_iter_compound)
    observed = [per_iter_compound[i]["observed_distinct_second_rate"] for i in its]
    chance = per_iter_compound[its[0]]["chance_distinct_second_rate"]
    counts = [per_iter_compound[i]["episodes_with_2plus_invocations"] for i in its]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _style(ax)
    ax.bar(its, observed, width=0.5, color=C_LEARNED, label="observed P(distinct 2nd tool)")
    ax.axhline(chance, color=C_HEURISTIC, linestyle="--", label="uniform-chance (2/3)")
    for it, o, c in zip(its, observed, counts):
        ax.text(it, o + 0.02, f"n={c}", ha="center", fontsize=8, color=C_MUTED)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("iteration")
    ax.set_ylabel("rate")
    ax.set_title("Compound tool sequences (2nd distinct tool) vs chance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _stage_action_dist(turns: np.ndarray, actions: np.ndarray) -> dict:
    out: dict[str, list[float]] = {}
    for stage, mask in (("turn 1", turns == 1), ("turn 2+", turns >= 2)):
        if mask.any():
            counts = np.bincount(actions[mask], minlength=4) / mask.sum()
        else:
            counts = np.zeros(4)
        out[stage] = counts.tolist()
    return out


def run_analysis(
    mcts_dir: Path,
    comparative_dir: Path,
    out_dir: Path,
    hidden_dim: int,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    history = load_metrics_history(mcts_dir)
    stats["training_metrics"] = [
        {
            k: m[k]
            for k in (
                "iteration",
                "policy_entropy",
                "kl_from_heuristic",
                "heuristic_agreement",
                "eval_score",
                "heuristic_eval_score",
                "train_loss",
                "value_correlation",
            )
        }
        for m in history
    ]

    iter_eps = load_iteration_episodes(mcts_dir)
    per_iter = {i: episode_tool_stats(eps) for i, eps in iter_eps.items()}
    per_iter_compound = {i: compound_sequence_stats(eps) for i, eps in iter_eps.items()}
    stats["per_iteration_episode_stats"] = {str(k): v for k, v in per_iter.items()}
    stats["per_iteration_compound"] = {str(k): v for k, v in per_iter_compound.items()}

    if comparative_dir.exists():
        stats["comparative_tool_usage"] = comparative_tool_usage(comparative_dir)

    fig_training_dynamics(plt, history, out_dir / "training_dynamics.png")
    fig_degenerate_dynamics(plt, per_iter, out_dir / "training_dynamics_degenerate.png")
    fig_compound_sequences(plt, per_iter_compound, out_dir / "compound_sequences.png")

    torch_ok = find_spec("torch") is not None
    stats["torch_available"] = torch_ok

    gaps = None
    final_it = max(iter_eps)

    if torch_ok:
        from bicameral_agent.policy_value_net import PolicyValueNetwork

        pipeline = TrainingDataPipeline()
        ref_states, ref_turns = reconstruct_states(iter_eps[final_it], pipeline)
        t1 = ref_turns == 1
        t2 = ref_turns >= 2

        # Per-iteration inhibition gap on the fixed final-iteration state set.
        gap_iters, g_t1, g_t2 = [], [], []
        for it in sorted(iter_eps):
            ckpt = mcts_dir / f"iteration-{it:03d}" / "policy_value.pt"
            if not ckpt.exists():
                continue
            net = PolicyValueNetwork.load(
                str(ckpt), input_dim=STATE_DIM, hidden_dim=hidden_dim
            )
            probs = policy_probs(net, ref_states)
            p_inv = 1.0 - probs[:, _DO_NOTHING_IDX]
            gap_iters.append(it)
            g_t1.append(float(p_inv[t1].mean()) if t1.any() else 0.0)
            g_t2.append(float(p_inv[t2].mean()) if t2.any() else 0.0)
        gaps = {
            "iterations": gap_iters,
            "p_invoke_turn1": g_t1,
            "p_invoke_turn2plus": g_t2,
        }
        stats["inhibition_gap"] = gaps

        final_net = PolicyValueNetwork.load(
            str(mcts_dir / f"iteration-{final_it:03d}" / "policy_value.pt"),
            input_dim=STATE_DIM,
            hidden_dim=hidden_dim,
        )
        probs = policy_probs(final_net, ref_states)
        argmax = probs.argmax(axis=1)
        learned_dist = _stage_action_dist(ref_turns, argmax)

        heur = heuristic_actions_for_episodes(iter_eps[final_it])
        heur_turns = np.array([h[0] for h in heur])
        heur_act = np.array([h[1] for h in heur])
        heuristic_dist = _stage_action_dist(heur_turns, heur_act)

        agreement = float(np.mean(argmax == heur_act))
        disagreements = []
        for k in range(len(argmax)):
            if argmax[k] != heur_act[k]:
                disagreements.append(
                    {
                        "turn": int(ref_turns[k]),
                        "policy_action": ACTION_ORDER[int(argmax[k])].value,
                        "heuristic_action": ACTION_ORDER[int(heur_act[k])].value,
                        "queue_depth": int(heur[k][2]),
                        "value_estimate": round(policy_value(final_net, ref_states[k]), 3),
                        "policy_probs": [round(float(x), 3) for x in probs[k]],
                    }
                )
        stats["final_heuristic_agreement"] = agreement
        stats["disagreements"] = disagreements

        # Queue-depth counterfactual on turn-1 states.
        base = ref_states[t1].copy()
        depths = [0, 1, 2, 3, 5, 8]
        p_invoke = []
        for d in depths:
            vs = base.copy()
            vs[:, _OFF_QUEUE] = min(d / _QUEUE_DEPTH_CAP, 1.0)
            pr = policy_probs(final_net, vs)
            p_invoke.append(float((1.0 - pr[:, _DO_NOTHING_IDX]).mean()))
        cf = {"depths": depths, "p_invoke": p_invoke}
        stats["queue_counterfactual"] = cf

        learned_marginal = np.bincount(argmax, minlength=4) / len(argmax)
        heur_marginal = np.bincount(heur_act, minlength=4) / len(heur_act)
        stats["learned_marginal"] = learned_marginal.tolist()
        stats["heuristic_marginal"] = heur_marginal.tolist()

        fig_emergent_inhibition(plt, per_iter, gaps, out_dir / "emergent_inhibition.png")
        fig_timing_by_turn(plt, learned_dist, heuristic_dist, out_dir / "timing_by_turn.png")
        fig_queue_counterfactual(plt, cf, out_dir / "queue_counterfactual.png")
        fig_learned_vs_heuristic(
            plt, learned_marginal, heur_marginal, out_dir / "learned_vs_heuristic.png"
        )
    else:
        logger.warning("torch unavailable: skipping network-probe figures")
        fig_emergent_inhibition(plt, per_iter, None, out_dir / "emergent_inhibition.png")

    (out_dir / "emergent_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcts-dir", type=Path, default=Path("data/mcts_training"))
    parser.add_argument("--comparative-dir", type=Path, default=Path("data/comparative"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures/emergent"))
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden width of the saved policy_value.pt checkpoints.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    stats = run_analysis(
        mcts_dir=args.mcts_dir,
        comparative_dir=args.comparative_dir,
        out_dir=args.out_dir,
        hidden_dim=args.hidden_dim,
    )
    logger.info(
        "wrote figures + emergent_stats.json to %s (torch=%s)",
        args.out_dir,
        stats["torch_available"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
