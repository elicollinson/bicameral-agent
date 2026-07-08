"""Smoke tests for scripts/analyze_emergent_behavior.py (issue #32).

The torch-free path (metrics + episode-derived figures) must run
end-to-end on a tiny synthetic corpus without the torch extra. A second
test exercises the network-probe path when torch is available.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bicameral_agent.schema import (
    Episode,
    EpisodeOutcome,
    Message,
    ToolInvocation,
    UserEvent,
    UserEventType,
)
from bicameral_agent.serialization import episodes_to_parquet

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import analyze_emergent_behavior as aeb  # noqa: E402


def _episode(ep_id: str, tools: dict[int, str], quality: float) -> Episode:
    """Two-turn synthetic episode with tools invoked at the given turns."""
    base = 1_000_000
    messages: list[Message] = []
    invocations: list[ToolInvocation] = []
    for turn in (1, 2):
        user_ts = base + (turn - 1) * 1000
        messages.append(
            Message(role="user", content=f"q{turn}", timestamp_ms=user_ts, token_count=10)
        )
        messages.append(
            Message(
                role="assistant",
                content=f"a{turn}",
                timestamp_ms=user_ts + 500,
                token_count=20,
            )
        )
        if turn in tools:
            invocations.append(
                ToolInvocation(
                    tool_id=tools[turn],
                    invoked_at_ms=user_ts + 600,
                    completed_at_ms=user_ts + 800,
                    input_tokens=50,
                    output_tokens=80,
                    result_deposited=False,
                    turn=turn,
                )
            )
    return Episode(
        episode_id=ep_id,
        messages=messages,
        user_events=[UserEvent(event_type=UserEventType.FOLLOW_UP, timestamp_ms=base + 700)],
        context_injections=[],
        tool_invocations=invocations,
        outcome=EpisodeOutcome(
            quality_score=quality, total_tokens=60, total_turns=2, wall_clock_ms=2000
        ),
        metadata={"interrupt_count": 0, "expired_queue_items": 0, "injection_mode": "breakpoint"},
    )


def _build_corpus(mcts_dir: Path, comparative_dir: Path) -> None:
    (mcts_dir / "episodes").mkdir(parents=True)
    comparative_dir.mkdir(parents=True)
    for it in (0, 1):
        episodes = [
            _episode(f"it{it}-single", {1: "research_gap_scanner"}, 0.6),
            # multi-tool episode to exercise compound-sequence stats
            _episode(
                f"it{it}-multi",
                {1: "research_gap_scanner", 2: "context_refresher"},
                0.7,
            ),
        ]
        episodes_to_parquet(episodes, str(mcts_dir / "episodes" / f"iteration-{it:03d}.parquet"))
    metrics = [
        {
            "iteration": it,
            "policy_entropy": 0.6 - 0.2 * it,
            "kl_from_heuristic": 0.5 - 0.2 * it,
            "heuristic_agreement": 0.99,
            "eval_score": 0.65,
            "heuristic_eval_score": 0.66,
            "train_loss": 0.5 - 0.1 * it,
            "train_policy_loss": 0.4,
            "train_value_loss": 0.03,
            "value_correlation": 0.5 + 0.1 * it,
        }
        for it in (0, 1)
    ]
    (mcts_dir / "metrics_history.json").write_text(json.dumps(metrics), encoding="utf-8")
    episodes_to_parquet(
        [_episode("cmp", {1: "research_gap_scanner"}, 0.6)],
        str(comparative_dir / "heuristic.parquet"),
    )


def test_runs_without_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mcts_dir = tmp_path / "mcts"
    comparative_dir = tmp_path / "comparative"
    out_dir = tmp_path / "out"
    _build_corpus(mcts_dir, comparative_dir)

    # Force the torch-free path regardless of whether torch is installed.
    monkeypatch.setattr(aeb, "find_spec", lambda name: None)

    stats = aeb.run_analysis(
        mcts_dir=mcts_dir, comparative_dir=comparative_dir, out_dir=out_dir, hidden_dim=8
    )

    assert stats["torch_available"] is False
    for name in (
        "training_dynamics.png",
        "training_dynamics_degenerate.png",
        "compound_sequences.png",
        "emergent_inhibition.png",
        "emergent_stats.json",
    ):
        assert (out_dir / name).exists(), name
    # Episode-derived stats are populated.
    assert stats["per_iteration_episode_stats"]["0"]["multi_tool_episodes"] == 1
    assert stats["per_iteration_compound"]["0"]["episodes_with_2plus_invocations"] == 1
    assert "heuristic" in stats["comparative_tool_usage"]


def test_runs_with_torch(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch extra not installed")
    from bicameral_agent.policy_value_net import PolicyValueNetwork

    mcts_dir = tmp_path / "mcts"
    comparative_dir = tmp_path / "comparative"
    out_dir = tmp_path / "out"
    _build_corpus(mcts_dir, comparative_dir)

    # Write tiny real checkpoints so the network-probe path executes.
    for it in (0, 1):
        ckpt_dir = mcts_dir / f"iteration-{it:03d}"
        ckpt_dir.mkdir(parents=True)
        PolicyValueNetwork(input_dim=108, hidden_dim=8).save(ckpt_dir / "policy_value.pt")

    stats = aeb.run_analysis(
        mcts_dir=mcts_dir, comparative_dir=comparative_dir, out_dir=out_dir, hidden_dim=8
    )

    assert stats["torch_available"] is True
    assert "final_heuristic_agreement" in stats
    assert "inhibition_gap" in stats
    assert "queue_counterfactual" in stats
    for name in ("timing_by_turn.png", "queue_counterfactual.png", "learned_vs_heuristic.png"):
        assert (out_dir / name).exists(), name
