"""Tests for supervised policy/value pre-training (issue #26).

The whole module requires torch (``bicameral_agent.pretrain`` imports it
at module level, matching policy_value_net), so it is skipped wholesale
when torch is unavailable.

The AC-threshold verification (80% action accuracy, r > 0.3 value
correlation, ...) is data-dependent and pending the completed #46
baseline re-run; these tests verify the training mechanics on synthetic
data plus a skippable smoke test on the partial pilot episodes.
"""

from __future__ import annotations

# ruff: noqa: E402  (imports below the importorskip guard are intentional)

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bicameral_agent.heuristic_controller import TOOL_IDS
from bicameral_agent.policy_value_net import NUM_ACTIONS, PolicyValueNetwork
from bicameral_agent.pretrain import (
    PretrainConfig,
    evaluate_policy_value,
    pretrain_policy_value,
)
from bicameral_agent.schema import Episode, EpisodeOutcome, Message, ToolInvocation
from bicameral_agent.serialization import episodes_to_parquet
from bicameral_agent.training_data_store import TrainingDataStore
from bicameral_agent.training_pipeline import (
    STATE_DIM,
    TrainingDataPipeline,
    TrainingExample,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Real pilot-episode directory for the smoke test: repo-relative by
# default, overridable via BICAMERAL_BASELINE_DATA.
_PARTIAL_DATA_DIR = Path(
    os.environ.get(
        "BICAMERAL_BASELINE_DATA",
        _REPO_ROOT / "data" / "baseline_rerun_partial1",
    )
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _synthetic_examples(
    n: int = 300,
    n_episodes: int = 15,
    seed: int = 0,
) -> list[TrainingExample]:
    """Learnable synthetic (state, action, return) examples.

    The "heuristic policy" is ``action = argmax(W @ state)`` for a fixed
    random projection ``W`` and the return is a fixed linear function of
    the state — deterministic functions a small MLP can fit, so training
    loss must decrease and validation accuracy must beat chance.
    """
    rng = np.random.default_rng(seed)
    w_policy = rng.standard_normal((NUM_ACTIONS, STATE_DIM)).astype(np.float32)
    w_value = rng.standard_normal(STATE_DIM).astype(np.float32) / np.sqrt(STATE_DIM)
    examples: list[TrainingExample] = []
    per_ep = n // n_episodes
    for i in range(n):
        state = rng.random(STATE_DIM).astype(np.float32)
        action = int(np.argmax(w_policy @ state))
        ret = float(w_value @ state)
        done = (i % per_ep) == per_ep - 1
        examples.append(
            TrainingExample(
                state=state,
                action=action,
                reward=ret,
                next_state=np.zeros(STATE_DIM, dtype=np.float32),
                done=done,
                discounted_return=ret,
                episode_id=f"ep-{i // per_ep}",
                decision_index=i % per_ep,
            )
        )
    return examples


def _synthetic_episode(num_turns: int, quality: float, base_ts: int) -> Episode:
    """Minimal valid Episode with ``num_turns`` user/assistant pairs."""
    messages: list[Message] = []
    tools: list[ToolInvocation] = []
    tool_ids = list(TOOL_IDS.values())
    for turn in range(1, num_turns + 1):
        user_ts = base_ts + (turn - 1) * 1000
        messages.append(
            Message(role="user", content=f"user {turn}", timestamp_ms=user_ts, token_count=10)
        )
        if turn % 2 == 0:  # alternate tool use so actions vary
            tools.append(
                ToolInvocation(
                    tool_id=tool_ids[turn % len(tool_ids)],
                    invoked_at_ms=user_ts + 100,
                    completed_at_ms=user_ts + 300,
                    input_tokens=20,
                    output_tokens=30,
                    result_deposited=False,
                )
            )
        messages.append(
            Message(
                role="assistant",
                content=f"assistant {turn}",
                timestamp_ms=user_ts + 500,
                token_count=20,
            )
        )
    return Episode(
        messages=messages,
        tool_invocations=tools,
        outcome=EpisodeOutcome(
            quality_score=quality,
            total_tokens=sum(m.token_count for m in messages),
            total_turns=num_turns,
            wall_clock_ms=num_turns * 1000,
        ),
    )


# ---------------------------------------------------------------------------
# Training behaviour
# ---------------------------------------------------------------------------


class TestTraining:
    def test_loss_decreases_and_beats_chance_on_learnable_policy(self) -> None:
        examples = _synthetic_examples(n=300)
        config = PretrainConfig(max_epochs=60, patience=60, seed=0)
        result = pretrain_policy_value(examples, config)
        first = result.history[0]["train_loss"]
        last = result.history[-1]["train_loss"]
        assert last < first * 0.5, f"loss did not decrease: {first:.4f} -> {last:.4f}"
        # Learnable deterministic policy: accuracy well above 1/4 chance.
        assert result.metrics["action_accuracy"] > 0.5
        # Learnable linear value target: strong correlation.
        assert result.metrics["value_correlation"] > 0.5

    def test_fit_rejects_empty_examples(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pretrain_policy_value([])

    def test_fit_rejects_single_episode(self) -> None:
        examples = _synthetic_examples(n=20, n_episodes=1)
        with pytest.raises(ValueError, match="2 episodes"):
            pretrain_policy_value(examples)

    def test_fit_is_deterministic_for_fixed_seed(self) -> None:
        examples = _synthetic_examples(n=120)
        config = PretrainConfig(max_epochs=5, min_epochs=5, seed=42)
        r1 = pretrain_policy_value(examples, config)
        r2 = pretrain_policy_value(examples, config)
        assert r1.history == r2.history
        assert r1.n_train == r2.n_train and r1.n_val == r2.n_val
        for p1, p2 in zip(r1.model.parameters(), r2.model.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_early_stopping_plateau(self) -> None:
        # A huge min_delta means no epoch ever counts as an improvement,
        # so training must stop at min_epochs, well before max_epochs.
        examples = _synthetic_examples(n=120)
        config = PretrainConfig(
            max_epochs=100, min_epochs=8, patience=3, min_delta=1e9, seed=0
        )
        result = pretrain_policy_value(examples, config)
        assert len(result.history) == 8
        assert result.best_epoch == 1

    def test_best_weights_are_restored(self) -> None:
        examples = _synthetic_examples(n=120)
        config = PretrainConfig(max_epochs=30, patience=30, seed=1)
        result = pretrain_policy_value(examples, config)
        best = result.history[result.best_epoch - 1]["val_loss"]
        assert all(e["val_loss"] >= best for e in result.history)
        # The returned model reproduces the best epoch's validation loss.
        assert result.metrics["loss"] == pytest.approx(best, rel=1e-5)

    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        examples = _synthetic_examples(n=120)
        result = pretrain_policy_value(
            examples, PretrainConfig(max_epochs=3, min_epochs=3)
        )
        path = tmp_path / "model.pt"
        result.model.save(path)
        loaded = PolicyValueNetwork.load(path, input_dim=STATE_DIM)
        state = np.random.default_rng(5).random(STATE_DIM).astype(np.float32)
        expected_probs, expected_value = result.model.predict(state)
        got_probs, got_value = loaded.predict(state)
        np.testing.assert_array_equal(expected_probs, got_probs)
        assert expected_value == got_value


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_metrics_structure(self) -> None:
        examples = _synthetic_examples(n=100)
        model = PolicyValueNetwork(input_dim=STATE_DIM)
        metrics = evaluate_policy_value(model, examples)
        assert metrics["n_examples"] == 100
        assert 0.0 <= metrics["action_accuracy"] <= 1.0
        assert 0.25 <= metrics["majority_action_fraction"] <= 1.0
        assert -1.0 <= metrics["value_correlation"] <= 1.0
        assert metrics["value_mse"] >= 0.0
        assert sum(metrics["true_action_counts"]) == 100
        assert sum(metrics["predicted_action_counts"]) == 100
        assert len(metrics["true_action_counts"]) == NUM_ACTIONS
        # JSON-serializable end to end
        json.dumps(metrics)

    def test_constant_returns_give_null_correlation(self) -> None:
        examples = [
            TrainingExample(
                state=np.random.default_rng(i).random(STATE_DIM).astype(np.float32),
                action=i % NUM_ACTIONS,
                reward=0.5,
                next_state=np.zeros(STATE_DIM, dtype=np.float32),
                done=False,
                discounted_return=0.5,
                episode_id=f"ep-{i}",
                decision_index=0,
            )
            for i in range(10)
        ]
        metrics = evaluate_policy_value(PolicyValueNetwork(input_dim=STATE_DIM), examples)
        assert metrics["value_correlation"] is None

    def test_empty_examples_yield_null_metrics(self) -> None:
        metrics = evaluate_policy_value(PolicyValueNetwork(input_dim=STATE_DIM), [])
        assert metrics["n_examples"] == 0
        assert metrics["action_accuracy"] is None
        assert metrics["value_correlation"] is None


# ---------------------------------------------------------------------------
# Integration: full harness from parquet episode files
# ---------------------------------------------------------------------------


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "pretrain_policy", _REPO_ROOT / "scripts" / "pretrain_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestIntegration:
    def test_cli_end_to_end_on_synthetic_episodes(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        episodes = [
            _synthetic_episode(num_turns=6, quality=0.2 + 0.1 * i, base_ts=1_000_000 + i * 60_000)
            for i in range(5)
        ]
        parquet_path = tmp_path / "episodes.parquet"
        episodes_to_parquet(episodes, str(parquet_path))

        out_dir = tmp_path / "out"
        cli = _load_cli()
        rc = cli.main(
            [
                str(parquet_path),
                "--out-dir", str(out_dir),
                "--max-epochs", "12",
                "--patience", "12",
                "--seed", "0",
            ]
        )
        assert rc == 0

        model_path = out_dir / "policy_value_pretrained.pt"
        metrics_path = out_dir / "metrics.json"
        curves_path = out_dir / "training_curves.png"
        assert model_path.exists()
        assert curves_path.exists() and curves_path.stat().st_size > 0
        with metrics_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["data"]["n_episodes"] == 5
        assert payload["data"]["n_train"] + payload["data"]["n_val"] == 30
        assert payload["train"]["epochs_run"] == len(payload["train"]["history"])
        assert set(payload["acceptance_criteria"]) == {
            "action_accuracy_ge_0.8",
            "value_correlation_gt_0.3",
            "train_loss_monotonic_10_epochs",
            "val_loss_within_20pct_of_train",
            "training_lt_30min_cpu",
        }
        # The mechanics-level AC must hold even on tiny synthetic data.
        assert payload["acceptance_criteria"]["training_lt_30min_cpu"] is True

        # Saved checkpoint loads and predicts.
        loaded = PolicyValueNetwork.load(model_path, input_dim=STATE_DIM)
        probs, value = loaded.predict(np.zeros(STATE_DIM, dtype=np.float32))
        assert probs.shape == (NUM_ACTIONS,)
        assert np.isfinite(probs).all() and np.isfinite(value)

    def test_cli_loads_examples_from_store(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        store_root = tmp_path / "store"
        TrainingDataStore(store_root).save_examples(
            _synthetic_examples(n=100, n_episodes=5), iteration=0
        )
        out_dir = tmp_path / "out"
        cli = _load_cli()
        rc = cli.main(
            [
                "--store", str(store_root),
                "--out-dir", str(out_dir),
                "--max-epochs", "3",
                "--min-epochs", "3",
            ]
        )
        assert rc == 0
        with (out_dir / "metrics.json").open(encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["data"]["n_examples"] == 100
        assert payload["data"]["store"] == str(store_root)

    @pytest.mark.skipif(
        not (
            _PARTIAL_DATA_DIR.exists()
            and any(_PARTIAL_DATA_DIR.glob("*.parquet"))
        ),
        reason="partial baseline re-run data not present on this machine",
    )
    def test_smoke_pretrain_on_partial_baseline_rerun(self) -> None:
        """Skippable smoke test: the harness trains on real pilot episodes."""
        from bicameral_agent.serialization import episodes_from_parquet

        episodes = []
        for path in sorted(_PARTIAL_DATA_DIR.glob("*.parquet")):
            episodes.extend(episodes_from_parquet(str(path)))
        assert episodes, "parquet files present but contained no episodes"

        examples = TrainingDataPipeline().process_episodes(episodes)
        assert examples

        result = pretrain_policy_value(
            examples, PretrainConfig(max_epochs=20, min_epochs=10, patience=20, seed=0)
        )
        assert np.isfinite(result.history[-1]["train_loss"])
        assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
        assert result.metrics["n_examples"] > 0
        assert 0.0 <= result.metrics["action_accuracy"] <= 1.0
        json.dumps(result.metrics)
