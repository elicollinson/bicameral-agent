"""Tests for the MCTS transition model (issue #27).

The whole module requires torch (``bicameral_agent.transition_model``
imports it at module level, matching policy_value_net), so it is skipped
wholesale when torch is unavailable — the module-level equivalent of
test_training_pipeline.py's per-test ``pytest.importorskip`` pattern.
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
from bicameral_agent.policy_value_net import NUM_ACTIONS
from bicameral_agent.schema import Episode, EpisodeOutcome, Message, ToolInvocation
from bicameral_agent.serialization import episodes_to_parquet
from bicameral_agent.training_pipeline import (
    STATE_DIM,
    TrainingDataPipeline,
    TrainingExample,
)
from bicameral_agent.transition_model import (
    TransitionModel,
    TransitionTrainingConfig,
    evaluate_transition_model,
    fit_transition_model,
    measure_forward_latency,
    split_examples,
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
    n: int = 200,
    n_episodes: int = 10,
    seed: int = 0,
) -> list[TrainingExample]:
    """Learnable synthetic transitions with contractive linear dynamics.

    ``next_state = 0.5 * state + action-dependent offset`` (clipped to
    [0, 1]) and ``reward = mean(state) - 0.1 * action`` — deterministic
    functions a small MLP can fit, so training loss must decrease and
    rollouts of the fitted model stay bounded.
    """
    rng = np.random.default_rng(seed)
    offsets = rng.random((NUM_ACTIONS, STATE_DIM)).astype(np.float32) * 0.3
    examples: list[TrainingExample] = []
    per_ep = n // n_episodes
    for i in range(n):
        state = rng.random(STATE_DIM).astype(np.float32)
        action = int(rng.integers(NUM_ACTIONS))
        next_state = np.clip(0.5 * state + offsets[action], 0.0, 1.0).astype(np.float32)
        reward = float(state.mean() - 0.1 * action)
        done = (i % per_ep) == per_ep - 1
        examples.append(
            TrainingExample(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state if not done else np.zeros(STATE_DIM, dtype=np.float32),
                done=done,
                discounted_return=reward,
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
# Shape / dtype contracts
# ---------------------------------------------------------------------------


class TestShapes:
    def test_forward_batch_shapes_and_dtypes(self) -> None:
        model = TransitionModel()
        states = torch.rand(7, STATE_DIM)
        onehots = torch.nn.functional.one_hot(
            torch.arange(7) % NUM_ACTIONS, NUM_ACTIONS
        ).float()
        next_states, rewards = model(states, onehots)
        assert next_states.shape == (7, STATE_DIM)
        assert rewards.shape == (7,)
        assert next_states.dtype == torch.float32
        assert rewards.dtype == torch.float32

    def test_predict_numpy_contract(self) -> None:
        model = TransitionModel()
        # float64 input must be accepted and cast
        state = np.random.default_rng(0).random(STATE_DIM)
        next_state, reward = model.predict(state, 2)
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (STATE_DIM,)
        assert next_state.dtype == np.float32
        assert isinstance(reward, float)

    def test_predict_rejects_bad_action(self) -> None:
        model = TransitionModel()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        with pytest.raises(ValueError, match="action"):
            model.predict(state, NUM_ACTIONS)
        with pytest.raises(ValueError, match="action"):
            model.predict(state, -1)

    def test_rollout_shapes(self) -> None:
        model = TransitionModel()
        state = np.random.default_rng(1).random(STATE_DIM).astype(np.float32)
        states, rewards = model.rollout(state, [0, 1, 2, 3, 0])
        assert states.shape == (5, STATE_DIM)
        assert rewards.shape == (5,)
        assert states.dtype == np.float32

    def test_architecture_spec(self) -> None:
        """3 hidden layers x 128 units, ReLU (issue #27)."""
        model = TransitionModel()
        linears = [m for m in model.trunk if isinstance(m, torch.nn.Linear)]
        relus = [m for m in model.trunk if isinstance(m, torch.nn.ReLU)]
        assert len(linears) == 3
        assert len(relus) == 3
        assert all(m.out_features == 128 for m in linears)
        assert linears[0].in_features == STATE_DIM + NUM_ACTIONS
        assert model.state_head.out_features == STATE_DIM
        assert model.reward_head.out_features == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_seeded_init_is_deterministic(self) -> None:
        torch.manual_seed(123)
        m1 = TransitionModel()
        torch.manual_seed(123)
        m2 = TransitionModel()
        for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_fit_is_deterministic_for_fixed_seed(self) -> None:
        examples = _synthetic_examples(n=80)
        config = TransitionTrainingConfig(epochs=5, seed=42)
        r1 = fit_transition_model(examples, config)
        r2 = fit_transition_model(examples, config)
        assert r1.epoch_losses == r2.epoch_losses
        for p1, p2 in zip(r1.model.parameters(), r2.model.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_split_is_deterministic_and_episode_grouped(self) -> None:
        examples = _synthetic_examples(n=100, n_episodes=10)
        t1, v1 = split_examples(examples, train_ratio=0.8, seed=7)
        t2, v2 = split_examples(examples, train_ratio=0.8, seed=7)
        assert [e.episode_id for e in t1] == [e.episode_id for e in t2]
        assert len(t1) + len(v1) == len(examples)
        assert len(t1) > 0 and len(v1) > 0
        assert {e.episode_id for e in t1}.isdisjoint({e.episode_id for e in v1})


# ---------------------------------------------------------------------------
# Training behaviour
# ---------------------------------------------------------------------------


class TestTraining:
    def test_loss_decreases_on_synthetic_data(self) -> None:
        examples = _synthetic_examples(n=200)
        config = TransitionTrainingConfig(epochs=50, seed=0)
        result = fit_transition_model(examples, config)
        first = result.epoch_losses[0]["total"]
        last = result.epoch_losses[-1]["total"]
        assert last < first * 0.5, f"loss did not decrease: {first:.4f} -> {last:.4f}"

    def test_fit_rejects_empty_examples(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            fit_transition_model([])

    def test_rollout_stays_bounded_after_fitting(self) -> None:
        examples = _synthetic_examples(n=200)
        result = fit_transition_model(examples, TransitionTrainingConfig(epochs=50, seed=0))
        rollout = result.metrics["rollout"]
        assert rollout["steps"] == 5
        assert rollout["bounded"] is True
        assert rollout["max_state_norm"] <= rollout["norm_bound"]

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        examples = _synthetic_examples(n=80)
        result = fit_transition_model(examples, TransitionTrainingConfig(epochs=3))
        path = tmp_path / "model.pt"
        result.model.save(path)
        loaded = TransitionModel.load(path)
        state = np.random.default_rng(5).random(STATE_DIM).astype(np.float32)
        expected_next, expected_reward = result.model.predict(state, 1)
        got_next, got_reward = loaded.predict(state, 1)
        np.testing.assert_array_equal(expected_next, got_next)
        assert expected_reward == got_reward


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_metrics_structure(self) -> None:
        examples = _synthetic_examples(n=100)
        model = TransitionModel()
        metrics = evaluate_transition_model(model, examples)
        assert metrics["n_examples"] == 100
        assert metrics["n_state_examples"] == 90  # 10 terminal steps excluded
        assert len(metrics["state_mse_per_dim"]) == STATE_DIM
        assert metrics["state_mse_per_dim_mean"] >= 0.0
        assert metrics["state_mse_per_dim_max"] >= metrics["state_mse_per_dim_mean"]
        assert metrics["reward_mse"] >= 0.0
        assert -1.0 <= metrics["reward_correlation"] <= 1.0
        assert metrics["latency_ms_median"] > 0.0
        # JSON-serializable end to end
        json.dumps(metrics)

    def test_constant_rewards_give_null_correlation(self) -> None:
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
        metrics = evaluate_transition_model(TransitionModel(), examples)
        assert metrics["reward_correlation"] is None

    def test_empty_examples_yield_null_metrics(self) -> None:
        metrics = evaluate_transition_model(TransitionModel(), [])
        assert metrics["n_examples"] == 0
        assert metrics["state_mse_per_dim_mean"] is None
        assert metrics["rollout"] is None

    def test_forward_latency_under_2ms(self) -> None:
        model = TransitionModel()
        median_ms = measure_forward_latency(model)
        assert median_ms < 2.0, f"forward pass median={median_ms:.3f}ms > 2ms"


# ---------------------------------------------------------------------------
# Integration: full harness from parquet episode files
# ---------------------------------------------------------------------------


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "train_transition_model", _REPO_ROOT / "scripts" / "train_transition_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestIntegration:
    def test_cli_end_to_end_on_synthetic_episodes(self, tmp_path: Path) -> None:
        episodes = [
            _synthetic_episode(num_turns=6, quality=0.2 + 0.1 * i, base_ts=1_000_000 + i * 60_000)
            for i in range(5)
        ]
        parquet_path = tmp_path / "episodes.parquet"
        episodes_to_parquet(episodes, str(parquet_path))

        out_dir = tmp_path / "out"
        cli = _load_cli()
        rc = cli.main(
            [str(parquet_path), "--out-dir", str(out_dir), "--epochs", "10", "--seed", "0"]
        )
        assert rc == 0

        model_path = out_dir / "transition_model.pt"
        metrics_path = out_dir / "metrics.json"
        assert model_path.exists()
        with metrics_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["data"]["n_episodes"] == 5
        assert payload["data"]["n_train"] + payload["data"]["n_val"] == 30
        assert set(payload["acceptance_criteria"]) == {
            "state_mse_per_dim_mean_lt_0.1",
            "reward_correlation_gt_0.4",
            "rollout_5_step_bounded",
            "forward_latency_lt_2ms",
            "training_lt_30min_cpu",
        }
        # The mechanics-level ACs must hold even on tiny synthetic data.
        assert payload["acceptance_criteria"]["forward_latency_lt_2ms"] is True
        assert payload["acceptance_criteria"]["training_lt_30min_cpu"] is True

        # Saved checkpoint loads and predicts.
        loaded = TransitionModel.load(model_path)
        next_state, reward = loaded.predict(np.zeros(STATE_DIM, dtype=np.float32), 0)
        assert np.isfinite(next_state).all()
        assert np.isfinite(reward)

    @pytest.mark.skipif(
        not (
            _PARTIAL_DATA_DIR.exists()
            and any(_PARTIAL_DATA_DIR.glob("*.parquet"))
        ),
        reason="partial baseline re-run data not present on this machine",
    )
    def test_smoke_fit_on_partial_baseline_rerun(self) -> None:
        """Skippable smoke test: the harness fits on real pilot episodes."""
        from bicameral_agent.serialization import episodes_from_parquet

        episodes = []
        for path in sorted(_PARTIAL_DATA_DIR.glob("*.parquet")):
            episodes.extend(episodes_from_parquet(str(path)))
        assert episodes, "parquet files present but contained no episodes"

        examples = TrainingDataPipeline().process_episodes(episodes)
        assert examples

        result = fit_transition_model(examples, TransitionTrainingConfig(epochs=20, seed=0))
        assert np.isfinite(result.epoch_losses[-1]["total"])
        assert result.epoch_losses[-1]["total"] < result.epoch_losses[0]["total"]
        metrics = result.metrics
        assert metrics["n_examples"] > 0
        assert np.isfinite(metrics["state_mse_per_dim_mean"])
        json.dumps(metrics)
