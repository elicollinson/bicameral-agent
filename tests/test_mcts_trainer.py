"""Tests for the MCTS training loop (issue #29).

Everything here is mechanically verifiable without live LLM calls:
episode collection runs through the real EpisodeRunner with mocked model
clients, and target generation / training / checkpointing operate on
synthetic episodes. The live acceptance criteria (monotonic eval
improvement, KL shift from the heuristic, no catastrophic forgetting,
entropy convergence, budget) are data/LLM-dependent; their *measurement*
is verified here (metrics computed and persisted per iteration) while
their verification is pending the #46 data era.

The module requires torch and is skipped wholesale when unavailable.
"""

from __future__ import annotations

# ruff: noqa: E402  (imports below the importorskip guard are intentional)

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.episode_runner import EpisodeConfig, EpisodeRunner
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.gemini import GeminiClient, GeminiResponse
from bicameral_agent.heuristic_controller import TOOL_IDS, Action
from bicameral_agent.mcts_trainer import (
    MCTSTrainer,
    MCTSTrainerConfig,
    TrainingMetrics,
)
from bicameral_agent.policy_value_net import NUM_ACTIONS, PolicyValueNetwork
from bicameral_agent.schema import (
    Episode,
    EpisodeOutcome,
    Message,
    UserEvent,
    UserEventType,
)
from bicameral_agent.simulated_user import ActionType, UserAction
from bicameral_agent.tool_primitive import ToolMetadata, ToolResult
from bicameral_agent.training_data_store import TrainingDataStore
from bicameral_agent.training_pipeline import STATE_DIM, TrainingDataPipeline
from bicameral_agent.transition_model import TransitionModel, TransitionTrainingConfig

_HIDDEN_DIM = 32
_TEST_CONFIG = MCTSTrainerConfig(epochs=12, batch_size=32, seed=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(seed: int = 0) -> PolicyValueNetwork:
    torch.manual_seed(seed)
    net = PolicyValueNetwork(input_dim=STATE_DIM, hidden_dim=_HIDDEN_DIM)
    net.eval()
    return net


def _make_transition(seed: int = 0) -> TransitionModel:
    torch.manual_seed(seed)
    model = TransitionModel(hidden_dim=_HIDDEN_DIM)
    model.eval()
    return model


def _make_trainer(tmp_path, seed: int = 0, **kwargs) -> MCTSTrainer:
    return MCTSTrainer(
        _make_policy(seed),
        _make_transition(seed),
        checkpoint_dir=tmp_path / "run",
        config=kwargs.pop("config", _TEST_CONFIG),
        **kwargs,
    )


def _build_episode(
    *, num_turns: int, quality: float | None = 0.6, tool_turns: dict[int, str] | None = None
) -> Episode:
    """Synthetic episode with ``num_turns`` user/assistant pairs."""
    messages: list[Message] = []
    tools = []
    base_ts = 1_000_000
    for turn in range(1, num_turns + 1):
        user_ts = base_ts + (turn - 1) * 1000
        messages.append(
            Message(
                role="user",
                content=f"user msg {turn} with varied words {turn * 7}",
                timestamp_ms=user_ts,
                token_count=10,
            )
        )
        if tool_turns and turn in tool_turns:
            from bicameral_agent.schema import ToolInvocation

            tools.append(
                ToolInvocation(
                    tool_id=tool_turns[turn],
                    invoked_at_ms=user_ts + 100,
                    completed_at_ms=user_ts + 300,
                    input_tokens=50,
                    output_tokens=80,
                )
            )
        messages.append(
            Message(
                role="assistant",
                content=f"assistant msg {turn}",
                timestamp_ms=user_ts + 500,
                token_count=20,
            )
        )
    user_events = [
        UserEvent(
            event_type=UserEventType.TASK_COMPLETE,
            timestamp_ms=base_ts + (num_turns - 1) * 1000 + 600,
        )
    ]
    return Episode(
        messages=messages,
        user_events=user_events,
        tool_invocations=tools,
        outcome=EpisodeOutcome(
            quality_score=quality,
            total_tokens=30 * num_turns,
            total_turns=num_turns,
            wall_clock_ms=num_turns * 1000,
        ),
    )


def _synthetic_episodes() -> list[Episode]:
    return [
        _build_episode(num_turns=3, quality=0.8, tool_turns={1: TOOL_IDS[Action.SCANNER]}),
        _build_episode(num_turns=4, quality=0.4, tool_turns={2: TOOL_IDS[Action.AUDITOR]}),
        _build_episode(num_turns=2, quality=0.6),
    ]


def _make_task(task_id: str = "task-1") -> ResearchQATask:
    return ResearchQATask(
        task_id=task_id,
        difficulty=TaskDifficulty.TYPICAL,
        split=TaskSplit.EVAL,
        question="What is photosynthesis?",
        gold_answer="Plants convert light energy into chemical energy.",
        known_gaps=None,
        known_assumptions=None,
        scoring_rubric="5: Complete. 3: Partial. 1: Wrong.",
    )


# ---------------------------------------------------------------------------
# MCTS target generation
# ---------------------------------------------------------------------------


class TestMCTSTargets:
    def test_shapes_normalization_and_determinism(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        pipeline = TrainingDataPipeline()
        examples = pipeline.process_episodes(_synthetic_episodes())
        assert examples

        dists_a, values_a = trainer.build_mcts_targets(examples, 8, seed=5)
        dists_b, values_b = trainer.build_mcts_targets(examples, 8, seed=5)

        assert dists_a.shape == (len(examples), NUM_ACTIONS)
        assert values_a.shape == (len(examples),)
        np.testing.assert_allclose(dists_a.sum(axis=1), 1.0, atol=1e-5)
        assert (dists_a >= 0).all()
        np.testing.assert_array_equal(dists_a, dists_b)
        np.testing.assert_array_equal(values_a, values_b)

    def test_different_seed_changes_noisy_targets(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        examples = TrainingDataPipeline().process_episodes(_synthetic_episodes())
        dists_a, _ = trainer.build_mcts_targets(examples, 8, seed=1)
        dists_b, _ = trainer.build_mcts_targets(examples, 8, seed=2)
        # Root Dirichlet noise is on by default for targets, so different
        # seeds should produce at least one different distribution.
        assert not np.array_equal(dists_a, dists_b)


# ---------------------------------------------------------------------------
# Offline iterations
# ---------------------------------------------------------------------------


class TestOfflineIteration:
    def test_run_iteration_populates_metrics_and_checkpoints(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        metrics = trainer.run_iteration(0, 8, episodes=_synthetic_episodes())

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.iteration == 0
        assert metrics.n_episodes == 3
        assert metrics.n_examples == 9
        assert len(metrics.epoch_losses) == _TEST_CONFIG.epochs
        assert metrics.policy_entropy > 0.0
        assert metrics.value_mse is not None
        assert isinstance(metrics.kl_from_heuristic, float)
        assert 0.0 <= metrics.heuristic_agreement <= 1.0
        # >= 2 episodes -> episode-grouped holdout metrics exist.
        assert metrics.holdout is not None
        assert metrics.holdout["n_examples"] > 0
        # Offline: no live evaluation.
        assert metrics.eval_score is None
        assert metrics.heuristic_eval_score is None
        assert metrics.eval_scores_by_difficulty is None
        assert metrics.transition_metrics is None

        run_dir = tmp_path / "run"
        it_dir = run_dir / "iteration-000"
        assert (it_dir / "policy_value.pt").exists()
        assert (it_dir / "transition.pt").exists()
        saved = json.loads((it_dir / "metrics.json").read_text())
        assert saved["iteration"] == 0
        history = json.loads((run_dir / "metrics_history.json").read_text())
        assert len(history) == 1

        store = TrainingDataStore(run_dir / "store")
        assert len(store) == metrics.n_examples
        assert store.iterations == [0]

    def test_training_reduces_loss(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        metrics = trainer.run_iteration(0, 8, episodes=_synthetic_episodes())
        assert metrics.epoch_losses[-1]["total"] < metrics.epoch_losses[0]["total"]
        assert metrics.train_loss == metrics.epoch_losses[-1]["total"]

    def test_checkpoint_round_trip(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.run_iteration(0, 8, episodes=_synthetic_episodes())

        path = tmp_path / "run" / "iteration-000" / "policy_value.pt"
        reloaded = PolicyValueNetwork.load(
            path, input_dim=STATE_DIM, hidden_dim=_HIDDEN_DIM
        )
        state = np.random.default_rng(0).random(STATE_DIM).astype(np.float32)
        probs_a, value_a = trainer._policy.predict(state)  # noqa: SLF001
        probs_b, value_b = reloaded.predict(state)
        np.testing.assert_allclose(probs_a, probs_b)
        assert value_a == pytest.approx(value_b)

        transition_path = tmp_path / "run" / "iteration-000" / "transition.pt"
        reloaded_t = TransitionModel.load(transition_path, hidden_dim=_HIDDEN_DIM)
        next_a, reward_a = trainer.transition_model.predict(state, 0)
        next_b, reward_b = reloaded_t.predict(state, 0)
        np.testing.assert_allclose(next_a, next_b)
        assert reward_a == pytest.approx(reward_b)

    def test_iteration_counter_and_history_append(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        episodes = _synthetic_episodes()
        m0 = trainer.run_iteration(0, 4, episodes=episodes)
        m1 = trainer.run_iteration(0, 4, episodes=episodes)
        assert (m0.iteration, m1.iteration) == (0, 1)
        history = json.loads((tmp_path / "run" / "metrics_history.json").read_text())
        assert [h["iteration"] for h in history] == [0, 1]
        # A fresh trainer over the same directory resumes the counter.
        resumed = MCTSTrainer(
            _make_policy(),
            _make_transition(),
            checkpoint_dir=tmp_path / "run",
            config=_TEST_CONFIG,
        )
        assert resumed.iteration == 2

    def test_deterministic_across_trainers(self, tmp_path):
        episodes = _synthetic_episodes()
        metrics = []
        for name in ("a", "b"):
            trainer = _make_trainer(tmp_path / name)
            metrics.append(trainer.run_iteration(0, 8, episodes=episodes))
        assert metrics[0].train_loss == metrics[1].train_loss
        assert metrics[0].policy_entropy == metrics[1].policy_entropy
        assert metrics[0].kl_from_heuristic == metrics[1].kl_from_heuristic

    def test_retrain_transition(self, tmp_path):
        config = MCTSTrainerConfig(
            epochs=3,
            seed=0,
            retrain_transition=True,
            transition_config=TransitionTrainingConfig(epochs=5),
        )
        trainer = _make_trainer(tmp_path, config=config)
        original = trainer.transition_model
        metrics = trainer.run_iteration(0, 4, episodes=_synthetic_episodes())
        assert metrics.transition_metrics is not None
        assert metrics.transition_metrics["n_train"] > 0
        assert trainer.transition_model is not original

    def test_empty_episodes_raise(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        with pytest.raises(ValueError, match="no training examples"):
            trainer.run_iteration(0, 4, episodes=[])

    def test_live_collection_without_runner_raises(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        with pytest.raises(ValueError, match="runner"):
            trainer.run_iteration(2, 4)


class TestParallelCollection:
    """Issue #91: bounded concurrent collection preserves episode order."""

    def _collect(self, tmp_path, parallel: int) -> list[Episode]:
        import time

        tasks = [_make_task(f"task-{i}") for i in range(4)]
        runner = MagicMock(spec=EpisodeRunner)

        def run_episode(task, controller):
            # Later tasks finish sooner, inverting completion order at N>1.
            idx = int(task.task_id.split("-")[1])
            time.sleep((len(tasks) - idx) * 0.01)
            episode = _build_episode(num_turns=2)
            return episode.model_copy(update={"metadata": {"task_id": task.task_id}})

        runner.run_episode.side_effect = run_episode
        trainer = _make_trainer(
            tmp_path,
            config=MCTSTrainerConfig(
                epochs=1, seed=0, parallel_episodes=parallel
            ),
            runner=runner,
            train_tasks=tasks,
        )
        return trainer._collect(len(tasks), n_simulations=2, base_seed=0)

    def test_parallel_collect_matches_sequential_order(self, tmp_path):
        sequential = self._collect(tmp_path / "seq", 1)
        parallel = self._collect(tmp_path / "par", 3)
        expected = [f"task-{i}" for i in range(4)]
        assert [e.metadata["task_id"] for e in sequential] == expected
        assert [e.metadata["task_id"] for e in parallel] == expected

    def test_parallel_collect_propagates_failures(self, tmp_path):
        tasks = [_make_task(f"task-{i}") for i in range(3)]
        runner = MagicMock(spec=EpisodeRunner)
        runner.run_episode.side_effect = RuntimeError("API meltdown")
        trainer = _make_trainer(
            tmp_path,
            config=MCTSTrainerConfig(epochs=1, seed=0, parallel_episodes=3),
            runner=runner,
            train_tasks=tasks,
        )
        with pytest.raises(RuntimeError, match="API meltdown"):
            trainer._collect(3, n_simulations=2, base_seed=0)


# ---------------------------------------------------------------------------
# Live-style collection through the real runner (mocked model client)
# ---------------------------------------------------------------------------


class TestMockedCollection:
    def _make_runner(self, max_turns: int) -> EpisodeRunner:
        client = MagicMock(spec=GeminiClient)
        client.generate.return_value = GeminiResponse(
            content="Mocked answer with enough words to count.",
            input_tokens=10,
            output_tokens=20,
            duration_ms=50.0,
            finish_reason="STOP",
        )
        return EpisodeRunner(client, EpisodeConfig(max_turns=max_turns))

    def test_collects_episodes_through_real_runner(self, tmp_path):
        max_turns = 5
        config = MCTSTrainerConfig(epochs=5, seed=0, max_turns=max_turns)
        trainer = MCTSTrainer(
            _make_policy(),
            _make_transition(),
            checkpoint_dir=tmp_path / "run",
            config=config,
            runner=self._make_runner(max_turns),
            train_tasks=[_make_task("task-1"), _make_task("task-2")],
        )

        def sim_respond(task, response, history, *, turn_number):
            if turn_number < 3:
                return UserAction(
                    action_type=ActionType.FOLLOW_UP,
                    message="Tell me more.",
                    followup_type=FollowUpType.ELABORATION,
                    response_delay_ms=100,
                    confidence=0.8,
                )
            return UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )

        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.side_effect = sim_respond
            MockSimUser.return_value = mock_sim
            with (
                patch("bicameral_agent.episode_runner.ResearchGapScanner") as m1,
                patch("bicameral_agent.episode_runner.AssumptionAuditor") as m2,
                patch("bicameral_agent.episode_runner.ContextRefresher") as m3,
            ):
                for mock_cls, tool_id in (
                    (m1, "research_gap_scanner"),
                    (m2, "assumption_auditor"),
                    (m3, "context_refresher"),
                ):
                    mock_tool = MagicMock()
                    mock_tool.execute.return_value = ToolResult(
                        queue_deposit=None,
                        metadata=ToolMetadata(
                            tool_id=tool_id,
                            action_taken="ran",
                            confidence=0.8,
                            items_found=0,
                            estimated_relevance=0.0,
                            tokens_consumed=30,
                        ),
                    )
                    mock_cls.return_value = mock_tool

                metrics = trainer.run_iteration(2, 4)

        assert metrics.n_episodes == 2
        assert metrics.n_examples == 6  # 3 decision points per episode
        # Collected episodes are persisted for offline reuse.
        parquet = tmp_path / "run" / "episodes" / "iteration-000.parquet"
        assert parquet.exists()
        # No eval tasks -> live evaluation skipped.
        assert metrics.eval_score is None
