"""Tests for the learned-policy controller (issue #29).

The module requires torch (it exercises the real PolicyValueNetwork /
TransitionModel), so it is skipped wholesale when torch is unavailable.
"""

from __future__ import annotations

# ruff: noqa: E402  (imports below the importorskip guard are intentional)

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.episode_runner import Controller, EpisodeConfig, EpisodeRunner
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.gemini import GeminiClient, GeminiResponse
from bicameral_agent.heuristic_controller import FullState
from bicameral_agent.learned_controller import LEARNED_RULE_ID, LearnedPolicyController
from bicameral_agent.mcts import MCTSEngine
from bicameral_agent.policy_value_net import ACTION_ORDER, PolicyValueNetwork
from bicameral_agent.queue import Priority, QueueItem
from bicameral_agent.schema import Episode, EpisodeOutcome, Message
from bicameral_agent.simulated_user import ActionType, UserAction
from bicameral_agent.tool_primitive import ToolMetadata, ToolResult
from bicameral_agent.training_pipeline import STATE_DIM, TrainingDataPipeline
from bicameral_agent.transition_model import TransitionModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(seed: int = 0) -> PolicyValueNetwork:
    torch.manual_seed(seed)
    net = PolicyValueNetwork(input_dim=STATE_DIM, hidden_dim=32)
    net.eval()
    return net


def _make_transition(seed: int = 0) -> TransitionModel:
    torch.manual_seed(seed)
    model = TransitionModel(hidden_dim=32)
    model.eval()
    return model


def _make_task(**overrides) -> ResearchQATask:
    defaults = dict(
        task_id="test-001",
        difficulty=TaskDifficulty.TYPICAL,
        split=TaskSplit.EVAL,
        question="What is photosynthesis?",
        gold_answer="Plants convert light energy into chemical energy.",
        known_gaps=None,
        known_assumptions=None,
        scoring_rubric="5: Complete explanation. 3: Partial. 1: Wrong.",
    )
    defaults.update(overrides)
    return ResearchQATask(**defaults)


def _make_mock_client():
    client = MagicMock(spec=GeminiClient)
    client.generate.return_value = GeminiResponse(
        content="Test response with some detail.",
        input_tokens=10,
        output_tokens=20,
        duration_ms=100.0,
        finish_reason="STOP",
    )
    return client


def _tool_result(tool_id: str, deposit: bool) -> ToolResult:
    queue_deposit = None
    if deposit:
        queue_deposit = QueueItem(
            content="context from " + tool_id,
            priority=Priority.MEDIUM,
            source_tool_id=tool_id,
            token_count=12,
        )
    return ToolResult(
        queue_deposit=queue_deposit,
        metadata=ToolMetadata(
            tool_id=tool_id,
            action_taken="ran",
            confidence=0.8,
            items_found=1,
            estimated_relevance=0.5,
            tokens_consumed=40,
        ),
    )


def _run_real_runner_episode(
    controller: LearnedPolicyController, max_turns: int, n_followups: int = 3
) -> Episode:
    """Run a full episode through the real EpisodeRunner with mocked LLM roles."""
    client = _make_mock_client()
    runner = EpisodeRunner(client, EpisodeConfig(max_turns=max_turns))

    def sim_respond(task, response, history, *, turn_number):
        if turn_number <= n_followups:
            return UserAction(
                action_type=ActionType.FOLLOW_UP,
                message=f"Tell me more, please ({turn_number}).",
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            )
        return UserAction(
            action_type=ActionType.TASK_COMPLETE, response_delay_ms=100, confidence=0.9
        )

    with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
        mock_sim = MagicMock()
        mock_sim.respond.side_effect = sim_respond
        MockSimUser.return_value = mock_sim
        with (
            patch("bicameral_agent.episode_runner.ResearchGapScanner") as MockScanner,
            patch("bicameral_agent.episode_runner.AssumptionAuditor") as MockAuditor,
            patch("bicameral_agent.episode_runner.ContextRefresher") as MockRefresher,
        ):
            for mock_cls, tool_id in (
                (MockScanner, "research_gap_scanner"),
                (MockAuditor, "assumption_auditor"),
                (MockRefresher, "context_refresher"),
            ):
                mock_tool = MagicMock()
                mock_tool.execute.return_value = _tool_result(tool_id, deposit=True)
                mock_cls.return_value = mock_tool
            return runner.run_episode(_make_task(), controller)


def _synthetic_snapshot(num_turns: int = 2) -> Episode:
    """Episode ending with an assistant message, as a live snapshot would."""
    messages: list[Message] = []
    base_ts = 1_000_000
    for turn in range(1, num_turns + 1):
        messages.append(
            Message(
                role="user",
                content=f"question part {turn}",
                timestamp_ms=base_ts + (turn - 1) * 1000,
                token_count=10,
            )
        )
        messages.append(
            Message(
                role="assistant",
                content=f"answer part {turn}",
                timestamp_ms=base_ts + (turn - 1) * 1000 + 500,
                token_count=20,
            )
        )
    return Episode(
        messages=messages,
        outcome=EpisodeOutcome(
            quality_score=None, total_tokens=0, total_turns=num_turns, wall_clock_ms=0
        ),
    )


def _full_state(turn_number: int) -> FullState:
    return FullState(
        turn_number=turn_number,
        stop_count=0,
        followup_type=FollowUpType.ELABORATION,
        queue_depth=0,
        executing_tools=(),
        predicted_latencies={},
    )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class TestContract:
    def test_satisfies_controller_protocol(self):
        ctrl = LearnedPolicyController(_make_policy())
        assert isinstance(ctrl, Controller)

    def test_rejects_encoder_dim_network(self):
        """A default (64-dim) PolicyValueNetwork cannot consume pipeline states."""
        with pytest.raises(ValueError, match="input_dim"):
            LearnedPolicyController(PolicyValueNetwork())

    def test_rejects_bad_temperature_and_budget(self):
        with pytest.raises(ValueError, match="temperature"):
            LearnedPolicyController(_make_policy(), temperature=-0.1)
        with pytest.raises(ValueError, match="num_simulations"):
            LearnedPolicyController(_make_policy(), num_simulations=0)

    def test_decide_without_snapshot_raises(self):
        ctrl = LearnedPolicyController(_make_policy())
        with pytest.raises(RuntimeError, match="observe_episode"):
            ctrl.decide(_full_state(1))

    def test_stale_snapshot_raises(self):
        ctrl = LearnedPolicyController(_make_policy())
        ctrl.observe_episode(_synthetic_snapshot(num_turns=2))
        with pytest.raises(RuntimeError, match="stale"):
            ctrl.decide(_full_state(5))

    def test_decisions_logged(self):
        ctrl = LearnedPolicyController(_make_policy())
        ctrl.observe_episode(_synthetic_snapshot(num_turns=1))
        action = ctrl.decide(_full_state(1))
        assert len(ctrl.decisions) == 1
        assert ctrl.decisions[0].action == action
        assert ctrl.decisions[0].rule_fired == LEARNED_RULE_ID
        assert ctrl.decisions[0].state.turn_number == 1
        assert len(ctrl.encoded_states) == 1
        assert ctrl.encoded_states[0].shape == (STATE_DIM,)
        assert len(ctrl.distributions) == 1
        assert ctrl.distributions[0].shape == (len(ACTION_ORDER),)


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------


class TestActionSelection:
    def test_temperature_zero_is_policy_argmax(self):
        policy = _make_policy(seed=1)
        ctrl = LearnedPolicyController(policy)
        ctrl.observe_episode(_synthetic_snapshot(num_turns=1))
        action = ctrl.decide(_full_state(1))
        probs, _ = policy.predict(ctrl.encoded_states[0])
        assert action == ACTION_ORDER[int(np.argmax(probs))]

    def test_sampling_deterministic_with_seed(self):
        snapshot = _synthetic_snapshot(num_turns=2)
        actions = []
        for _ in range(2):
            ctrl = LearnedPolicyController(
                _make_policy(seed=2), temperature=1.5, seed=99
            )
            run: list = []
            for turn in (1, 2):
                partial = Episode(
                    messages=snapshot.messages[: turn * 2],
                    outcome=snapshot.outcome,
                )
                ctrl.observe_episode(partial)
                run.append(ctrl.decide(_full_state(turn)))
            actions.append(run)
        assert actions[0] == actions[1]

    def test_mcts_mode_matches_engine_search(self):
        """The MCTS decision equals argmax of an identical engine's search."""
        policy = _make_policy(seed=3)
        transition = _make_transition(seed=3)
        ctrl = LearnedPolicyController(
            policy,
            mcts_engine=MCTSEngine(policy, transition, seed=7),
            num_simulations=16,
        )
        ctrl.observe_episode(_synthetic_snapshot(num_turns=1))
        action = ctrl.decide(_full_state(1))

        fresh_engine = MCTSEngine(policy, transition, seed=7)
        result = fresh_engine.search(ctrl.encoded_states[0], num_simulations=16)
        assert action == ACTION_ORDER[int(np.argmax(result.action_distribution))]
        np.testing.assert_allclose(ctrl.distributions[0], result.action_distribution)


# ---------------------------------------------------------------------------
# Train/serve consistency through the real runner
# ---------------------------------------------------------------------------


class TestTrainServeConsistency:
    def test_serve_time_encoding_matches_pipeline(self):
        """Live encodings equal the pipeline's post-hoc encodings bit-for-bit.

        Runs a real EpisodeRunner episode (mocked LLM roles, real logger /
        queue / replay) with a sampling controller so tool actions occur,
        then re-encodes the finalized episode with an independent
        TrainingDataPipeline and compares state vectors per decision point.
        """
        max_turns = 6
        ctrl = LearnedPolicyController(
            _make_policy(seed=4),
            temperature=1.0,
            seed=11,
            pipeline=TrainingDataPipeline(max_turns=max_turns),
        )
        episode = _run_real_runner_episode(ctrl, max_turns=max_turns, n_followups=4)

        pipeline = TrainingDataPipeline(max_turns=max_turns)
        examples = pipeline.process_episode(episode)

        assert len(examples) == len(ctrl.encoded_states) > 1
        for example, served in zip(examples, ctrl.encoded_states):
            np.testing.assert_array_equal(example.state, served)

    def test_runner_calls_observe_episode_each_turn(self):
        ctrl = LearnedPolicyController(_make_policy(seed=5))
        episode = _run_real_runner_episode(ctrl, max_turns=4, n_followups=2)
        assistant_msgs = [m for m in episode.messages if m.role == "assistant"]
        assert len(ctrl.decisions) == len(assistant_msgs) == 3
