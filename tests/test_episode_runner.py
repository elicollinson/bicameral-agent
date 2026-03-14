"""Tests for the end-to-end episode runner."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.episode_runner import (
    Controller,
    EpisodeConfig,
    EpisodeRunner,
    InjectionMode,
    RandomController,
)
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.gemini import GeminiClient, GeminiResponse
from bicameral_agent.heuristic_controller import Action, FullState, HeuristicController
from bicameral_agent.schema import Episode, UserEventType
from bicameral_agent.simulated_user import ActionType, UserAction
from bicameral_agent.tool_primitive import BudgetExceededError


def _make_task(**overrides) -> ResearchQATask:
    defaults = dict(
        task_id="test-001",
        difficulty=TaskDifficulty.TYPICAL,
        split=TaskSplit.EVAL,
        question="What is photosynthesis?",
        gold_answer="Photosynthesis is the process by which plants convert light energy into chemical energy.",
        known_gaps=None,
        known_assumptions=None,
        scoring_rubric="5: Complete explanation. 3: Partial. 1: Wrong.",
    )
    defaults.update(overrides)
    return ResearchQATask(**defaults)


def _mock_gemini_response(content="Test response", input_tokens=10, output_tokens=20):
    return GeminiResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=100.0,
        finish_reason="STOP",
    )


def _make_mock_client():
    client = MagicMock(spec=GeminiClient)
    client.generate.return_value = _mock_gemini_response()
    return client


def _make_state(**overrides) -> FullState:
    defaults = dict(
        turn_number=1,
        stop_count=0,
        followup_type=FollowUpType.ELABORATION,
        queue_depth=0,
        executing_tools=(),
        predicted_latencies={},
    )
    defaults.update(overrides)
    return FullState(**defaults)


# ---------------------------------------------------------------------------
# TestControllerProtocol
# ---------------------------------------------------------------------------


class TestControllerProtocol:
    def test_heuristic_controller_satisfies_protocol(self):
        assert isinstance(HeuristicController(), Controller)

    def test_random_controller_satisfies_protocol(self):
        assert isinstance(RandomController(), Controller)


# ---------------------------------------------------------------------------
# TestRandomController
# ---------------------------------------------------------------------------


class TestRandomController:
    def test_queue_depth_guard(self):
        """Queue depth >= 3 forces DO_NOTHING regardless of probability."""
        ctrl = RandomController(action_probability=1.0, seed=42)
        action = ctrl.decide(_make_state(queue_depth=3))
        assert action == Action.DO_NOTHING

    def test_seed_reproducibility(self):
        """Same seed produces identical action sequences."""
        state = _make_state()
        result_a = RandomController(seed=123).decide(state)
        result_b = RandomController(seed=123).decide(state)
        assert result_a == result_b

    def test_action_distribution(self):
        """With probability 1.0, all decisions should be tool actions."""
        ctrl = RandomController(action_probability=1.0, seed=42)
        state = _make_state()
        actions = [ctrl.decide(state) for _ in range(100)]
        assert all(a != Action.DO_NOTHING for a in actions)
        assert len(set(actions)) >= 2

    def test_zero_probability(self):
        """With probability 0, always DO_NOTHING."""
        ctrl = RandomController(action_probability=0.0, seed=42)
        state = _make_state()
        actions = [ctrl.decide(state) for _ in range(20)]
        assert all(a == Action.DO_NOTHING for a in actions)

    def test_one_probability(self):
        """With probability 1.0 and depth < 3, never DO_NOTHING."""
        ctrl = RandomController(action_probability=1.0, seed=42)
        state = _make_state(turn_number=5)
        actions = [ctrl.decide(state) for _ in range(50)]
        assert all(a != Action.DO_NOTHING for a in actions)

    def test_decisions_recorded(self):
        ctrl = RandomController(seed=42)
        state = _make_state()
        ctrl.decide(state)
        ctrl.decide(state)
        assert len(ctrl.decisions) == 2
        assert all(d.rule_fired == 0 for d in ctrl.decisions)


# ---------------------------------------------------------------------------
# TestEpisodeConfig
# ---------------------------------------------------------------------------


class TestEpisodeConfig:
    def test_defaults(self):
        cfg = EpisodeConfig()
        assert cfg.max_turns == 25
        assert cfg.thinking_level == "medium"
        assert cfg.score_episode is False
        assert cfg.use_lexical_scorer is False

    def test_custom_values(self):
        cfg = EpisodeConfig(max_turns=10, thinking_level="low", score_episode=True)
        assert cfg.max_turns == 10
        assert cfg.thinking_level == "low"
        assert cfg.score_episode is True


# ---------------------------------------------------------------------------
# TestEpisodeRunner
# ---------------------------------------------------------------------------


class TestEpisodeRunner:
    def _run_with_user_actions(
        self,
        user_actions: list[UserAction],
        controller_actions: list[Action] | None = None,
        config: EpisodeConfig | None = None,
    ) -> Episode:
        """Helper: run episode with mocked components."""
        client = _make_mock_client()

        # SimulatedUser mock
        action_iter = iter(user_actions)

        def sim_respond(task, response, history):
            return next(action_iter)

        # Controller mock
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        if controller_actions:
            ctrl.decide.side_effect = controller_actions
        else:
            ctrl.decide.return_value = Action.DO_NOTHING

        runner = EpisodeRunner(client, config or EpisodeConfig(max_turns=len(user_actions) + 5))

        with patch(
            "bicameral_agent.episode_runner.SimulatedUser"
        ) as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.side_effect = sim_respond
            MockSimUser.return_value = mock_sim

            episode = runner.run_episode(_make_task(), ctrl)

        return episode

    def test_single_turn_task_complete(self):
        """TASK_COMPLETE on first turn produces a valid single-turn episode."""
        episode = self._run_with_user_actions(
            [UserAction(action_type=ActionType.TASK_COMPLETE, response_delay_ms=100, confidence=0.9)]
        )
        assert isinstance(episode, Episode)
        assert episode.outcome.total_turns == 1
        # Should have user + assistant messages
        user_msgs = [m for m in episode.messages if m.role == "user"]
        assistant_msgs = [m for m in episode.messages if m.role == "assistant"]
        assert len(user_msgs) == 1
        assert len(assistant_msgs) == 1

    def test_multi_turn_follow_up(self):
        """FOLLOW_UP extends the conversation, TASK_COMPLETE ends it."""
        episode = self._run_with_user_actions([
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="Can you elaborate?",
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            ),
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="What about the details?",
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            ),
            UserAction(action_type=ActionType.TASK_COMPLETE, response_delay_ms=100, confidence=0.9),
        ])
        assert episode.outcome.total_turns == 3
        # Should have FOLLOW_UP events
        followup_events = [e for e in episode.user_events if e.event_type == UserEventType.FOLLOW_UP]
        assert len(followup_events) == 2

    def test_stop_termination(self):
        """STOP action ends the episode and logs a STOP event."""
        episode = self._run_with_user_actions([
            UserAction(action_type=ActionType.STOP, response_delay_ms=100, confidence=0.5),
        ])
        assert episode.outcome.total_turns == 1
        stop_events = [e for e in episode.user_events if e.event_type == UserEventType.STOP]
        assert len(stop_events) == 1

    def test_max_turns_enforcement(self):
        """Episode stops after max_turns even with continuous FOLLOW_UP."""
        actions = [
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                message=f"Follow up {i}",
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            )
            for i in range(10)
        ]
        config = EpisodeConfig(max_turns=3)
        episode = self._run_with_user_actions(actions[:3], config=config)
        assert episode.outcome.total_turns == 3

    def test_do_nothing_no_tool(self):
        """DO_NOTHING means no tool invocations are logged."""
        episode = self._run_with_user_actions([
            UserAction(action_type=ActionType.TASK_COMPLETE, response_delay_ms=100, confidence=0.9),
        ])
        assert len(episode.tool_invocations) == 0

    def test_tool_invocation_logging(self):
        """Tool invocations are logged when controller selects an action."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.SCANNER

        runner = EpisodeRunner(
            client,
            EpisodeConfig(max_turns=1),
        )

        with patch(
            "bicameral_agent.episode_runner.SimulatedUser"
        ) as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = mock_sim

            # Mock the tool's execute method to return a result
            with patch(
                "bicameral_agent.episode_runner.ResearchGapScanner"
            ) as MockScanner:
                from bicameral_agent.tool_primitive import ToolMetadata, ToolResult

                mock_tool = MagicMock()
                mock_tool.execute.return_value = ToolResult(
                    queue_deposit=None,
                    metadata=ToolMetadata(
                        tool_id="research_gap_scanner",
                        action_taken="scanned",
                        confidence=0.8,
                        items_found=2,
                        estimated_relevance=0.7,
                        tokens_consumed=50,
                    ),
                )
                MockScanner.return_value = mock_tool

                episode = runner.run_episode(_make_task(), ctrl)

        assert len(episode.tool_invocations) == 1
        assert episode.tool_invocations[0].tool_id == "research_gap_scanner"

    def test_context_injection_logging(self):
        """Queue deposits are logged as context injections."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.SCANNER

        runner = EpisodeRunner(
            client,
            EpisodeConfig(max_turns=1),
        )

        with patch(
            "bicameral_agent.episode_runner.SimulatedUser"
        ) as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = mock_sim

            with patch(
                "bicameral_agent.episode_runner.ResearchGapScanner"
            ) as MockScanner:
                from bicameral_agent.queue import Priority, QueueItem
                from bicameral_agent.tool_primitive import ToolMetadata, ToolResult

                mock_tool = MagicMock()
                mock_tool.execute.return_value = ToolResult(
                    queue_deposit=QueueItem(
                        content="New research context",
                        priority=Priority.HIGH,
                        source_tool_id="research_gap_scanner",
                        token_count=15,
                    ),
                    metadata=ToolMetadata(
                        tool_id="research_gap_scanner",
                        action_taken="scanned",
                        confidence=0.8,
                        items_found=1,
                        estimated_relevance=0.9,
                        tokens_consumed=30,
                    ),
                )
                MockScanner.return_value = mock_tool

                episode = runner.run_episode(_make_task(), ctrl)

        assert len(episode.context_injections) == 1
        assert episode.context_injections[0].source_tool_id == "research_gap_scanner"

    def test_budget_exceeded_handling(self):
        """BudgetExceededError is caught gracefully; episode continues."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.SCANNER

        runner = EpisodeRunner(
            client,
            EpisodeConfig(max_turns=1),
        )

        with patch(
            "bicameral_agent.episode_runner.SimulatedUser"
        ) as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = mock_sim

            with patch(
                "bicameral_agent.episode_runner.ResearchGapScanner"
            ) as MockScanner:
                mock_tool = MagicMock()
                mock_tool.execute.side_effect = BudgetExceededError("budget exceeded")
                MockScanner.return_value = mock_tool

                episode = runner.run_episode(_make_task(), ctrl)

        # Episode completes despite budget error
        assert isinstance(episode, Episode)
        # Tool invocation logged with 0 output
        assert len(episode.tool_invocations) == 1
        assert episode.tool_invocations[0].output_tokens == 0

    def test_controller_state_correctness(self):
        """Controller receives correct turn number and queue depth."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.DO_NOTHING
        recorded_states: list[FullState] = []

        def capture_state(state):
            recorded_states.append(state)
            return Action.DO_NOTHING

        ctrl.decide.side_effect = capture_state

        runner = EpisodeRunner(
            client,
            EpisodeConfig(max_turns=3),
        )

        with patch(
            "bicameral_agent.episode_runner.SimulatedUser"
        ) as MockSimUser:
            mock_sim = MagicMock()
            follow_up = UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="More please",
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            )
            complete = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            mock_sim.respond.side_effect = [follow_up, follow_up, complete]
            MockSimUser.return_value = mock_sim

            runner.run_episode(_make_task(), ctrl)

        assert len(recorded_states) == 3
        assert recorded_states[0].turn_number == 1
        assert recorded_states[1].turn_number == 2
        assert recorded_states[2].turn_number == 3

    def test_schema_validation(self):
        """Produced episode passes EpisodeValidator."""
        from bicameral_agent.validation import EpisodeValidator

        episode = self._run_with_user_actions([
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="Tell me more",
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            ),
            UserAction(action_type=ActionType.TASK_COMPLETE, response_delay_ms=100, confidence=0.9),
        ])
        result = EpisodeValidator().validate(episode)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_user_events_recorded(self):
        """STOP and FOLLOW_UP events are recorded in the episode."""
        episode = self._run_with_user_actions([
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="More detail please",
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            ),
            UserAction(action_type=ActionType.STOP, response_delay_ms=100, confidence=0.5),
        ])
        event_types = [e.event_type for e in episode.user_events]
        assert UserEventType.FOLLOW_UP in event_types
        assert UserEventType.STOP in event_types


# ---------------------------------------------------------------------------
# TestInjectionModes
# ---------------------------------------------------------------------------


class TestInjectionModes:
    """Tests for SYNCHRONOUS, BREAKPOINT, and INTERRUPT injection modes."""

    def _run_with_mode(
        self,
        mode: InjectionMode,
        controller_actions: list[Action] | None = None,
    ) -> Episode:
        """Run a single-turn episode with a specific injection mode."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        if controller_actions:
            ctrl.decide.side_effect = controller_actions
        else:
            ctrl.decide.return_value = Action.DO_NOTHING

        config = EpisodeConfig(max_turns=1, injection_mode=mode)
        runner = EpisodeRunner(client, config)

        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = mock_sim
            episode = runner.run_episode(_make_task(), ctrl)

        return episode

    def test_breakpoint_preserves_existing_behavior(self):
        """BREAKPOINT mode should produce the same structure as before."""
        episode = self._run_with_mode(InjectionMode.BREAKPOINT)
        assert isinstance(episode, Episode)
        assert episode.outcome.total_turns == 1
        assert episode.metadata.get("injection_mode") == "breakpoint"

    def test_synchronous_mode_metadata(self):
        """SYNCHRONOUS mode stores mode in metadata."""
        episode = self._run_with_mode(InjectionMode.SYNCHRONOUS)
        assert episode.metadata.get("injection_mode") == "synchronous"

    def test_interrupt_mode_metadata(self):
        """INTERRUPT mode stores mode in metadata."""
        episode = self._run_with_mode(InjectionMode.INTERRUPT)
        assert episode.metadata.get("injection_mode") == "interrupt"

    def test_synchronous_triggers_regeneration(self):
        """SYNCHRONOUS mode regenerates after tool deposit."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.SCANNER

        config = EpisodeConfig(max_turns=1, injection_mode=InjectionMode.SYNCHRONOUS)
        runner = EpisodeRunner(client, config)

        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = mock_sim

            with patch("bicameral_agent.episode_runner.ResearchGapScanner") as MockScanner:
                from bicameral_agent.queue import Priority, QueueItem
                from bicameral_agent.tool_primitive import ToolMetadata, ToolResult

                mock_tool = MagicMock()
                mock_tool.execute.return_value = ToolResult(
                    queue_deposit=QueueItem(
                        content="New context from scanner",
                        priority=Priority.HIGH,
                        source_tool_id="research_gap_scanner",
                        token_count=10,
                    ),
                    metadata=ToolMetadata(
                        tool_id="research_gap_scanner",
                        action_taken="scanned",
                        confidence=0.8,
                        items_found=1,
                        estimated_relevance=0.9,
                        tokens_consumed=30,
                    ),
                )
                MockScanner.return_value = mock_tool

                episode = runner.run_episode(_make_task(), ctrl)

        # In synchronous mode, tool deposit should trigger regeneration
        # The generate method should be called more than once (original + regen)
        assert client.generate.call_count >= 2
        assert episode.metadata.get("injection_mode") == "synchronous"

    def test_interrupt_mode_threshold(self):
        """INTERRUPT mode only regenerates when threshold exceeded."""
        from bicameral_agent.queue import InterruptConfig, Priority, QueueItem
        from bicameral_agent.tool_primitive import ToolMetadata, ToolResult

        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.SCANNER

        # Set high threshold so interrupt is NOT triggered
        config = EpisodeConfig(
            max_turns=1,
            injection_mode=InjectionMode.INTERRUPT,
            interrupt_config=InterruptConfig(
                count_threshold=100,
                token_threshold=100000,
                priority_threshold=Priority.CRITICAL,
            ),
        )
        runner = EpisodeRunner(client, config)

        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = mock_sim

            with patch("bicameral_agent.episode_runner.ResearchGapScanner") as MockScanner:
                mock_tool = MagicMock()
                mock_tool.execute.return_value = ToolResult(
                    queue_deposit=QueueItem(
                        content="Low priority context",
                        priority=Priority.LOW,
                        source_tool_id="research_gap_scanner",
                        token_count=5,
                    ),
                    metadata=ToolMetadata(
                        tool_id="research_gap_scanner",
                        action_taken="scanned",
                        confidence=0.5,
                        items_found=1,
                        estimated_relevance=0.3,
                        tokens_consumed=20,
                    ),
                )
                MockScanner.return_value = mock_tool

                episode = runner.run_episode(_make_task(), ctrl)

        # Threshold not exceeded → no regeneration, only 1 generate call per turn
        # (conscious loop's run_turn calls generate once, plus possible interrupt check)
        assert episode.metadata.get("interrupt_count") == 0
        assert episode.metadata.get("injection_mode") == "interrupt"

    def test_interrupt_count_tracked(self):
        """interrupt_count is tracked in metadata."""
        episode = self._run_with_mode(InjectionMode.INTERRUPT)
        assert "interrupt_count" in episode.metadata
        assert isinstance(episode.metadata["interrupt_count"], int)

    def test_injection_mode_in_episode_config(self):
        """EpisodeConfig defaults to BREAKPOINT."""
        cfg = EpisodeConfig()
        assert cfg.injection_mode == InjectionMode.BREAKPOINT

    def test_injection_mode_configurable(self):
        """EpisodeConfig accepts custom injection mode."""
        cfg = EpisodeConfig(injection_mode=InjectionMode.SYNCHRONOUS)
        assert cfg.injection_mode == InjectionMode.SYNCHRONOUS


# ---------------------------------------------------------------------------
# Integration test stubs (require GEMINI_API_KEY)
# ---------------------------------------------------------------------------

_SKIP_REASON = "GEMINI_API_KEY not set"
_has_key = os.environ.get("GEMINI_API_KEY") is not None


@pytest.mark.skipif(not _has_key, reason=_SKIP_REASON)
class TestIntegrationHeuristic:
    def test_typical_task(self):
        client = GeminiClient()
        task = _make_task()
        runner = EpisodeRunner(client, EpisodeConfig(max_turns=5))
        episode = runner.run_episode(task, HeuristicController())
        assert isinstance(episode, Episode)

    def test_hard_task(self):
        client = GeminiClient()
        task = _make_task(difficulty=TaskDifficulty.HARD)
        runner = EpisodeRunner(client, EpisodeConfig(max_turns=5))
        episode = runner.run_episode(task, HeuristicController())
        assert isinstance(episode, Episode)

    def test_tricky_task(self):
        client = GeminiClient()
        task = _make_task(
            difficulty=TaskDifficulty.TRICKY,
            known_assumptions=["Plants need sunlight"],
        )
        runner = EpisodeRunner(client, EpisodeConfig(max_turns=5))
        episode = runner.run_episode(task, HeuristicController())
        assert isinstance(episode, Episode)


@pytest.mark.skipif(not _has_key, reason=_SKIP_REASON)
class TestIntegrationRandom:
    def test_typical_task(self):
        client = GeminiClient()
        task = _make_task()
        runner = EpisodeRunner(client, EpisodeConfig(max_turns=5))
        episode = runner.run_episode(task, RandomController(seed=42))
        assert isinstance(episode, Episode)

    def test_hard_task(self):
        client = GeminiClient()
        task = _make_task(difficulty=TaskDifficulty.HARD)
        runner = EpisodeRunner(client, EpisodeConfig(max_turns=5))
        episode = runner.run_episode(task, RandomController(seed=42))
        assert isinstance(episode, Episode)

    def test_tricky_task(self):
        client = GeminiClient()
        task = _make_task(
            difficulty=TaskDifficulty.TRICKY,
            known_assumptions=["Plants need sunlight"],
        )
        runner = EpisodeRunner(client, EpisodeConfig(max_turns=5))
        episode = runner.run_episode(task, RandomController(seed=42))
        assert isinstance(episode, Episode)


@pytest.mark.skipif(not _has_key, reason=_SKIP_REASON)
class TestIntegrationReplay:
    def test_episode_replayable(self):
        from bicameral_agent.replay import EpisodeReplayer

        client = GeminiClient()
        task = _make_task()
        runner = EpisodeRunner(client, EpisodeConfig(max_turns=3))
        episode = runner.run_episode(task, HeuristicController())
        replayer = EpisodeReplayer(episode)
        assert replayer.total_turns >= 1


@pytest.mark.skipif(not _has_key, reason=_SKIP_REASON)
class TestIntegrationControllerSwap:
    def test_drop_in_replacement(self):
        """Both controllers produce valid episodes for the same task."""
        client = GeminiClient()
        task = _make_task()
        config = EpisodeConfig(max_turns=3)
        runner = EpisodeRunner(client, config)
        ep_h = runner.run_episode(task, HeuristicController())
        ep_r = runner.run_episode(task, RandomController(seed=42))
        assert isinstance(ep_h, Episode)
        assert isinstance(ep_r, Episode)
