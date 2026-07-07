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
)
from bicameral_agent.random_controller import RandomController
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.gemini import GeminiClient, GeminiResponse
from bicameral_agent.heuristic_controller import Action, FullState, HeuristicController
from bicameral_agent.schema import Episode, UserEventType
from bicameral_agent.simulated_user import ActionType, Strictness, UserAction
from bicameral_agent.cost_tracker import (
    CostBudgetExceeded,
    CostTrackedClient,
    CostTracker,
)
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
        assert cfg.metric == "llm_judge"

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

        def sim_respond(task, response, history, *, turn_number):
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
        # Tool invocation logged with 0 output and flagged as budget-exceeded
        assert len(episode.tool_invocations) == 1
        assert episode.tool_invocations[0].output_tokens == 0
        assert episode.tool_invocations[0].budget_exceeded is True

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

    def test_task_complete_event_recorded(self):
        """TASK_COMPLETE termination is recorded as a user event."""
        episode = self._run_with_user_actions([
            UserAction(action_type=ActionType.TASK_COMPLETE, response_delay_ms=100, confidence=0.9),
        ])
        event_types = [e.event_type for e in episode.user_events]
        assert UserEventType.TASK_COMPLETE in event_types

    def test_follow_up_event_records_followup_type(self):
        """FOLLOW_UP events carry the sim-user's own followup_type in metadata."""
        episode = self._run_with_user_actions([
            UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="Actually, back to my question.",
                followup_type=FollowUpType.REDIRECT,
                response_delay_ms=100,
                confidence=0.8,
            ),
            UserAction(action_type=ActionType.TASK_COMPLETE, response_delay_ms=100, confidence=0.9),
        ])
        followup_events = [
            e for e in episode.user_events if e.event_type == UserEventType.FOLLOW_UP
        ]
        assert len(followup_events) == 1
        assert followup_events[0].metadata["followup_type"] == "redirect"

    def test_assistant_message_logged_before_tool_events(self):
        """Assistant messages are logged at generation time, before tool events."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.SCANNER

        runner = EpisodeRunner(client, EpisodeConfig(max_turns=1))

        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = mock_sim

            with patch("bicameral_agent.episode_runner.ResearchGapScanner") as MockScanner:
                from bicameral_agent.tool_primitive import ToolMetadata, ToolResult

                mock_tool = MagicMock()
                mock_tool.execute.return_value = ToolResult(
                    queue_deposit=None,
                    metadata=ToolMetadata(
                        tool_id="research_gap_scanner",
                        action_taken="scanned",
                        confidence=0.8,
                        items_found=0,
                        estimated_relevance=0.0,
                        tokens_consumed=50,
                    ),
                )
                MockScanner.return_value = mock_tool

                episode = runner.run_episode(_make_task(), ctrl)

        assistant_msg = next(m for m in episode.messages if m.role == "assistant")
        assert assistant_msg.timestamp_ms <= episode.tool_invocations[0].invoked_at_ms

    def test_sim_user_receives_runner_turn(self):
        """The runner passes its own turn to the sim-user (no off-by-one).

        The runner appends the current exchange to the history before calling
        respond(); the sim-user must not re-derive the turn from that history.
        """
        import json

        sim_prompts: list[str] = []

        def generate(messages, **kwargs):
            if kwargs.get("response_schema") is not None:
                # SimulatedUser call: follow up twice, then complete
                sim_prompts.append(messages[0]["content"])
                if len(sim_prompts) < 3:
                    data = {
                        "action_type": "follow_up",
                        "message": "Tell me more.",
                        "followup_type": "elaboration",
                        "response_delay_ms": 100,
                        "confidence": 0.8,
                    }
                else:
                    data = {
                        "action_type": "task_complete",
                        "response_delay_ms": 100,
                        "confidence": 0.9,
                    }
                return GeminiResponse(
                    content=json.dumps(data),
                    input_tokens=5,
                    output_tokens=5,
                    duration_ms=10.0,
                    finish_reason="STOP",
                )
            return _mock_gemini_response()

        client = MagicMock(spec=GeminiClient)
        client.generate.side_effect = generate

        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.DO_NOTHING

        runner = EpisodeRunner(client, EpisodeConfig(max_turns=5))
        episode = runner.run_episode(_make_task(), ctrl)

        assert episode.outcome.total_turns == 3
        assert len(sim_prompts) == 3
        for turn, prompt in enumerate(sim_prompts, start=1):
            assert f"This is turn {turn}." in prompt


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

    def _run_mode_with_deposit(
        self,
        mode: InjectionMode,
        interrupt_config=None,
    ) -> Episode:
        """Run a single-turn episode where the tool deposits a zero-token item."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.SCANNER

        config = EpisodeConfig(
            max_turns=1, injection_mode=mode, interrupt_config=interrupt_config
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
                from bicameral_agent.queue import Priority, QueueItem
                from bicameral_agent.tool_primitive import ToolMetadata, ToolResult

                mock_tool = MagicMock()
                mock_tool.execute.return_value = ToolResult(
                    # token_count=0 so any total_tokens difference between
                    # modes is attributable purely to regeneration waste.
                    queue_deposit=QueueItem(
                        content="New context",
                        priority=Priority.HIGH,
                        source_tool_id="research_gap_scanner",
                        token_count=0,
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

                return runner.run_episode(_make_task(), ctrl)

    def test_regen_episode_costs_more_than_non_regen(self):
        """A regenerating episode charges the discarded generation's tokens."""
        regen_episode = self._run_mode_with_deposit(InjectionMode.SYNCHRONOUS)
        no_regen_episode = self._run_mode_with_deposit(InjectionMode.BREAKPOINT)

        # Mock generations cost input=10 + output=20 tokens; the synchronous
        # regen discards one full generation.
        assert regen_episode.metadata["wasted_tokens"] == 30
        assert no_regen_episode.metadata["wasted_tokens"] == 0
        assert (
            regen_episode.outcome.total_tokens
            > no_regen_episode.outcome.total_tokens
        )

    def test_interrupt_fires_with_reachable_config(self):
        """A HIGH-priority deposit fires an interrupt under AB_INTERRUPT_CONFIG."""
        from bicameral_agent.ab_test import AB_INTERRUPT_CONFIG

        episode = self._run_mode_with_deposit(
            InjectionMode.INTERRUPT, interrupt_config=AB_INTERRUPT_CONFIG
        )

        assert episode.metadata["interrupt_count"] == 1
        # The interrupt discarded one full mock generation (input 10 + output 20)
        assert episode.metadata["wasted_tokens"] == 30
        # The deposited context was consumed by the regeneration
        assert len(episode.context_injections) == 1
        assert episode.context_injections[0].consumed is True

    def test_injection_mode_in_episode_config(self):
        """EpisodeConfig defaults to BREAKPOINT."""
        cfg = EpisodeConfig()
        assert cfg.injection_mode == InjectionMode.BREAKPOINT

    def test_injection_mode_configurable(self):
        """EpisodeConfig accepts custom injection mode."""
        cfg = EpisodeConfig(injection_mode=InjectionMode.SYNCHRONOUS)
        assert cfg.injection_mode == InjectionMode.SYNCHRONOUS


# ---------------------------------------------------------------------------
# TestMultiTurnQueueIntegration (issue #45)
# ---------------------------------------------------------------------------

_COMPLETE_ACTION = {
    "action_type": "task_complete",
    "response_delay_ms": 100,
    "confidence": 0.9,
}

_SECRET = "SECRET-FINDING-XYZ"


def _scripted_client(sim_actions: list[dict]) -> MagicMock:
    """Mock GeminiClient transport routing sim-user calls vs answerer calls.

    Calls from the real SimulatedUser are identified by their response_schema
    (it has an "action_type" property) and consume the scripted actions; all
    other calls are answerer generations, whose content reflects whether the
    injected tool context is visible in the prompt.
    """
    import json

    actions = iter(sim_actions)

    def generate(messages, **kwargs):
        schema = kwargs.get("response_schema")
        if schema is not None and "action_type" in schema.get("properties", {}):
            return GeminiResponse(
                content=json.dumps(next(actions)),
                input_tokens=5,
                output_tokens=5,
                duration_ms=10.0,
                finish_reason="STOP",
            )
        joined = " ".join(
            m["content"] if isinstance(m, dict) else m.content for m in messages
        )
        if _SECRET in joined:
            return _mock_gemini_response(content=f"Refined answer using {_SECRET}.")
        return _mock_gemini_response(content="Preliminary answer without tool context.")

    client = MagicMock(spec=GeminiClient)
    client.generate.side_effect = generate
    return client


def _deposit_tool() -> MagicMock:
    """Mock tool that deposits a recognizable context item into the queue."""
    from bicameral_agent.queue import Priority, QueueItem
    from bicameral_agent.tool_primitive import ToolMetadata, ToolResult

    mock_tool = MagicMock()
    mock_tool.execute.return_value = ToolResult(
        queue_deposit=QueueItem(
            content=f"{_SECRET}: key evidence found.",
            priority=Priority.HIGH,
            source_tool_id="research_gap_scanner",
            token_count=12,
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
    return mock_tool


class TestMultiTurnQueueIntegration:
    """Multi-turn wiring tests with the real SimulatedUser, queue, and loop.

    Only the LLM transport and the tools are mocked; EpisodeRunner,
    ConsciousLoop, ContextQueue, and SimulatedUser run for real. The
    sim-user LLM is scripted to say task_complete on every turn, so any
    extra turns come from the strictness completion floor (issue #45).
    """

    def _run(self, config=None, decide=None):
        client = _scripted_client([_COMPLETE_ACTION] * 8)
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        if decide is None:
            ctrl.decide.return_value = Action.DO_NOTHING
        else:
            ctrl.decide.side_effect = decide
        runner = EpisodeRunner(client, config or EpisodeConfig(max_turns=8))
        with patch("bicameral_agent.episode_runner.ResearchGapScanner") as MockScanner:
            MockScanner.return_value = _deposit_tool()
            episode = runner.run_episode(_make_task(), ctrl)
        return episode

    def test_completion_floor_makes_default_episode_multiturn(self):
        """Default (medium) strictness: turn-1 task_complete becomes a probe."""
        episode = self._run()
        assert episode.outcome.total_turns == 2
        followups = [
            e for e in episode.user_events if e.event_type == UserEventType.FOLLOW_UP
        ]
        assert len(followups) == 1
        assert any(
            e.event_type == UserEventType.TASK_COMPLETE for e in episode.user_events
        )

    def test_high_strictness_floor_gives_three_turns(self):
        episode = self._run(
            config=EpisodeConfig(max_turns=8, strictness=Strictness.HIGH)
        )
        assert episode.outcome.total_turns == 3

    def test_turn1_deposit_consumed_turn2_and_changes_scored_answer(self):
        recorded: list[FullState] = []

        def decide(state):
            recorded.append(state)
            return Action.SCANNER if state.turn_number == 1 else Action.DO_NOTHING

        episode = self._run(decide=decide)

        assert episode.outcome.total_turns == 2
        # The turn-1 deposit was drained and consumed at turn 2
        assert len(episode.context_injections) == 1
        inj = episode.context_injections[0]
        assert inj.consumed is True
        assert inj.consumed_at_turn == 2
        # Pre-drain snapshot: the turn-2 decision saw the pending item
        assert recorded[0].queue_depth == 0
        assert recorded[1].queue_depth == 1
        # The final (scored) answer reflects the turn-1 tool deposit
        final = [m for m in episode.messages if m.role == "assistant"][-1]
        assert _SECRET in final.content

    def test_without_tools_final_answer_lacks_injected_context(self):
        episode = self._run()
        assert len(episode.context_injections) == 0
        final = [m for m in episode.messages if m.role == "assistant"][-1]
        assert _SECRET not in final.content
        assert "Preliminary answer" in final.content

    def test_short_expiry_expires_deposit_before_consumption(self):
        def decide(state):
            return Action.SCANNER if state.turn_number == 1 else Action.DO_NOTHING

        episode = self._run(
            config=EpisodeConfig(max_turns=8, queue_expiry_turns=1),
            decide=decide,
        )

        assert episode.metadata["expired_queue_items"] == 1
        assert episode.context_injections[0].consumed is False
        final = [m for m in episode.messages if m.role == "assistant"][-1]
        assert _SECRET not in final.content

    def test_queue_metrics_nontrivial_with_real_controller(self):
        """avg_queue_depth and drain_count are non-zero in a multi-turn run."""
        from bicameral_agent.baseline_benchmark import extract_task_metrics

        client = _scripted_client([_COMPLETE_ACTION] * 8)
        ctrl = RandomController(action_probability=1.0, seed=7)
        runner = EpisodeRunner(client, EpisodeConfig(max_turns=8))
        with (
            patch("bicameral_agent.episode_runner.ResearchGapScanner") as MockScanner,
            patch("bicameral_agent.episode_runner.AssumptionAuditor") as MockAuditor,
            patch("bicameral_agent.episode_runner.ContextRefresher") as MockRefresher,
        ):
            MockScanner.return_value = _deposit_tool()
            MockAuditor.return_value = _deposit_tool()
            MockRefresher.return_value = _deposit_tool()
            episode = runner.run_episode(_make_task(), ctrl)

        metrics = extract_task_metrics(episode, ctrl.decisions)
        assert metrics.total_turns >= 2
        assert metrics.drain_count >= 1
        assert metrics.avg_queue_depth > 0
        assert metrics.task_completed == 1


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


# ---------------------------------------------------------------------------
# TestCostBudgetHandling (issue #47)
# ---------------------------------------------------------------------------


class TestPerRoleClients:
    """Issue #53: judge and sim-user clients are selectable per role."""

    _PRICED_MODEL = "gemini-3.1-flash-lite-preview"

    def _run_episode(self, runner: EpisodeRunner):
        """Run a one-turn scored episode; return the SimulatedUser and
        TaskScorer mock classes so tests can inspect construction kwargs."""
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.DO_NOTHING

        with patch(
            "bicameral_agent.episode_runner.SimulatedUser"
        ) as MockSimUser, patch(
            "bicameral_agent.verifiers.TaskScorer"
        ) as MockScorer:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = mock_sim
            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(overall=0.5, detail=None)
            MockScorer.return_value = mock_scorer

            runner.run_episode(_make_task(), ctrl)

        return MockSimUser, MockScorer

    def test_roles_default_to_main_client(self):
        """Back-compat: without per-role clients, everyone gets the main one."""
        client = _make_mock_client()
        runner = EpisodeRunner(
            client, EpisodeConfig(max_turns=1, score_episode=True)
        )
        MockSimUser, MockScorer = self._run_episode(runner)
        assert MockSimUser.call_args.kwargs["client"] is client
        assert MockScorer.call_args.kwargs["client"] is client

    def test_judge_and_sim_user_clients_override(self):
        """Per-role clients reach the scorer and sim-user, not the answerer's."""
        client = _make_mock_client()
        judge = _make_mock_client()
        sim = _make_mock_client()
        runner = EpisodeRunner(
            client,
            EpisodeConfig(max_turns=1, score_episode=True),
            judge_client=judge,
            sim_user_client=sim,
        )
        MockSimUser, MockScorer = self._run_episode(runner)
        assert MockSimUser.call_args.kwargs["client"] is sim
        assert MockScorer.call_args.kwargs["client"] is judge

    def test_measurement_clients_cost_tracked(self):
        """With a cost tracker, judge and sim-user clients are wrapped too."""
        client = _make_mock_client()
        client.model = self._PRICED_MODEL
        judge = _make_mock_client()
        judge.model = self._PRICED_MODEL
        sim = _make_mock_client()
        sim.model = self._PRICED_MODEL
        runner = EpisodeRunner(
            client,
            EpisodeConfig(max_turns=1, score_episode=True),
            cost_tracker=CostTracker(),
            judge_client=judge,
            sim_user_client=sim,
        )
        MockSimUser, MockScorer = self._run_episode(runner)
        judge_arg = MockScorer.call_args.kwargs["client"]
        sim_arg = MockSimUser.call_args.kwargs["client"]
        assert isinstance(judge_arg, CostTrackedClient)
        assert isinstance(sim_arg, CostTrackedClient)
        assert judge_arg._inner is judge
        assert sim_arg._inner is sim

    def test_default_judge_cost_tracked(self):
        """Judge calls no longer bypass cost tracking (issue #53)."""
        client = _make_mock_client()
        client.model = self._PRICED_MODEL
        runner = EpisodeRunner(
            client,
            EpisodeConfig(max_turns=1, score_episode=True),
            cost_tracker=CostTracker(),
        )
        _, MockScorer = self._run_episode(runner)
        judge_arg = MockScorer.call_args.kwargs["client"]
        assert isinstance(judge_arg, CostTrackedClient)
        assert judge_arg._inner is client


class TestScoringWiring:
    """Issue #56: EpisodeConfig.metric selects the verifier via build_verifier."""

    def _run_scored(self, metric: str, answer: str):
        client = _make_mock_client()
        client.generate.return_value = _mock_gemini_response(content=answer)
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.DO_NOTHING
        runner = EpisodeRunner(
            client, EpisodeConfig(max_turns=1, score_episode=True, metric=metric)
        )
        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.TASK_COMPLETE,
                response_delay_ms=100,
                confidence=0.9,
            )
            MockSimUser.return_value = mock_sim
            return runner.run_episode(_make_task(), ctrl)

    def test_lexical_metric_scores_without_llm_judge(self):
        gold = _make_task().gold_answer
        episode = self._run_scored("lexical", gold)
        assert episode.outcome.quality_score == 1.0
        verification = episode.metadata["verification"]
        assert verification["metric"] == "lexical"
        assert verification["detail"] is None

    def test_deterministic_metric_detail_lands_in_metadata(self):
        episode = self._run_scored("exact_match", "Something unrelated")
        assert episode.outcome.quality_score == 0.0
        assert "exact_match" in episode.metadata["verification"]["detail"]

    def test_unknown_metric_fails_loudly(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            self._run_scored("not_a_metric", "x")


class TestCostBudgetHandling:
    def test_budget_trip_in_tool_ends_episode_gracefully(self):
        """CostBudgetExceeded from a tool call ends the episode, not the run."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.SCANNER

        runner = EpisodeRunner(client, EpisodeConfig(max_turns=5))

        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.return_value = UserAction(
                action_type=ActionType.FOLLOW_UP,
                message="More?",
                followup_type=FollowUpType.ELABORATION,
                response_delay_ms=100,
                confidence=0.8,
            )
            MockSimUser.return_value = mock_sim

            with patch(
                "bicameral_agent.episode_runner.ResearchGapScanner"
            ) as MockScanner:
                mock_tool = MagicMock()
                mock_tool.execute.side_effect = CostBudgetExceeded("episode budget")
                MockScanner.return_value = mock_tool

                episode = runner.run_episode(_make_task(), ctrl)

        assert isinstance(episode, Episode)
        # Episode ended on the first turn (no sim-user follow-ups consumed)
        assert episode.outcome.total_turns == 1
        assert len(episode.tool_invocations) == 1
        assert episode.tool_invocations[0].budget_exceeded is True
        # Sim user never reached: episode ended before the respond() call
        mock_sim.respond.assert_not_called()

    def test_budget_trip_in_sim_user_ends_episode_gracefully(self):
        """CostBudgetExceeded from the sim-user call ends the episode, not the run."""
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.DO_NOTHING

        runner = EpisodeRunner(client, EpisodeConfig(max_turns=5))

        with patch("bicameral_agent.episode_runner.SimulatedUser") as MockSimUser:
            mock_sim = MagicMock()
            mock_sim.respond.side_effect = CostBudgetExceeded("episode budget")
            MockSimUser.return_value = mock_sim

            episode = runner.run_episode(_make_task(), ctrl)

        assert isinstance(episode, Episode)
        assert episode.outcome.total_turns == 1
        assert mock_sim.respond.call_count == 1


# ---------------------------------------------------------------------------
# TestParseDegradationMetadata
# ---------------------------------------------------------------------------


class TestParseDegradationMetadata:
    """Per-episode structured-output degradation counts (issue #82)."""

    @staticmethod
    def _run(client) -> Episode:
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.DO_NOTHING
        runner = EpisodeRunner(client, EpisodeConfig(max_turns=1))
        return runner.run_episode(_make_task(), ctrl)

    def test_clean_episode_reports_empty_counts(self):
        client = _scripted_client(
            [{"action_type": "task_complete", "response_delay_ms": 100, "confidence": 0.9}]
        )
        episode = self._run(client)
        assert episode.metadata["parse_degradations"] == {}

    def test_sim_user_degradation_counted(self):
        # _make_mock_client returns non-JSON prose for every call, so the
        # real SimulatedUser's safe-parse degrades on its one call.
        episode = self._run(_make_mock_client())
        assert episode.metadata["parse_degradations"] == {"SimulatedUser.respond": 1}

    def test_counts_do_not_cross_episodes(self):
        client = _make_mock_client()
        self._run(client)
        second = self._run(client)
        assert second.metadata["parse_degradations"] == {"SimulatedUser.respond": 1}

    def test_scorer_degradation_counted(self):
        client = _make_mock_client()
        ctrl = MagicMock(spec=Controller)
        ctrl.decisions = []
        ctrl.decide.return_value = Action.DO_NOTHING
        runner = EpisodeRunner(
            client, EpisodeConfig(max_turns=1, score_episode=True, metric="llm_judge")
        )
        episode = runner.run_episode(_make_task(), ctrl)
        counts = episode.metadata["parse_degradations"]
        assert counts["TaskScorer"] == 1
        assert counts["SimulatedUser.respond"] == 1
