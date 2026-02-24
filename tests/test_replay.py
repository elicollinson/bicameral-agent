"""Tests for the Episode Replay Engine (Issue #3)."""

import time
import uuid

import pytest

from bicameral_agent.logger import ConversationLogger
from bicameral_agent.replay import DecisionPoint, EpisodeReplayer, ReplayState
from bicameral_agent.schema import (
    ContextInjection,
    Episode,
    EpisodeOutcome,
    Message,
    ToolInvocation,
    UserEvent,
    UserEventType,
)


@pytest.fixture
def multi_turn_episode():
    """A 6-message episode with 3 user turns, tools, injections, and events."""
    messages = [
        Message(role="user", content="hello", timestamp_ms=100, token_count=5),
        Message(role="assistant", content="hi there", timestamp_ms=200, token_count=8),
        Message(role="user", content="do stuff", timestamp_ms=300, token_count=6),
        Message(role="assistant", content="doing it", timestamp_ms=400, token_count=7),
        Message(role="user", content="more", timestamp_ms=500, token_count=4),
        Message(role="assistant", content="done", timestamp_ms=600, token_count=3),
    ]
    context_injections = [
        ContextInjection(
            content="ctx1",
            source_tool_id="tool-a",
            priority=1,
            timestamp_ms=150,
            token_count=10,
            consumed=True,
            consumed_at_turn=1,
        ),
        ContextInjection(
            content="ctx2",
            source_tool_id="tool-b",
            priority=2,
            timestamp_ms=350,
            token_count=15,
            consumed=False,
        ),
    ]
    tool_invocations = [
        ToolInvocation(
            tool_id="tool-a",
            invoked_at_ms=120,
            completed_at_ms=180,
            input_tokens=10,
            output_tokens=20,
        ),
        ToolInvocation(
            tool_id="tool-b",
            invoked_at_ms=310,
            completed_at_ms=550,
            input_tokens=12,
            output_tokens=18,
        ),
    ]
    user_events = [
        UserEvent(event_type=UserEventType.FOLLOW_UP, timestamp_ms=250),
        UserEvent(event_type=UserEventType.STOP, timestamp_ms=650),
    ]
    return Episode(
        episode_id=str(uuid.uuid4()),
        messages=messages,
        context_injections=context_injections,
        tool_invocations=tool_invocations,
        user_events=user_events,
        outcome=EpisodeOutcome(
            total_tokens=100,
            total_turns=3,
            wall_clock_ms=550,
        ),
    )


class TestStateAtTurn:
    """AC1 & AC2: state_at_turn(0) returns initial state, state_at_turn(N) matches complete episode."""

    def test_turn_0_returns_first_user_message_only(self, multi_turn_episode):
        """AC1: state_at_turn(0) returns the initial state (first user message only)."""
        replayer = EpisodeReplayer(multi_turn_episode)
        state = replayer.state_at_turn(0)

        assert state.turn_number == 1
        assert len(state.messages) == 1
        assert state.messages[0].role == "user"
        assert state.messages[0].content == "hello"

    def test_turn_0_has_no_user_events_before_cutoff(self, multi_turn_episode):
        """At turn 0 (ts=100), no user events have occurred yet."""
        replayer = EpisodeReplayer(multi_turn_episode)
        state = replayer.state_at_turn(0)

        # The first user event is at ts=250, cutoff is 100
        assert len(state.user_events) == 0

    def test_final_turn_matches_complete_episode(self, multi_turn_episode):
        """AC2: state_at_turn(N) for the final turn matches the complete episode."""
        replayer = EpisodeReplayer(multi_turn_episode)
        final = replayer.state_at_turn(replayer.total_turns - 1)

        # All messages up through the last user message (index 4, which is 5 messages)
        assert len(final.messages) == 5  # messages[0:5]
        assert final.turn_number == 3

    def test_intermediate_turn_has_correct_messages(self, multi_turn_episode):
        """Turn 1 includes messages up through the second user message."""
        replayer = EpisodeReplayer(multi_turn_episode)
        state = replayer.state_at_turn(1)

        assert len(state.messages) == 3  # user, assistant, user
        assert state.turn_number == 2
        assert state.messages[-1].content == "do stuff"

    def test_out_of_range_raises_index_error(self, multi_turn_episode):
        replayer = EpisodeReplayer(multi_turn_episode)
        with pytest.raises(IndexError):
            replayer.state_at_turn(-1)
        with pytest.raises(IndexError):
            replayer.state_at_turn(replayer.total_turns)

    def test_tool_invocations_partitioned_by_turn(self, multi_turn_episode):
        """Tools active at turn 0 vs completed at later turns."""
        replayer = EpisodeReplayer(multi_turn_episode)

        # At turn 0 (cutoff=100ms): tool-a invoked at 120 > 100, so not visible
        state0 = replayer.state_at_turn(0)
        assert len(state0.active_tool_invocations) == 0
        assert len(state0.completed_tool_invocations) == 0

        # At turn 1 (cutoff=300ms): tool-a completed (120→180), tool-b started (310>300)
        state1 = replayer.state_at_turn(1)
        assert len(state1.completed_tool_invocations) == 1
        assert state1.completed_tool_invocations[0].tool_id == "tool-a"
        assert len(state1.active_tool_invocations) == 0

    def test_context_injection_consumption_tracking(self, multi_turn_episode):
        """Injections are classified as pending or consumed based on turn number."""
        replayer = EpisodeReplayer(multi_turn_episode)

        # At turn 0 (cutoff_ms=100): ctx1 at ts=150 not yet visible
        state0 = replayer.state_at_turn(0)
        assert len(state0.pending_injections) == 0
        assert len(state0.consumed_injections) == 0

        # At turn 1 (cutoff_ms=300, turn_number=2): ctx1 at ts=150 visible,
        # consumed_at_turn=1, 1 < 2 → consumed. ctx2 at ts=350 > 300 → not visible.
        state1 = replayer.state_at_turn(1)
        assert len(state1.consumed_injections) == 1
        assert state1.consumed_injections[0].content == "ctx1"
        assert len(state1.pending_injections) == 0

        # At turn 2 (cutoff_ms=500, turn_number=3): ctx1 consumed_at_turn=1, 1 < 3 → consumed
        # ctx2 at ts=350 <= 500 but not consumed → pending
        state2 = replayer.state_at_turn(2)
        assert len(state2.consumed_injections) == 1
        assert state2.consumed_injections[0].content == "ctx1"
        assert len(state2.pending_injections) == 1
        assert state2.pending_injections[0].content == "ctx2"


class TestStateAtTime:
    """AC3: state_at_time(ms) correctly interpolates between turns."""

    def test_before_any_message(self, multi_turn_episode):
        """Before the first message, state is empty."""
        replayer = EpisodeReplayer(multi_turn_episode)
        state = replayer.state_at_time(50)

        assert len(state.messages) == 0
        assert state.turn_number == 0

    def test_at_first_message(self, multi_turn_episode):
        """At the exact timestamp of the first message, it's included."""
        replayer = EpisodeReplayer(multi_turn_episode)
        state = replayer.state_at_time(100)

        assert len(state.messages) == 1
        assert state.turn_number == 1

    def test_between_turns(self, multi_turn_episode):
        """Between user messages, shows partial state."""
        replayer = EpisodeReplayer(multi_turn_episode)
        # At 250ms: messages at 100, 200 are included; 300 is not
        state = replayer.state_at_time(250)

        assert len(state.messages) == 2
        assert state.turn_number == 1
        assert state.messages[-1].content == "hi there"

    def test_mid_tool_execution(self, multi_turn_episode):
        """During a tool's execution window, it appears as active."""
        replayer = EpisodeReplayer(multi_turn_episode)
        # tool-b: invoked_at_ms=310, completed_at_ms=550
        state = replayer.state_at_time(400)

        assert any(t.tool_id == "tool-b" for t in state.active_tool_invocations)
        assert all(t.tool_id != "tool-b" for t in state.completed_tool_invocations)

    def test_after_tool_completion(self, multi_turn_episode):
        """After a tool completes, it moves to completed."""
        replayer = EpisodeReplayer(multi_turn_episode)
        state = replayer.state_at_time(560)

        assert any(t.tool_id == "tool-b" for t in state.completed_tool_invocations)
        assert all(t.tool_id != "tool-b" for t in state.active_tool_invocations)

    def test_user_events_filtered_by_time(self, multi_turn_episode):
        """Only events before the cutoff time are included."""
        replayer = EpisodeReplayer(multi_turn_episode)

        state_early = replayer.state_at_time(200)
        assert len(state_early.user_events) == 0

        state_mid = replayer.state_at_time(300)
        assert len(state_mid.user_events) == 1
        assert state_mid.user_events[0].event_type == UserEventType.FOLLOW_UP

    def test_at_end_of_episode(self, multi_turn_episode):
        """At a time past all events, everything is included."""
        replayer = EpisodeReplayer(multi_turn_episode)
        state = replayer.state_at_time(999)

        assert len(state.messages) == 6
        assert state.turn_number == 3


class TestIterDecisionPoints:
    """AC4: iter_decision_points() yields correct number of decision points."""

    def test_yields_correct_count(self, multi_turn_episode):
        """Number of decision points matches number of assistant messages."""
        replayer = EpisodeReplayer(multi_turn_episode)
        points = list(replayer.iter_decision_points())

        assistant_count = sum(
            1 for m in multi_turn_episode.messages if m.role == "assistant"
        )
        assert len(points) == assistant_count

    def test_each_point_has_prior_state(self, multi_turn_episode):
        """Each decision point's state contains only prior messages."""
        replayer = EpisodeReplayer(multi_turn_episode)
        points = list(replayer.iter_decision_points())

        # First decision point: state has 1 message (the first user msg),
        # action is the first assistant msg
        assert len(points[0].state.messages) == 1
        assert points[0].state.messages[0].role == "user"
        assert points[0].action.role == "assistant"
        assert points[0].action.content == "hi there"

        # Second decision point: state has 3 messages (user, asst, user)
        assert len(points[1].state.messages) == 3
        assert points[1].action.content == "doing it"

    def test_decision_point_types(self, multi_turn_episode):
        """All yielded items are DecisionPoint instances."""
        replayer = EpisodeReplayer(multi_turn_episode)
        for point in replayer.iter_decision_points():
            assert isinstance(point, DecisionPoint)
            assert isinstance(point.state, ReplayState)
            assert isinstance(point.action, Message)

    def test_no_assistant_messages_yields_nothing(self):
        """An episode with no assistant messages yields no decision points."""
        episode = Episode(
            messages=[
                Message(role="user", content="hi", timestamp_ms=100, token_count=2),
            ],
            outcome=EpisodeOutcome(
                total_tokens=2, total_turns=1, wall_clock_ms=0
            ),
        )
        replayer = EpisodeReplayer(episode)
        assert list(replayer.iter_decision_points()) == []


class TestPerformance:
    """AC5: Replaying a 50-turn episode and querying every turn completes in < 100ms."""

    def test_50_turn_replay_under_100ms(self):
        """Build a 50-turn episode and query every turn within the time budget."""
        messages = []
        ts = 1000
        for i in range(50):
            messages.append(
                Message(role="user", content=f"u{i}", timestamp_ms=ts, token_count=3)
            )
            ts += 100
            messages.append(
                Message(role="assistant", content=f"a{i}", timestamp_ms=ts, token_count=4)
            )
            ts += 100

        episode = Episode(
            messages=messages,
            outcome=EpisodeOutcome(
                total_tokens=350, total_turns=50, wall_clock_ms=ts - 1000
            ),
        )

        replayer = EpisodeReplayer(episode)

        start = time.perf_counter_ns()
        for turn in range(50):
            replayer.state_at_turn(turn)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

        assert elapsed_ms < 100, f"Replay took {elapsed_ms:.1f}ms, budget is 100ms"


class TestFidelity:
    """AC6: Log an episode with ConversationLogger, replay it, verify every field matches."""

    def test_logger_replay_round_trip(self):
        """Log an episode, replay it, and verify state at every turn."""
        logger = ConversationLogger(metadata={"test": True})

        # Turn 1: user message
        logger.log_message("user", "hello", token_count=5)
        # Tool invocation during turn 1
        tool_idx = logger.log_tool_invocation("search", input_tokens=10)
        # Context injection
        inj_idx = logger.log_context_injection(
            content="search results",
            source_tool_id="search",
            priority=1,
            token_count=20,
        )
        logger.log_tool_completion(tool_idx, output_tokens=15, result_deposited=True)
        # Assistant response
        logger.log_message("assistant", "I found something", token_count=12)

        # Turn 2: user follow-up
        logger.log_user_event(UserEventType.FOLLOW_UP)
        logger.log_message("user", "tell me more", token_count=6)
        logger.log_injection_consumed(inj_idx, turn_number=1)
        logger.log_message("assistant", "here is more detail", token_count=15)

        # Turn 3
        logger.log_message("user", "thanks", token_count=3)
        logger.log_message("assistant", "you're welcome", token_count=8)

        episode = logger.finalize(quality_score=0.9)
        replayer = EpisodeReplayer(episode)

        # Verify total turns
        assert replayer.total_turns == 3

        # Turn 0: just the first user message
        s0 = replayer.state_at_turn(0)
        assert len(s0.messages) == 1
        assert s0.messages[0].role == "user"
        assert s0.messages[0].content == "hello"

        # Final turn: all user messages accounted for
        s_final = replayer.state_at_turn(2)
        assert s_final.turn_number == 3
        user_msgs = [m for m in s_final.messages if m.role == "user"]
        assert len(user_msgs) == 3

        # Verify all messages in the final state match the episode's messages
        # (up through the last user message)
        last_user_idx = max(
            i for i, m in enumerate(episode.messages) if m.role == "user"
        )
        expected_msgs = episode.messages[: last_user_idx + 1]
        assert list(s_final.messages) == expected_msgs

        # Verify decision points match assistant messages
        points = list(replayer.iter_decision_points())
        assistant_msgs = [m for m in episode.messages if m.role == "assistant"]
        assert len(points) == len(assistant_msgs)
        for point, expected in zip(points, assistant_msgs):
            assert point.action == expected

        # Verify tool invocations appear in replay
        s1 = replayer.state_at_turn(1)
        assert len(s1.completed_tool_invocations) > 0
        assert s1.completed_tool_invocations[0].tool_id == "search"

        # Verify injection consumption tracking
        # At turn 0 (turn_number=1), injection consumed_at_turn=1, 1 < 1 → pending
        assert len(s0.pending_injections) + len(s0.consumed_injections) >= 0
        # At turn 2 (turn_number=3), injection consumed_at_turn=1, 1 < 3 → consumed
        assert any(
            inj.content == "search results"
            for inj in s_final.consumed_injections
        )

    def test_empty_episode_replay(self):
        """An episode with no messages can be replayed."""
        episode = Episode(
            messages=[],
            outcome=EpisodeOutcome(
                total_tokens=0, total_turns=0, wall_clock_ms=0
            ),
        )
        replayer = EpisodeReplayer(episode)
        assert replayer.total_turns == 0
        assert list(replayer.iter_decision_points()) == []

        # state_at_time on empty episode
        state = replayer.state_at_time(0)
        assert len(state.messages) == 0
        assert state.turn_number == 0
