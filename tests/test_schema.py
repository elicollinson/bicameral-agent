"""Tests for Pydantic schema model construction and field constraints."""

import uuid

import pytest
from pydantic import ValidationError

from bicameral_agent.schema import (
    ContextInjection,
    Episode,
    EpisodeOutcome,
    Message,
    ToolInvocation,
    UserEvent,
    UserEventType,
)


class TestMessage:
    def test_valid_construction(self):
        msg = Message(role="user", content="hi", timestamp_ms=1000, token_count=5)
        assert msg.role == "user"
        assert msg.content == "hi"
        assert msg.timestamp_ms == 1000
        assert msg.token_count == 5

    def test_negative_token_count_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="user", content="hi", timestamp_ms=1000, token_count=-1)

    def test_negative_timestamp_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="user", content="hi", timestamp_ms=-1, token_count=5)

    def test_zero_values_accepted(self):
        msg = Message(role="user", content="", timestamp_ms=0, token_count=0)
        assert msg.timestamp_ms == 0
        assert msg.token_count == 0


class TestUserEvent:
    def test_valid_construction(self):
        event = UserEvent(event_type=UserEventType.STOP, timestamp_ms=500)
        assert event.event_type == UserEventType.STOP
        assert event.metadata == {}

    def test_invalid_event_type_rejected(self):
        with pytest.raises(ValidationError):
            UserEvent(event_type="invalid", timestamp_ms=500)

    def test_all_event_types(self):
        for et in UserEventType:
            event = UserEvent(event_type=et, timestamp_ms=100)
            assert event.event_type == et

    def test_metadata_default_empty(self):
        event = UserEvent(event_type=UserEventType.EDIT, timestamp_ms=100)
        assert event.metadata == {}

    def test_metadata_with_data(self):
        event = UserEvent(
            event_type=UserEventType.EDIT, timestamp_ms=100, metadata={"key": "value"}
        )
        assert event.metadata == {"key": "value"}


class TestContextInjection:
    def test_valid_construction(self):
        ci = ContextInjection(
            content="context",
            source_tool_id="tool-1",
            priority=1,
            timestamp_ms=1000,
            token_count=10,
        )
        assert ci.consumed is False
        assert ci.consumed_at_turn is None

    def test_negative_token_count_rejected(self):
        with pytest.raises(ValidationError):
            ContextInjection(
                content="ctx",
                source_tool_id="t",
                priority=0,
                timestamp_ms=0,
                token_count=-1,
            )

    def test_consumed_at_turn_accepts_none(self):
        ci = ContextInjection(
            content="ctx",
            source_tool_id="t",
            priority=0,
            timestamp_ms=0,
            token_count=0,
            consumed_at_turn=None,
        )
        assert ci.consumed_at_turn is None

    def test_consumed_at_turn_accepts_zero(self):
        ci = ContextInjection(
            content="ctx",
            source_tool_id="t",
            priority=0,
            timestamp_ms=0,
            token_count=0,
            consumed_at_turn=0,
        )
        assert ci.consumed_at_turn == 0


class TestToolInvocation:
    def test_valid_construction(self):
        ti = ToolInvocation(
            tool_id="tool-1",
            invoked_at_ms=100,
            completed_at_ms=200,
            input_tokens=10,
            output_tokens=20,
        )
        assert ti.result_deposited is False

    def test_completed_before_invoked_rejected(self):
        with pytest.raises(ValidationError):
            ToolInvocation(
                tool_id="tool-1",
                invoked_at_ms=200,
                completed_at_ms=100,
                input_tokens=10,
                output_tokens=20,
            )

    def test_same_time_accepted(self):
        ti = ToolInvocation(
            tool_id="tool-1",
            invoked_at_ms=100,
            completed_at_ms=100,
            input_tokens=0,
            output_tokens=0,
        )
        assert ti.completed_at_ms == ti.invoked_at_ms

    def test_negative_input_tokens_rejected(self):
        with pytest.raises(ValidationError):
            ToolInvocation(
                tool_id="tool-1",
                invoked_at_ms=100,
                completed_at_ms=200,
                input_tokens=-1,
                output_tokens=0,
            )


class TestEpisodeOutcome:
    def test_valid_construction(self):
        outcome = EpisodeOutcome(total_tokens=100, total_turns=5, wall_clock_ms=3000)
        assert outcome.quality_score is None

    def test_quality_score_in_range(self):
        outcome = EpisodeOutcome(
            quality_score=0.5, total_tokens=100, total_turns=5, wall_clock_ms=3000
        )
        assert outcome.quality_score == 0.5

    def test_quality_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            EpisodeOutcome(
                quality_score=1.5, total_tokens=100, total_turns=5, wall_clock_ms=3000
            )

    def test_quality_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            EpisodeOutcome(
                quality_score=-0.1, total_tokens=100, total_turns=5, wall_clock_ms=3000
            )

    def test_quality_score_boundaries(self):
        EpisodeOutcome(quality_score=0.0, total_tokens=0, total_turns=0, wall_clock_ms=0)
        EpisodeOutcome(quality_score=1.0, total_tokens=0, total_turns=0, wall_clock_ms=0)


class TestEpisode:
    def test_default_uuid(self):
        ep = Episode(outcome=EpisodeOutcome(total_tokens=0, total_turns=0, wall_clock_ms=0))
        uuid.UUID(ep.episode_id)  # validates it's a valid UUID

    def test_missing_outcome_rejected(self):
        with pytest.raises(ValidationError):
            Episode()

    def test_empty_lists_accepted(self):
        ep = Episode(outcome=EpisodeOutcome(total_tokens=0, total_turns=0, wall_clock_ms=0))
        assert ep.messages == []
        assert ep.user_events == []
        assert ep.context_injections == []
        assert ep.tool_invocations == []
        assert ep.metadata == {}

    def test_full_construction(self, make_episode):
        ep = make_episode(num_messages=3)
        assert len(ep.messages) == 3
        assert len(ep.user_events) == 1
        assert len(ep.context_injections) == 1
        assert len(ep.tool_invocations) == 1
        assert ep.outcome.total_tokens == 100
        assert ep.metadata == {"source": "test"}
