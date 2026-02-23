"""Tests for EpisodeValidator semantic validation."""

from bicameral_agent.schema import (
    ContextInjection,
    Episode,
    EpisodeOutcome,
    Message,
    ToolInvocation,
    UserEvent,
    UserEventType,
)
from bicameral_agent.validation import EpisodeValidator


class TestEpisodeValidator:
    def setup_method(self):
        self.validator = EpisodeValidator()

    def test_valid_episode_passes(self, make_episode):
        ep = make_episode()
        result = self.validator.validate(ep)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_empty_episode_passes(self):
        ep = Episode(
            outcome=EpisodeOutcome(total_tokens=0, total_turns=0, wall_clock_ms=0)
        )
        result = self.validator.validate(ep)
        assert result.is_valid is True
        assert result.errors == []

    def test_messages_out_of_order(self, make_episode):
        ep = make_episode(
            messages=[
                Message(role="user", content="first", timestamp_ms=2000, token_count=5),
                Message(role="assistant", content="second", timestamp_ms=1000, token_count=5),
            ]
        )
        result = self.validator.validate(ep)
        assert result.is_valid is False
        assert any("Messages out of order" in e for e in result.errors)

    def test_messages_equal_timestamps_accepted(self, make_episode):
        ep = make_episode(
            messages=[
                Message(role="user", content="first", timestamp_ms=1000, token_count=5),
                Message(role="assistant", content="second", timestamp_ms=1000, token_count=5),
            ]
        )
        result = self.validator.validate(ep)
        assert result.is_valid is True

    def test_user_events_out_of_order(self, make_episode):
        ep = make_episode(
            user_events=[
                UserEvent(event_type=UserEventType.EDIT, timestamp_ms=2000),
                UserEvent(event_type=UserEventType.STOP, timestamp_ms=1000),
            ]
        )
        result = self.validator.validate(ep)
        assert result.is_valid is False
        assert any("User events out of order" in e for e in result.errors)

    def test_context_injections_out_of_order(self, make_episode):
        ep = make_episode(
            context_injections=[
                ContextInjection(
                    content="a", source_tool_id="t", priority=0,
                    timestamp_ms=2000, token_count=5,
                ),
                ContextInjection(
                    content="b", source_tool_id="t", priority=0,
                    timestamp_ms=1000, token_count=5,
                ),
            ]
        )
        result = self.validator.validate(ep)
        assert result.is_valid is False
        assert any("Context injections out of order" in e for e in result.errors)

    def test_tool_invocations_out_of_order(self, make_episode):
        ep = make_episode(
            tool_invocations=[
                ToolInvocation(
                    tool_id="t1", invoked_at_ms=2000, completed_at_ms=2100,
                    input_tokens=5, output_tokens=5,
                ),
                ToolInvocation(
                    tool_id="t2", invoked_at_ms=1000, completed_at_ms=1100,
                    input_tokens=5, output_tokens=5,
                ),
            ]
        )
        result = self.validator.validate(ep)
        assert result.is_valid is False
        assert any("Tool invocations out of order" in e for e in result.errors)

    def test_outcome_zero_turns_with_messages_warns(self, make_episode):
        ep = make_episode(
            num_messages=3,
            outcome=EpisodeOutcome(total_tokens=50, total_turns=0, wall_clock_ms=1000),
        )
        result = self.validator.validate(ep)
        assert result.is_valid is True  # warnings don't affect validity
        assert any("total_turns is 0" in w for w in result.warnings)

    def test_multiple_errors_collected(self):
        ep = Episode(
            messages=[
                Message(role="user", content="a", timestamp_ms=2000, token_count=5),
                Message(role="assistant", content="b", timestamp_ms=1000, token_count=5),
            ],
            user_events=[
                UserEvent(event_type=UserEventType.EDIT, timestamp_ms=2000),
                UserEvent(event_type=UserEventType.STOP, timestamp_ms=1000),
            ],
            outcome=EpisodeOutcome(total_tokens=10, total_turns=2, wall_clock_ms=500),
        )
        result = self.validator.validate(ep)
        assert result.is_valid is False
        assert len(result.errors) >= 2
