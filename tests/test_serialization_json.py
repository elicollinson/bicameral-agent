"""Tests for JSON round-trip serialization."""

from bicameral_agent.schema import Episode, EpisodeOutcome, UserEventType


class TestJsonRoundTrip:
    def test_single_episode(self, make_episode):
        ep = make_episode()
        json_str = ep.to_json()
        restored = Episode.from_json(json_str)
        assert ep.model_dump() == restored.model_dump()

    def test_five_episodes(self, five_episodes):
        for ep in five_episodes:
            json_str = ep.to_json()
            restored = Episode.from_json(json_str)
            assert ep.model_dump() == restored.model_dump()

    def test_preserves_enum_values(self, make_episode):
        ep = make_episode()
        json_str = ep.to_json()
        restored = Episode.from_json(json_str)
        assert restored.user_events[0].event_type == UserEventType.FOLLOW_UP
        assert isinstance(restored.user_events[0].event_type, UserEventType)

    def test_preserves_optional_none(self):
        ep = Episode(
            outcome=EpisodeOutcome(
                quality_score=None, total_tokens=50, total_turns=2, wall_clock_ms=1000
            )
        )
        json_str = ep.to_json()
        restored = Episode.from_json(json_str)
        assert restored.outcome.quality_score is None

    def test_preserves_optional_value(self):
        ep = Episode(
            outcome=EpisodeOutcome(
                quality_score=0.75, total_tokens=50, total_turns=2, wall_clock_ms=1000
            )
        )
        json_str = ep.to_json()
        restored = Episode.from_json(json_str)
        assert restored.outcome.quality_score == 0.75

    def test_preserves_metadata_dict(self, make_episode):
        ep = make_episode(
            metadata={
                "controller": "bicameral",
                "model": "gpt-4",
                "hyperparams": {"temperature": 0.7, "max_tokens": 1024},
                "tags": ["test", "v1"],
            }
        )
        json_str = ep.to_json()
        restored = Episode.from_json(json_str)
        assert restored.metadata == ep.metadata

    def test_json_is_valid_string(self, make_episode):
        ep = make_episode()
        json_str = ep.to_json()
        assert isinstance(json_str, str)
        assert '"episode_id"' in json_str
