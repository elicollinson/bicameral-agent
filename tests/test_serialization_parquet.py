"""Tests for Parquet round-trip serialization."""

import os

import pyarrow.parquet as pq

from bicameral_agent.schema import Episode, EpisodeOutcome
from bicameral_agent.serialization import episodes_from_parquet, episodes_to_parquet


class TestParquetRoundTrip:
    def test_single_episode(self, make_episode, tmp_path):
        ep = make_episode()
        path = str(tmp_path / "episode.parquet")
        ep.to_parquet(path)
        restored = Episode.from_parquet(path)
        assert ep.model_dump() == restored.model_dump()

    def test_five_episodes(self, five_episodes, tmp_path):
        path = str(tmp_path / "episodes.parquet")
        episodes_to_parquet(five_episodes, path)
        restored = episodes_from_parquet(path)
        assert len(restored) == 5
        for orig, rest in zip(five_episodes, restored):
            assert orig.model_dump() == rest.model_dump()

    def test_parquet_file_is_valid(self, make_episode, tmp_path):
        ep = make_episode()
        path = str(tmp_path / "episode.parquet")
        ep.to_parquet(path)
        assert os.path.exists(path)
        table = pq.read_table(path)
        assert table.num_rows == 1
        assert "episode_id" in table.column_names
        assert "payload" in table.column_names

    def test_preserves_all_fields(self, make_episode, tmp_path):
        ep = make_episode(
            num_messages=4,
            metadata={"nested": {"key": [1, 2, 3]}},
        )
        path = str(tmp_path / "episode.parquet")
        ep.to_parquet(path)
        restored = Episode.from_parquet(path)
        assert restored.episode_id == ep.episode_id
        assert len(restored.messages) == 4
        assert len(restored.user_events) == 1
        assert len(restored.context_injections) == 1
        assert len(restored.tool_invocations) == 1
        assert restored.outcome.model_dump() == ep.outcome.model_dump()
        assert restored.metadata == {"nested": {"key": [1, 2, 3]}}

    def test_empty_episode(self, tmp_path):
        ep = Episode(
            outcome=EpisodeOutcome(total_tokens=0, total_turns=0, wall_clock_ms=0)
        )
        path = str(tmp_path / "empty.parquet")
        ep.to_parquet(path)
        restored = Episode.from_parquet(path)
        assert ep.model_dump() == restored.model_dump()
