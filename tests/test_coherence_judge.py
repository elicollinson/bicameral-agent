"""Tests for the CoherenceJudge."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from bicameral_agent.coherence_judge import CoherenceJudge, CoherenceScore
from bicameral_agent.gemini import GeminiClient, GeminiResponse
from bicameral_agent.schema import Message


def _make_messages(*pairs: tuple[str, str]) -> list[Message]:
    """Create messages from (role, content) pairs."""
    return [
        Message(role=role, content=content, timestamp_ms=0, token_count=len(content.split()))
        for role, content in pairs
    ]


def _mock_judge_response(logical_flow: int = 4, consistency: int = 4, overall: int = 4):
    return GeminiResponse(
        content=json.dumps({
            "logical_flow": logical_flow,
            "consistency": consistency,
            "overall": overall,
        }),
        input_tokens=50,
        output_tokens=20,
        duration_ms=100.0,
        finish_reason="STOP",
    )


class TestCoherenceScore:
    def test_valid_scores(self):
        score = CoherenceScore(logical_flow=0.5, consistency=0.75, overall=0.6)
        assert score.logical_flow == 0.5
        assert score.consistency == 0.75
        assert score.overall == 0.6

    def test_boundary_values(self):
        score = CoherenceScore(logical_flow=0.0, consistency=1.0, overall=0.0)
        assert score.logical_flow == 0.0
        assert score.consistency == 1.0

    def test_invalid_scores_rejected(self):
        with pytest.raises(Exception):
            CoherenceScore(logical_flow=-0.1, consistency=0.5, overall=0.5)
        with pytest.raises(Exception):
            CoherenceScore(logical_flow=0.5, consistency=1.1, overall=0.5)

    def test_from_raw(self):
        score = CoherenceScore.from_raw(5, 3, 1)
        assert score.logical_flow == 1.0
        assert score.consistency == 0.5
        assert score.overall == 0.0

    def test_from_raw_clamping(self):
        score = CoherenceScore.from_raw(0, 6, 3)
        assert score.logical_flow == 0.0  # clamped to 1 → (1-1)/4 = 0
        assert score.consistency == 1.0  # clamped to 5 → (5-1)/4 = 1


class TestCoherenceJudge:
    def test_score_returns_coherence_score(self):
        client = MagicMock(spec=GeminiClient)
        client.generate.return_value = _mock_judge_response(4, 5, 4)

        judge = CoherenceJudge(client=client)
        messages = _make_messages(
            ("user", "What is photosynthesis?"),
            ("assistant", "Photosynthesis is the process by which plants convert light."),
        )
        result = judge.score(messages)
        assert isinstance(result, CoherenceScore)
        assert result.logical_flow == 0.75  # (4-1)/4
        assert result.consistency == 1.0  # (5-1)/4
        assert result.overall == 0.75

    def test_cache_hit(self):
        client = MagicMock(spec=GeminiClient)
        client.generate.return_value = _mock_judge_response()

        judge = CoherenceJudge(client=client)
        messages = _make_messages(
            ("user", "Hello"),
            ("assistant", "Hi there!"),
        )
        judge.score(messages)
        judge.score(messages)  # should hit cache

        assert client.generate.call_count == 1
        assert judge.cache_size == 1

    def test_cache_miss_different_messages(self):
        client = MagicMock(spec=GeminiClient)
        client.generate.return_value = _mock_judge_response()

        judge = CoherenceJudge(client=client)
        msgs1 = _make_messages(("user", "Hello"), ("assistant", "Hi!"))
        msgs2 = _make_messages(("user", "Goodbye"), ("assistant", "Bye!"))

        judge.score(msgs1)
        judge.score(msgs2)

        assert client.generate.call_count == 2
        assert judge.cache_size == 2

    def test_batch_scoring(self):
        client = MagicMock(spec=GeminiClient)
        client.generate.return_value = _mock_judge_response(3, 3, 3)

        judge = CoherenceJudge(client=client)
        conversations = [
            _make_messages(("user", f"Question {i}"), ("assistant", f"Answer {i}"))
            for i in range(5)
        ]
        results = judge.score_batch(conversations)

        assert len(results) == 5
        assert all(isinstance(r, CoherenceScore) for r in results)

    def test_batch_uses_cache(self):
        client = MagicMock(spec=GeminiClient)
        client.generate.return_value = _mock_judge_response()

        judge = CoherenceJudge(client=client)
        msgs = _make_messages(("user", "Hello"), ("assistant", "Hi!"))

        # Pre-cache one conversation
        judge.score(msgs)

        # Batch with same conversation + a new one
        msgs2 = _make_messages(("user", "New"), ("assistant", "Response"))
        results = judge.score_batch([msgs, msgs2])

        assert len(results) == 2
        # Only the new one should trigger a call (1 original + 1 new = 2 total)
        assert client.generate.call_count == 2


class TestIntegration:
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    )
    def test_live_scoring(self):
        judge = CoherenceJudge()
        messages = _make_messages(
            ("user", "What causes rain?"),
            ("assistant", "Rain is caused by water vapor condensing in the atmosphere."),
            ("user", "Can you explain more?"),
            ("assistant", "When warm air rises, it cools and water vapor condenses into droplets."),
        )
        result = judge.score(messages)
        assert 0.0 <= result.logical_flow <= 1.0
        assert 0.0 <= result.consistency <= 1.0
        assert 0.0 <= result.overall <= 1.0
