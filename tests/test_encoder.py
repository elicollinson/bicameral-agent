"""Tests for the reasoning state encoder and embeddings."""

import time

import numpy as np
import pytest

from bicameral_agent.embeddings import Embedder, HashEmbedder, get_default_embedder
from bicameral_agent.encoder import (
    FEATURE_DIM,
    StateEncoder,
    _sentiment_score,
)
from bicameral_agent.schema import (
    Message,
    ToolInvocation,
    UserEvent,
    UserEventType,
)


# ── Embedding tests ──────────────────────────────────────────────────


class TestHashEmbedder:
    def test_output_shape_and_dtype(self):
        emb = HashEmbedder(seed=42)
        vec = emb.embed("hello world")
        assert vec.shape == (32,)
        assert vec.dtype == np.float32

    def test_unit_norm(self):
        emb = HashEmbedder(seed=42)
        vec = emb.embed("some text")
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_deterministic(self):
        a = HashEmbedder(seed=42).embed("test")
        b = HashEmbedder(seed=42).embed("test")
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = HashEmbedder(seed=1).embed("test")
        b = HashEmbedder(seed=2).embed("test")
        assert not np.array_equal(a, b)

    def test_different_texts_differ(self):
        emb = HashEmbedder(seed=42)
        a = emb.embed("alpha")
        b = emb.embed("beta")
        assert not np.array_equal(a, b)

    def test_satisfies_protocol(self):
        assert isinstance(HashEmbedder(), Embedder)


class TestGetDefaultEmbedder:
    def test_returns_embedder(self):
        emb = get_default_embedder()
        assert isinstance(emb, Embedder)


# ── Acceptance criterion tests ───────────────────────────────────────


class TestAC1ConsistentShape:
    """AC1: Output shape is (53,), dtype float32, FEATURE_DIM matches."""

    def test_shape_and_dtype(self, encoder, simple_conversation):
        msgs, events, tools = simple_conversation
        vec = encoder.encode(msgs, events, tools)
        assert vec.shape == (FEATURE_DIM,)
        assert vec.dtype == np.float32

    def test_feature_dim_value(self):
        assert FEATURE_DIM == 53


class TestAC2Deterministic:
    """AC2: Same input → identical output across instances with same seed."""

    def test_same_instance(self, simple_conversation):
        msgs, events, tools = simple_conversation
        enc = StateEncoder(HashEmbedder(seed=42))
        a = enc.encode(msgs, events, tools)
        b = enc.encode(msgs, events, tools)
        np.testing.assert_array_equal(a, b)

    def test_across_instances(self, simple_conversation):
        msgs, events, tools = simple_conversation
        a = StateEncoder(HashEmbedder(seed=42)).encode(msgs, events, tools)
        b = StateEncoder(HashEmbedder(seed=42)).encode(msgs, events, tools)
        np.testing.assert_array_equal(a, b)


class TestAC3Discriminative:
    """AC3: 10 diverse states produce pairwise L2 with std > 0.1 × mean."""

    def test_pairwise_distances(self, encoder):
        states = []
        for i in range(10):
            msgs = [
                Message(
                    role="user",
                    content=f"topic number {i} about {'math' if i % 2 == 0 else 'art'}",
                    timestamp_ms=1000 * i,
                    token_count=10 + i * 5,
                ),
                Message(
                    role="assistant",
                    content=f"response {i}" + (" maybe perhaps" if i % 3 == 0 else " definitely"),
                    timestamp_ms=1000 * i + 500,
                    token_count=15 + i * 3,
                ),
            ]
            events = (
                [UserEvent(event_type=UserEventType.STOP, timestamp_ms=1000 * i + 600)]
                if i % 4 == 0
                else [UserEvent(event_type=UserEventType.FOLLOW_UP, timestamp_ms=1000 * i + 600)]
            )
            tools = (
                [
                    ToolInvocation(
                        tool_id="research_gap_scanner" if i % 2 == 0 else "assumption_auditor",
                        invoked_at_ms=1000 * i + 100,
                        completed_at_ms=1000 * i + 200,
                        input_tokens=10,
                        output_tokens=20,
                    )
                ]
                if i % 3 != 0
                else []
            )
            states.append(encoder.encode(msgs, events, tools))

        distances = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                distances.append(np.linalg.norm(states[i] - states[j]))

        distances = np.array(distances)
        assert distances.std() > 0.1 * distances.mean()


class TestAC4Performance:
    """AC4: < 50ms per encode averaged over 10 calls with HashEmbedder."""

    def test_encode_speed(self, encoder, simple_conversation):
        msgs, events, tools = simple_conversation
        # Warm up
        encoder.encode(msgs, events, tools)

        start = time.perf_counter()
        for _ in range(10):
            encoder.encode(msgs, events, tools)
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.050, f"Average encode took {elapsed*1000:.1f}ms (limit 50ms)"


class TestAC5EdgeCases:
    """AC5: empty, single-turn, and 1000+ messages — no NaN/Inf, correct shape."""

    def test_empty_conversation(self, encoder):
        vec = encoder.encode([], [], [])
        assert vec.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_single_turn(self, encoder):
        msgs = [Message(role="user", content="hi", timestamp_ms=100, token_count=1)]
        vec = encoder.encode(msgs)
        assert vec.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_many_messages(self, encoder):
        msgs = [
            Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"message {i}",
                timestamp_ms=i * 100,
                token_count=5,
            )
            for i in range(1001)
        ]
        vec = encoder.encode(msgs)
        assert vec.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))


class TestAC6Documentation:
    """AC6: Module docstring documents layout, FEATURE_DIM exported."""

    def test_module_docstring_mentions_layout(self):
        import bicameral_agent.encoder as mod

        assert mod.__doc__ is not None
        assert "53" in mod.__doc__
        assert "topic_embedding" in mod.__doc__

    def test_feature_dim_exported(self):
        from bicameral_agent import FEATURE_DIM as exported

        assert exported == 53


# ── Unit tests for private helpers ───────────────────────────────────


class TestComputeConfidence:
    def test_no_assistant_message(self):
        msgs = [Message(role="user", content="hello", timestamp_ms=0, token_count=1)]
        assert StateEncoder._compute_confidence(msgs) == 0.5

    def test_hedging_reduces_confidence(self):
        msgs = [
            Message(
                role="assistant",
                content="I think maybe this could possibly work",
                timestamp_ms=0,
                token_count=10,
            )
        ]
        conf = StateEncoder._compute_confidence(msgs)
        assert conf < 1.0

    def test_no_hedging_full_confidence(self):
        msgs = [
            Message(
                role="assistant",
                content="The answer is forty two",
                timestamp_ms=0,
                token_count=5,
            )
        ]
        conf = StateEncoder._compute_confidence(msgs)
        assert conf == 1.0

    def test_empty_content(self):
        msgs = [Message(role="assistant", content="", timestamp_ms=0, token_count=0)]
        assert StateEncoder._compute_confidence(msgs) == 0.5


class TestEncodeLastTool:
    def test_no_tools(self):
        out = StateEncoder._encode_last_tool([])
        expected = np.array([0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_known_tool(self):
        tool = ToolInvocation(
            tool_id="research_gap_scanner",
            invoked_at_ms=0,
            completed_at_ms=100,
            input_tokens=1,
            output_tokens=1,
        )
        out = StateEncoder._encode_last_tool([tool])
        assert out[0] == 1.0
        assert out.sum() == 1.0

    def test_unknown_tool(self):
        tool = ToolInvocation(
            tool_id="mystery_tool",
            invoked_at_ms=0,
            completed_at_ms=100,
            input_tokens=1,
            output_tokens=1,
        )
        out = StateEncoder._encode_last_tool([tool])
        assert out[3] == 1.0
        assert out.sum() == 1.0


class TestComputeTurnsSinceTool:
    def test_no_tools_no_messages(self):
        assert StateEncoder._compute_turns_since_tool([], []) == 0.0

    def test_no_tools_with_messages(self):
        msgs = [Message(role="user", content="hi", timestamp_ms=100, token_count=1)]
        val = StateEncoder._compute_turns_since_tool(msgs, [])
        assert val == 1.0 / 20.0

    def test_tool_then_messages(self):
        tool = ToolInvocation(
            tool_id="t",
            invoked_at_ms=0,
            completed_at_ms=100,
            input_tokens=1,
            output_tokens=1,
        )
        msgs = [
            Message(role="user", content="a", timestamp_ms=50, token_count=1),
            Message(role="assistant", content="b", timestamp_ms=150, token_count=1),
            Message(role="user", content="c", timestamp_ms=200, token_count=1),
        ]
        val = StateEncoder._compute_turns_since_tool(msgs, [tool])
        # Two messages after tool completed at 100
        assert val == 2.0 / 20.0


class TestComputeStopCount:
    def test_no_events(self):
        assert StateEncoder._compute_stop_count([]) == 0.0

    def test_multiple_stops(self):
        events = [
            UserEvent(event_type=UserEventType.STOP, timestamp_ms=i * 100)
            for i in range(3)
        ]
        assert StateEncoder._compute_stop_count(events) == 3.0 / 5.0

    def test_cap(self):
        events = [
            UserEvent(event_type=UserEventType.STOP, timestamp_ms=i * 100)
            for i in range(10)
        ]
        assert StateEncoder._compute_stop_count(events) == 1.0


class TestClassifyFollowup:
    def _make_followup(self, content: str, ts: int = 1000):
        msgs = [Message(role="user", content=content, timestamp_ms=ts, token_count=5)]
        events = [UserEvent(event_type=UserEventType.FOLLOW_UP, timestamp_ms=ts)]
        return StateEncoder._classify_followup(msgs, events)

    def test_correction(self):
        out = self._make_followup("No, that's wrong")
        assert out[0] == 1.0

    def test_redirect(self):
        out = self._make_followup("Let's talk about a different topic instead")
        assert out[1] == 1.0

    def test_elaboration(self):
        out = self._make_followup("Can you explain more about that?")
        assert out[2] == 1.0

    def test_new_task(self):
        out = self._make_followup("Next, additionally handle that task")
        assert out[3] == 1.0

    def test_encouragement(self):
        out = self._make_followup("Great, exactly what I needed")
        assert out[4] == 1.0

    def test_no_followup_events(self):
        msgs = [Message(role="user", content="hi", timestamp_ms=0, token_count=1)]
        out = StateEncoder._classify_followup(msgs, [])
        np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))

    def test_priority_correction_over_encouragement(self):
        # "no" is both correction and potentially something else
        out = self._make_followup("No, that's not right, fix it")
        assert out[0] == 1.0
        assert out.sum() == 1.0


class TestComputeLatencyBucket:
    def _make_pair(self, user_ts: int, assistant_ts: int):
        return [
            Message(role="user", content="q", timestamp_ms=user_ts, token_count=1),
            Message(role="assistant", content="a", timestamp_ms=assistant_ts, token_count=1),
        ]

    def test_fast(self):
        out = StateEncoder._compute_latency_bucket(self._make_pair(1000, 2500))
        assert out[0] == 1.0  # < 2s

    def test_normal(self):
        out = StateEncoder._compute_latency_bucket(self._make_pair(1000, 6000))
        assert out[1] == 1.0  # 2–10s

    def test_slow(self):
        out = StateEncoder._compute_latency_bucket(self._make_pair(1000, 20000))
        assert out[2] == 1.0  # > 10s

    def test_no_messages(self):
        out = StateEncoder._compute_latency_bucket([])
        assert out[1] == 1.0  # default: normal

    def test_only_user(self):
        msgs = [Message(role="user", content="q", timestamp_ms=100, token_count=1)]
        out = StateEncoder._compute_latency_bucket(msgs)
        assert out[1] == 1.0


class TestComputeLengthRatio:
    def test_equal_lengths(self):
        msgs = [
            Message(role="user", content="hello", timestamp_ms=0, token_count=1),
            Message(role="assistant", content="world", timestamp_ms=100, token_count=1),
        ]
        ratio = StateEncoder._compute_length_ratio(msgs)
        assert ratio == pytest.approx(1.0 / 5.0)  # 1.0 / _LENGTH_RATIO_CAP

    def test_no_messages(self):
        assert StateEncoder._compute_length_ratio([]) == 0.0

    def test_empty_user_content(self):
        msgs = [
            Message(role="user", content="", timestamp_ms=0, token_count=0),
            Message(role="assistant", content="answer", timestamp_ms=100, token_count=1),
        ]
        assert StateEncoder._compute_length_ratio(msgs) == 0.0


class TestComputeSentimentShift:
    def test_positive_shift(self):
        msgs = [
            Message(role="user", content="this is bad", timestamp_ms=0, token_count=3),
            Message(role="user", content="actually great thanks", timestamp_ms=100, token_count=3),
        ]
        out = StateEncoder._compute_sentiment_shift(msgs)
        assert out[0] == 1.0  # positive

    def test_negative_shift(self):
        msgs = [
            Message(role="user", content="great work", timestamp_ms=0, token_count=2),
            Message(role="user", content="this is terrible", timestamp_ms=100, token_count=3),
        ]
        out = StateEncoder._compute_sentiment_shift(msgs)
        assert out[2] == 1.0  # negative

    def test_neutral(self):
        msgs = [
            Message(role="user", content="tell me about X", timestamp_ms=0, token_count=4),
            Message(role="user", content="tell me about Y", timestamp_ms=100, token_count=4),
        ]
        out = StateEncoder._compute_sentiment_shift(msgs)
        assert out[1] == 1.0  # neutral

    def test_single_user_message(self):
        msgs = [Message(role="user", content="hello", timestamp_ms=0, token_count=1)]
        out = StateEncoder._compute_sentiment_shift(msgs)
        assert out[1] == 1.0  # neutral default


class TestSentimentScore:
    def test_positive(self):
        assert _sentiment_score("great awesome") > 0

    def test_negative(self):
        assert _sentiment_score("terrible awful") < 0

    def test_neutral(self):
        assert _sentiment_score("the weather today") == 0
