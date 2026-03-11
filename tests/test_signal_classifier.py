"""Tests for the behavioral signal aggregator."""

from __future__ import annotations

import time

import numpy as np
import pytest

from bicameral_agent.encoder import FEATURE_DIM, StateEncoder
from bicameral_agent.embeddings import HashEmbedder
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.schema import Message, UserEvent, UserEventType
from bicameral_agent.signal_classifier import (
    SIGNAL_DIM,
    LengthRatio,
    ResponseLatency,
    SentimentShift,
    SignalClassifier,
    SignalVector,
    StopCount,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str, ts: int = 1000, tokens: int = 5) -> Message:
    return Message(role=role, content=content, timestamp_ms=ts, token_count=tokens)


def _stop(ts: int = 1000) -> UserEvent:
    return UserEvent(event_type=UserEventType.STOP, timestamp_ms=ts)


def _edit(ts: int = 1000) -> UserEvent:
    return UserEvent(event_type=UserEventType.EDIT, timestamp_ms=ts)


def _followup(ts: int = 1000) -> UserEvent:
    return UserEvent(event_type=UserEventType.FOLLOW_UP, timestamp_ms=ts)


# ---------------------------------------------------------------------------
# TestStopCount
# ---------------------------------------------------------------------------


class TestStopCount:
    """Stop count classification from user events."""

    def test_zero_stops_empty_events(self):
        assert SignalClassifier._classify_stop_count([]) == StopCount.ZERO

    def test_zero_stops_no_stop_events(self):
        events = [_edit(100), _followup(200)]
        assert SignalClassifier._classify_stop_count(events) == StopCount.ZERO

    def test_one_trailing_stop(self):
        events = [_edit(100), _stop(200)]
        assert SignalClassifier._classify_stop_count(events) == StopCount.ONE

    def test_two_trailing_stops(self):
        events = [_edit(100), _stop(200), _stop(300)]
        assert SignalClassifier._classify_stop_count(events) == StopCount.TWO

    def test_three_plus_trailing_stops(self):
        events = [_stop(100), _stop(200), _stop(300), _stop(400)]
        assert SignalClassifier._classify_stop_count(events) == StopCount.THREE_PLUS

    def test_reset_on_non_stop(self):
        """Non-STOP events reset the consecutive counter."""
        events = [_stop(100), _stop(200), _edit(300), _stop(400)]
        assert SignalClassifier._classify_stop_count(events) == StopCount.ONE


# ---------------------------------------------------------------------------
# TestResponseLatency
# ---------------------------------------------------------------------------


class TestResponseLatency:
    """Response latency classification from message timestamps."""

    def test_fast_latency(self):
        msgs = [
            _msg("user", "hi", ts=1000),
            _msg("assistant", "hello", ts=2000),
            _msg("user", "thanks", ts=5000),  # 3s after assistant
        ]
        assert SignalClassifier._classify_latency(msgs) == ResponseLatency.FAST

    def test_normal_latency(self):
        msgs = [
            _msg("user", "hi", ts=1000),
            _msg("assistant", "hello", ts=2000),
            _msg("user", "ok", ts=32_000),  # 30s after assistant
        ]
        assert SignalClassifier._classify_latency(msgs) == ResponseLatency.NORMAL

    def test_slow_latency(self):
        msgs = [
            _msg("user", "hi", ts=1000),
            _msg("assistant", "hello", ts=2000),
            _msg("user", "back", ts=200_000),  # 198s after assistant
        ]
        assert SignalClassifier._classify_latency(msgs) == ResponseLatency.SLOW

    def test_boundary_10s_is_normal(self):
        """Exactly 10s -> NORMAL (boundary: <10s is FAST, 10-60s is NORMAL)."""
        msgs = [
            _msg("user", "hi", ts=1000),
            _msg("assistant", "hello", ts=2000),
            _msg("user", "ok", ts=12_000),  # exactly 10s
        ]
        assert SignalClassifier._classify_latency(msgs) == ResponseLatency.NORMAL

    def test_default_when_no_assistant(self):
        msgs = [_msg("user", "hi", ts=1000)]
        assert SignalClassifier._classify_latency(msgs) == ResponseLatency.NORMAL


# ---------------------------------------------------------------------------
# TestLengthRatio
# ---------------------------------------------------------------------------


class TestLengthRatio:
    """Message length ratio classification."""

    def test_longer_ratio(self):
        """3:1 ratio -> LONGER."""
        msgs = [
            _msg("user", "hi", ts=1000),
            _msg("assistant", "response", ts=2000),
            _msg("user", "this is a much longer message than the first", ts=3000),
        ]
        assert SignalClassifier._classify_length_ratio(msgs) == LengthRatio.LONGER

    def test_similar_ratio(self):
        """~1:1 ratio -> SIMILAR."""
        msgs = [
            _msg("user", "hello world", ts=1000),
            _msg("assistant", "response", ts=2000),
            _msg("user", "hello there", ts=3000),
        ]
        assert SignalClassifier._classify_length_ratio(msgs) == LengthRatio.SIMILAR

    def test_shorter_ratio(self):
        """1:3 ratio -> SHORTER."""
        msgs = [
            _msg("user", "this is a very long first message indeed", ts=1000),
            _msg("assistant", "response", ts=2000),
            _msg("user", "ok", ts=3000),
        ]
        assert SignalClassifier._classify_length_ratio(msgs) == LengthRatio.SHORTER

    def test_single_message(self):
        """Single user message -> ratio is 1.0 -> SIMILAR."""
        msgs = [_msg("user", "hello", ts=1000)]
        assert SignalClassifier._classify_length_ratio(msgs) == LengthRatio.SIMILAR

    def test_empty_first_message(self):
        """Empty first user message -> SIMILAR (avoid division by zero)."""
        msgs = [
            _msg("user", "", ts=1000),
            _msg("assistant", "response", ts=2000),
            _msg("user", "hello", ts=3000),
        ]
        assert SignalClassifier._classify_length_ratio(msgs) == LengthRatio.SIMILAR


# ---------------------------------------------------------------------------
# TestSentimentShift
# ---------------------------------------------------------------------------


class TestSentimentShift:
    """Sentiment shift classification between user messages."""

    def test_positive_shift(self):
        msgs = [
            _msg("user", "this is bad and wrong", ts=1000),
            _msg("assistant", "let me fix that", ts=2000),
            _msg("user", "great, that's perfect and awesome", ts=3000),
        ]
        assert SignalClassifier._classify_sentiment(msgs) == SentimentShift.POSITIVE

    def test_negative_shift(self):
        msgs = [
            _msg("user", "great job, perfect", ts=1000),
            _msg("assistant", "thanks!", ts=2000),
            _msg("user", "actually this is terrible and wrong", ts=3000),
        ]
        assert SignalClassifier._classify_sentiment(msgs) == SentimentShift.NEGATIVE

    def test_neutral_shift(self):
        msgs = [
            _msg("user", "tell me about Python", ts=1000),
            _msg("assistant", "Python is...", ts=2000),
            _msg("user", "tell me about Java", ts=3000),
        ]
        assert SignalClassifier._classify_sentiment(msgs) == SentimentShift.NEUTRAL

    def test_single_user_message(self):
        msgs = [_msg("user", "hello", ts=1000)]
        assert SignalClassifier._classify_sentiment(msgs) == SentimentShift.NEUTRAL

    def test_no_messages(self):
        assert SignalClassifier._classify_sentiment([]) == SentimentShift.NEUTRAL


# ---------------------------------------------------------------------------
# TestSignalVector
# ---------------------------------------------------------------------------


class TestSignalVector:
    """SignalVector dataclass and one-hot encoding."""

    @pytest.fixture
    def default_vector(self):
        return SignalVector(
            stop_count=StopCount.ZERO,
            followup_type=FollowUpType.NEW_TASK,
            response_latency=ResponseLatency.NORMAL,
            message_length_ratio=LengthRatio.SIMILAR,
            sentiment_shift=SentimentShift.NEUTRAL,
        )

    def test_shape_and_dtype(self, default_vector):
        arr = default_vector.to_array()
        assert arr.shape == (SIGNAL_DIM,)
        assert arr.dtype == np.float32

    def test_one_hot_sums(self, default_vector):
        """Each signal group should sum to exactly 1.0."""
        arr = default_vector.to_array()
        assert arr[0:4].sum() == 1.0  # stop_count
        assert arr[4:9].sum() == 1.0  # followup_type
        assert arr[9:12].sum() == 1.0  # latency
        assert arr[12:15].sum() == 1.0  # length_ratio
        assert arr[15:18].sum() == 1.0  # sentiment

    def test_specific_values(self):
        sv = SignalVector(
            stop_count=StopCount.TWO,
            followup_type=FollowUpType.CORRECTION,
            response_latency=ResponseLatency.FAST,
            message_length_ratio=LengthRatio.LONGER,
            sentiment_shift=SentimentShift.POSITIVE,
        )
        arr = sv.to_array()
        # StopCount.TWO -> index 2
        assert arr[2] == 1.0
        assert arr[0] == 0.0
        # FollowUpType.CORRECTION -> type_index 0 -> offset 4
        assert arr[4] == 1.0
        # ResponseLatency.FAST -> index 9
        assert arr[9] == 1.0
        # LengthRatio.LONGER -> index 14
        assert arr[14] == 1.0
        # SentimentShift.POSITIVE -> index 15
        assert arr[15] == 1.0

    def test_frozen(self, default_vector):
        with pytest.raises(AttributeError):
            default_vector.stop_count = StopCount.ONE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestSignalClassifier
# ---------------------------------------------------------------------------


class TestSignalClassifier:
    """Full SignalClassifier.classify() integration."""

    def test_full_scenario(self):
        msgs = [
            _msg("user", "Tell me about Python", ts=1000),
            _msg("assistant", "Python is a language", ts=2000),
            _msg("user", "Can you explain more?", ts=7000),  # 5s -> FAST
        ]
        events = [
            _stop(500),
            _followup(7000),
        ]
        sv = SignalClassifier.classify(msgs, events)
        assert sv.stop_count == StopCount.ZERO  # stop then followup resets
        assert isinstance(sv.followup_type, FollowUpType)
        assert sv.response_latency == ResponseLatency.FAST
        assert isinstance(sv.message_length_ratio, LengthRatio)
        assert isinstance(sv.sentiment_shift, SentimentShift)

    def test_empty_inputs(self):
        sv = SignalClassifier.classify([], [])
        assert sv.stop_count == StopCount.ZERO
        assert sv.followup_type == FollowUpType.NEW_TASK
        assert sv.response_latency == ResponseLatency.NORMAL
        assert sv.message_length_ratio == LengthRatio.SIMILAR
        assert sv.sentiment_shift == SentimentShift.NEUTRAL

    def test_all_fields_populated(self):
        """Verify all 5 fields are set and to_array works."""
        msgs = [
            _msg("user", "bad wrong terrible", ts=1000),
            _msg("assistant", "sorry", ts=2000),
            _msg("user", "good great perfect awesome wonderful thanks", ts=70_000),
        ]
        events = [_stop(500), _stop(800), _followup(70_000)]
        sv = SignalClassifier.classify(msgs, events)
        arr = sv.to_array()
        assert arr.shape == (SIGNAL_DIM,)
        assert arr.sum() == 5.0  # exactly 5 one-hot bits


# ---------------------------------------------------------------------------
# TestPerformance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Performance bounds for signal classification."""

    def test_all_signals_under_100ms(self):
        msgs = [
            _msg("user", f"message {i}", ts=1000 + i * 5000)
            if i % 2 == 0
            else _msg("assistant", f"response {i}", ts=1000 + i * 5000)
            for i in range(20)
        ]
        events = [_stop(ts=500 + i * 100) for i in range(5)]

        start = time.perf_counter()
        for _ in range(100):
            SignalClassifier.classify(msgs, events)
        elapsed = (time.perf_counter() - start) / 100

        assert elapsed < 0.1, f"Average classify took {elapsed:.4f}s, expected <100ms"


# ---------------------------------------------------------------------------
# TestCompatibility
# ---------------------------------------------------------------------------


class TestCompatibility:
    """Verify SignalVector can concatenate with encoder output."""

    def test_concatenation_with_encoder(self):
        encoder = StateEncoder(HashEmbedder(seed=42))
        msgs = [
            _msg("user", "Tell me about Python", ts=1000),
            _msg("assistant", "Python is great", ts=3000),
            _msg("user", "Thanks, that's helpful", ts=8000),
        ]
        events = [_followup(8000)]

        state_vec = encoder.encode(msgs, events)
        signal_vec = SignalClassifier.classify(msgs, events).to_array()
        combined = np.concatenate([state_vec, signal_vec])

        assert combined.shape == (FEATURE_DIM + SIGNAL_DIM,)
        assert combined.dtype == np.float32
