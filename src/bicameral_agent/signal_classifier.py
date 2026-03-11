"""Behavioral signal aggregator for conversation analysis.

Computes 5 categorical behavioral signals from conversation history
and user events, returning a ``SignalVector`` whose one-hot encoding
can be concatenated onto other feature vectors.

Signal dimensions (18 total)
----------------------------

=====  ==================  ====
Index  Signal              Dims
=====  ==================  ====
0–3    stop_count             4
4–8    followup_type          5
9–11   response_latency       3
12–14  message_length_ratio   3
15–17  sentiment_shift        3
=====  ==================  ====
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np

from bicameral_agent.followup_classifier import FollowUpClassifier, FollowUpType
from bicameral_agent.schema import Message, UserEvent, UserEventType

SIGNAL_DIM: int = 18
"""Dimensionality of the one-hot signal vector."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StopCount(enum.Enum):
    """Number of consecutive trailing stop events."""

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE_PLUS = 3


class ResponseLatency(enum.Enum):
    """Response latency bucket."""

    FAST = "fast"  # <10s
    NORMAL = "normal"  # 10-60s
    SLOW = "slow"  # >60s


class LengthRatio(enum.Enum):
    """Ratio of last user message length to first user message length."""

    SHORTER = "shorter"  # <0.5
    SIMILAR = "similar"  # 0.5-2.0
    LONGER = "longer"  # >2.0


class SentimentShift(enum.Enum):
    """Sentiment direction between last two user messages."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


# ---------------------------------------------------------------------------
# One-hot offsets (stop_count uses StopCount.value directly: 0,1,2,3)
# ---------------------------------------------------------------------------
_FOLLOWUP_INDEX = {
    FollowUpType.CORRECTION: 4,
    FollowUpType.REDIRECT: 5,
    FollowUpType.ELABORATION: 6,
    FollowUpType.NEW_TASK: 7,
    FollowUpType.ENCOURAGEMENT: 8,
}
_LATENCY_INDEX = {ResponseLatency.FAST: 9, ResponseLatency.NORMAL: 10, ResponseLatency.SLOW: 11}
_RATIO_INDEX = {LengthRatio.SHORTER: 12, LengthRatio.SIMILAR: 13, LengthRatio.LONGER: 14}
_SENTIMENT_INDEX = {SentimentShift.POSITIVE: 15, SentimentShift.NEUTRAL: 16, SentimentShift.NEGATIVE: 17}


# ---------------------------------------------------------------------------
# Sentiment keywords (duplicated from encoder.py for module independence)
# ---------------------------------------------------------------------------
_POSITIVE_KEYWORDS: frozenset[str] = frozenset(
    {
        "good",
        "great",
        "thanks",
        "perfect",
        "nice",
        "excellent",
        "awesome",
        "helpful",
        "love",
        "wonderful",
        "amazing",
        "appreciate",
        "yes",
        "correct",
        "right",
    }
)
_NEGATIVE_KEYWORDS: frozenset[str] = frozenset(
    {
        "bad",
        "wrong",
        "terrible",
        "awful",
        "hate",
        "useless",
        "horrible",
        "worst",
        "no",
        "incorrect",
        "broken",
        "fail",
        "error",
        "annoying",
        "frustrating",
    }
)


def _sentiment_score(text: str) -> int:
    """Simple keyword-counting sentiment score."""
    text_lower = text.lower()
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)
    return pos - neg


# ---------------------------------------------------------------------------
# SignalVector
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SignalVector:
    """Five categorical behavioral signals with one-hot encoding."""

    stop_count: StopCount
    followup_type: FollowUpType
    response_latency: ResponseLatency
    message_length_ratio: LengthRatio
    sentiment_shift: SentimentShift

    def to_array(self) -> np.ndarray:
        """Return an 18-dim float32 one-hot vector (4+5+3+3+3)."""
        vec = np.zeros(SIGNAL_DIM, dtype=np.float32)
        vec[self.stop_count.value] = 1.0
        vec[_FOLLOWUP_INDEX[self.followup_type]] = 1.0
        vec[_LATENCY_INDEX[self.response_latency]] = 1.0
        vec[_RATIO_INDEX[self.message_length_ratio]] = 1.0
        vec[_SENTIMENT_INDEX[self.sentiment_shift]] = 1.0
        return vec


# ---------------------------------------------------------------------------
# SignalClassifier
# ---------------------------------------------------------------------------


class SignalClassifier:
    """Classifies conversation state into 5 categorical behavioral signals.

    All methods are static. Call :meth:`classify` with the full message
    history and user events to get a :class:`SignalVector`.
    """

    @staticmethod
    def classify(
        messages: list[Message],
        user_events: list[UserEvent],
    ) -> SignalVector:
        """Compute all 5 behavioral signals from conversation state.

        Parameters
        ----------
        messages:
            Ordered list of conversation messages.
        user_events:
            User-initiated events (stops, edits, follow-ups).

        Returns
        -------
        SignalVector
            Frozen dataclass with 5 categorical fields.
        """
        user_msgs = [m for m in messages if m.role == "user"]
        return SignalVector(
            stop_count=SignalClassifier._classify_stop_count(user_events),
            followup_type=SignalClassifier._classify_followup(messages, user_events),
            response_latency=SignalClassifier._classify_latency(messages),
            message_length_ratio=SignalClassifier._classify_length_ratio(user_msgs),
            sentiment_shift=SignalClassifier._classify_sentiment(user_msgs),
        )

    @staticmethod
    def _classify_stop_count(user_events: list[UserEvent]) -> StopCount:
        """Count consecutive trailing STOP events.

        Walks events chronologically, counting consecutive STOPs.
        Resets on any non-STOP event.
        """
        consecutive = 0
        for event in user_events:
            if event.event_type == UserEventType.STOP:
                consecutive += 1
            else:
                consecutive = 0
        return StopCount(min(consecutive, 3))

    @staticmethod
    def _classify_followup(
        messages: list[Message], user_events: list[UserEvent]
    ) -> FollowUpType:
        """Delegate to FollowUpClassifier on the last follow-up message."""
        last_followup_ts = max(
            (e.timestamp_ms for e in user_events if e.event_type == UserEventType.FOLLOW_UP),
            default=None,
        )
        if last_followup_ts is None:
            return FollowUpType.NEW_TASK
        followup_msg = None
        for msg in reversed(messages):
            if msg.role == "user" and msg.timestamp_ms >= last_followup_ts:
                followup_msg = msg
                break

        if followup_msg is None:
            # Fall back to last user message
            for msg in reversed(messages):
                if msg.role == "user":
                    followup_msg = msg
                    break

        if followup_msg is None:
            return FollowUpType.NEW_TASK

        return FollowUpClassifier.classify(followup_msg.content, messages)

    @staticmethod
    def _classify_latency(messages: list[Message]) -> ResponseLatency:
        """Time between last assistant msg and next user msg.

        <10s = FAST, 10-60s = NORMAL, >60s = SLOW.
        """
        if len(messages) < 2:
            return ResponseLatency.NORMAL

        # Find last assistant message, then the user message after it
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            return ResponseLatency.NORMAL

        # Find the next user message after the assistant
        next_user = None
        for i in range(last_assistant_idx + 1, len(messages)):
            if messages[i].role == "user":
                next_user = messages[i]
                break

        if next_user is None:
            return ResponseLatency.NORMAL

        latency_ms = next_user.timestamp_ms - messages[last_assistant_idx].timestamp_ms
        if latency_ms < 10_000:
            return ResponseLatency.FAST
        elif latency_ms <= 60_000:
            return ResponseLatency.NORMAL
        else:
            return ResponseLatency.SLOW

    @staticmethod
    def _classify_length_ratio(user_msgs: list[Message]) -> LengthRatio:
        """Ratio of last user message length to first user message length.

        <0.5 = SHORTER, 0.5-2.0 = SIMILAR, >2.0 = LONGER.
        """
        if len(user_msgs) < 1:
            return LengthRatio.SIMILAR

        first_user = user_msgs[0]
        last_user = user_msgs[-1]

        if len(first_user.content) == 0:
            return LengthRatio.SIMILAR

        ratio = len(last_user.content) / len(first_user.content)
        if ratio < 0.5:
            return LengthRatio.SHORTER
        elif ratio <= 2.0:
            return LengthRatio.SIMILAR
        else:
            return LengthRatio.LONGER

    @staticmethod
    def _classify_sentiment(user_msgs: list[Message]) -> SentimentShift:
        """Keyword-based sentiment delta between last two user messages."""
        if len(user_msgs) < 2:
            return SentimentShift.NEUTRAL

        prev_score = _sentiment_score(user_msgs[-2].content)
        curr_score = _sentiment_score(user_msgs[-1].content)
        delta = curr_score - prev_score

        if delta > 0:
            return SentimentShift.POSITIVE
        elif delta < 0:
            return SentimentShift.NEGATIVE
        else:
            return SentimentShift.NEUTRAL
