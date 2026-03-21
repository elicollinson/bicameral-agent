"""Reasoning state encoder for the MCTS controller.

Compresses conversation history, user events, and tool history into a
fixed-dimensional feature vector suitable for policy/value network input.

Feature vector layout (64 dimensions)
--------------------------------------

=====  ======================================  ====
Index  Feature                                 Dims
=====  ======================================  ====
0      turn_number                                1
1      total_tokens_so_far                        1
2–33   topic_embedding                           32
34     estimated_confidence                       1
35–38  last_tool_invoked (one-hot)                4
39     turns_since_last_tool                      1
40     user_stop_count                            1
41–45  last_followup_type (one-hot)               5
46–48  response_latency_bucket (one-hot)          3
49     message_length_ratio                       1
50–52  sentiment_shift (one-hot)                  3
53     queue_depth                                1
54     queue_token_total                          1
55     queue_max_priority (ordinal)               1
56     queue_time_since_last_drain                1
57     queue_pending_tool_count                   1
58     queue_estimated_next_arrival               1
59     latency_research_gap_scanner               1
60     latency_assumption_auditor                 1
61     latency_context_refresher                  1
62     episode_turn_progress                      1
63     episode_completion_fraction                1
=====  ======================================  ====

Scalar normalization uses cap-and-divide: ``min(val, cap) / cap``, producing
values deterministically bounded to [0, 1].
"""

from __future__ import annotations

import numpy as np

from bicameral_agent.embeddings import Embedder, get_default_embedder
from bicameral_agent.followup_classifier import FollowUpClassifier
from bicameral_agent.queue import QueueState
from bicameral_agent.schema import Message, ToolInvocation, UserEvent, UserEventType

FEATURE_DIM: int = 64
"""Dimensionality of the encoded state vector."""

# ---------------------------------------------------------------------------
# Normalization caps
# ---------------------------------------------------------------------------
_TURN_CAP = 100
_TOKEN_CAP = 100_000
_TURNS_SINCE_TOOL_CAP = 20
_STOP_CAP = 5
_LENGTH_RATIO_CAP = 5.0

# Queue state caps
_QUEUE_DEPTH_CAP = 20
_QUEUE_TOKEN_TOTAL_CAP = 10_000
_PRIORITY_MAX = 3.0
_TIME_SINCE_DRAIN_CAP = 60.0
_PENDING_TOOL_COUNT_CAP = 3
_ESTIMATED_ARRIVAL_CAP = 30.0

# Latency caps
_LATENCY_MS_CAP = 30_000.0

# Episode progress caps
_MAX_TURNS_CAP = 200

# ---------------------------------------------------------------------------
# Tool vocabulary (one-hot with 4 slots)
# ---------------------------------------------------------------------------
_TOOL_VOCAB: list[str] = ["research_gap_scanner", "assumption_auditor", "context_refresher"]
# Index 3 is "none / unknown"

# ---------------------------------------------------------------------------
# Hedging keywords (for confidence estimation)
# ---------------------------------------------------------------------------
_HEDGE_KEYWORDS: frozenset[str] = frozenset(
    {
        "maybe",
        "perhaps",
        "possibly",
        "might",
        "could",
        "uncertain",
        "not sure",
        "i think",
        "it seems",
        "probably",
        "likely",
        "unclear",
    }
)

# ---------------------------------------------------------------------------
# Sentiment keyword sets
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


class StateEncoder:
    """Encodes conversation state into a fixed-dimensional feature vector.

    Parameters
    ----------
    embedder:
        An :class:`~bicameral_agent.embeddings.Embedder` instance.
        When *None*, the best available embedder is selected automatically
        (``FastEmbedEmbedder`` if installed, otherwise ``HashEmbedder``).
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder: Embedder = embedder if embedder is not None else get_default_embedder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        conversation_history: list[Message],
        user_events: list[UserEvent] | None = None,
        tool_history: list[ToolInvocation] | None = None,
        *,
        queue_state: QueueState | None = None,
        latency_predictions: dict[str, float] | None = None,
        turn_number: int | None = None,
        max_turns: int | None = None,
    ) -> np.ndarray:
        """Encode the current conversation state into a 64-dim float32 vector.

        Parameters
        ----------
        conversation_history:
            Ordered list of messages exchanged so far.
        user_events:
            User-initiated events (stops, edits, follow-ups).
        tool_history:
            Tool invocations that have occurred.
        queue_state:
            Snapshot from ``ContextQueue.get_state()``.  When *None*, the
            queue feature slots (53–58) are left as zeros.
        latency_predictions:
            Mapping of tool_id to predicted ``mean_ms`` latency.  When
            *None*, the latency feature slots (59–61) are left as zeros.
        turn_number:
            Current episode turn (1-indexed loop counter).
        max_turns:
            Configured maximum turns for the episode.

        Returns
        -------
        numpy.ndarray
            Shape ``(64,)`` float32 vector with values in [0, 1].
        """
        user_events = user_events or []
        tool_history = tool_history or []

        vec = np.zeros(FEATURE_DIM, dtype=np.float32)

        # 0: turn_number (normalized)
        msg_count = len(conversation_history)
        vec[0] = min(msg_count, _TURN_CAP) / _TURN_CAP

        # 1: total_tokens_so_far (normalized)
        total_tokens = sum(m.token_count for m in conversation_history)
        vec[1] = min(total_tokens, _TOKEN_CAP) / _TOKEN_CAP

        # 2–33: topic embedding
        last_user_msg = self._last_message_by_role(conversation_history, "user")
        if last_user_msg is not None:
            vec[2:34] = self._embedder.embed(last_user_msg.content)
        # else: zeros (no user message yet)

        # 34: estimated confidence
        vec[34] = self._compute_confidence(conversation_history)

        # 35–38: last tool invoked (one-hot, 4 slots)
        vec[35:39] = self._encode_last_tool(tool_history)

        # 39: turns since last tool (normalized)
        vec[39] = self._compute_turns_since_tool(conversation_history, tool_history)

        # 40: user stop count (normalized)
        vec[40] = self._compute_stop_count(user_events)

        # 41–45: last followup type (one-hot, 5 slots)
        vec[41:46] = self._classify_followup(conversation_history, user_events)

        # 46–48: response latency bucket (one-hot, 3 slots)
        vec[46:49] = self._compute_latency_bucket(conversation_history)

        # 49: message length ratio
        vec[49] = self._compute_length_ratio(conversation_history)

        # 50–52: sentiment shift (one-hot, 3 slots)
        vec[50:53] = self._compute_sentiment_shift(conversation_history)

        # 53–58: queue state (6 dims)
        vec[53:59] = self._encode_queue_state(queue_state)

        # 59–61: latency context (3 dims)
        vec[59:62] = self._encode_latency_context(latency_predictions)

        # 62–63: episode progress (2 dims)
        vec[62:64] = self._encode_episode_progress(turn_number, max_turns)

        return vec

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_message_by_role(
        messages: list[Message], role: str
    ) -> Message | None:
        """Return the most recent message with the given *role*, or None."""
        for msg in reversed(messages):
            if msg.role == role:
                return msg
        return None

    @staticmethod
    def _compute_confidence(messages: list[Message]) -> float:
        """Estimate assistant confidence from hedging language.

        Returns ``1.0 - min(hedge_count / word_count, 1.0)``.
        Defaults to 0.5 when there is no assistant message.
        """
        last_assistant = StateEncoder._last_message_by_role(messages, "assistant")
        if last_assistant is None:
            return 0.5

        text_lower = last_assistant.content.lower()
        words = text_lower.split()
        if not words:
            return 0.5

        hedge_count = sum(1 for kw in _HEDGE_KEYWORDS if kw in text_lower)
        return 1.0 - min(hedge_count / len(words), 1.0)

    @staticmethod
    def _encode_last_tool(tool_history: list[ToolInvocation]) -> np.ndarray:
        """One-hot encode the last tool invoked (4 slots).

        Slots 0–2 correspond to the known tool vocabulary; slot 3 is
        "none / unknown".
        """
        out = np.zeros(4, dtype=np.float32)
        if not tool_history:
            out[3] = 1.0
            return out
        last_tool_id = tool_history[-1].tool_id
        if last_tool_id in _TOOL_VOCAB:
            out[_TOOL_VOCAB.index(last_tool_id)] = 1.0
        else:
            out[3] = 1.0
        return out

    @staticmethod
    def _compute_turns_since_tool(
        messages: list[Message], tool_history: list[ToolInvocation]
    ) -> float:
        """Normalized count of messages since the last tool completion."""
        if not tool_history or not messages:
            return min(len(messages), _TURNS_SINCE_TOOL_CAP) / _TURNS_SINCE_TOOL_CAP

        last_tool_time = tool_history[-1].completed_at_ms
        turns_after = sum(1 for m in messages if m.timestamp_ms > last_tool_time)
        return min(turns_after, _TURNS_SINCE_TOOL_CAP) / _TURNS_SINCE_TOOL_CAP

    @staticmethod
    def _compute_stop_count(user_events: list[UserEvent]) -> float:
        """Normalized count of user stop events."""
        stops = sum(1 for e in user_events if e.event_type == UserEventType.STOP)
        return min(stops, _STOP_CAP) / _STOP_CAP

    @staticmethod
    def _classify_followup(
        messages: list[Message], user_events: list[UserEvent]
    ) -> np.ndarray:
        """One-hot encode the last follow-up type (5 slots).

        Delegates to :class:`~bicameral_agent.followup_classifier.FollowUpClassifier`
        for the actual classification logic.

        Priority: correction(0) > redirect(1) > elaboration(2) >
        new_task(3) > encouragement(4).
        """
        out = np.zeros(5, dtype=np.float32)

        followup_times = {
            e.timestamp_ms for e in user_events if e.event_type == UserEventType.FOLLOW_UP
        }
        if not followup_times:
            return out

        # Find the user message closest to the last follow-up event, or fall back
        # to the last user message overall
        last_followup_ts = max(followup_times)
        followup_msg = None
        for msg in reversed(messages):
            if msg.role == "user" and msg.timestamp_ms >= last_followup_ts:
                followup_msg = msg
                break
        if followup_msg is None:
            followup_msg = StateEncoder._last_message_by_role(messages, "user")
        if followup_msg is None:
            return out

        followup_type = FollowUpClassifier.classify(followup_msg.content, messages)
        out[FollowUpClassifier.type_index(followup_type)] = 1.0
        return out

    @staticmethod
    def _last_user_and_assistant(
        messages: list[Message],
    ) -> tuple[Message | None, Message | None]:
        """Return the last user and last assistant messages, scanning once."""
        last_user = None
        last_assistant = None
        for msg in reversed(messages):
            if msg.role == "user" and last_user is None:
                last_user = msg
            elif msg.role == "assistant" and last_assistant is None:
                last_assistant = msg
            if last_user is not None and last_assistant is not None:
                break
        return last_user, last_assistant

    @staticmethod
    def _compute_latency_bucket(messages: list[Message]) -> np.ndarray:
        """One-hot encode response latency: fast(<2s), normal(2-10s), slow(>10s).

        Measures the time between the last user message and the last
        assistant message.  If either is missing, returns the "normal"
        bucket as default.
        """
        out = np.zeros(3, dtype=np.float32)
        last_user, last_assistant = StateEncoder._last_user_and_assistant(messages)

        if last_user is None or last_assistant is None:
            out[1] = 1.0  # default: normal
            return out

        # Only meaningful if assistant replied after user
        if last_assistant.timestamp_ms <= last_user.timestamp_ms:
            out[1] = 1.0  # default: normal
            return out

        latency_ms = last_assistant.timestamp_ms - last_user.timestamp_ms
        if latency_ms < 2000:
            out[0] = 1.0  # fast
        elif latency_ms <= 10_000:
            out[1] = 1.0  # normal
        else:
            out[2] = 1.0  # slow
        return out

    @staticmethod
    def _compute_length_ratio(messages: list[Message]) -> float:
        """Ratio of last assistant message length to last user message length.

        Returns 0 if either message is missing.  Capped and normalized.
        """
        last_user, last_assistant = StateEncoder._last_user_and_assistant(messages)

        if last_user is None or last_assistant is None:
            return 0.0

        user_len = len(last_user.content)
        if user_len == 0:
            return 0.0
        ratio = len(last_assistant.content) / user_len
        return min(ratio, _LENGTH_RATIO_CAP) / _LENGTH_RATIO_CAP

    @staticmethod
    def _compute_sentiment_shift(messages: list[Message]) -> np.ndarray:
        """One-hot sentiment shift between the last two user messages.

        Slots: positive(0), neutral(1), negative(2).
        Returns neutral if fewer than two user messages exist.
        """
        out = np.zeros(3, dtype=np.float32)

        user_msgs: list[Message] = [m for m in messages if m.role == "user"]
        if len(user_msgs) < 2:
            out[1] = 1.0  # neutral
            return out

        prev_score = _sentiment_score(user_msgs[-2].content)
        curr_score = _sentiment_score(user_msgs[-1].content)
        delta = curr_score - prev_score

        if delta > 0:
            out[0] = 1.0  # positive shift
        elif delta < 0:
            out[2] = 1.0  # negative shift
        else:
            out[1] = 1.0  # neutral
        return out

    @staticmethod
    def _encode_queue_state(queue_state: QueueState | None) -> np.ndarray:
        """Encode queue state into 6 normalized features.

        Indices: depth, token_total, max_priority (ordinal),
        time_since_last_drain, pending_tool_count, estimated_next_arrival.
        """
        out = np.zeros(6, dtype=np.float32)
        if queue_state is None:
            return out

        out[0] = min(queue_state.depth, _QUEUE_DEPTH_CAP) / _QUEUE_DEPTH_CAP
        out[1] = min(queue_state.token_total, _QUEUE_TOKEN_TOTAL_CAP) / _QUEUE_TOKEN_TOTAL_CAP
        out[2] = (queue_state.max_priority.value / _PRIORITY_MAX) if queue_state.max_priority is not None else 0.0
        out[3] = min(queue_state.time_since_last_drain, _TIME_SINCE_DRAIN_CAP) / _TIME_SINCE_DRAIN_CAP
        out[4] = min(queue_state.pending_tool_count, _PENDING_TOOL_COUNT_CAP) / _PENDING_TOOL_COUNT_CAP
        out[5] = min(queue_state.estimated_next_arrival, _ESTIMATED_ARRIVAL_CAP) / _ESTIMATED_ARRIVAL_CAP
        return out

    @staticmethod
    def _encode_latency_context(
        latency_predictions: dict[str, float] | None,
    ) -> np.ndarray:
        """Encode predicted latency for each tool into 3 normalized features.

        One slot per tool in ``_TOOL_VOCAB`` order, using pre-computed
        ``mean_ms`` values capped at ``_LATENCY_MS_CAP``.
        """
        out = np.zeros(3, dtype=np.float32)
        if latency_predictions is None:
            return out

        for i, tool_id in enumerate(_TOOL_VOCAB):
            mean_ms = latency_predictions.get(tool_id, 0.0)
            out[i] = min(mean_ms, _LATENCY_MS_CAP) / _LATENCY_MS_CAP
        return out

    @staticmethod
    def _encode_episode_progress(
        turn_number: int | None,
        max_turns: int | None,
    ) -> np.ndarray:
        """Encode episode progress into 2 normalized features.

        [0] Absolute turn progress: ``turn_number / _MAX_TURNS_CAP``.
        [1] Relative completion fraction: ``turn_number / max_turns``.
        """
        out = np.zeros(2, dtype=np.float32)
        if turn_number is None:
            return out

        out[0] = min(turn_number, _MAX_TURNS_CAP) / _MAX_TURNS_CAP
        max_turns_eff = max_turns if max_turns is not None else _MAX_TURNS_CAP
        out[1] = min(turn_number / max_turns_eff, 1.0)
        return out


def _sentiment_score(text: str) -> int:
    """Simple keyword-counting sentiment score."""
    text_lower = text.lower()
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)
    return pos - neg
