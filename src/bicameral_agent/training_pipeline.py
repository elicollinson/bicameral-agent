"""Episode-to-training-data pipeline.

Converts logged :class:`~bicameral_agent.schema.Episode` records into
``(state, action, reward, next_state, done)`` tuples for training the
policy/value network.

State vector layout (108 dimensions)
------------------------------------

=========  =========================================  ====
Index      Component                                  Dims
=========  =========================================  ====
0–63       Reasoning state (StateEncoder.encode)        64
64–81      User signal vector (SignalClassifier)        18
82–93      One-hot of last 3 tool invocations           12
94–96      Seconds since last 3 tool invocations         3
97–102     Queue state                                   6
103–105    Predicted latency per tool                    3
106–107    Turn number + episode completion fraction     2
=========  =========================================  ====

There is intentional redundancy with the StateEncoder slice — the issue
spec lists each component as a separate concatenation. The network can
learn to ignore duplicated dimensions.

Reward construction
-------------------

Per-step intermediate reward components:

- user STOP event: −0.3
- user FOLLOW_UP (default): +0.1
- follow-up classified as ENCOURAGEMENT: +0.2 (replaces the +0.1)
- follow-up classified as REDIRECT or CORRECTION: −0.2 (replaces +0.1)
- assistant + tool tokens this turn: −0.01 × (tokens / 100)
- queue interrupt this turn: −0.15
- expired queue item this turn: −0.05 (attributed to terminal step)
- drained/consumed injection this turn: +0.05

Terminal reward = ``episode.outcome.quality_score`` (or 0.0 if absent),
added to the final decision point's reward.

Discounted returns are computed backwards with γ = 0.95.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from bicameral_agent.encoder import (
    FEATURE_DIM,
    QUEUE_STATE_DIM,
    StateEncoder,
    _cap_norm,
    encode_queue_state,
)
from bicameral_agent.followup_classifier import FollowUpClassifier, FollowUpType
from bicameral_agent.heuristic_controller import TOOL_IDS, Action
from bicameral_agent.queue import Priority, QueueState
from bicameral_agent.replay import EpisodeReplayer, ReplayState
from bicameral_agent.schema import (
    ContextInjection,
    Episode,
    Message,
    ToolInvocation,
    UserEventType,
)
from bicameral_agent.signal_classifier import SIGNAL_DIM, SignalClassifier
from bicameral_agent.token_estimator import ContextFeatures
from bicameral_agent.tool_latency import ToolLatencyModel

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.data import TensorDataset

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

# Tool vocabulary (order matters for one-hot indexing)
_TOOL_VOCAB: tuple[str, ...] = tuple(TOOL_IDS.values())  # 3 tools

_TOOL_HISTORY_SLOTS = 3  # how many recent invocations are encoded
# Per-slot one-hot width: one index per tool plus a trailing "none/unknown"
# entry. Derived from the vocabulary so adding a tool cannot silently
# collide with the "none" index.
_TOOL_ONEHOT_WIDTH = len(_TOOL_VOCAB) + 1
_TOOL_HISTORY_ONEHOT = _TOOL_HISTORY_SLOTS * _TOOL_ONEHOT_WIDTH  # 12
_TOOL_HISTORY_TIME = _TOOL_HISTORY_SLOTS
_QUEUE_DIMS = QUEUE_STATE_DIM
_LATENCY_DIMS = len(_TOOL_VOCAB)  # one slot per tool in the vocabulary
_PROGRESS_DIMS = 2

STATE_DIM: int = (
    FEATURE_DIM
    + SIGNAL_DIM
    + _TOOL_HISTORY_ONEHOT
    + _TOOL_HISTORY_TIME
    + _QUEUE_DIMS
    + _LATENCY_DIMS
    + _PROGRESS_DIMS
)
"""Total dimensionality of the training state vector (108)."""

# Slice offsets
_OFF_ENCODER = 0
_OFF_SIGNALS = _OFF_ENCODER + FEATURE_DIM
_OFF_TOOL_ONEHOT = _OFF_SIGNALS + SIGNAL_DIM
_OFF_TOOL_TIMES = _OFF_TOOL_ONEHOT + _TOOL_HISTORY_ONEHOT
_OFF_QUEUE = _OFF_TOOL_TIMES + _TOOL_HISTORY_TIME
_OFF_LATENCY = _OFF_QUEUE + _QUEUE_DIMS
_OFF_PROGRESS = _OFF_LATENCY + _LATENCY_DIMS

# Action -> categorical index. Mirrors policy_value_net.ACTION_ORDER but
# defined here to avoid a hard dependency on torch. A cross-check test
# asserts the two stay identical (tests/test_training_pipeline.py).
_ACTION_ORDER: tuple[Action, ...] = (
    Action.SCANNER,
    Action.AUDITOR,
    Action.REFRESHER,
    Action.DO_NOTHING,
)
_ACTION_INDEX: dict[Action, int] = {a: i for i, a in enumerate(_ACTION_ORDER)}

# ---------------------------------------------------------------------------
# Normalization caps (queue caps live in encoder.encode_queue_state, which is
# shared with the StateEncoder slice of the vector)
# ---------------------------------------------------------------------------
_TOOL_RECENCY_SECONDS_CAP = 300.0  # elapsed seconds since a tool invocation
_LATENCY_MS_CAP = 30_000.0
_TURN_CAP = 200

# Default configured episode turn limit; must match the EpisodeConfig used
# to generate the episodes (episode_runner.EpisodeConfig.max_turns).
DEFAULT_MAX_TURNS = 25

# Reward weights (per issue spec)
R_STOP: float = -0.3
R_FOLLOWUP: float = 0.1
R_ENCOURAGEMENT: float = 0.2
R_REDIRECT_CORRECTION: float = -0.2
R_TOKEN_PER_100: float = -0.01
R_INTERRUPT: float = -0.15
R_EXPIRED: float = -0.05
R_DRAIN_CONSUMED: float = 0.05

DISCOUNT_GAMMA: float = 0.95
"""Discount factor for computing returns."""


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingExample:
    """A single (state, action, reward, next_state, done) tuple plus diagnostics.

    Fields
    ------
    state, next_state:
        108-dim float32 numpy arrays.
    action:
        Categorical action index (0–3) matching ``ACTION_ORDER``.
    reward:
        Per-step shaped reward at this decision point.
    done:
        True if this is the final decision point in the episode.
    discounted_return:
        Sum of discounted future rewards from this step onward
        (computed during pipeline processing).
    episode_id, decision_index:
        Provenance: which episode this example came from and its
        position within that episode (0-indexed).
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    discounted_return: float
    episode_id: str
    decision_index: int


class TrainingDataPipeline:
    """Convert :class:`Episode` records into training examples for RL.

    Parameters
    ----------
    encoder:
        StateEncoder instance. If *None*, a default encoder is created.
    latency_model:
        ToolLatencyModel for the predicted-latency feature slice. If
        *None*, a fresh model is created (predictions use built-in
        defaults). This intentionally matches serving time, where
        ``episode_runner`` also constructs a fresh ``ToolLatencyModel``
        per episode: the default predictions are a deterministic function
        of context features, so training and serving stay consistent.
        Once a trained latency model exists (#44), pass the same trained
        model here and at serving time.
    max_turns:
        The configured episode turn limit used when the episodes were
        generated (``EpisodeConfig.max_turns``). Used for the
        completion-fraction features (dims 63/107). The episode's actual
        final turn count must NOT be used here: it is unavailable at
        inference time and leaks the episode's outcome (e.g. early STOPs)
        into every decision-point state.
    """

    def __init__(
        self,
        encoder: StateEncoder | None = None,
        latency_model: ToolLatencyModel | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        if max_turns < 1:
            msg = f"max_turns must be >= 1, got {max_turns}"
            raise ValueError(msg)
        self._encoder = encoder or StateEncoder()
        self._latency_model = latency_model or ToolLatencyModel()
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_episode(self, episode: Episode) -> list[TrainingExample]:
        """Convert an episode into a list of training examples.

        One example is produced per decision point (each assistant message).
        Returns an empty list if the episode has no assistant messages.
        """
        replayer = EpisodeReplayer(episode)
        decision_points = list(replayer.iter_decision_points())
        if not decision_points:
            return []

        # Pre-compute per-turn metadata derived from the whole episode.
        injection_mode = episode.metadata.get("injection_mode", "")
        is_interrupt_mode = injection_mode == "interrupt"

        # Map: turn_number -> (action_int, tool_invocation_or_none)
        turn_actions = self._infer_actions(episode)

        # Build state vectors and per-step rewards
        n = len(decision_points)
        states = [
            self._build_state_vector(dp.state, dp.action)
            for dp in decision_points
        ]
        next_states = [
            states[i + 1] if i + 1 < n else np.zeros(STATE_DIM, dtype=np.float32)
            for i in range(n)
        ]
        actions = [
            turn_actions.get(dp.state.turn_number, _ACTION_INDEX[Action.DO_NOTHING])
            for dp in decision_points
        ]

        # Per-step intermediate rewards
        rewards: list[float] = []
        for i, dp in enumerate(decision_points):
            next_dp = decision_points[i + 1] if i + 1 < n else None
            r = self._compute_intermediate_reward(
                episode=episode,
                this_dp_action_msg=dp.action,
                next_dp_action_msg=next_dp.action if next_dp is not None else None,
                turn_number=dp.state.turn_number,
                is_interrupt_mode=is_interrupt_mode,
            )
            rewards.append(r)

        # Terminal reward: quality score + expired-item penalty applied to
        # the final step only.
        if rewards:
            terminal_quality = (
                episode.outcome.quality_score
                if episode.outcome.quality_score is not None
                else 0.0
            )
            unconsumed = sum(
                1 for inj in episode.context_injections if not inj.consumed
            )
            rewards[-1] += terminal_quality + unconsumed * R_EXPIRED

        # Discounted returns (backward pass)
        returns = [0.0] * n
        running = 0.0
        for i in range(n - 1, -1, -1):
            running = rewards[i] + DISCOUNT_GAMMA * running
            returns[i] = running

        examples: list[TrainingExample] = []
        for i in range(n):
            examples.append(
                TrainingExample(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_states[i],
                    done=(i == n - 1),
                    discounted_return=returns[i],
                    episode_id=episode.episode_id,
                    decision_index=i,
                )
            )
        return examples

    def process_episodes(
        self, episodes: list[Episode]
    ) -> list[TrainingExample]:
        """Process multiple episodes; flatten their training examples."""
        out: list[TrainingExample] = []
        for ep in episodes:
            out.extend(self.process_episode(ep))
        return out

    # ------------------------------------------------------------------
    # State vector construction
    # ------------------------------------------------------------------

    def _build_state_vector(
        self,
        state: ReplayState,
        action_msg: Message,
    ) -> np.ndarray:
        """Build the 108-dim state vector at a decision point.

        ``state`` reflects everything before ``action_msg`` (the assistant
        message about to be produced). No information from after
        ``action_msg.timestamp_ms`` is used.
        """
        vec = np.zeros(STATE_DIM, dtype=np.float32)

        messages = list(state.messages)
        user_events = list(state.user_events)
        tool_history = list(state.completed_tool_invocations)
        queue_snapshot = self._reconstruct_queue_state(
            state.pending_injections, action_msg.timestamp_ms
        )

        # Latency context features need ContextFeatures from the
        # conversation length BEFORE the assistant message.
        ctx_features = ContextFeatures(
            conversation_length_tokens=sum(m.token_count for m in messages),
            conversation_turn_count=state.turn_number,
        )
        latency_predictions = {
            tool_id: self._latency_model.predict_tool_duration(
                tool_id, ctx_features
            ).mean_ms
            for tool_id in _TOOL_VOCAB
        }

        # Use the configured turn limit — never the episode's actual final
        # turn count, which would leak future information (the label) into
        # the completion-fraction features.
        max_turns = self._max_turns

        # 0–63: encoder output
        encoded = self._encoder.encode(
            messages,
            user_events=user_events,
            tool_history=tool_history,
            queue_state=queue_snapshot,
            latency_predictions=latency_predictions,
            turn_number=state.turn_number,
            max_turns=max_turns,
        )
        vec[_OFF_ENCODER : _OFF_ENCODER + FEATURE_DIM] = encoded

        # 64–81: signal classifier output
        signals = SignalClassifier.classify(messages, user_events)
        vec[_OFF_SIGNALS : _OFF_SIGNALS + SIGNAL_DIM] = signals.to_array()

        # 82–93: one-hot of last 3 tool invocations
        vec[_OFF_TOOL_ONEHOT : _OFF_TOOL_ONEHOT + _TOOL_HISTORY_ONEHOT] = (
            self._encode_tool_history_onehot(tool_history)
        )

        # 94–96: time since each of the last 3 invocations
        vec[_OFF_TOOL_TIMES : _OFF_TOOL_TIMES + _TOOL_HISTORY_TIME] = (
            self._encode_tool_history_times(tool_history, action_msg.timestamp_ms)
        )

        # 97–102: queue state (shared encoding with the StateEncoder slice)
        vec[_OFF_QUEUE : _OFF_QUEUE + _QUEUE_DIMS] = encode_queue_state(queue_snapshot)

        # 103–105: latency predictions
        for i, tool_id in enumerate(_TOOL_VOCAB):
            vec[_OFF_LATENCY + i] = _cap_norm(
                latency_predictions[tool_id], _LATENCY_MS_CAP
            )

        # 106–107: turn + completion fraction
        vec[_OFF_PROGRESS] = _cap_norm(state.turn_number, _TURN_CAP)
        vec[_OFF_PROGRESS + 1] = min(state.turn_number / max_turns, 1.0)

        return vec

    @staticmethod
    def _encode_tool_history_onehot(
        tool_history: list[ToolInvocation],
    ) -> np.ndarray:
        """One-hot encode the last ``_TOOL_HISTORY_SLOTS`` tool invocations.

        Layout: consecutive ``_TOOL_ONEHOT_WIDTH``-wide chunks, ordered
        most-recent-first. Each chunk: one index per tool in
        ``_TOOL_VOCAB`` order, then a trailing "none/unknown" index.
        """
        out = np.zeros(_TOOL_HISTORY_ONEHOT, dtype=np.float32)
        none_idx = len(_TOOL_VOCAB)
        recent = list(tool_history)[-_TOOL_HISTORY_SLOTS:][::-1]  # most recent first
        for slot in range(_TOOL_HISTORY_SLOTS):
            base = slot * _TOOL_ONEHOT_WIDTH
            if slot < len(recent):
                tool_id = recent[slot].tool_id
                if tool_id in _TOOL_VOCAB:
                    out[base + _TOOL_VOCAB.index(tool_id)] = 1.0
                else:
                    out[base + none_idx] = 1.0
            else:
                out[base + none_idx] = 1.0  # "none"
        return out

    @staticmethod
    def _encode_tool_history_times(
        tool_history: list[ToolInvocation],
        cutoff_ms: int,
    ) -> np.ndarray:
        """Time since each recent invocation (normalized seconds).

        Slot order matches the one-hot encoding (most recent first).
        Elapsed seconds are capped at ``_TOOL_RECENCY_SECONDS_CAP``.
        Missing slots are 1.0 (treated as "long ago / never").
        """
        out = np.ones(_TOOL_HISTORY_TIME, dtype=np.float32)
        recent = list(tool_history)[-_TOOL_HISTORY_SLOTS:][::-1]
        for slot, inv in enumerate(recent):
            elapsed_s = max(0.0, (cutoff_ms - inv.completed_at_ms) / 1000.0)
            out[slot] = _cap_norm(elapsed_s, _TOOL_RECENCY_SECONDS_CAP)
        return out

    @staticmethod
    def _reconstruct_queue_state(
        pending: tuple[ContextInjection, ...],
        cutoff_ms: int,
    ) -> QueueState:
        """Reconstruct a queue snapshot from pending injections at this point."""
        if not pending:
            return QueueState(
                depth=0,
                token_total=0,
                max_priority=None,
                time_since_last_drain=0.0,
                pending_tool_count=0,
                estimated_next_arrival=0.0,
            )
        max_p_int = max(inj.priority for inj in pending)
        # Time since most-recent enqueue (bounded by what we know — the
        # injections' timestamps).
        latest_ts = max(inj.timestamp_ms for inj in pending)
        time_since_last_ms = max(0, cutoff_ms - latest_ts)
        return QueueState(
            depth=len(pending),
            token_total=sum(inj.token_count for inj in pending),
            max_priority=Priority(max_p_int) if max_p_int <= 3 else Priority.CRITICAL,
            time_since_last_drain=time_since_last_ms / 1000.0,
            pending_tool_count=len({inj.source_tool_id for inj in pending}),
            estimated_next_arrival=0.0,
        )

    # ------------------------------------------------------------------
    # Action inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_actions(episode: Episode) -> dict[int, int]:
        """Infer the controller's action for each turn from tool invocations.

        Returns a dict ``{turn_number: action_index}``. Missing turns
        default to DO_NOTHING.
        """
        # Reverse mapping: tool_id -> Action
        tool_to_action: dict[str, Action] = {v: k for k, v in TOOL_IDS.items()}

        # Pair each user message with the next assistant message; any tool
        # invocation falling in that window is the action for that turn.
        actions: dict[int, int] = {}
        turn_no = 0
        i = 0
        msgs = episode.messages
        while i < len(msgs):
            if msgs[i].role != "user":
                i += 1
                continue
            turn_no += 1
            user_ts = msgs[i].timestamp_ms

            # Find next assistant message
            j = i + 1
            while j < len(msgs) and msgs[j].role != "assistant":
                j += 1
            if j >= len(msgs):
                break
            assistant_ts = msgs[j].timestamp_ms

            # Find a tool invocation in (user_ts, assistant_ts]
            found_action = Action.DO_NOTHING
            for inv in episode.tool_invocations:
                if user_ts < inv.invoked_at_ms <= assistant_ts:
                    action = tool_to_action.get(inv.tool_id, Action.DO_NOTHING)
                    found_action = action
                    break
            actions[turn_no] = _ACTION_INDEX[found_action]
            i = j + 1
        return actions

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_intermediate_reward(
        episode: Episode,
        this_dp_action_msg: Message,
        next_dp_action_msg: Message | None,
        turn_number: int,
        is_interrupt_mode: bool,
    ) -> float:
        """Compute the per-step intermediate reward for one decision point.

        Events between this assistant message and the next assistant
        message are attributed to this turn.
        """
        start_ts = this_dp_action_msg.timestamp_ms
        end_ts = (
            next_dp_action_msg.timestamp_ms
            if next_dp_action_msg is not None
            else _max_episode_timestamp(episode)
        )

        reward = 0.0

        # User events in this window
        followup_msg: Message | None = None
        for evt in episode.user_events:
            # Use a half-open window for events, but include events at the
            # final boundary on the last step.
            in_window = (
                start_ts <= evt.timestamp_ms < end_ts
                if next_dp_action_msg is not None
                else start_ts <= evt.timestamp_ms <= end_ts
            )
            if not in_window:
                continue
            if evt.event_type == UserEventType.STOP:
                reward += R_STOP
            elif evt.event_type == UserEventType.FOLLOW_UP:
                # Find the user message corresponding to this follow-up
                # (the next user message after the assistant action).
                if followup_msg is None:
                    followup_msg = _find_next_user_msg(episode.messages, start_ts)
                followup_type = (
                    FollowUpClassifier.classify(followup_msg.content, episode.messages)
                    if followup_msg is not None
                    else FollowUpType.NEW_TASK
                )
                if followup_type == FollowUpType.ENCOURAGEMENT:
                    reward += R_ENCOURAGEMENT
                elif followup_type in (
                    FollowUpType.REDIRECT,
                    FollowUpType.CORRECTION,
                ):
                    reward += R_REDIRECT_CORRECTION
                else:
                    reward += R_FOLLOWUP

        # Token cost: assistant message + tools invoked during this turn
        # (between the previous assistant message and this one).
        prev_action_ts = _prev_assistant_ts(episode.messages, start_ts)
        tool_tokens = sum(
            inv.input_tokens + inv.output_tokens
            for inv in episode.tool_invocations
            if prev_action_ts < inv.invoked_at_ms <= start_ts
        )
        total_tokens = this_dp_action_msg.token_count + tool_tokens
        reward += R_TOKEN_PER_100 * (total_tokens / 100.0)

        # Drains consumed at this turn: count injections consumed_at_turn == turn_number
        drains = sum(
            1
            for inj in episode.context_injections
            if inj.consumed and inj.consumed_at_turn == turn_number
        )
        reward += drains * R_DRAIN_CONSUMED

        # Interrupts: in INTERRUPT mode, an interrupt is signaled by an
        # injection enqueued AND consumed within the same turn (a forced
        # mid-turn drain). Each such injection contributes one penalty.
        if is_interrupt_mode:
            interrupts = sum(
                1
                for inj in episode.context_injections
                if (
                    inj.consumed
                    and inj.consumed_at_turn == turn_number
                    and _enqueued_at_turn(inj, episode) == turn_number
                )
            )
            reward += interrupts * R_INTERRUPT

        return reward

    # ------------------------------------------------------------------
    # PyTorch interop (optional)
    # ------------------------------------------------------------------

    @staticmethod
    def to_torch_dataset(examples: list[TrainingExample]) -> TensorDataset:
        """Build a ``torch.utils.data.TensorDataset`` from training examples.

        The dataset yields ``(state, action, reward, next_state, done,
        discounted_return)`` tensors per item, suitable for use with
        ``torch.utils.data.DataLoader``.

        Raises
        ------
        ImportError
            If torch is not installed.
        """
        import torch
        from torch.utils.data import TensorDataset

        if not examples:
            empty_state = torch.zeros((0, STATE_DIM), dtype=torch.float32)
            return TensorDataset(
                empty_state,
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32),
                empty_state,
                torch.zeros((0,), dtype=torch.bool),
                torch.zeros((0,), dtype=torch.float32),
            )

        states = torch.from_numpy(np.stack([e.state for e in examples])).float()
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in examples])
        ).float()
        actions = torch.tensor([e.action for e in examples], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in examples], dtype=torch.float32)
        dones = torch.tensor([e.done for e in examples], dtype=torch.bool)
        returns = torch.tensor(
            [e.discounted_return for e in examples], dtype=torch.float32
        )
        return TensorDataset(states, actions, rewards, next_states, dones, returns)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _max_episode_timestamp(episode: Episode) -> int:
    """Return the latest timestamp anywhere in the episode (for terminal window)."""
    candidates: list[int] = []
    candidates.extend(m.timestamp_ms for m in episode.messages)
    candidates.extend(e.timestamp_ms for e in episode.user_events)
    candidates.extend(c.timestamp_ms for c in episode.context_injections)
    candidates.extend(t.completed_at_ms for t in episode.tool_invocations)
    return max(candidates) if candidates else 0


def _find_next_user_msg(messages: list[Message], after_ts: int) -> Message | None:
    """Return the first user message with timestamp_ms > after_ts."""
    for m in messages:
        if m.role == "user" and m.timestamp_ms > after_ts:
            return m
    return None


def _prev_assistant_ts(messages: list[Message], before_ts: int) -> int:
    """Return the timestamp_ms of the assistant message strictly before before_ts.

    Returns -1 if there is none, so the comparison ``prev_ts < x`` admits
    everything from the start of the episode.
    """
    prev = -1
    for m in messages:
        if m.role == "assistant" and m.timestamp_ms < before_ts:
            prev = m.timestamp_ms
        elif m.timestamp_ms >= before_ts:
            break
    return prev


def _enqueued_at_turn(inj: ContextInjection, episode: Episode) -> int:
    """Best-effort estimate of the turn during which an injection was enqueued.

    Counts user messages with timestamp <= injection timestamp.
    """
    return sum(
        1 for m in episode.messages if m.role == "user" and m.timestamp_ms <= inj.timestamp_ms
    )
