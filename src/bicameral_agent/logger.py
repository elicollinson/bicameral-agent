"""Conversation logger that captures all events needed for training episodes.

Wraps LLM conversations to produce validated Episode objects. Thread-safe
via a single lock guarding all mutable state.
"""

from __future__ import annotations

import threading
import time
import uuid

from bicameral_agent.queue import QueueItem
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


class ConversationLogger:
    """Thread-safe logger that accumulates episode events and produces validated Episodes.

    Uses a hybrid timestamp scheme: wall-clock epoch captured at construction,
    with monotonic offsets thereafter for ordering guarantees.
    """

    def __init__(self, metadata: dict | None = None) -> None:
        self._epoch_ms = int(time.time() * 1000)
        self._mono_origin_ns = time.monotonic_ns()

        self._messages: list[Message] = []
        self._user_events: list[UserEvent] = []
        self._context_injections: list[ContextInjection] = []
        self._tool_invocations: list[tuple[int, ToolInvocation]] = []  # (index, invocation)

        self._pending_tools: dict[int, tuple[str, int, int, int | None]] = {}
        self._next_tool_index = 0
        self._next_injection_index = 0
        self._wasted_tokens = 0

        self._lock = threading.Lock()
        self._finalized = False
        self._metadata: dict = metadata or {}

    def set_metadata(self, key: str, value: object) -> None:
        """Set a metadata key-value pair.

        Args:
            key: Metadata key.
            value: Metadata value (must be JSON-serializable).
        """
        with self._lock:
            self._check_not_finalized()
            self._metadata[key] = value

    def _now_ms(self) -> int:
        return self._epoch_ms + (time.monotonic_ns() - self._mono_origin_ns) // 1_000_000

    def _check_not_finalized(self) -> None:
        if self._finalized:
            raise RuntimeError("Cannot log events after finalize()")

    def log_message(self, role: str, content: str, token_count: int) -> None:
        """Append a message to the episode.

        Args:
            role: Sender role (e.g. 'user', 'assistant', 'system').
            content: Text content of the message.
            token_count: Number of tokens in this message.
        """
        with self._lock:
            self._check_not_finalized()
            ts = self._now_ms()
            self._messages.append(
                Message(
                    role=role,
                    content=content,
                    timestamp_ms=ts,
                    token_count=token_count,
                )
            )

    def replace_last_message(self, content: str, token_count: int) -> None:
        """Replace the most recent message's content (e.g. after regeneration).

        The replacement keeps the original role but takes a fresh timestamp,
        reflecting that the final content was produced after any intervening
        tool/injection events.

        Args:
            content: New text content for the message.
            token_count: Number of tokens in the new content.

        Raises:
            ValueError: If no message has been logged yet.
        """
        with self._lock:
            self._check_not_finalized()
            if not self._messages:
                raise ValueError("No message to replace")
            self._messages[-1] = Message(
                role=self._messages[-1].role,
                content=content,
                timestamp_ms=self._now_ms(),
                token_count=token_count,
            )

    def log_wasted_tokens(self, token_count: int) -> None:
        """Record tokens spent on discarded generations (interrupt/regeneration).

        Wasted tokens are folded into the episode outcome's total_tokens so
        that conditions which regenerate are not undercounted.

        Args:
            token_count: Number of tokens in the discarded generation.
        """
        with self._lock:
            self._check_not_finalized()
            self._wasted_tokens += token_count

    @property
    def wasted_tokens(self) -> int:
        """Total tokens recorded via log_wasted_tokens() so far."""
        with self._lock:
            return self._wasted_tokens

    def log_user_event(
        self, event_type: UserEventType, metadata: dict | None = None
    ) -> None:
        """Record a user-initiated event.

        Args:
            event_type: The type of user event.
            metadata: Optional metadata dict for the event.
        """
        with self._lock:
            self._check_not_finalized()
            ts = self._now_ms()
            self._user_events.append(
                UserEvent(
                    event_type=event_type,
                    timestamp_ms=ts,
                    metadata=metadata or {},
                )
            )

    def log_tool_invocation(
        self, tool_id: str, input_tokens: int, turn: int | None = None
    ) -> int:
        """Record the start of a tool invocation.

        Args:
            tool_id: Identifier of the tool being invoked.
            input_tokens: Number of tokens in the tool's input.
            turn: 1-based conversational turn making the invocation, if known.
                Recorded on the ToolInvocation so consumers can attribute the
                invocation to its turn without timestamp heuristics.

        Returns:
            An opaque invocation index to pass to log_tool_completion().
        """
        with self._lock:
            self._check_not_finalized()
            ts = self._now_ms()
            idx = self._next_tool_index
            self._next_tool_index += 1
            self._pending_tools[idx] = (tool_id, input_tokens, ts, turn)
            return idx

    def log_tool_completion(
        self,
        invocation_index: int,
        output_tokens: int,
        result_deposited: bool = False,
        budget_exceeded: bool = False,
    ) -> None:
        """Record the completion of a tool invocation.

        Args:
            invocation_index: Index returned by log_tool_invocation().
            output_tokens: Number of tokens in the tool's output.
            result_deposited: Whether the result was deposited into the conversation.
            budget_exceeded: Whether the invocation aborted on budget exhaustion.

        Raises:
            ValueError: If invocation_index is not a pending tool.
        """
        with self._lock:
            self._check_not_finalized()
            ts = self._now_ms()
            pending = self._pending_tools.pop(invocation_index, None)
            if pending is None:
                raise ValueError(
                    f"Unknown invocation index: {invocation_index}"
                )
            tool_id, input_tokens, invoked_at_ms, turn = pending
            self._tool_invocations.append((
                invocation_index,
                ToolInvocation(
                    tool_id=tool_id,
                    invoked_at_ms=invoked_at_ms,
                    completed_at_ms=ts,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    result_deposited=result_deposited,
                    budget_exceeded=budget_exceeded,
                    turn=turn,
                ),
            ))

    def log_context_injection(self, item: QueueItem) -> int:
        """Record a context injection from a queue item.

        Args:
            item: The queue item being deposited; converted to the schema-level
                ContextInjection via ContextInjection.from_queue_item().

        Returns:
            An opaque injection index to pass to log_injection_consumed().
        """
        with self._lock:
            self._check_not_finalized()
            ts = self._now_ms()
            idx = self._next_injection_index
            self._next_injection_index += 1
            self._context_injections.append(
                ContextInjection.from_queue_item(item, timestamp_ms=ts)
            )
            return idx

    def log_injection_consumed(self, injection_index: int, turn_number: int) -> None:
        """Mark a context injection as consumed.

        Args:
            injection_index: Index returned by log_context_injection().
            turn_number: The turn number at which this injection was consumed.

        Raises:
            ValueError: If injection_index is invalid or already consumed.
        """
        with self._lock:
            self._check_not_finalized()
            if injection_index < 0 or injection_index >= len(self._context_injections):
                raise ValueError(f"Invalid injection index: {injection_index}")
            inj = self._context_injections[injection_index]
            if inj.consumed:
                raise ValueError(
                    f"Injection {injection_index} already consumed"
                )
            self._context_injections[injection_index] = inj.model_copy(
                update={"consumed": True, "consumed_at_turn": turn_number}
            )

    def snapshot(self) -> Episode:
        """Return a non-finalizing Episode view of everything logged so far.

        Used by controllers that need the live episode context at decision
        time (issue #29): the returned Episode contains the messages, user
        events, context injections, and *completed* tool invocations logged
        so far — the same records a post-hoc
        :class:`~bicameral_agent.replay.EpisodeReplayer` would see at this
        point. Pending (started, uncompleted) tool invocations are excluded,
        matching how they would appear mid-flight. The outcome carries
        placeholder totals; it exists only to satisfy the Episode schema.

        Does not finalize: logging can continue afterwards.
        """
        with self._lock:
            tool_invocations = [
                t
                for _, t in sorted(
                    self._tool_invocations, key=lambda pair: (pair[1].invoked_at_ms, pair[0])
                )
            ]
            return Episode(
                messages=list(self._messages),
                user_events=list(self._user_events),
                context_injections=list(self._context_injections),
                tool_invocations=tool_invocations,
                outcome=EpisodeOutcome(
                    quality_score=None,
                    total_tokens=0,
                    total_turns=sum(1 for m in self._messages if m.role == "user"),
                    wall_clock_ms=0,
                ),
                metadata=dict(self._metadata),
            )

    def finalize(self, quality_score: float | None = None) -> Episode:
        """Finalize the episode and return a validated Episode object.

        Args:
            quality_score: Optional quality score in [0.0, 1.0].

        Returns:
            A validated Episode containing all logged events.

        Raises:
            RuntimeError: If called twice or if tools are still pending.
        """
        with self._lock:
            if self._finalized:
                raise RuntimeError("finalize() already called")
            if self._pending_tools:
                pending_ids = list(self._pending_tools.keys())
                raise RuntimeError(
                    f"Cannot finalize with pending tool invocations: {pending_ids}"
                )
            self._finalized = True

            # Sort tool invocations by (invoked_at_ms, original_index) to handle
            # out-of-order completions and same-millisecond invocations
            tool_invocations = [
                t for _, t in sorted(
                    self._tool_invocations, key=lambda pair: (pair[1].invoked_at_ms, pair[0])
                )
            ]

            # Compute outcome
            total_tokens = (
                sum(m.token_count for m in self._messages)
                + sum(t.input_tokens + t.output_tokens for t in tool_invocations)
                + sum(
                    c.token_count
                    for c in self._context_injections
                    if c.consumed
                )
                + self._wasted_tokens
            )
            total_turns = sum(1 for m in self._messages if m.role == "user")

            all_timestamps: list[int] = []
            all_timestamps.extend(m.timestamp_ms for m in self._messages)
            all_timestamps.extend(e.timestamp_ms for e in self._user_events)
            all_timestamps.extend(c.timestamp_ms for c in self._context_injections)
            all_timestamps.extend(t.invoked_at_ms for t in tool_invocations)
            all_timestamps.extend(t.completed_at_ms for t in tool_invocations)

            if all_timestamps:
                wall_clock_ms = max(all_timestamps) - min(all_timestamps)
            else:
                wall_clock_ms = 0

            episode = Episode(
                episode_id=str(uuid.uuid4()),
                messages=list(self._messages),
                user_events=list(self._user_events),
                context_injections=list(self._context_injections),
                tool_invocations=tool_invocations,
                outcome=EpisodeOutcome(
                    quality_score=quality_score,
                    total_tokens=total_tokens,
                    total_turns=total_turns,
                    wall_clock_ms=wall_clock_ms,
                ),
                metadata=dict(self._metadata),
            )

        # Validate outside the lock (read-only operation on immutable episode)
        result = EpisodeValidator().validate(episode)
        if not result.is_valid:
            raise RuntimeError(
                f"Episode validation failed: {'; '.join(result.errors)}"
            )
        return episode
