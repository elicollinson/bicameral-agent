"""Conversation logger that captures all events needed for training episodes.

Wraps LLM conversations to produce validated Episode objects. Thread-safe
via a single lock guarding all mutable state.
"""

from __future__ import annotations

import threading
import time
import uuid

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

        self._pending_tools: dict[int, tuple[str, int, int]] = {}
        self._next_tool_index = 0
        self._next_injection_index = 0

        self._lock = threading.Lock()
        self._finalized = False
        self._metadata: dict = metadata if metadata is not None else {}

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
                    metadata=metadata if metadata is not None else {},
                )
            )

    def log_tool_invocation(self, tool_id: str, input_tokens: int) -> int:
        """Record the start of a tool invocation.

        Args:
            tool_id: Identifier of the tool being invoked.
            input_tokens: Number of tokens in the tool's input.

        Returns:
            An opaque invocation index to pass to log_tool_completion().
        """
        with self._lock:
            self._check_not_finalized()
            ts = self._now_ms()
            idx = self._next_tool_index
            self._next_tool_index += 1
            self._pending_tools[idx] = (tool_id, input_tokens, ts)
            return idx

    def log_tool_completion(
        self,
        invocation_index: int,
        output_tokens: int,
        result_deposited: bool = False,
    ) -> None:
        """Record the completion of a tool invocation.

        Args:
            invocation_index: Index returned by log_tool_invocation().
            output_tokens: Number of tokens in the tool's output.
            result_deposited: Whether the result was deposited into the conversation.

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
            tool_id, input_tokens, invoked_at_ms = pending
            self._tool_invocations.append((
                invocation_index,
                ToolInvocation(
                    tool_id=tool_id,
                    invoked_at_ms=invoked_at_ms,
                    completed_at_ms=ts,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    result_deposited=result_deposited,
                ),
            ))

    def log_context_injection(
        self,
        content: str,
        source_tool_id: str,
        priority: int,
        token_count: int,
    ) -> int:
        """Record a context injection.

        Args:
            content: The injected context text.
            source_tool_id: Identifier of the tool that produced this context.
            priority: Priority level (higher = more important).
            token_count: Number of tokens in the injected content.

        Returns:
            An opaque injection index to pass to log_injection_consumed().
        """
        with self._lock:
            self._check_not_finalized()
            ts = self._now_ms()
            idx = self._next_injection_index
            self._next_injection_index += 1
            self._context_injections.append(
                ContextInjection(
                    content=content,
                    source_tool_id=source_tool_id,
                    priority=priority,
                    timestamp_ms=ts,
                    token_count=token_count,
                )
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
