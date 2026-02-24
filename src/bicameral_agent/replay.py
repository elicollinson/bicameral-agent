"""Episode replay engine for reconstructing state at any point in time.

Provides tools for debugging, training data validation, and offline evaluation
by reconstructing the full conversation state at any turn or timestamp.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator
from dataclasses import dataclass, field

from bicameral_agent.schema import (
    ContextInjection,
    Episode,
    Message,
    ToolInvocation,
    UserEvent,
)


@dataclass(frozen=True, slots=True)
class ReplayState:
    """Snapshot of the full conversation state at a specific point."""

    turn_number: int
    """Current turn number (count of user messages seen so far)."""

    messages: tuple[Message, ...]
    """Conversation history up to this point."""

    pending_injections: tuple[ContextInjection, ...]
    """Context injections not yet consumed at this point."""

    consumed_injections: tuple[ContextInjection, ...]
    """Context injections already consumed at this point."""

    active_tool_invocations: tuple[ToolInvocation, ...]
    """Tool invocations that have started but not yet completed."""

    completed_tool_invocations: tuple[ToolInvocation, ...]
    """Tool invocations that have completed."""

    user_events: tuple[UserEvent, ...]
    """User events received up to this point."""


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    """A point where the controller made a decision, with full context."""

    state: ReplayState
    """The full conversation state at this decision point."""

    action: Message
    """The assistant message (action) taken at this point."""


class EpisodeReplayer:
    """Reconstructs episode state at any turn or timestamp.

    Takes a completed Episode and provides efficient random-access to
    the conversation state at any point during the episode.
    """

    def __init__(self, episode: Episode) -> None:
        self._episode = episode

        # Precompute turn boundaries: _turn_indices[t] = index into messages
        # of the (t+1)-th user message. Turn 0 = after first user message.
        self._turn_indices: list[int] = []
        for i, msg in enumerate(episode.messages):
            if msg.role == "user":
                self._turn_indices.append(i)

        # Precompute sorted message timestamps for bisect lookups
        self._msg_timestamps = [m.timestamp_ms for m in episode.messages]

    @property
    def total_turns(self) -> int:
        """Total number of turns (user messages) in the episode."""
        return len(self._turn_indices)

    def state_at_turn(self, n: int) -> ReplayState:
        """Return the full conversation state at turn N.

        Turn 0 is the state after the first user message.
        Turn total_turns-1 is the final turn matching the complete episode.

        Args:
            n: Turn number (0-indexed).

        Returns:
            ReplayState at that turn.

        Raises:
            IndexError: If n is out of range.
        """
        if n < 0 or n >= len(self._turn_indices):
            raise IndexError(
                f"Turn {n} out of range [0, {len(self._turn_indices) - 1}]"
            )

        # Messages up through the n-th user message
        msg_end = self._turn_indices[n] + 1
        messages = self._episode.messages[:msg_end]
        cutoff_ms = messages[-1].timestamp_ms

        return self._build_state(n + 1, messages, cutoff_ms)

    def state_at_time(self, ms: int) -> ReplayState:
        """Return the state at a specific timestamp.

        Returns the state including all messages with timestamp <= ms.

        Args:
            ms: Timestamp in milliseconds since epoch.

        Returns:
            ReplayState at that time.
        """
        # Find how many messages have timestamp <= ms
        msg_end = bisect.bisect_right(self._msg_timestamps, ms)
        messages = self._episode.messages[:msg_end]

        # Count user turns in the included messages
        turn_number = sum(1 for m in messages if m.role == "user")

        return self._build_state(turn_number, messages, ms)

    def iter_decision_points(self) -> Iterator[DecisionPoint]:
        """Yield each point where the controller made a decision.

        A decision point is each assistant message, with the state being
        everything up to (but not including) that assistant message.

        Yields:
            DecisionPoint with full state and the action taken.
        """
        turn_number = 0
        for i, msg in enumerate(self._episode.messages):
            if msg.role == "user":
                turn_number += 1
            elif msg.role == "assistant":
                # State is everything before this assistant message
                prior_messages = self._episode.messages[:i]
                cutoff_ms = msg.timestamp_ms
                state = self._build_state(turn_number, prior_messages, cutoff_ms)
                yield DecisionPoint(state=state, action=msg)

    def _build_state(
        self,
        turn_number: int,
        messages: list[Message],
        cutoff_ms: int,
    ) -> ReplayState:
        """Build a ReplayState from the given parameters.

        Args:
            turn_number: Number of user turns seen.
            messages: Messages included in the state.
            cutoff_ms: Timestamp cutoff for injections, tools, and events.

        Returns:
            A fully populated ReplayState.
        """
        episode = self._episode

        # Partition context injections by time and consumption status
        pending: list[ContextInjection] = []
        consumed: list[ContextInjection] = []
        for inj in episode.context_injections:
            if inj.timestamp_ms > cutoff_ms:
                continue
            if inj.consumed and inj.consumed_at_turn is not None and inj.consumed_at_turn < turn_number:
                consumed.append(inj)
            else:
                pending.append(inj)

        # Partition tool invocations
        active: list[ToolInvocation] = []
        completed: list[ToolInvocation] = []
        for tool in episode.tool_invocations:
            if tool.invoked_at_ms > cutoff_ms:
                continue
            if tool.completed_at_ms <= cutoff_ms:
                completed.append(tool)
            else:
                active.append(tool)

        # Filter user events
        events = [e for e in episode.user_events if e.timestamp_ms <= cutoff_ms]

        return ReplayState(
            turn_number=turn_number,
            messages=tuple(messages),
            pending_injections=tuple(pending),
            consumed_injections=tuple(consumed),
            active_tool_invocations=tuple(active),
            completed_tool_invocations=tuple(completed),
            user_events=tuple(events),
        )
