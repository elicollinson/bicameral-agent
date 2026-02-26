"""Context injection priority queue between tool primitives and the conscious loop.

Provides a thread-safe priority queue that accepts context items from multiple
tool sources, handles deduplication, expiry, and priority-ordered draining.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Priority(enum.IntEnum):
    """Priority levels for queue items, ordered low to high.

    Uses IntEnum so that comparisons (critical > high > medium > low)
    work naturally with Python comparison operators.
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class QueueItem(BaseModel):
    """An item in the context injection queue.

    This is the queue-internal representation, separate from the
    schema-level ContextInjection which records consumed/unconsumed
    state for episode logging.
    """

    content: str
    """The context text to inject."""

    priority: Priority
    """Priority level for consumption ordering."""

    source_tool_id: str
    """Identifier of the tool that produced this context."""

    timestamp: float = Field(default_factory=time.monotonic)
    """Monotonic timestamp when the item was created."""

    token_count: int = Field(ge=0)
    """Number of tokens in the content."""

    expiry_turns: int = Field(default=10, ge=1)
    """Number of turns after which this item expires."""

    dedup_key: str | None = None
    """Optional key for deduplication. Items with the same dedup_key
    are deduplicated: newer or higher-priority replaces older."""

    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this queue item."""

    enqueued_at_turn: int = Field(default=0, ge=0)
    """Turn number when this item was enqueued."""


@dataclass(frozen=True, slots=True)
class QueueState:
    """Snapshot of the queue's current state vector.

    Used by the controller to make decisions about context consumption
    and scheduling.
    """

    depth: int
    """Number of items currently in the queue."""

    token_total: int
    """Sum of token_count across all queued items."""

    max_priority: Priority | None
    """Highest priority among queued items, or None if empty."""

    time_since_last_drain: float
    """Seconds since the last dequeue_all() call."""

    pending_tool_count: int
    """Number of distinct source_tool_ids with items in the queue."""

    estimated_next_arrival: float
    """Estimated seconds until the next enqueue, based on arrival rate EMA."""


@dataclass(frozen=True, slots=True)
class InterruptConfig:
    """Configuration for interrupt threshold checks.

    Defines the thresholds at which the queue should signal the conscious
    loop to interrupt the current generation and retry with injected context.
    """

    count_threshold: int = 5
    """Interrupt if queue depth reaches this count."""

    priority_threshold: Priority = Priority.CRITICAL
    """Interrupt if any item has at least this priority."""

    token_threshold: int = 1000
    """Interrupt if total pending tokens reach this count."""


class ContextQueue:
    """Thread-safe priority queue for context injections.

    Sits between tool primitives and the conscious loop. Supports
    concurrent enqueue from multiple tools, deduplication by key,
    priority-ordered draining, and turn-based expiry.
    """

    _EMA_ALPHA = 0.3

    def __init__(self) -> None:
        self._items: dict[str, QueueItem] = {}
        self._dedup_index: dict[str, str] = {}
        self._lock = threading.Lock()
        self._last_drain_time: float = time.monotonic()
        self._last_enqueue_time: float | None = None
        self._arrival_interval_ema: float = 0.0
        self._enqueue_count: int = 0
        self._frozen: bool = False
        self._wasted_tokens: int = 0

    def enqueue(self, item: QueueItem) -> None:
        """Add an item to the queue, handling deduplication.

        If the item has a dedup_key that matches an existing item, the
        newer or higher-priority item replaces the older one. If priorities
        are equal, the newer item wins.

        Args:
            item: The queue item to add.
        """
        with self._lock:
            now = time.monotonic()

            # Update arrival rate EMA
            if self._last_enqueue_time is not None:
                interval = now - self._last_enqueue_time
                if self._enqueue_count <= 1:
                    self._arrival_interval_ema = interval
                else:
                    self._arrival_interval_ema = (
                        self._EMA_ALPHA * interval
                        + (1 - self._EMA_ALPHA) * self._arrival_interval_ema
                    )
            self._last_enqueue_time = now
            self._enqueue_count += 1

            # Deduplication
            if item.dedup_key is not None:
                existing_item_id = self._dedup_index.get(item.dedup_key)
                if existing_item_id is not None:
                    existing = self._items.get(existing_item_id)
                    if existing is not None:
                        if item.priority >= existing.priority:
                            del self._items[existing_item_id]
                            self._items[item.item_id] = item
                            self._dedup_index[item.dedup_key] = item.item_id
                        return
                self._dedup_index[item.dedup_key] = item.item_id

            self._items[item.item_id] = item

    def dequeue_all(self) -> list[QueueItem]:
        """Return all items ordered by priority descending, then FIFO. Clears the queue.

        Returns:
            Items sorted by priority (critical first), with FIFO order within
            the same priority level (based on timestamp).
        """
        with self._lock:
            self._last_drain_time = time.monotonic()
            if not self._items:
                return []
            items = list(self._items.values())
            self._items.clear()
            self._dedup_index.clear()

        items.sort(key=lambda x: (-x.priority, x.timestamp))
        return items

    def peek(self) -> list[QueueItem]:
        """Non-destructive view of all items, ordered by priority descending then FIFO.

        Returns:
            A copy of all items in priority order. The queue is not modified.
        """
        with self._lock:
            if not self._items:
                return []
            items = list(self._items.values())

        items.sort(key=lambda x: (-x.priority, x.timestamp))
        return items

    def expire_stale(self, current_turn: int) -> list[QueueItem]:
        """Remove items that have exceeded their expiry_turns.

        An item is stale if ``current_turn - item.enqueued_at_turn >= item.expiry_turns``.

        Args:
            current_turn: The current conversation turn number.

        Returns:
            List of expired items that were removed.
        """
        expired: list[QueueItem] = []
        with self._lock:
            stale_ids = [
                item_id
                for item_id, item in self._items.items()
                if current_turn - item.enqueued_at_turn >= item.expiry_turns
            ]
            for item_id in stale_ids:
                item = self._items.pop(item_id)
                expired.append(item)
                if (
                    item.dedup_key is not None
                    and self._dedup_index.get(item.dedup_key) == item_id
                ):
                    del self._dedup_index[item.dedup_key]

        for item in expired:
            logger.info(
                "Expired queue item: item_id=%s, dedup_key=%s, priority=%s, "
                "enqueued_at_turn=%d, expiry_turns=%d, current_turn=%d",
                item.item_id,
                item.dedup_key,
                item.priority.name,
                item.enqueued_at_turn,
                item.expiry_turns,
                current_turn,
            )

        return expired

    def get_state(self) -> QueueState:
        """Return the current state vector of the queue.

        Returns:
            A QueueState snapshot with depth, token total, max priority,
            time since last drain, pending tool count, and estimated next arrival.
        """
        with self._lock:
            items = list(self._items.values())
            return QueueState(
                depth=len(items),
                token_total=sum(item.token_count for item in items),
                max_priority=max((item.priority for item in items), default=None),
                time_since_last_drain=time.monotonic() - self._last_drain_time,
                pending_tool_count=len({item.source_tool_id for item in items}),
                estimated_next_arrival=self._arrival_interval_ema,
            )

    def drain_at_breakpoint(self) -> str | None:
        """Dequeue all items and return a formatted bundle string.

        Used at natural breakpoints in the conscious loop to inject all
        pending context. Returns None if the queue is empty.

        Returns:
            A formatted string with a header and one line per item,
            ordered by priority descending, or None if empty.
        """
        items = self.dequeue_all()
        if not items:
            return None

        lines = [f"--- Context updates ({len(items)} items) ---"]
        for item in items:
            lines.append(f"[{item.source_tool_id}] {item.content}")
        return "\n".join(lines)

    def check_interrupt_threshold(self, config: InterruptConfig) -> bool:
        """Check whether the queue state exceeds any interrupt threshold.

        Used by the conscious loop to decide whether to abort the current
        generation and retry with injected context.

        Args:
            config: Threshold configuration to check against.

        Returns:
            True if any threshold is exceeded and the queue is not frozen.
        """
        with self._lock:
            if self._frozen or not self._items:
                return False

            if len(self._items) >= config.count_threshold:
                return True

            if any(item.priority >= config.priority_threshold for item in self._items.values()):
                return True

            token_total = sum(item.token_count for item in self._items.values())
            return token_total >= config.token_threshold

    def freeze(self) -> None:
        """Freeze the queue to suppress interrupt threshold checks.

        While frozen, check_interrupt_threshold() always returns False.
        Used during the retry step after an interrupt to prevent
        degenerate loops.
        """
        with self._lock:
            self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the queue to re-enable interrupt threshold checks."""
        with self._lock:
            self._frozen = False

    def report_wasted_tokens(self, token_count: int) -> None:
        """Record tokens wasted when an interrupt discards a partial generation.

        Args:
            token_count: Number of tokens in the discarded partial generation.
        """
        with self._lock:
            self._wasted_tokens += token_count

    @property
    def wasted_tokens(self) -> int:
        """Total tokens wasted across all interrupts."""
        with self._lock:
            return self._wasted_tokens
