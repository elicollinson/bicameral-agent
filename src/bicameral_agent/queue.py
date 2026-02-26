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
from typing import Optional

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

    dedup_key: Optional[str] = None
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

    max_priority: Optional[Priority]
    """Highest priority among queued items, or None if empty."""

    time_since_last_drain: float
    """Seconds since the last dequeue_all() call."""

    pending_tool_count: int
    """Number of distinct source_tool_ids with items in the queue."""

    estimated_next_arrival: float
    """Estimated seconds until the next enqueue, based on arrival rate EMA."""


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
        self._last_enqueue_time: Optional[float] = None
        self._arrival_interval_ema: float = 0.0
        self._enqueue_count: int = 0

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
            stale_ids: list[str] = []
            for item_id, item in self._items.items():
                if current_turn - item.enqueued_at_turn >= item.expiry_turns:
                    stale_ids.append(item_id)
                    expired.append(item)

            for item_id in stale_ids:
                item = self._items.pop(item_id)
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
            if not self._items:
                return QueueState(
                    depth=0,
                    token_total=0,
                    max_priority=None,
                    time_since_last_drain=time.monotonic() - self._last_drain_time,
                    pending_tool_count=0,
                    estimated_next_arrival=self._arrival_interval_ema,
                )

            items = self._items.values()
            depth = len(self._items)
            token_total = sum(item.token_count for item in items)
            max_priority = max(item.priority for item in items)
            tool_ids = {item.source_tool_id for item in items}

            return QueueState(
                depth=depth,
                token_total=token_total,
                max_priority=max_priority,
                time_since_last_drain=time.monotonic() - self._last_drain_time,
                pending_tool_count=len(tool_ids),
                estimated_next_arrival=self._arrival_interval_ema,
            )
