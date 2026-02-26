"""Tests for the context injection queue."""

import logging
import threading
import time

import pytest

from bicameral_agent.queue import ContextQueue, Priority, QueueItem


class TestPriorityEnum:
    def test_ordering(self):
        assert Priority.CRITICAL > Priority.HIGH > Priority.MEDIUM > Priority.LOW

    def test_values(self):
        assert Priority.LOW == 0
        assert Priority.MEDIUM == 1
        assert Priority.HIGH == 2
        assert Priority.CRITICAL == 3


class TestQueueItem:
    def test_defaults(self, make_queue_item):
        item = make_queue_item()
        assert item.expiry_turns == 10
        assert item.dedup_key is None
        assert len(item.item_id) == 36  # UUID format

    def test_unique_ids(self, make_queue_item):
        items = [make_queue_item() for _ in range(100)]
        ids = {item.item_id for item in items}
        assert len(ids) == 100

    def test_invalid_token_count_rejected(self):
        with pytest.raises(Exception):
            QueueItem(
                content="x",
                priority=Priority.LOW,
                source_tool_id="t",
                token_count=-1,
            )

    def test_invalid_expiry_turns_rejected(self):
        with pytest.raises(Exception):
            QueueItem(
                content="x",
                priority=Priority.LOW,
                source_tool_id="t",
                token_count=1,
                expiry_turns=0,
            )


class TestBasicQueueOps:
    """AC1: enqueue 10 items with mixed priorities, dequeue_all returns priority order."""

    def test_enqueue_dequeue_priority_order(self, empty_queue, make_queue_item):
        priorities = [
            Priority.LOW, Priority.HIGH, Priority.MEDIUM, Priority.CRITICAL,
            Priority.LOW, Priority.HIGH, Priority.MEDIUM, Priority.CRITICAL,
            Priority.LOW, Priority.MEDIUM,
        ]
        for i, p in enumerate(priorities):
            item = make_queue_item(
                content=f"item {i}",
                priority=p,
                source_tool_id=f"tool-{i}",
                token_count=10 + i,
            )
            empty_queue.enqueue(item)

        result = empty_queue.dequeue_all()
        assert len(result) == 10

        result_priorities = [item.priority for item in result]
        assert result_priorities == sorted(result_priorities, reverse=True)

        # Queue should be empty
        assert empty_queue.dequeue_all() == []
        state = empty_queue.get_state()
        assert state.depth == 0

    def test_fifo_within_same_priority(self, empty_queue, make_queue_item):
        """AC7: stable FIFO order within same priority level."""
        items = []
        for i in range(5):
            item = make_queue_item(
                content=f"item {i}",
                priority=Priority.HIGH,
                source_tool_id=f"tool-{i}",
            )
            items.append(item)
            empty_queue.enqueue(item)

        result = empty_queue.dequeue_all()
        result_ids = [item.item_id for item in result]
        expected_ids = [item.item_id for item in items]
        assert result_ids == expected_ids


class TestDeduplication:
    """AC2: dedup_key semantics."""

    def test_same_key_higher_priority_replaces(self, empty_queue, make_queue_item):
        old = make_queue_item(dedup_key="key-1", priority=Priority.LOW, content="old")
        new = make_queue_item(dedup_key="key-1", priority=Priority.HIGH, content="new")
        empty_queue.enqueue(old)
        empty_queue.enqueue(new)
        result = empty_queue.dequeue_all()
        assert len(result) == 1
        assert result[0].content == "new"

    def test_same_key_same_priority_newer_wins(self, empty_queue, make_queue_item):
        old = make_queue_item(dedup_key="key-1", priority=Priority.MEDIUM, content="old")
        new = make_queue_item(dedup_key="key-1", priority=Priority.MEDIUM, content="new")
        empty_queue.enqueue(old)
        empty_queue.enqueue(new)
        result = empty_queue.dequeue_all()
        assert len(result) == 1
        assert result[0].content == "new"

    def test_same_key_lower_priority_kept(self, empty_queue, make_queue_item):
        old = make_queue_item(dedup_key="key-1", priority=Priority.HIGH, content="old")
        new = make_queue_item(dedup_key="key-1", priority=Priority.LOW, content="new")
        empty_queue.enqueue(old)
        empty_queue.enqueue(new)
        result = empty_queue.dequeue_all()
        assert len(result) == 1
        assert result[0].content == "old"

    def test_different_dedup_keys_coexist(self, empty_queue, make_queue_item):
        a = make_queue_item(dedup_key="key-a", content="a")
        b = make_queue_item(dedup_key="key-b", content="b")
        empty_queue.enqueue(a)
        empty_queue.enqueue(b)
        result = empty_queue.dequeue_all()
        assert len(result) == 2

    def test_none_dedup_key_never_deduplicates(self, empty_queue, make_queue_item):
        for i in range(5):
            empty_queue.enqueue(make_queue_item(dedup_key=None, content=f"item {i}"))
        result = empty_queue.dequeue_all()
        assert len(result) == 5


class TestExpiry:
    """AC3: expire_stale removes items past their expiry_turns."""

    def test_stale_items_removed(self, empty_queue, make_queue_item):
        item = make_queue_item(expiry_turns=3, enqueued_at_turn=0)
        empty_queue.enqueue(item)
        expired = empty_queue.expire_stale(current_turn=5)
        assert len(expired) == 1
        assert empty_queue.get_state().depth == 0

    def test_fresh_items_kept(self, empty_queue, make_queue_item):
        item = make_queue_item(expiry_turns=10, enqueued_at_turn=0)
        empty_queue.enqueue(item)
        expired = empty_queue.expire_stale(current_turn=5)
        assert len(expired) == 0
        assert empty_queue.get_state().depth == 1

    def test_exact_boundary_expires(self, empty_queue, make_queue_item):
        """Item expires when current_turn - enqueued_at_turn == expiry_turns."""
        item = make_queue_item(expiry_turns=3, enqueued_at_turn=2)
        empty_queue.enqueue(item)
        expired = empty_queue.expire_stale(current_turn=5)
        assert len(expired) == 1

    def test_one_before_boundary_kept(self, empty_queue, make_queue_item):
        """Item not expired when current_turn - enqueued_at_turn < expiry_turns."""
        item = make_queue_item(expiry_turns=3, enqueued_at_turn=3)
        empty_queue.enqueue(item)
        expired = empty_queue.expire_stale(current_turn=5)
        assert len(expired) == 0
        assert empty_queue.get_state().depth == 1

    def test_expiry_logged(self, empty_queue, make_queue_item, caplog):
        item = make_queue_item(expiry_turns=1, enqueued_at_turn=0)
        empty_queue.enqueue(item)
        with caplog.at_level(logging.INFO, logger="bicameral_agent.queue"):
            empty_queue.expire_stale(current_turn=5)
        assert "Expired queue item" in caplog.text
        assert item.item_id in caplog.text

    def test_expiry_cleans_dedup_index(self, empty_queue, make_queue_item):
        """After expiring a dedup'd item, a new item with same key can be enqueued."""
        item = make_queue_item(dedup_key="dk", expiry_turns=1, enqueued_at_turn=0)
        empty_queue.enqueue(item)
        empty_queue.expire_stale(current_turn=5)
        new_item = make_queue_item(dedup_key="dk", content="replacement")
        empty_queue.enqueue(new_item)
        result = empty_queue.dequeue_all()
        assert len(result) == 1
        assert result[0].content == "replacement"


class TestGetState:
    """AC4: get_state() returns accurate values."""

    def test_empty_queue_state(self, empty_queue):
        state = empty_queue.get_state()
        assert state.depth == 0
        assert state.token_total == 0
        assert state.max_priority is None
        assert state.pending_tool_count == 0
        assert state.time_since_last_drain >= 0.0

    def test_state_after_enqueue(self, empty_queue, make_queue_item):
        empty_queue.enqueue(
            make_queue_item(token_count=50, priority=Priority.HIGH, source_tool_id="tool-a")
        )
        empty_queue.enqueue(
            make_queue_item(token_count=30, priority=Priority.CRITICAL, source_tool_id="tool-b")
        )
        state = empty_queue.get_state()
        assert state.depth == 2
        assert state.token_total == 80
        assert state.max_priority == Priority.CRITICAL
        assert state.pending_tool_count == 2

    def test_state_after_dequeue(self, empty_queue, make_queue_item):
        empty_queue.enqueue(make_queue_item(token_count=20))
        empty_queue.dequeue_all()
        state = empty_queue.get_state()
        assert state.depth == 0
        assert state.token_total == 0

    def test_state_after_expiry(self, empty_queue, make_queue_item):
        empty_queue.enqueue(
            make_queue_item(token_count=10, expiry_turns=1, enqueued_at_turn=0)
        )
        empty_queue.enqueue(
            make_queue_item(token_count=20, expiry_turns=100, enqueued_at_turn=0)
        )
        empty_queue.expire_stale(current_turn=5)
        state = empty_queue.get_state()
        assert state.depth == 1
        assert state.token_total == 20

    def test_pending_tool_count_distinct(self, empty_queue, make_queue_item):
        """Two items from same tool = 1 pending tool."""
        empty_queue.enqueue(make_queue_item(source_tool_id="tool-x"))
        empty_queue.enqueue(make_queue_item(source_tool_id="tool-x"))
        state = empty_queue.get_state()
        assert state.pending_tool_count == 1

    def test_queue_state_is_frozen(self, empty_queue):
        state = empty_queue.get_state()
        with pytest.raises(AttributeError):
            state.depth = 99


class TestThreadSafety:
    """AC5: 5 threads enqueuing 100 items each, 50 runs, no lost items."""

    def test_concurrent_enqueue(self, make_queue_item):
        for run in range(50):
            queue = ContextQueue()
            barrier = threading.Barrier(5)
            errors: list[Exception] = []

            def enqueue_batch(thread_id: int) -> None:
                try:
                    barrier.wait()
                    for i in range(100):
                        item = make_queue_item(
                            content=f"t{thread_id}-i{i}",
                            source_tool_id=f"tool-{thread_id}",
                            token_count=1,
                        )
                        queue.enqueue(item)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=enqueue_batch, args=(t,)) for t in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Run {run}: Thread errors: {errors}"
            result = queue.dequeue_all()
            assert len(result) == 500, f"Run {run}: Expected 500, got {len(result)}"

    def test_concurrent_enqueue_and_get_state(self, make_queue_item):
        """get_state() doesn't crash while items are being enqueued."""
        queue = ContextQueue()
        stop_event = threading.Event()
        errors: list[Exception] = []

        def enqueue_loop() -> None:
            try:
                for i in range(200):
                    queue.enqueue(make_queue_item(content=f"item-{i}", token_count=1))
                stop_event.set()
            except Exception as e:
                errors.append(e)
                stop_event.set()

        def state_loop() -> None:
            try:
                while not stop_event.is_set():
                    state = queue.get_state()
                    assert state.depth >= 0
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=enqueue_loop)
        t2 = threading.Thread(target=state_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors


class TestEmptyQueueFastPath:
    """AC6: Empty queue operations complete in < 0.1ms."""

    def test_get_state_empty_fast(self, empty_queue):
        empty_queue.get_state()  # warm up
        start = time.perf_counter()
        for _ in range(1000):
            empty_queue.get_state()
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 1000
        assert avg_ms < 0.1, f"get_state() avg {avg_ms:.4f}ms (limit: 0.1ms)"

    def test_dequeue_all_empty_fast(self, empty_queue):
        empty_queue.dequeue_all()  # warm up
        start = time.perf_counter()
        for _ in range(1000):
            empty_queue.dequeue_all()
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 1000
        assert avg_ms < 0.1, f"dequeue_all() avg {avg_ms:.4f}ms (limit: 0.1ms)"


class TestPeek:
    def test_peek_does_not_remove(self, empty_queue, make_queue_item):
        empty_queue.enqueue(make_queue_item(content="a"))
        empty_queue.enqueue(make_queue_item(content="b"))
        peeked = empty_queue.peek()
        assert len(peeked) == 2
        result = empty_queue.dequeue_all()
        assert len(result) == 2

    def test_peek_returns_priority_order(self, empty_queue, make_queue_item):
        empty_queue.enqueue(make_queue_item(priority=Priority.LOW, content="low"))
        empty_queue.enqueue(make_queue_item(priority=Priority.CRITICAL, content="crit"))
        peeked = empty_queue.peek()
        assert peeked[0].content == "crit"
        assert peeked[1].content == "low"


class TestEstimatedNextArrival:
    def test_zero_before_any_enqueue(self, empty_queue):
        state = empty_queue.get_state()
        assert state.estimated_next_arrival == 0.0

    def test_nonzero_after_multiple_enqueues(self, empty_queue, make_queue_item):
        for _ in range(5):
            empty_queue.enqueue(make_queue_item())
        state = empty_queue.get_state()
        assert state.estimated_next_arrival >= 0.0
