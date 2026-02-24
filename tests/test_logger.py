"""Tests for ConversationLogger."""

import threading
import time

import pytest

from bicameral_agent.logger import ConversationLogger
from bicameral_agent.schema import UserEventType
from bicameral_agent.validation import EpisodeValidator


class TestBasicConstruction:
    def test_empty_logger_produces_valid_episode(self):
        logger = ConversationLogger()
        episode = logger.finalize()
        assert episode.outcome.total_tokens == 0
        assert episode.outcome.total_turns == 0
        assert episode.outcome.wall_clock_ms == 0
        assert episode.messages == []
        result = EpisodeValidator().validate(episode)
        assert result.is_valid

    def test_metadata_passed_through(self):
        logger = ConversationLogger(metadata={"model": "gpt-4", "version": 2})
        episode = logger.finalize()
        assert episode.metadata == {"model": "gpt-4", "version": 2}

    def test_single_message(self):
        logger = ConversationLogger()
        logger.log_message("user", "hello", token_count=3)
        episode = logger.finalize()
        assert len(episode.messages) == 1
        assert episode.messages[0].role == "user"
        assert episode.messages[0].content == "hello"
        assert episode.messages[0].token_count == 3
        assert episode.outcome.total_turns == 1


class TestRoundTrip:
    """AC1: 20+ events, serialize/deserialize, all fields match."""

    def test_round_trip_20_plus_events(self):
        logger = ConversationLogger(metadata={"test": "round_trip"})

        # 8 messages
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            logger.log_message(role, f"message {i}", token_count=10 + i)

        # 3 user events
        for evt_type in [UserEventType.STOP, UserEventType.EDIT, UserEventType.FOLLOW_UP]:
            logger.log_user_event(evt_type, metadata={"key": evt_type.value})

        # 4 tool invocations (all completed)
        for i in range(4):
            idx = logger.log_tool_invocation(f"tool-{i}", input_tokens=5 + i)
            logger.log_tool_completion(idx, output_tokens=8 + i, result_deposited=i % 2 == 0)

        # 6 context injections (3 consumed)
        for i in range(6):
            inj_idx = logger.log_context_injection(
                content=f"context {i}",
                source_tool_id=f"src-{i}",
                priority=i,
                token_count=15 + i,
            )
            if i < 3:
                logger.log_injection_consumed(inj_idx, turn_number=i)

        episode = logger.finalize(quality_score=0.85)

        # Verify event counts (21 total: 8+3+4+6)
        assert len(episode.messages) == 8
        assert len(episode.user_events) == 3
        assert len(episode.tool_invocations) == 4
        assert len(episode.context_injections) == 6
        total_events = (
            len(episode.messages)
            + len(episode.user_events)
            + len(episode.tool_invocations)
            + len(episode.context_injections)
        )
        assert total_events >= 20

        # Serialize to JSON and back
        json_str = episode.to_json()
        restored = episode.from_json(json_str)

        assert restored.episode_id == episode.episode_id
        assert len(restored.messages) == 8
        assert len(restored.user_events) == 3
        assert len(restored.tool_invocations) == 4
        assert len(restored.context_injections) == 6
        assert restored.outcome.quality_score == 0.85
        assert restored.metadata == {"test": "round_trip"}

        # Verify consumed injections
        consumed = [c for c in restored.context_injections if c.consumed]
        assert len(consumed) == 3
        for i, c in enumerate(consumed):
            assert c.consumed_at_turn == i

        # Verify tool details survived
        for i, t in enumerate(restored.tool_invocations):
            assert t.tool_id == f"tool-{i}"
            assert t.result_deposited == (i % 2 == 0)

        # Validate restored episode
        result = EpisodeValidator().validate(restored)
        assert result.is_valid


class TestMonotonicTimestamps:
    """AC2: Timestamps are monotonically non-decreasing."""

    def test_message_timestamps_sorted(self):
        logger = ConversationLogger()
        for i in range(50):
            logger.log_message("user", f"msg {i}", token_count=1)
        episode = logger.finalize()
        timestamps = [m.timestamp_ms for m in episode.messages]
        assert timestamps == sorted(timestamps)

    def test_user_event_timestamps_sorted(self):
        logger = ConversationLogger()
        for _ in range(50):
            logger.log_user_event(UserEventType.FOLLOW_UP)
        episode = logger.finalize()
        timestamps = [e.timestamp_ms for e in episode.user_events]
        assert timestamps == sorted(timestamps)

    def test_context_injection_timestamps_sorted(self):
        logger = ConversationLogger()
        for i in range(50):
            logger.log_context_injection(f"ctx {i}", "tool", priority=0, token_count=1)
        episode = logger.finalize()
        timestamps = [c.timestamp_ms for c in episode.context_injections]
        assert timestamps == sorted(timestamps)

    def test_tool_invocation_timestamps_sorted(self):
        logger = ConversationLogger()
        indices = []
        for i in range(50):
            idx = logger.log_tool_invocation(f"tool-{i}", input_tokens=1)
            indices.append(idx)
        # Complete in reverse order to test sorting
        for idx in reversed(indices):
            logger.log_tool_completion(idx, output_tokens=1)
        episode = logger.finalize()
        timestamps = [t.invoked_at_ms for t in episode.tool_invocations]
        assert timestamps == sorted(timestamps)


class TestThreadSafety:
    """AC3: 3 threads × 20 events, no lost events, 100 runs."""

    def test_concurrent_logging(self):
        for _ in range(100):
            logger = ConversationLogger()
            barrier = threading.Barrier(3)
            errors: list[Exception] = []

            def log_messages():
                try:
                    barrier.wait()
                    for i in range(20):
                        logger.log_message("user", f"msg {i}", token_count=1)
                except Exception as e:
                    errors.append(e)

            def log_events():
                try:
                    barrier.wait()
                    for _ in range(20):
                        logger.log_user_event(UserEventType.FOLLOW_UP)
                except Exception as e:
                    errors.append(e)

            def log_tools():
                try:
                    barrier.wait()
                    for i in range(20):
                        idx = logger.log_tool_invocation(f"tool-{i}", input_tokens=1)
                        logger.log_tool_completion(idx, output_tokens=1)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=log_messages),
                threading.Thread(target=log_events),
                threading.Thread(target=log_tools),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Thread errors: {errors}"
            episode = logger.finalize()
            assert len(episode.messages) == 20
            assert len(episode.user_events) == 20
            assert len(episode.tool_invocations) == 20


class TestValidatorCompliance:
    """AC4: Finalized episode passes EpisodeValidator."""

    def test_finalized_episode_validates(self):
        logger = ConversationLogger()
        logger.log_message("user", "hi", token_count=2)
        logger.log_message("assistant", "hello", token_count=3)
        idx = logger.log_tool_invocation("search", input_tokens=5)
        logger.log_tool_completion(idx, output_tokens=10, result_deposited=True)
        inj = logger.log_context_injection("extra context", "search", priority=1, token_count=8)
        logger.log_injection_consumed(inj, turn_number=0)
        logger.log_user_event(UserEventType.FOLLOW_UP)
        episode = logger.finalize(quality_score=0.9)

        result = EpisodeValidator().validate(episode)
        assert result.is_valid
        assert not result.errors


class TestPerformanceBenchmark:
    """AC5: 1000 log_message calls < 500ms."""

    def test_1000_messages_under_500ms(self):
        logger = ConversationLogger()
        start = time.perf_counter()
        for i in range(1000):
            logger.log_message("user", f"message {i}", token_count=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"1000 messages took {elapsed_ms:.1f}ms (limit: 500ms)"
        episode = logger.finalize()
        assert len(episode.messages) == 1000


class TestFinalizeGuards:
    def test_pending_tools_block_finalize(self):
        logger = ConversationLogger()
        logger.log_tool_invocation("tool-1", input_tokens=5)
        with pytest.raises(RuntimeError, match="pending tool invocations"):
            logger.finalize()

    def test_double_finalize_raises(self):
        logger = ConversationLogger()
        logger.finalize()
        with pytest.raises(RuntimeError, match="already called"):
            logger.finalize()

    def test_log_after_finalize_raises(self):
        logger = ConversationLogger()
        logger.finalize()
        with pytest.raises(RuntimeError, match="Cannot log events after finalize"):
            logger.log_message("user", "late", token_count=1)
        with pytest.raises(RuntimeError, match="Cannot log events after finalize"):
            logger.log_user_event(UserEventType.STOP)
        with pytest.raises(RuntimeError, match="Cannot log events after finalize"):
            logger.log_tool_invocation("tool", input_tokens=1)
        with pytest.raises(RuntimeError, match="Cannot log events after finalize"):
            logger.log_context_injection("ctx", "tool", priority=0, token_count=1)


class TestToolInvocationSorting:
    def test_out_of_order_completions_sorted_by_invocation_time(self):
        logger = ConversationLogger()
        idx0 = logger.log_tool_invocation("first", input_tokens=1)
        idx1 = logger.log_tool_invocation("second", input_tokens=1)
        idx2 = logger.log_tool_invocation("third", input_tokens=1)
        # Complete in reverse order
        logger.log_tool_completion(idx2, output_tokens=1)
        logger.log_tool_completion(idx1, output_tokens=1)
        logger.log_tool_completion(idx0, output_tokens=1)

        episode = logger.finalize()
        assert [t.tool_id for t in episode.tool_invocations] == [
            "first", "second", "third"
        ]
        timestamps = [t.invoked_at_ms for t in episode.tool_invocations]
        assert timestamps == sorted(timestamps)


class TestInjectionEdgeCases:
    def test_invalid_injection_index_raises(self):
        logger = ConversationLogger()
        with pytest.raises(ValueError, match="Invalid injection index"):
            logger.log_injection_consumed(0, turn_number=0)
        with pytest.raises(ValueError, match="Invalid injection index"):
            logger.log_injection_consumed(-1, turn_number=0)

    def test_double_consume_raises(self):
        logger = ConversationLogger()
        idx = logger.log_context_injection("ctx", "tool", priority=0, token_count=5)
        logger.log_injection_consumed(idx, turn_number=0)
        with pytest.raises(ValueError, match="already consumed"):
            logger.log_injection_consumed(idx, turn_number=1)

    def test_log_tool_completion_unknown_index_raises(self):
        logger = ConversationLogger()
        with pytest.raises(ValueError, match="Unknown invocation index"):
            logger.log_tool_completion(999, output_tokens=1)


class TestOutcomeComputation:
    def test_total_tokens_includes_messages_tools_consumed_injections(self):
        logger = ConversationLogger()
        logger.log_message("user", "hi", token_count=10)
        logger.log_message("assistant", "hello", token_count=20)

        idx = logger.log_tool_invocation("tool", input_tokens=5)
        logger.log_tool_completion(idx, output_tokens=15)

        inj0 = logger.log_context_injection("consumed", "src", priority=0, token_count=8)
        logger.log_context_injection("not consumed", "src", priority=0, token_count=100)
        logger.log_injection_consumed(inj0, turn_number=0)

        episode = logger.finalize()
        # messages: 10+20=30, tools: 5+15=20, consumed injections: 8
        assert episode.outcome.total_tokens == 58

    def test_total_turns_counts_user_messages(self):
        logger = ConversationLogger()
        logger.log_message("user", "q1", token_count=1)
        logger.log_message("assistant", "a1", token_count=1)
        logger.log_message("user", "q2", token_count=1)
        logger.log_message("assistant", "a2", token_count=1)
        logger.log_message("system", "sys", token_count=1)
        episode = logger.finalize()
        assert episode.outcome.total_turns == 2

    def test_wall_clock_ms_spans_all_events(self):
        logger = ConversationLogger()
        logger.log_message("user", "start", token_count=1)
        # Small sleep to ensure nonzero wall clock
        time.sleep(0.01)
        logger.log_message("assistant", "end", token_count=1)
        episode = logger.finalize()
        assert episode.outcome.wall_clock_ms >= 10
