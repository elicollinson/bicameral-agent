"""Tests for safe parsing of structured LLM output (issue #47)."""

from __future__ import annotations

import contextvars
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from bicameral_agent.gemini import GeminiResponse
from bicameral_agent.llm_output import (
    clamp,
    coerce_int,
    count_degradations,
    safe_parse_json,
)


def _response(content: str, finish_reason: str = "STOP") -> GeminiResponse:
    return GeminiResponse(
        content=content,
        input_tokens=10,
        output_tokens=10,
        duration_ms=1.0,
        finish_reason=finish_reason,
    )


class TestSafeParseJson:
    def test_valid_json(self):
        parsed = safe_parse_json(_response('{"a": 1}'), context="t")
        assert parsed == {"a": 1}

    def test_json_with_preamble(self):
        parsed = safe_parse_json(
            _response('Here is the JSON:\n{"a": 1} trailing'), context="t"
        )
        assert parsed == {"a": 1}

    def test_truncated_json_returns_default(self):
        truncated = json.dumps({"has_gaps": True, "gaps": [{"description": "x"}]})[:-8]
        parsed = safe_parse_json(
            _response(truncated, finish_reason="MAX_TOKENS"),
            context="t",
            default={"has_gaps": False},
        )
        assert parsed == {"has_gaps": False}

    def test_empty_content_returns_default(self):
        assert safe_parse_json(_response(""), context="t", default=None) is None

    def test_non_dict_json_returns_default(self):
        parsed = safe_parse_json(_response("[1, 2]"), context="t", default={})
        assert parsed == {}

    def test_logs_finish_reason(self, caplog):
        with caplog.at_level("WARNING"):
            safe_parse_json(
                _response('{"broken', finish_reason="MAX_TOKENS"), context="my_tool"
            )
        assert "MAX_TOKENS" in caplog.text
        assert "my_tool" in caplog.text


class TestClamp:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.5, 0.5), (-1, 0.0), (4, 1.0), ("0.7", 0.7), (0, 0.0), (1, 1.0)],
    )
    def test_numeric_values(self, value, expected):
        assert clamp(value, 0.0, 1.0, 0.5) == expected

    @pytest.mark.parametrize("value", [None, "high", {}, float("nan")])
    def test_non_numeric_returns_default(self, value):
        assert clamp(value, 0.0, 1.0, 0.5) == 0.5


class TestCoerceInt:
    @pytest.mark.parametrize(
        ("value", "expected"), [(4, 4), (4.7, 4), ("4", 4), ("4.2", 4), (-2, -2)]
    )
    def test_numeric_values(self, value, expected):
        assert coerce_int(value, 3) == expected

    @pytest.mark.parametrize("value", [None, "good", [], float("nan")])
    def test_non_numeric_returns_default(self, value):
        assert coerce_int(value, 3) == 3


class TestCountDegradations:
    """Episode-scoped degradation counting (issue #82)."""

    def test_counts_per_component(self):
        with count_degradations() as counter:
            safe_parse_json(_response("not json"), context="TaskScorer")
            safe_parse_json(_response("also bad"), context="TaskScorer")
            safe_parse_json(_response("nope"), context="SimulatedUser.respond")
        assert counter.counts == {"TaskScorer": 2, "SimulatedUser.respond": 1}

    def test_successful_parse_not_counted(self):
        with count_degradations() as counter:
            safe_parse_json(_response('{"a": 1}'), context="TaskScorer")
        assert counter.counts == {}

    def test_detached_after_exit(self):
        with count_degradations() as counter:
            pass
        safe_parse_json(_response("bad"), context="TaskScorer")
        assert counter.counts == {}

    def test_ignores_untagged_warnings(self):
        import logging

        with count_degradations() as counter:
            logging.getLogger("bicameral_agent.llm_output").warning("unrelated")
        assert counter.counts == {}

    def test_nested_counters_are_independent(self):
        with count_degradations() as outer:
            safe_parse_json(_response("bad"), context="A")
            with count_degradations() as inner:
                safe_parse_json(_response("bad"), context="B")
        assert inner.counts == {"B": 1}
        assert outer.counts == {"A": 1, "B": 1}

    def test_concurrent_episodes_do_not_cross_contaminate(self):
        """Issue #91: overlapping episode counters each see only their own.

        A barrier holds both threads inside their ``count_degradations``
        blocks simultaneously, so a regression to a shared module-level
        counter would deterministically leak counts across threads.
        """
        barrier = threading.Barrier(2)
        results: dict[str, dict[str, int]] = {}

        def episode(name: str, n: int) -> None:
            with count_degradations() as counter:
                barrier.wait(timeout=5)
                for _ in range(n):
                    safe_parse_json(_response("bad"), context=name)
                barrier.wait(timeout=5)
            results[name] = dict(counter.counts)

        threads = [
            threading.Thread(target=episode, args=("A", 2)),
            threading.Thread(target=episode, args=("B", 3)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == {"A": {"A": 2}, "B": {"B": 3}}

    def test_copied_context_worker_threads_report_to_counter(self):
        """Threads submitted with a copied context (scorer-pool style, issue
        #91) attribute their degradations to the submitting episode."""
        with count_degradations() as counter:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        contextvars.copy_context().run,
                        safe_parse_json,
                        _response("bad"),
                        context="TaskScorer",
                    )
                    for _ in range(3)
                ]
                for f in futures:
                    f.result()
        assert counter.counts == {"TaskScorer": 3}
