"""Tests for safe parsing of structured LLM output (issue #47)."""

from __future__ import annotations

import json

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
