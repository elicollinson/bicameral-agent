"""Tests for the unified EvalReport model (Issue #56)."""

import json

from bicameral_agent.ab_test import compute_summary
from bicameral_agent.baseline_benchmark import BenchmarkResult, ConditionReport
from bicameral_agent.eval_report import EvalReport

# Metrics the ui/ Review screen reads from summary.json
# (ui/src/core/runs.ts HEADLINE_METRICS); their shape must not break.
_HEADLINE_METRICS = (
    "quality_score",
    "total_tokens",
    "total_turns",
    "wall_clock_ms",
    "tool_cost_usd",
    "avg_queue_depth",
    "drain_count",
)


def _condition_report(condition: str) -> ConditionReport:
    return ConditionReport(
        condition=condition,
        n_episodes=2,
        summaries={name: compute_summary([0.5, 1.0]) for name in _HEADLINE_METRICS},
        latency_mape_percent=12.5,
        latency_n_pairs=3,
    )


def _make_report(result: BenchmarkResult) -> EvalReport:
    return EvalReport.from_benchmark(
        result,
        dataset="builtin",
        metric="llm_judge",
        answerer={"provider": "gemini", "model": "gemini-x"},
        measurement={"provider": "gemini", "model": "gemini-x"},
        tasks_per_condition=2,
        max_turns=10,
    )


class TestEvalReport:
    def test_summary_shape_backward_compatible(self, make_episode):
        """The serialized shape keeps every key the ui/ Review screen reads."""
        result = BenchmarkResult(
            episodes={"heuristic": [make_episode()]},
            reports={"heuristic": _condition_report("heuristic")},
        )
        payload = json.loads(_make_report(result).to_json())

        assert payload["dataset"] == "builtin"
        assert payload["metric"] == "llm_judge"
        assert payload["tasks_per_condition"] == 2
        assert payload["max_turns"] == 10
        assert payload["answerer"] == {"provider": "gemini", "model": "gemini-x"}
        condition = payload["conditions"]["heuristic"]
        assert condition["n_episodes"] == 2
        for name in _HEADLINE_METRICS:
            summary = condition["summaries"][name]
            assert set(summary) == {"mean", "std", "ci_lower", "ci_upper", "n"}

    def test_results_capture_scores_and_verification_detail(self, make_episode):
        episode = make_episode(
            metadata={
                "task_id": "t9",
                "verification": {
                    "metric": "exact_match",
                    "detail": "exact_match: expected 'x', got 'y' -> no match",
                },
            }
        )
        result = BenchmarkResult(
            episodes={"random": [episode]},
            reports={"random": _condition_report("random")},
        )
        (task_result,) = _make_report(result).results
        assert task_result.task_id == "t9"
        assert task_result.condition == "random"
        assert task_result.score == episode.outcome.quality_score
        assert "no match" in task_result.detail

    def test_results_tolerate_unverified_episodes(self, make_episode):
        result = BenchmarkResult(
            episodes={"random": [make_episode(metadata={"task_id": "t1"})]},
            reports={"random": _condition_report("random")},
        )
        (task_result,) = _make_report(result).results
        assert task_result.detail is None
