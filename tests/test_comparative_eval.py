"""Tests for comparative_eval — Issue #30.

All tests are mock-LLM: episodes are synthetic and the runner is a
MagicMock. The learned-condition wiring tests require torch and are
skipped when it is unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bicameral_agent.ab_test import compute_summary
from bicameral_agent.baseline_benchmark import TaskMetrics
from bicameral_agent.comparative_eval import (
    CONDITION_NAMES,
    METRIC_NAMES,
    ComparativeEvaluator,
    ComparativeResult,
    baseline_condition_factories,
    build_report,
    condition_seed,
    difficulty_breakdown,
    metric_values,
    pairwise_tests,
    parse_task_mix,
    select_tasks,
    student_t_two_tailed_p,
    welch_test_with_p,
)
from bicameral_agent.dataset import (
    ResearchQADataset,
    ResearchQATask,
    TaskDifficulty,
    TaskSplit,
)
from bicameral_agent.episode_runner import EpisodeRunner
from bicameral_agent.heuristic_controller import HeuristicController
from bicameral_agent.no_subconscious_controller import NoSubconsciousController
from bicameral_agent.random_controller import RandomController
from bicameral_agent.schema import Episode, EpisodeOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(
    task_id: str = "t1",
    difficulty: TaskDifficulty = TaskDifficulty.TYPICAL,
    split: TaskSplit = TaskSplit.EVAL,
) -> ResearchQATask:
    return ResearchQATask(
        task_id=task_id,
        difficulty=difficulty,
        split=split,
        question="q",
        gold_answer="a",
        scoring_rubric="rubric",
        known_assumptions=(
            ["assume"] if difficulty == TaskDifficulty.TRICKY else None
        ),
    )


def _metrics(
    *,
    quality_score: float | None = 0.7,
    total_tokens: int = 100,
    total_turns: int = 4,
    user_stops: int = 0,
    task_completed: int = 1,
    wall_clock_ms: int = 5000,
    tool_invocation_count: int = 2,
    avg_queue_depth: float = 0.5,
    interrupt_count: int = 1,
    drain_count: int = 1,
    expired_count: int = 1,
    latency_pairs: tuple[tuple[float, float], ...] = ((100.0, 200.0),),
) -> TaskMetrics:
    return TaskMetrics(
        quality_score=quality_score,
        total_tokens=total_tokens,
        total_turns=total_turns,
        user_stops=user_stops,
        task_completed=task_completed,
        wall_clock_ms=wall_clock_ms,
        tool_invocation_count=tool_invocation_count,
        tool_cost_usd=0.01,
        avg_queue_depth=avg_queue_depth,
        interrupt_count=interrupt_count,
        drain_count=drain_count,
        expired_count=expired_count,
        latency_pairs=latency_pairs,
    )


def _episode(quality_score: float | None = 0.7, task_id: str = "t1") -> Episode:
    return Episode(
        outcome=EpisodeOutcome(
            quality_score=quality_score,
            total_tokens=100,
            total_turns=3,
            wall_clock_ms=5000,
        ),
        metadata={"task_id": task_id},
    )


def _synthetic_result(
    conditions: tuple[str, ...] = ("no_subconscious", "heuristic"),
    qualities: dict[str, list[float]] | None = None,
) -> ComparativeResult:
    tasks = [
        _task("t1", TaskDifficulty.TYPICAL),
        _task("t2", TaskDifficulty.TYPICAL),
        _task("t3", TaskDifficulty.HARD),
        _task("t4", TaskDifficulty.TRICKY),
    ]
    qualities = qualities or {
        c: [0.1 + 0.05 * i + 0.2 * j for i in range(len(tasks))]
        for j, c in enumerate(conditions)
    }
    result = ComparativeResult(tasks=tasks)
    for condition in conditions:
        qs = qualities[condition]
        result.episodes[condition] = [
            _episode(q, task_id=t.task_id) for q, t in zip(qs, tasks)
        ]
        result.metrics[condition] = [_metrics(quality_score=q) for q in qs]
    return result


def _report(result: ComparativeResult | None = None):
    return build_report(
        result or _synthetic_result(),
        dataset="builtin",
        metric="llm_judge",
        answerer={"provider": "gemini", "model": "m"},
        measurement={"provider": "gemini", "model": "m"},
        max_turns=10,
        base_seed=42,
    )


# ---------------------------------------------------------------------------
# Condition wiring
# ---------------------------------------------------------------------------


class TestConditionFactories:
    def test_baseline_factories_construct_expected_types(self):
        factories = baseline_condition_factories(HeuristicController, base_seed=7)
        assert isinstance(factories["no_subconscious"](0), NoSubconsciousController)
        assert isinstance(factories["random"](0), RandomController)
        assert isinstance(factories["heuristic"](0), HeuristicController)

    def test_random_seeds_are_per_condition_and_per_task(self):
        factories = baseline_condition_factories(
            HeuristicController, random_probability=1.0, base_seed=7
        )
        # Same idx -> same decision stream; different idx -> distinct rng seed.
        a = factories["random"](3)
        b = factories["random"](3)
        assert a._rng.random() == b._rng.random()

    def test_condition_seeds_disjoint(self):
        seeds = {condition_seed(42, c) for c in CONDITION_NAMES}
        assert len(seeds) == len(CONDITION_NAMES)

    def test_learned_factories_load_checkpoints(self, tmp_path):
        pytest.importorskip("torch")
        from bicameral_agent.comparative_eval import learned_condition_factories
        from bicameral_agent.learned_controller import LearnedPolicyController
        from bicameral_agent.policy_value_net import PolicyValueNetwork
        from bicameral_agent.training_pipeline import STATE_DIM
        from bicameral_agent.transition_model import TransitionModel

        PolicyValueNetwork(input_dim=STATE_DIM).save(tmp_path / "policy.pt")
        TransitionModel().save(tmp_path / "transition.pt")

        factories = learned_condition_factories(
            str(tmp_path / "policy.pt"),
            str(tmp_path / "transition.pt"),
            num_simulations=5,
        )
        assert set(factories) == {"learned_no_search", "learned_with_search"}
        no_search = factories["learned_no_search"](0)
        with_search = factories["learned_with_search"](0)
        assert isinstance(no_search, LearnedPolicyController)
        assert isinstance(with_search, LearnedPolicyController)
        assert no_search._engine is None
        assert with_search._engine is not None

    def test_all_five_conditions_wired(self, tmp_path):
        pytest.importorskip("torch")
        from bicameral_agent.comparative_eval import learned_condition_factories
        from bicameral_agent.policy_value_net import PolicyValueNetwork
        from bicameral_agent.training_pipeline import STATE_DIM
        from bicameral_agent.transition_model import TransitionModel

        PolicyValueNetwork(input_dim=STATE_DIM).save(tmp_path / "policy.pt")
        TransitionModel().save(tmp_path / "transition.pt")
        conditions = {
            **baseline_condition_factories(HeuristicController),
            **learned_condition_factories(
                str(tmp_path / "policy.pt"), str(tmp_path / "transition.pt")
            ),
        }
        assert tuple(conditions) == CONDITION_NAMES


# ---------------------------------------------------------------------------
# Paired execution
# ---------------------------------------------------------------------------


class TestComparativeEvaluator:
    def _mock_runner(self, seen: list[tuple[str, str]]) -> MagicMock:
        runner = MagicMock(spec=EpisodeRunner)

        def run_episode(task, controller):
            seen.append((type(controller).__name__, task.task_id))
            return _episode(task_id=task.task_id)

        runner.run_episode.side_effect = run_episode
        return runner

    def test_task_order_identical_across_conditions(self):
        seen: list[tuple[str, str]] = []
        runner = self._mock_runner(seen)
        tasks = [_task(f"t{i}") for i in range(3)]
        conditions = {
            "no_subconscious": lambda _idx: NoSubconsciousController(),
            "random": lambda idx: RandomController(seed=idx),
        }
        result = ComparativeEvaluator(runner).run(tasks, conditions)

        orders = {
            condition: [tid for controller_name, tid in seen
                        if controller_name == cls_name]
            for condition, cls_name in [
                ("no_subconscious", "NoSubconsciousController"),
                ("random", "RandomController"),
            ]
        }
        assert orders["no_subconscious"] == orders["random"] == result.task_ids
        assert result.task_ids == ["t0", "t1", "t2"]

    def test_episodes_and_metrics_collected_per_condition(self):
        runner = self._mock_runner([])
        tasks = [_task("t1"), _task("t2")]
        result = ComparativeEvaluator(runner).run(
            tasks, {"no_subconscious": lambda _idx: NoSubconsciousController()}
        )
        assert len(result.episodes["no_subconscious"]) == 2
        assert len(result.metrics["no_subconscious"]) == 2

    def test_on_episode_callback_fires_incrementally(self):
        calls: list[tuple[str, int, str]] = []
        runner = self._mock_runner([])
        evaluator = ComparativeEvaluator(
            runner,
            on_episode=lambda cond, idx, ep: calls.append(
                (cond, idx, ep.metadata["task_id"])
            ),
        )
        evaluator.run(
            [_task("t1"), _task("t2")],
            {"no_subconscious": lambda _idx: NoSubconsciousController()},
        )
        assert calls == [("no_subconscious", 0, "t1"), ("no_subconscious", 1, "t2")]


class _DelayedRunner:
    """Thread-safe EpisodeRunner stand-in with reversed per-task delays.

    Later tasks finish sooner, so with ``parallel_episodes > 1`` completion
    order inverts task order; episode content depends only on the task, so
    a parallel run must collect the exact same episodes as a sequential one.
    """

    def __init__(self, n_tasks: int, delay_step_s: float = 0.01) -> None:
        self._n_tasks = n_tasks
        self._delay_step_s = delay_step_s

    def run_episode(self, task, controller):
        idx = int(task.task_id[1:])
        time.sleep((self._n_tasks - idx) * self._delay_step_s)
        return _episode(quality_score=0.1 * (idx + 1), task_id=task.task_id)


class TestParallelEpisodes:
    """Issue #91 concurrency, reused for the comparative harness.

    run_condition keys results by task index and returns them in task
    order regardless of completion order, so the paired design needs no
    extra work here — asserted by the report-identity test below.
    """

    def _run(self, parallel: int) -> ComparativeResult:
        tasks = [
            _task("t0", TaskDifficulty.TYPICAL),
            _task("t1", TaskDifficulty.TYPICAL),
            _task("t2", TaskDifficulty.HARD),
            _task("t3", TaskDifficulty.TRICKY),
        ]
        conditions = {
            "no_subconscious": lambda _idx: NoSubconsciousController(),
            "random": lambda idx: RandomController(seed=idx),
        }
        evaluator = ComparativeEvaluator(
            _DelayedRunner(len(tasks)), parallel_episodes=parallel
        )
        return evaluator.run(tasks, conditions)

    def test_parallel_report_identical_to_sequential(self):
        seq = _report(self._run(1))
        par = _report(self._run(3))
        assert par.to_json() == seq.to_json()
        assert par.to_markdown() == seq.to_markdown()

    def test_parallel_episodes_stay_in_task_order(self):
        result = self._run(3)
        for condition in ("no_subconscious", "random"):
            assert [
                e.metadata["task_id"] for e in result.episodes[condition]
            ] == result.task_ids

    def test_parallel_episodes_forwarded_to_run_condition(self):
        with patch(
            "bicameral_agent.comparative_eval.run_condition",
            return_value=([], [], []),
        ) as mock_run:
            ComparativeEvaluator(
                MagicMock(spec=EpisodeRunner), parallel_episodes=4
            ).run(
                [_task("t0")],
                {"no_subconscious": lambda _idx: NoSubconsciousController()},
            )
        assert mock_run.call_args.kwargs["parallel_episodes"] == 4

    def test_default_is_sequential(self):
        with patch(
            "bicameral_agent.comparative_eval.run_condition",
            return_value=([], [], []),
        ) as mock_run:
            ComparativeEvaluator(MagicMock(spec=EpisodeRunner)).run(
                [_task("t0")],
                {"no_subconscious": lambda _idx: NoSubconsciousController()},
            )
        assert mock_run.call_args.kwargs["parallel_episodes"] == 1


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


class TestMetricValues:
    def test_derived_rates(self):
        m = _metrics(
            total_turns=4,
            interrupt_count=2,
            tool_invocation_count=4,
            drain_count=3,
            expired_count=1,
            latency_pairs=((100.0, 200.0), (300.0, 200.0)),
        )
        assert metric_values([m], "tool_precision") == [pytest.approx(0.75)]
        assert metric_values([m], "interrupt_rate") == [pytest.approx(0.5)]
        assert metric_values([m], "queue_expiry_rate") == [pytest.approx(0.25)]
        # MAPE: mean(|100-200|/200, |300-200|/200) * 100 = mean(0.5, 0.5)*100
        assert metric_values([m], "latency_mape") == [pytest.approx(50.0)]

    def test_undefined_metrics_are_skipped(self):
        m = _metrics(
            quality_score=None,
            tool_invocation_count=0,
            drain_count=0,
            expired_count=0,
            latency_pairs=(),
        )
        for name in ("task_quality", "tool_precision", "queue_expiry_rate",
                     "latency_mape"):
            assert metric_values([m], name) == []
        # Defined ones still present.
        assert metric_values([m], "token_efficiency") == [100.0]
        assert metric_values([m], "time_to_completion_ms") == [5000.0]

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            metric_values([_metrics()], "nope")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_t_p_value_matches_table(self):
        # t-table: two-tailed critical value at df=10, alpha=0.05 is 2.228.
        assert student_t_two_tailed_p(2.228, 10) == pytest.approx(0.05, abs=1e-3)
        assert student_t_two_tailed_p(0.0, 10) == pytest.approx(1.0)
        # Large |t| -> tiny p.
        assert student_t_two_tailed_p(10.0, 30) < 1e-6

    def test_welch_clear_difference_is_significant(self):
        a = [0.9, 0.85, 0.95, 0.9, 0.88]
        b = [0.1, 0.15, 0.05, 0.1, 0.12]
        t_stat, p_value, significant = welch_test_with_p(a, b)
        assert significant
        assert p_value < 0.001
        assert t_stat > 0

    def test_welch_identical_samples_not_significant(self):
        a = [0.5, 0.5, 0.5]
        t_stat, p_value, significant = welch_test_with_p(a, list(a))
        assert not significant
        assert t_stat == 0.0
        assert p_value == 1.0

    def test_welch_degenerate_inputs(self):
        assert welch_test_with_p([0.5], [0.4, 0.6]) == (0.0, 1.0, False)

    def test_pairwise_covers_all_metric_pairs(self):
        result = _synthetic_result(conditions=("a", "b", "c"))
        tests = pairwise_tests(result.metrics)
        assert len(tests) == len(METRIC_NAMES) * 3  # 3 pairs per metric
        quality = [t for t in tests if t.metric == "task_quality"]
        assert [(t.condition_a, t.condition_b) for t in quality] == [
            ("a", "b"), ("a", "c"), ("b", "c"),
        ]

    def test_pairwise_detects_planted_difference(self):
        low = [_metrics(quality_score=0.1 + 0.01 * i) for i in range(10)]
        high = [_metrics(quality_score=0.9 - 0.01 * i) for i in range(10)]
        tests = pairwise_tests({"low": low, "high": high})
        quality = next(t for t in tests if t.metric == "task_quality")
        assert quality.significant
        assert quality.p_value < 0.05
        # Identical distributions elsewhere are not flagged.
        tokens = next(t for t in tests if t.metric == "token_efficiency")
        assert not tokens.significant


# ---------------------------------------------------------------------------
# Difficulty breakdown
# ---------------------------------------------------------------------------


class TestDifficultyBreakdown:
    def test_groups_by_task_difficulty(self):
        result = _synthetic_result()
        breakdown = difficulty_breakdown(result.tasks, result.metrics)
        assert list(breakdown) == ["typical", "hard", "tricky"]
        for condition in result.metrics:
            assert breakdown["typical"][condition]["task_quality"].n == 2
            assert breakdown["hard"][condition]["task_quality"].n == 1
            assert breakdown["tricky"][condition]["task_quality"].n == 1

    def test_per_tier_means_use_paired_indices(self):
        result = _synthetic_result(
            conditions=("a",), qualities={"a": [0.2, 0.4, 0.6, 0.8]}
        )
        breakdown = difficulty_breakdown(result.tasks, result.metrics)
        assert breakdown["typical"]["a"]["task_quality"].mean == pytest.approx(0.3)
        assert breakdown["hard"]["a"]["task_quality"].mean == pytest.approx(0.6)
        assert breakdown["tricky"]["a"]["task_quality"].mean == pytest.approx(0.8)

    def test_absent_tier_omitted(self):
        tasks = [_task("t1"), _task("t2")]
        metrics = {"a": [_metrics(), _metrics()]}
        assert list(difficulty_breakdown(tasks, metrics)) == ["typical"]


# ---------------------------------------------------------------------------
# Report outputs
# ---------------------------------------------------------------------------


class TestReport:
    def test_json_schema(self):
        payload = json.loads(_report().to_json())
        assert payload["dataset"] == "builtin"
        assert payload["tasks_per_condition"] == 4
        assert payload["base_seed"] == 42
        assert payload["task_mix"] == {"typical": 2, "hard": 1, "tricky": 1}
        assert payload["task_ids"] == ["t1", "t2", "t3", "t4"]
        for condition in ("no_subconscious", "heuristic"):
            summaries = payload["conditions"][condition]["summaries"]
            assert set(summaries) == set(METRIC_NAMES)
            for summary in summaries.values():
                assert set(summary) == {"mean", "std", "ci_lower", "ci_upper", "n"}
        assert {t["metric"] for t in payload["pairwise"]} == set(METRIC_NAMES)
        first = payload["pairwise"][0]
        assert {"condition_a", "condition_b", "t_stat", "p_value",
                "significant"} <= set(first)
        assert set(payload["by_difficulty"]) == {"typical", "hard", "tricky"}
        assert len(payload["results"]) == 8  # 2 conditions x 4 tasks
        assert payload["results"][0]["difficulty"] == "typical"

    def test_summary_values_match_compute_summary(self):
        result = _synthetic_result(conditions=("a",))
        report = _report(result)
        values = metric_values(result.metrics["a"], "task_quality")
        expected = compute_summary(values)
        got = report.conditions["a"]["summaries"]["task_quality"]
        assert got["mean"] == pytest.approx(expected.mean)
        assert got["ci_lower"] == pytest.approx(expected.ci_lower)
        assert got["n"] == expected.n

    def test_markdown_structure(self):
        md = _report().to_markdown()
        assert "# Comparative Evaluation Report" in md
        assert "## Summary" in md
        assert "## Pairwise Welch t-tests: task_quality" in md
        assert "## Breakdown by difficulty" in md
        assert "## Transport failures\n\n- none" in md
        assert "### typical" in md and "### hard" in md and "### tricky" in md
        for name in METRIC_NAMES:
            assert name in md
        for condition in ("no_subconscious", "heuristic"):
            assert condition in md

    def test_downstream_determinism(self):
        result = _synthetic_result()
        first = _report(result)
        second = _report(result)
        assert first.to_json() == second.to_json()
        assert first.to_markdown() == second.to_markdown()

    def test_extends_eval_report(self):
        from bicameral_agent.comparative_eval import ComparativeReport
        from bicameral_agent.eval_report import EvalReport

        assert issubclass(ComparativeReport, EvalReport)

    def test_from_benchmark_is_unsupported(self):
        from bicameral_agent.comparative_eval import ComparativeReport

        with pytest.raises(NotImplementedError, match="build_report"):
            ComparativeReport.from_benchmark(MagicMock())


# ---------------------------------------------------------------------------
# Transport failures and pairing (issue #81 containment)
# ---------------------------------------------------------------------------


class TestTransportFailures:
    """run_condition contains TransportExhausted per episode; a failed task
    is dropped from paired analyses for all conditions and recorded.
    """

    def _failing_runner(self, fail_on: tuple[str, str]) -> MagicMock:
        """Runner that fails one (controller class name, task_id) episode."""
        from bicameral_agent.model_client import TransportExhausted

        runner = MagicMock(spec=EpisodeRunner)

        def run_episode(task, controller):
            if (type(controller).__name__, task.task_id) == fail_on:
                raise TransportExhausted(4, RuntimeError("boom"))
            return _episode(task_id=task.task_id)

        runner.run_episode.side_effect = run_episode
        return runner

    def _run(self):
        # 4 tasks: one failure is 25%, under the default 30% abort threshold.
        runner = self._failing_runner(("RandomController", "t1"))
        tasks = [_task(f"t{i}") for i in range(4)]
        conditions = {
            "no_subconscious": lambda _idx: NoSubconsciousController(),
            "random": lambda idx: RandomController(seed=idx),
        }
        return ComparativeEvaluator(runner).run(tasks, conditions)

    def test_failure_recorded_and_run_continues(self):
        result = self._run()
        assert [f.task_id for f in result.failures["random"]] == ["t1"]
        assert result.failures["random"][0].episode_index == 1
        assert "boom" in result.failures["random"][0].error
        assert result.failures["no_subconscious"] == []
        assert len(result.episodes["random"]) == 3
        assert len(result.episodes["no_subconscious"]) == 4

    def test_failed_task_dropped_from_paired_analyses_for_all_conditions(self):
        result = self._run()
        assert result.paired_indices == [0, 2, 3]
        assert result.excluded_task_ids == ["t1"]
        paired = result.paired_metrics()
        assert len(paired["no_subconscious"]) == 3
        assert len(paired["random"]) == 3
        assert result.completed_indices("random") == [0, 2, 3]
        assert result.completed_indices("no_subconscious") == [0, 1, 2, 3]

    def test_report_records_failures_and_exclusions(self):
        report = _report(self._run())
        assert [f.task_id for f in report.failures] == ["t1"]
        assert report.failures[0].condition == "random"
        assert report.excluded_task_ids == ["t1"]
        # Statistics are computed over the paired subset only...
        for condition in ("no_subconscious", "random"):
            summaries = report.conditions[condition]["summaries"]
            assert summaries["task_quality"]["n"] == 3
        # ...while n_episodes and per-task rows keep every completed episode,
        # mapped back to the right task despite the skipped index.
        assert report.conditions["no_subconscious"]["n_episodes"] == 4
        assert report.conditions["random"]["n_episodes"] == 3
        random_rows = [r for r in report.results if r.condition == "random"]
        assert [r.task_id for r in random_rows] == ["t0", "t2", "t3"]

    def test_markdown_lists_failures_and_paired_coverage(self):
        md = _report(self._run()).to_markdown()
        assert "## Transport failures" in md
        assert "random / t1: " in md
        assert "Paired analyses cover 3/4 tasks" in md
        assert "excluded (failed in >= 1 condition): t1" in md

    def test_failure_threshold_forwarded_to_run_condition(self):
        from bicameral_agent.baseline_benchmark import ConditionAbortedError

        runner = self._failing_runner(("NoSubconsciousController", "t0"))
        evaluator = ComparativeEvaluator(runner, failure_threshold=0.0)
        with pytest.raises(ConditionAbortedError):
            evaluator.run(
                [_task("t0"), _task("t1")],
                {"no_subconscious": lambda _idx: NoSubconsciousController()},
            )


# ---------------------------------------------------------------------------
# Evaluation integrity: judge blinding
# ---------------------------------------------------------------------------


class TestJudgeBlinding:
    """Scoring is LLM-judged by design (no human-eval pathway; issue #53
    pins the judge model), so "human eval blinded" reduces to: the judge
    must not see which condition produced an answer.
    """

    def test_judge_prompt_contains_no_condition_identity(self):
        from bicameral_agent.scorer import TaskScorer

        client = MagicMock()
        response = MagicMock()
        response.content = json.dumps(
            {"quality": 4, "completeness": 3, "accuracy": 5}
        )
        client.generate.return_value = response

        task = _task("t1").model_copy(
            update={"question": "What is the melting point of gallium?"}
        )
        TaskScorer(client=client).score(task, "the agent answer text")

        call = client.generate.call_args
        prompt_text = f"{call.args} {call.kwargs}".lower()
        for condition in CONDITION_NAMES:
            assert condition not in prompt_text
        assert "controller" not in prompt_text
        # The judge sees only task-derived fields and the answer.
        assert task.question.lower() in prompt_text
        assert "the agent answer text" in prompt_text

    def test_runner_and_verifier_have_no_condition_channel(self):
        import inspect

        from bicameral_agent.verifiers import Verifier

        # run_condition passes the condition name only to the persistence
        # callback; run_episode has no parameter that could carry it.
        run_params = inspect.signature(EpisodeRunner.run_episode).parameters
        assert set(run_params) == {"self", "task", "controller"}
        # Verifiers score (task, agent_answer) only.
        score_params = inspect.signature(Verifier.score).parameters
        assert set(score_params) == {"self", "task", "agent_answer"}


# ---------------------------------------------------------------------------
# Task mix / selection
# ---------------------------------------------------------------------------


class TestTaskSelection:
    def _dataset(self, typical=3, hard=2, tricky=2, non_eval_typical=1):
        tasks = (
            [_task(f"ty{i}", TaskDifficulty.TYPICAL) for i in range(typical)]
            + [_task(f"h{i}", TaskDifficulty.HARD) for i in range(hard)]
            + [_task(f"tr{i}", TaskDifficulty.TRICKY) for i in range(tricky)]
            + [
                _task(f"tt{i}", TaskDifficulty.TYPICAL, split=TaskSplit.TOOL_TEST)
                for i in range(non_eval_typical)
            ]
        )
        return ResearchQADataset(tasks)

    def test_parse_plain_integer(self):
        assert parse_task_mix("100") == {
            TaskDifficulty.TYPICAL: 50,
            TaskDifficulty.HARD: 25,
            TaskDifficulty.TRICKY: 25,
        }

    def test_parse_explicit_mix(self):
        assert parse_task_mix("typical=2, hard=1") == {
            TaskDifficulty.TYPICAL: 2,
            TaskDifficulty.HARD: 1,
            TaskDifficulty.TRICKY: 0,
        }

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid task mix"):
            parse_task_mix("weird=3")

    def test_select_is_deterministic_and_ordered(self):
        dataset = self._dataset()
        mix = parse_task_mix("typical=2,hard=1,tricky=1")
        tasks = select_tasks(dataset, mix)
        assert [t.task_id for t in tasks] == ["ty0", "ty1", "h0", "tr0"]
        assert tasks == select_tasks(dataset, mix)

    def test_select_uses_eval_split_only(self):
        tasks = select_tasks(self._dataset(), {TaskDifficulty.TYPICAL: 3})
        assert all(t.split == TaskSplit.EVAL for t in tasks)

    def test_select_raises_on_shortfall(self):
        with pytest.raises(ValueError, match="hard: want 5, have 2"):
            select_tasks(self._dataset(), {TaskDifficulty.HARD: 5})


# ---------------------------------------------------------------------------
# Script wiring: --parallel-episodes
# ---------------------------------------------------------------------------


def _load_comparative_script():
    """Import scripts/run_comparative_eval.py (not a package) by path."""
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_comparative_eval.py"
    )
    spec = importlib.util.spec_from_file_location("run_comparative_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScriptParallelEpisodes:
    """--parallel-episodes resolves CLI > config [run] > 1 and reaches
    ComparativeEvaluator (issue #91 wiring for the comparative script)."""

    def _evaluator_kwargs(self, tmp_path, extra_args=(), config_toml=None):
        script = _load_comparative_script()
        argv = [
            "--output-dir", str(tmp_path / "out"),
            "--tasks", "typical=1",
            "--policy-checkpoint", "policy.pt",
            "--transition-checkpoint", "transition.pt",
            "--quiet",
            *extra_args,
        ]
        if config_toml is not None:
            config_path = tmp_path / "config.toml"
            config_path.write_text(config_toml)
            argv += ["--config", str(config_path)]
        report = MagicMock()
        report.to_json.return_value = "{}"
        report.to_markdown.return_value = "md"
        provenance = {"answerer": {}, "measurement": {}}
        with (
            patch.object(script, "learned_condition_factories", return_value={}),
            patch.object(
                script,
                "resolve_runner_clients",
                return_value=(MagicMock(), MagicMock(), provenance),
            ),
            patch.object(script, "EpisodeRunner"),
            patch.object(script, "ComparativeEvaluator") as mock_evaluator,
            patch.object(script, "build_report", return_value=report),
        ):
            assert script.main(argv) == 0
        return mock_evaluator.call_args.kwargs

    def test_flag_wins_over_config(self, tmp_path):
        kwargs = self._evaluator_kwargs(
            tmp_path,
            extra_args=("--parallel-episodes", "3"),
            config_toml="[run]\nparallel_episodes = 5\n",
        )
        assert kwargs["parallel_episodes"] == 3

    def test_config_used_when_flag_unset(self, tmp_path):
        kwargs = self._evaluator_kwargs(
            tmp_path, config_toml="[run]\nparallel_episodes = 5\n"
        )
        assert kwargs["parallel_episodes"] == 5

    def test_default_is_one(self, tmp_path):
        kwargs = self._evaluator_kwargs(tmp_path)
        assert kwargs["parallel_episodes"] == 1

    def test_rejects_non_positive(self, tmp_path):
        script = _load_comparative_script()
        with pytest.raises(SystemExit):
            script.main(
                [
                    "--output-dir", str(tmp_path / "out"),
                    "--tasks", "typical=1",
                    "--policy-checkpoint", "policy.pt",
                    "--transition-checkpoint", "transition.pt",
                    "--parallel-episodes", "0",
                ]
            )
