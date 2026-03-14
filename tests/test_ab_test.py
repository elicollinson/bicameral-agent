"""Tests for the A/B test framework."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from bicameral_agent.ab_test import (
    ABTestResult,
    ABTestRunner,
    Condition,
    EpisodeMetrics,
    MetricSummary,
    compute_summary,
    count_derailments,
    default_conditions,
    extract_metrics,
    t_critical_95,
    welch_t_test,
)
from bicameral_agent.coherence_judge import CoherenceScore
from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.episode_runner import (
    Controller,
    EpisodeConfig,
    EpisodeRunner,
    InjectionMode,
)
from bicameral_agent.gemini import GeminiClient, GeminiResponse
from bicameral_agent.heuristic_controller import Action, HeuristicController
from bicameral_agent.schema import (
    Episode,
    EpisodeOutcome,
    Message,
    UserEventType,
)


def _make_task(**overrides) -> ResearchQATask:
    defaults = dict(
        task_id="test-001",
        difficulty=TaskDifficulty.TYPICAL,
        split=TaskSplit.EVAL,
        question="What is photosynthesis?",
        gold_answer="Photosynthesis converts light to chemical energy.",
        scoring_rubric="5: Complete. 3: Partial. 1: Wrong.",
    )
    defaults.update(overrides)
    return ResearchQATask(**defaults)


def _make_episode(
    quality_score=0.75,
    total_tokens=1000,
    total_turns=5,
    wall_clock_ms=5000.0,
    messages=None,
    metadata=None,
) -> Episode:
    if messages is None:
        messages = [
            Message(role="user", content="What is photosynthesis?", timestamp_ms=0, token_count=4),
            Message(role="assistant", content="It is a process in plants.", timestamp_ms=100, token_count=6),
        ]
    return Episode(
        episode_id="test-ep-001",
        messages=messages,
        user_events=[],
        context_injections=[],
        tool_invocations=[],
        outcome=EpisodeOutcome(
            quality_score=quality_score,
            total_tokens=total_tokens,
            total_turns=total_turns,
            wall_clock_ms=wall_clock_ms,
        ),
        metadata=metadata or {"interrupt_count": 0, "injection_mode": "breakpoint"},
    )


# ---------------------------------------------------------------------------
# TestCondition
# ---------------------------------------------------------------------------


class TestCondition:
    def test_construction(self):
        cond = Condition(
            name="test",
            controller_factory=HeuristicController,
            episode_config=EpisodeConfig(injection_mode=InjectionMode.SYNCHRONOUS),
        )
        assert cond.name == "test"
        assert cond.injection_mode == InjectionMode.SYNCHRONOUS

    def test_frozen(self):
        cond = Condition(
            name="test",
            controller_factory=HeuristicController,
            episode_config=EpisodeConfig(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cond.name = "changed"

    def test_default_conditions(self):
        conditions = default_conditions(HeuristicController)
        assert len(conditions) == 3
        names = {c.name for c in conditions}
        assert names == {"synchronous", "breakpoint", "interrupt"}


# ---------------------------------------------------------------------------
# TestTCritical
# ---------------------------------------------------------------------------


class TestTCritical:
    def test_known_values(self):
        assert t_critical_95(1) == 12.706
        assert t_critical_95(10) == 2.228
        assert t_critical_95(30) == 2.042

    def test_large_df(self):
        assert t_critical_95(1000) == 1.96
        assert t_critical_95(5000) == 1.96

    def test_interpolation(self):
        # df=5 → 2.571, df=6 → 2.447
        val = t_critical_95(5)
        assert val == 2.571
        # Between 5 and 6: should interpolate
        val_mid = t_critical_95(5)
        assert 2.4 <= val_mid <= 2.6

    def test_invalid_df(self):
        with pytest.raises(ValueError):
            t_critical_95(0)


# ---------------------------------------------------------------------------
# TestMetricSummary
# ---------------------------------------------------------------------------


class TestMetricSummary:
    def test_empty_values(self):
        result = compute_summary([])
        assert result.n == 0
        assert result.mean == 0.0

    def test_single_value(self):
        result = compute_summary([5.0])
        assert result.n == 1
        assert result.mean == 5.0
        assert result.ci_lower == 5.0
        assert result.ci_upper == 5.0

    def test_known_data(self):
        # Known: [2, 4, 6, 8, 10] → mean=6, std≈3.162
        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = compute_summary(values)
        assert result.n == 5
        assert result.mean == pytest.approx(6.0)
        assert result.std == pytest.approx(math.sqrt(10), rel=1e-3)

    def test_ci_bounds(self):
        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = compute_summary(values)
        # CI should be symmetric around mean
        margin = result.ci_upper - result.mean
        assert margin == pytest.approx(result.mean - result.ci_lower, rel=1e-10)
        # CI should contain the mean
        assert result.ci_lower < result.mean < result.ci_upper

    def test_hand_calculated_ci(self):
        # [10, 20, 30] → mean=20, std=10, n=3, df=2, t_crit=4.303
        # margin = 4.303 * (10/sqrt(3)) ≈ 24.841
        values = [10.0, 20.0, 30.0]
        result = compute_summary(values)
        assert result.mean == pytest.approx(20.0)
        assert result.std == pytest.approx(10.0)
        expected_margin = 4.303 * (10.0 / math.sqrt(3))
        assert result.ci_lower == pytest.approx(20.0 - expected_margin, rel=1e-3)
        assert result.ci_upper == pytest.approx(20.0 + expected_margin, rel=1e-3)


# ---------------------------------------------------------------------------
# TestWelchTTest
# ---------------------------------------------------------------------------


class TestWelchTTest:
    def test_identical_samples(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        t_stat, sig = welch_t_test(a, a)
        assert t_stat == 0.0
        assert not sig

    def test_different_samples(self):
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        t_stat, sig = welch_t_test(a, b)
        assert t_stat > 0
        assert sig  # these should be significantly different

    def test_insufficient_samples(self):
        t_stat, sig = welch_t_test([1.0], [2.0])
        assert t_stat == 0.0
        assert not sig

    def test_significance_detection(self):
        # Large separation, small variance → significant
        a = [100.0, 100.1, 100.2, 99.9, 100.0]
        b = [0.0, 0.1, 0.2, -0.1, 0.0]
        _, sig = welch_t_test(a, b)
        assert sig


# ---------------------------------------------------------------------------
# TestDerailmentCounting
# ---------------------------------------------------------------------------


class TestDerailmentCounting:
    def test_no_derailments(self):
        messages = [
            Message(role="user", content="What is photosynthesis?", timestamp_ms=0, token_count=4),
            Message(role="assistant", content="It converts light.", timestamp_ms=100, token_count=3),
            Message(role="user", content="Can you elaborate more?", timestamp_ms=200, token_count=4),
        ]
        assert count_derailments(messages) == 0

    def test_redirect_counted(self):
        messages = [
            Message(role="user", content="What is photosynthesis?", timestamp_ms=0, token_count=4),
            Message(role="assistant", content="It converts light.", timestamp_ms=100, token_count=3),
            Message(role="user", content="Actually, let's talk about something else instead.", timestamp_ms=200, token_count=8),
        ]
        assert count_derailments(messages) >= 1

    def test_correction_counted(self):
        messages = [
            Message(role="user", content="What is photosynthesis?", timestamp_ms=0, token_count=4),
            Message(role="assistant", content="It converts light.", timestamp_ms=100, token_count=3),
            Message(role="user", content="No, that's wrong. It also involves carbon dioxide.", timestamp_ms=200, token_count=9),
        ]
        assert count_derailments(messages) >= 1


# ---------------------------------------------------------------------------
# TestMetricExtraction
# ---------------------------------------------------------------------------


class TestMetricExtraction:
    def test_extract_metrics(self):
        episode = _make_episode()
        coherence = CoherenceScore(logical_flow=0.8, consistency=0.9, overall=0.85)
        metrics = extract_metrics(episode, "breakpoint", "test-001", coherence)

        assert metrics.task_id == "test-001"
        assert metrics.condition == "breakpoint"
        assert metrics.quality_score == 0.75
        assert metrics.total_tokens == 1000
        assert metrics.coherence_score == 0.85
        assert metrics.total_turns == 5
        assert metrics.interrupt_count == 0

    def test_extract_with_interrupt_count(self):
        episode = _make_episode(metadata={"interrupt_count": 3, "injection_mode": "interrupt"})
        coherence = CoherenceScore(logical_flow=0.5, consistency=0.5, overall=0.5)
        metrics = extract_metrics(episode, "interrupt", "test-001", coherence)
        assert metrics.interrupt_count == 3


# ---------------------------------------------------------------------------
# TestABTestRunner
# ---------------------------------------------------------------------------


class TestABTestRunner:
    def _make_mock_runner(self):
        """Create ABTestRunner with fully mocked dependencies."""
        client = MagicMock(spec=GeminiClient)
        # Mock the generate for coherence judge
        client.generate.return_value = GeminiResponse(
            content=json.dumps({"logical_flow": 4, "consistency": 4, "overall": 4}),
            input_tokens=10,
            output_tokens=20,
            duration_ms=100.0,
            finish_reason="STOP",
        )
        return client

    def test_all_conditions_run(self):
        client = self._make_mock_runner()
        tasks = [_make_task(task_id="t1"), _make_task(task_id="t2")]

        episodes_created = []

        def mock_run_episode(task, controller):
            ep = _make_episode(
                quality_score=0.8,
                metadata={"interrupt_count": 0, "injection_mode": "breakpoint"},
            )
            episodes_created.append(ep)
            return ep

        conditions = [
            Condition("sync", HeuristicController, EpisodeConfig(injection_mode=InjectionMode.SYNCHRONOUS)),
            Condition("bp", HeuristicController, EpisodeConfig()),
        ]

        runner = ABTestRunner(client, score_episodes=True)

        with patch.object(EpisodeRunner, "run_episode", side_effect=mock_run_episode):
            result = runner.run(tasks, conditions)

        assert isinstance(result, ABTestResult)
        # 2 conditions x 2 tasks = 4 episodes
        assert len(result.episode_metrics) == 4
        assert len(result.conditions) == 2

    def test_same_tasks_across_conditions(self):
        client = self._make_mock_runner()
        tasks = [_make_task(task_id="t1"), _make_task(task_id="t2")]

        task_ids_per_condition: dict[str, list[str]] = {}

        def mock_run_episode(task, controller):
            return _make_episode(
                metadata={"interrupt_count": 0, "injection_mode": "breakpoint"},
            )

        conditions = default_conditions(HeuristicController)

        runner = ABTestRunner(client)

        with patch.object(EpisodeRunner, "run_episode", side_effect=mock_run_episode):
            result = runner.run(tasks, conditions)

        # Each condition should have metrics for both tasks
        for cond_name in ["synchronous", "breakpoint", "interrupt"]:
            cond_tasks = [
                m["task_id"] for m in result.episode_metrics if m["condition"] == cond_name
            ]
            assert set(cond_tasks) == {"t1", "t2"}

    def test_fresh_controller_per_episode(self):
        client = self._make_mock_runner()
        tasks = [_make_task(task_id="t1"), _make_task(task_id="t2")]
        controllers_created = []

        def factory():
            ctrl = HeuristicController()
            controllers_created.append(ctrl)
            return ctrl

        conditions = [
            Condition("bp", factory, EpisodeConfig()),
        ]

        def mock_run_episode(task, controller):
            return _make_episode(
                metadata={"interrupt_count": 0, "injection_mode": "breakpoint"},
            )

        runner = ABTestRunner(client)

        with patch.object(EpisodeRunner, "run_episode", side_effect=mock_run_episode):
            runner.run(tasks, conditions)

        # One fresh controller per task
        assert len(controllers_created) == 2
        assert controllers_created[0] is not controllers_created[1]

    def test_best_condition_selection(self):
        client = self._make_mock_runner()
        tasks = [_make_task(task_id="t1")]

        call_count = 0

        def mock_run_episode(task, controller):
            nonlocal call_count
            # First condition (sync) gets quality=0.9, second (bp) gets quality=0.5
            q = 0.9 if call_count == 0 else 0.5
            call_count += 1
            return _make_episode(
                quality_score=q,
                metadata={"interrupt_count": 0, "injection_mode": "breakpoint"},
            )

        conditions = [
            Condition("sync", HeuristicController, EpisodeConfig(injection_mode=InjectionMode.SYNCHRONOUS)),
            Condition("bp", HeuristicController, EpisodeConfig()),
        ]

        runner = ABTestRunner(client)

        with patch.object(EpisodeRunner, "run_episode", side_effect=mock_run_episode):
            result = runner.run(tasks, conditions)

        assert result.best_condition == "sync"

    def test_json_output(self):
        client = self._make_mock_runner()
        tasks = [_make_task(task_id="t1")]

        def mock_run_episode(task, controller):
            return _make_episode(
                metadata={"interrupt_count": 0, "injection_mode": "breakpoint"},
            )

        conditions = [
            Condition("bp", HeuristicController, EpisodeConfig()),
        ]
        runner = ABTestRunner(client)

        with patch.object(EpisodeRunner, "run_episode", side_effect=mock_run_episode):
            result = runner.run(tasks, conditions)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "conditions" in data
            assert "episode_metrics" in data
            assert "best_condition" in data
        finally:
            os.unlink(path)

    def test_csv_output(self):
        client = self._make_mock_runner()
        tasks = [_make_task(task_id="t1"), _make_task(task_id="t2")]

        def mock_run_episode(task, controller):
            return _make_episode(
                metadata={"interrupt_count": 0, "injection_mode": "breakpoint"},
            )

        conditions = [
            Condition("bp", HeuristicController, EpisodeConfig()),
        ]
        runner = ABTestRunner(client)

        with patch.object(EpisodeRunner, "run_episode", side_effect=mock_run_episode):
            result = runner.run(tasks, conditions)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            result.to_csv(path)
            with open(path) as f:
                lines = f.readlines()
            # Header + 2 data rows
            assert len(lines) == 3
            assert "task_id" in lines[0]
        finally:
            os.unlink(path)

    def test_justification_contains_stats(self):
        client = self._make_mock_runner()
        tasks = [_make_task(task_id="t1")]

        def mock_run_episode(task, controller):
            return _make_episode(
                metadata={"interrupt_count": 0, "injection_mode": "breakpoint"},
            )

        conditions = default_conditions(HeuristicController)
        runner = ABTestRunner(client)

        with patch.object(EpisodeRunner, "run_episode", side_effect=mock_run_episode):
            result = runner.run(tasks, conditions)

        assert "quality=" in result.justification
        assert "tokens=" in result.justification
        assert "coherence=" in result.justification


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

_has_key = os.environ.get("GEMINI_API_KEY") is not None


@pytest.mark.skipif(not _has_key, reason="GEMINI_API_KEY not set")
class TestIntegration:
    def test_full_ab_test(self):
        client = GeminiClient()
        tasks = [
            _make_task(task_id="int-t1"),
            _make_task(task_id="int-t2", question="What is gravity?",
                       gold_answer="Gravity is a fundamental force of attraction."),
        ]
        conditions = default_conditions(HeuristicController)
        # Use short episodes for speed
        conditions = [
            Condition(
                name=c.name,
                controller_factory=c.controller_factory,
                episode_config=dataclasses.replace(c.episode_config, max_turns=3),
            )
            for c in conditions
        ]
        runner = ABTestRunner(client, use_lexical_scorer=True)
        result = runner.run(tasks, conditions)

        assert isinstance(result, ABTestResult)
        assert len(result.conditions) == 3
        assert len(result.episode_metrics) == 6  # 3 conditions x 2 tasks
        assert result.best_condition in {"synchronous", "breakpoint", "interrupt"}
