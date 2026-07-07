"""Tests for baseline_benchmark — Issue #23."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bicameral_agent.baseline_benchmark import (
    aggregate,
    BenchmarkResult,
    ConditionAbortedError,
    CONDITION_NAMES,
    EpisodeFailure,
    extract_task_metrics,
    format_report,
    heuristic_outperforms,
    latency_mape,
    parse_conditions,
    run_benchmark,
    run_condition,
    TaskMetrics,
)
from bicameral_agent.dataset import ResearchQATask, TaskDifficulty, TaskSplit
from bicameral_agent.episode_runner import Controller, EpisodeRunner
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.heuristic_controller import (
    Action,
    DecisionLog,
    FullState,
    TOOL_IDS,
)
from bicameral_agent.model_client import TransportExhausted
from bicameral_agent.schema import (
    ContextInjection,
    Episode,
    EpisodeOutcome,
    ToolInvocation,
    UserEvent,
    UserEventType,
)


def _state(predicted: dict[str, float] | None = None, queue_depth: int = 0) -> FullState:
    return FullState(
        turn_number=1,
        stop_count=0,
        followup_type=FollowUpType.ELABORATION,
        queue_depth=queue_depth,
        executing_tools=(),
        predicted_latencies=predicted or {},
    )


def _decision(action: Action, predicted: dict[str, float] | None = None,
              queue_depth: int = 0) -> DecisionLog:
    return DecisionLog(
        action=action,
        rule_fired=0,
        state=_state(predicted, queue_depth),
        timestamp_ms=0.0,
    )


def _make_episode(
    *,
    quality_score: float | None = 0.7,
    total_tokens: int = 100,
    total_turns: int = 3,
    wall_clock_ms: int = 5000,
    tool_invocations: list[ToolInvocation] | None = None,
    context_injections: list[ContextInjection] | None = None,
    user_events: list[UserEvent] | None = None,
    metadata: dict | None = None,
) -> Episode:
    return Episode(
        outcome=EpisodeOutcome(
            quality_score=quality_score,
            total_tokens=total_tokens,
            total_turns=total_turns,
            wall_clock_ms=wall_clock_ms,
        ),
        tool_invocations=tool_invocations or [],
        context_injections=context_injections or [],
        user_events=user_events or [],
        metadata=metadata or {},
    )


def _task(task_id: str = "t1") -> ResearchQATask:
    return ResearchQATask(
        task_id=task_id,
        difficulty=TaskDifficulty.TYPICAL,
        split=TaskSplit.EVAL,
        question="q",
        gold_answer="a",
        scoring_rubric="rubric",
    )


class TestExtractTaskMetrics:
    def test_basic_fields(self):
        episode = _make_episode(
            quality_score=0.5,
            total_tokens=200,
            total_turns=4,
            wall_clock_ms=2000,
            user_events=[
                UserEvent(event_type=UserEventType.STOP, timestamp_ms=10),
                UserEvent(event_type=UserEventType.FOLLOW_UP, timestamp_ms=20),
            ],
            metadata={
                "interrupt_count": 2,
                "expired_queue_items": 1,
                "episode_cost": {"total": 0.0123},
            },
        )
        decisions = [_decision(Action.DO_NOTHING, queue_depth=0),
                     _decision(Action.DO_NOTHING, queue_depth=2)]
        m = extract_task_metrics(episode, decisions)
        assert m.quality_score == 0.5
        assert m.total_tokens == 200
        assert m.total_turns == 4
        assert m.user_stops == 1
        assert m.task_completed == 0
        assert m.wall_clock_ms == 2000
        assert m.tool_invocation_count == 0
        assert m.tool_cost_usd == pytest.approx(0.0123)
        assert m.avg_queue_depth == 1.0
        assert m.interrupt_count == 2
        assert m.expired_count == 1

    def test_drain_count_from_consumed_injections(self):
        injections = [
            ContextInjection(
                content="c", source_tool_id="x", priority=1,
                timestamp_ms=1, token_count=5, consumed=True, consumed_at_turn=1,
            ),
            ContextInjection(
                content="c2", source_tool_id="x", priority=1,
                timestamp_ms=2, token_count=5, consumed=False,
            ),
        ]
        episode = _make_episode(context_injections=injections)
        m = extract_task_metrics(episode, [])
        assert m.drain_count == 1

    def test_latency_pairs_align_with_invocations(self):
        scanner_id = TOOL_IDS[Action.SCANNER]
        auditor_id = TOOL_IDS[Action.AUDITOR]
        episode = _make_episode(
            tool_invocations=[
                ToolInvocation(
                    tool_id=scanner_id, invoked_at_ms=100, completed_at_ms=600,
                    input_tokens=0, output_tokens=10,
                ),
                ToolInvocation(
                    tool_id=auditor_id, invoked_at_ms=700, completed_at_ms=900,
                    input_tokens=0, output_tokens=5,
                ),
            ],
        )
        decisions = [
            _decision(Action.DO_NOTHING),
            _decision(Action.SCANNER, predicted={scanner_id: 400.0}),
            _decision(Action.DO_NOTHING),
            _decision(Action.AUDITOR, predicted={auditor_id: 250.0}),
        ]
        m = extract_task_metrics(episode, decisions)
        assert m.latency_pairs == ((400.0, 500.0), (250.0, 200.0))

    def test_latency_pairs_drop_zero_duration(self):
        scanner_id = TOOL_IDS[Action.SCANNER]
        episode = _make_episode(
            tool_invocations=[
                ToolInvocation(
                    tool_id=scanner_id, invoked_at_ms=100, completed_at_ms=100,
                    input_tokens=0, output_tokens=0,
                ),
            ],
        )
        decisions = [_decision(Action.SCANNER, predicted={scanner_id: 400.0})]
        m = extract_task_metrics(episode, decisions)
        assert m.latency_pairs == ()

    def test_latency_pairs_drop_budget_exceeded(self):
        """Budget-exceeded invocations are excluded even with positive durations."""
        scanner_id = TOOL_IDS[Action.SCANNER]
        episode = _make_episode(
            tool_invocations=[
                ToolInvocation(
                    tool_id=scanner_id, invoked_at_ms=100, completed_at_ms=600,
                    input_tokens=0, output_tokens=0, budget_exceeded=True,
                ),
            ],
        )
        decisions = [_decision(Action.SCANNER, predicted={scanner_id: 400.0})]
        m = extract_task_metrics(episode, decisions)
        assert m.latency_pairs == ()

    def test_latency_pairs_drop_tool_id_mismatch(self):
        """A decision/invocation tool_id mismatch means misalignment; drop it."""
        scanner_id = TOOL_IDS[Action.SCANNER]
        auditor_id = TOOL_IDS[Action.AUDITOR]
        episode = _make_episode(
            tool_invocations=[
                ToolInvocation(
                    tool_id=auditor_id, invoked_at_ms=100, completed_at_ms=600,
                    input_tokens=0, output_tokens=10,
                ),
            ],
        )
        decisions = [_decision(Action.SCANNER, predicted={scanner_id: 400.0})]
        m = extract_task_metrics(episode, decisions)
        assert m.latency_pairs == ()

    def test_task_completed_extracted(self):
        episode = _make_episode(
            user_events=[
                UserEvent(event_type=UserEventType.TASK_COMPLETE, timestamp_ms=10),
            ],
        )
        m = extract_task_metrics(episode, [])
        assert m.task_completed == 1


def _metrics(latency_pairs=(), **fields) -> TaskMetrics:
    defaults = dict(
        quality_score=0.5,
        total_tokens=100,
        total_turns=3,
        user_stops=0,
        task_completed=0,
        wall_clock_ms=1000,
        tool_invocation_count=0,
        tool_cost_usd=0.0,
        avg_queue_depth=0.0,
        interrupt_count=0,
        drain_count=0,
        expired_count=0,
        latency_pairs=tuple(latency_pairs),
    )
    defaults.update(fields)
    return TaskMetrics(**defaults)


class TestLatencyMAPE:
    def test_empty(self):
        assert latency_mape([]) == (0.0, 0)
        assert latency_mape([_metrics()]) == (0.0, 0)

    def test_basic(self):
        # |400-500|/500 = 0.20 ; |250-200|/200 = 0.25 → mean = 0.225 = 22.5%
        m = _metrics(latency_pairs=[(400.0, 500.0), (250.0, 200.0)])
        mape, n = latency_mape([m])
        assert mape == pytest.approx(22.5)
        assert n == 2

    def test_perfect_predictions(self):
        m = _metrics(latency_pairs=[(100.0, 100.0)])
        assert latency_mape([m]) == (0.0, 1)


class TestAggregate:
    def test_quality_score_drops_none(self):
        report = aggregate("c", [_metrics(quality_score=0.5),
                                 _metrics(quality_score=None),
                                 _metrics(quality_score=0.7)])
        assert report.summaries["quality_score"].n == 2
        assert report.summaries["quality_score"].mean == pytest.approx(0.6)

    def test_all_metrics_summarized(self):
        report = aggregate("c", [_metrics() for _ in range(3)])
        assert report.n_episodes == 3
        for name in ("quality_score", "total_tokens", "tool_cost_usd",
                     "avg_queue_depth", "interrupt_count", "drain_count",
                     "expired_count", "task_completed"):
            assert name in report.summaries


def _quality_reports(
    heuristic: list[float], baseline: list[float], name: str = "random"
) -> dict:
    return {
        "heuristic": aggregate(
            "heuristic", [_metrics(quality_score=q) for q in heuristic]
        ),
        name: aggregate(name, [_metrics(quality_score=q) for q in baseline]),
    }


class TestComparisons:
    def test_heuristic_outperforms_random(self):
        # Clearly separated samples: higher mean AND Welch-significant.
        reports = _quality_reports([0.8, 0.82, 0.78], [0.4, 0.42, 0.38])
        assert heuristic_outperforms(reports, "random") is True

    def test_heuristic_does_not_outperform(self):
        reports = _quality_reports([0.3, 0.32, 0.28], [0.4, 0.42, 0.38])
        assert heuristic_outperforms(reports, "random") is False

    def test_higher_mean_but_not_significant(self):
        # Means differ slightly but overlap heavily: no winner declared.
        reports = _quality_reports([0.5, 0.7, 0.3], [0.45, 0.65, 0.25])
        assert heuristic_outperforms(reports, "random") is False

    def test_single_episode_is_never_significant(self):
        # n=1 per condition cannot reach significance under Welch's t-test.
        reports = _quality_reports([0.8], [0.4])
        assert heuristic_outperforms(reports, "random") is False

    def test_missing_baseline_returns_false(self):
        reports = {"heuristic": aggregate("heuristic", [_metrics()])}
        assert heuristic_outperforms(reports, "random") is False


class TestFormatReport:
    def test_includes_all_conditions(self):
        def _report(name: str, qualities: list[float]):
            return aggregate(name, [_metrics(quality_score=q) for q in qualities])

        result = BenchmarkResult()
        result.reports = {
            "heuristic": _report("heuristic", [0.6, 0.62, 0.58]),
            "random": _report("random", [0.4, 0.42, 0.38]),
            "no_subconscious": _report("no_subconscious", [0.3, 0.32, 0.28]),
        }
        text = format_report(result)
        assert "heuristic" in text
        assert "random" in text
        assert "no_subconscious" in text
        assert "MAPE" in text
        assert "quality_score" in text
        assert "heuristic > random on quality_score (Welch 95%): YES" in text
        assert "heuristic > no_subconscious on quality_score (Welch 95%): YES" in text

    def test_subset_compares_only_present_conditions(self):
        """A --conditions subset run reports comparisons among present conditions only."""
        result = BenchmarkResult()
        result.reports = {
            "heuristic": aggregate(
                "heuristic", [_metrics(quality_score=q) for q in (0.6, 0.62, 0.58)]
            ),
            "random": aggregate(
                "random", [_metrics(quality_score=q) for q in (0.4, 0.42, 0.38)]
            ),
        }
        text = format_report(result)
        assert "heuristic > random on quality_score" in text
        assert "no_subconscious" not in text

    def test_single_condition_report_has_no_comparisons(self):
        result = BenchmarkResult()
        result.reports = {
            "no_subconscious": aggregate("no_subconscious", [_metrics(), _metrics()])
        }
        text = format_report(result)
        assert "no_subconscious" in text
        assert "heuristic >" not in text

    def test_lists_transport_failures(self):
        result = BenchmarkResult()
        result.reports = {"random": aggregate("random", [_metrics()])}
        result.failures = {
            "random": [
                EpisodeFailure(
                    condition="random", episode_index=1, task_id="t7",
                    error="transport error persisted after 4 attempts",
                )
            ]
        }
        text = format_report(result)
        assert "transport_failures       n=1 (tasks: t7)" in text


class TestParseConditions:
    def test_default_spec_returns_all(self):
        assert parse_conditions(",".join(CONDITION_NAMES)) == CONDITION_NAMES

    def test_single_condition(self):
        assert parse_conditions("heuristic") == ("heuristic",)

    def test_canonical_order_and_dedup(self):
        assert parse_conditions("heuristic, random,heuristic") == (
            "random",
            "heuristic",
        )

    def test_unknown_condition_raises(self):
        with pytest.raises(ValueError, match="Unknown condition"):
            parse_conditions("heuristic,bogus")

    def test_empty_spec_raises(self):
        with pytest.raises(ValueError, match="No conditions"):
            parse_conditions(" , ")


class _StubController:
    def __init__(self) -> None:
        self._decisions: list[DecisionLog] = [_decision(Action.DO_NOTHING)]

    def decide(self, state: FullState) -> Action:
        return Action.DO_NOTHING

    @property
    def decisions(self) -> list[DecisionLog]:
        return list(self._decisions)


class TestRunCondition:
    def test_runs_each_task_with_fresh_controller(self):
        runner = MagicMock(spec=EpisodeRunner)
        runner.run_episode.side_effect = lambda task, ctrl: _make_episode()

        factory_calls: list[int] = []

        def factory(idx: int) -> Controller:
            factory_calls.append(idx)
            return _StubController()

        tasks = [_task(), _task(), _task()]
        episodes, metrics, failures = run_condition(runner, tasks, factory)

        assert factory_calls == [0, 1, 2]
        assert len(episodes) == 3
        assert len(metrics) == 3
        assert failures == []
        assert runner.run_episode.call_count == 3

    def test_run_benchmark_aggregates_each_condition(self):
        runner = MagicMock(spec=EpisodeRunner)
        runner.run_episode.side_effect = lambda task, ctrl: _make_episode(
            quality_score=0.5
        )

        result = run_benchmark(
            client=MagicMock(),
            tasks=[_task(), _task()],
            conditions={
                "heuristic": lambda _idx: _StubController(),
                "random": lambda _idx: _StubController(),
            },
            runner=runner,
        )
        assert set(result.episodes) == {"heuristic", "random"}
        assert all(len(eps) == 2 for eps in result.episodes.values())
        assert result.reports["heuristic"].n_episodes == 2


class TestIncrementalPersistence:
    """on_episode fires per completed episode so a late crash keeps prior results."""

    def test_on_episode_called_per_episode(self):
        runner = MagicMock(spec=EpisodeRunner)
        runner.run_episode.side_effect = lambda task, ctrl: _make_episode()

        seen: list[tuple[str, int, str]] = []
        run_condition(
            runner,
            [_task(), _task()],
            lambda _idx: _StubController(),
            condition="heuristic",
            on_episode=lambda cond, idx, ep: seen.append((cond, idx, ep.episode_id)),
        )
        assert [(c, i) for c, i, _ in seen] == [("heuristic", 0), ("heuristic", 1)]

    def test_completed_episodes_persisted_before_crash(self):
        """Episodes completed before a mid-run crash have already been reported."""
        runner = MagicMock(spec=EpisodeRunner)
        episode = _make_episode()
        runner.run_episode.side_effect = [episode, RuntimeError("API meltdown")]

        seen: list[Episode] = []
        with pytest.raises(RuntimeError, match="API meltdown"):
            run_condition(
                runner,
                [_task(), _task()],
                lambda _idx: _StubController(),
                condition="random",
                on_episode=lambda cond, idx, ep: seen.append(ep),
            )
        assert seen == [episode]

    def test_run_benchmark_forwards_callback_with_condition(self):
        runner = MagicMock(spec=EpisodeRunner)
        runner.run_episode.side_effect = lambda task, ctrl: _make_episode()

        seen: list[str] = []
        run_benchmark(
            client=MagicMock(),
            tasks=[_task()],
            conditions={
                "heuristic": lambda _idx: _StubController(),
                "random": lambda _idx: _StubController(),
            },
            runner=runner,
            on_episode=lambda cond, idx, ep: seen.append(cond),
        )
        assert seen == ["heuristic", "random"]


def _transport_exhausted() -> TransportExhausted:
    return TransportExhausted(4, TimeoutError("read timed out"))


class TestTransportFailureContainment:
    """Issue #81: a transport failure that outlives client retries fails only
    its episode; the condition continues unless failures exceed the threshold."""

    def test_failed_episode_recorded_and_run_continues(self):
        runner = MagicMock(spec=EpisodeRunner)
        episode = _make_episode()
        runner.run_episode.side_effect = [
            episode, _transport_exhausted(), episode, episode
        ]
        tasks = [_task(f"t{i}") for i in range(4)]

        episodes, metrics, failures = run_condition(
            runner, tasks, lambda _idx: _StubController(), condition="random"
        )

        assert runner.run_episode.call_count == 4
        assert len(episodes) == 3
        assert len(metrics) == 3
        (failure,) = failures
        assert failure.condition == "random"
        assert failure.episode_index == 1
        assert failure.task_id == "t1"
        assert "read timed out" in failure.error

    def test_on_episode_skips_failed_episode(self):
        runner = MagicMock(spec=EpisodeRunner)
        episode = _make_episode()
        runner.run_episode.side_effect = [
            episode, _transport_exhausted(), episode, episode
        ]

        seen: list[int] = []
        run_condition(
            runner,
            [_task(f"t{i}") for i in range(4)],
            lambda _idx: _StubController(),
            condition="random",
            on_episode=lambda cond, idx, ep: seen.append(idx),
        )
        assert seen == [0, 2, 3]

    def test_threshold_abort_names_failure_rate(self):
        runner = MagicMock(spec=EpisodeRunner)
        runner.run_episode.side_effect = _transport_exhausted()
        tasks = [_task(f"t{i}") for i in range(3)]

        # 1 failure out of 3 tasks (33%) already exceeds the 30% default.
        with pytest.raises(ConditionAbortedError, match=r"1/3 episodes \(33%\)"):
            run_condition(
                runner, tasks, lambda _idx: _StubController(), condition="random"
            )
        assert runner.run_episode.call_count == 1

    def test_failures_below_threshold_do_not_abort(self):
        runner = MagicMock(spec=EpisodeRunner)
        episode = _make_episode()
        # 3 failures out of 10 tasks = 30%, not strictly above the threshold.
        effects: list = [_transport_exhausted()] * 3 + [episode] * 7
        runner.run_episode.side_effect = effects

        episodes, _, failures = run_condition(
            runner,
            [_task(f"t{i}") for i in range(10)],
            lambda _idx: _StubController(),
            condition="heuristic",
        )
        assert len(episodes) == 7
        assert len(failures) == 3

    def test_abort_preserves_transport_error_as_cause(self):
        runner = MagicMock(spec=EpisodeRunner)
        runner.run_episode.side_effect = _transport_exhausted()

        with pytest.raises(ConditionAbortedError) as exc_info:
            run_condition(
                runner,
                [_task()],
                lambda _idx: _StubController(),
                condition="random",
            )
        assert isinstance(exc_info.value.__cause__, TransportExhausted)

    def test_non_transport_error_still_propagates(self):
        runner = MagicMock(spec=EpisodeRunner)
        runner.run_episode.side_effect = ValueError("programming bug")

        with pytest.raises(ValueError, match="programming bug"):
            run_condition(
                runner, [_task()], lambda _idx: _StubController(), condition="random"
            )

    def test_run_benchmark_records_failures_per_condition(self):
        runner = MagicMock(spec=EpisodeRunner)
        episode = _make_episode()
        runner.run_episode.side_effect = [
            episode, _transport_exhausted(), episode, episode,  # heuristic
            episode, episode, episode, episode,  # random
        ]

        result = run_benchmark(
            client=MagicMock(),
            tasks=[_task(f"t{i}") for i in range(4)],
            conditions={
                "heuristic": lambda _idx: _StubController(),
                "random": lambda _idx: _StubController(),
            },
            runner=runner,
        )
        assert [f.task_id for f in result.failures["heuristic"]] == ["t1"]
        assert result.failures["random"] == []
        assert result.reports["heuristic"].n_episodes == 3
        assert result.reports["random"].n_episodes == 4


def _load_benchmark_script():
    """Import scripts/run_baseline_benchmark.py (not a package) by path."""
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_baseline_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("run_baseline_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBuildConditions:
    """The script's factories must stay in sync with CONDITION_NAMES."""

    def test_factories_cover_all_condition_names(self):
        script = _load_benchmark_script()
        conditions = script.build_conditions(MagicMock(), MagicMock(), CONDITION_NAMES)
        assert tuple(conditions) == CONDITION_NAMES

    def test_subset_builds_only_selected(self):
        script = _load_benchmark_script()
        conditions = script.build_conditions(MagicMock(), MagicMock(), ("heuristic",))
        assert list(conditions) == ["heuristic"]
