"""Tests for the NoSubconsciousController."""

from __future__ import annotations

import time

from bicameral_agent.episode_runner import Controller
from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.heuristic_controller import Action, ExecutingTool, FullState
from bicameral_agent.no_subconscious_controller import NoSubconsciousController


def _make_state(
    turn: int = 1,
    queue_depth: int = 0,
    followup_type: FollowUpType = FollowUpType.ELABORATION,
    stop_count: int = 0,
    executing_tools: tuple[ExecutingTool, ...] = (),
) -> FullState:
    return FullState(
        turn_number=turn,
        stop_count=stop_count,
        followup_type=followup_type,
        queue_depth=queue_depth,
        executing_tools=executing_tools,
        predicted_latencies={},
    )


class TestAlwaysDoNothing:
    """Every call must return DO_NOTHING regardless of state."""

    def test_basic(self) -> None:
        ctrl = NoSubconsciousController()
        assert ctrl.decide(_make_state()) == Action.DO_NOTHING

    def test_various_turns(self) -> None:
        ctrl = NoSubconsciousController()
        for turn in (1, 2, 5, 8, 10, 100):
            assert ctrl.decide(_make_state(turn=turn)) == Action.DO_NOTHING

    def test_various_queue_depths(self) -> None:
        ctrl = NoSubconsciousController()
        for depth in (0, 1, 2, 3, 10):
            assert ctrl.decide(_make_state(queue_depth=depth)) == Action.DO_NOTHING

    def test_various_followup_types(self) -> None:
        ctrl = NoSubconsciousController()
        for ft in FollowUpType:
            assert ctrl.decide(_make_state(followup_type=ft)) == Action.DO_NOTHING

    def test_with_executing_tools(self) -> None:
        ctrl = NoSubconsciousController()
        tools = (ExecutingTool(tool_id="scanner", predicted_remaining_ms=500.0),)
        assert ctrl.decide(_make_state(executing_tools=tools)) == Action.DO_NOTHING

    def test_high_stop_count(self) -> None:
        ctrl = NoSubconsciousController()
        assert ctrl.decide(_make_state(stop_count=5)) == Action.DO_NOTHING


class TestDecisionLogging:
    """Decisions must be logged with full state."""

    def test_decisions_grow(self) -> None:
        ctrl = NoSubconsciousController()
        for i in range(5):
            ctrl.decide(_make_state(turn=i + 1))
        assert len(ctrl.decisions) == 5

    def test_decisions_capture_state(self) -> None:
        ctrl = NoSubconsciousController()
        state = _make_state(turn=7, queue_depth=2)
        ctrl.decide(state)
        log = ctrl.decisions[0]
        assert log.action == Action.DO_NOTHING
        assert log.rule_fired == 0
        assert log.state is state
        assert log.timestamp_ms > 0

    def test_decisions_returns_copy(self) -> None:
        ctrl = NoSubconsciousController()
        ctrl.decide(_make_state())
        d1 = ctrl.decisions
        d2 = ctrl.decisions
        assert d1 == d2
        assert d1 is not d2


class TestControllerProtocol:
    """Must satisfy the Controller protocol from episode_runner."""

    def test_isinstance_check(self) -> None:
        ctrl = NoSubconsciousController()
        assert isinstance(ctrl, Controller)


class TestPerformance:
    """Single decision must complete in under 1ms."""

    def test_decision_under_one_ms(self) -> None:
        ctrl = NoSubconsciousController()
        state = _make_state()
        # Warm up
        ctrl.decide(state)
        start = time.perf_counter()
        ctrl.decide(state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1.0, f"Decision took {elapsed_ms:.3f}ms"
