"""Tests for the rule-based heuristic controller."""

from __future__ import annotations

import time

from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.heuristic_controller import (
    Action,
    ExecutingTool,
    FullState,
    HeuristicController,
)


def _state(
    *,
    turn_number: int = 2,
    stop_count: int = 0,
    followup_type: FollowUpType = FollowUpType.ELABORATION,
    queue_depth: int = 0,
    executing_tools: tuple[ExecutingTool, ...] = (),
    predicted_latencies: dict[str, float] | None = None,
) -> FullState:
    """Build a FullState with sensible defaults."""
    return FullState(
        turn_number=turn_number,
        stop_count=stop_count,
        followup_type=followup_type,
        queue_depth=queue_depth,
        executing_tools=executing_tools,
        predicted_latencies=predicted_latencies or {},
    )


# --- Rule isolation tests ---


class TestRuleIsolation:
    """Each rule fires in isolation with the right mock state."""

    def test_rule1_scanner_on_turn1(self) -> None:
        ctrl = HeuristicController()
        action = ctrl.decide(_state(turn_number=1))
        assert action == Action.SCANNER
        assert ctrl.decisions[-1].rule_fired == 1

    def test_rule2_scanner_on_turn_multiple_of_5(self) -> None:
        ctrl = HeuristicController()
        action = ctrl.decide(_state(turn_number=10))
        assert action == Action.SCANNER
        assert ctrl.decisions[-1].rule_fired == 2

    def test_rule3_auditor_on_stop_count_1(self) -> None:
        ctrl = HeuristicController()
        action = ctrl.decide(_state(stop_count=1))
        assert action == Action.AUDITOR
        assert ctrl.decisions[-1].rule_fired == 3

    def test_rule4_auditor_on_stop_count_2(self) -> None:
        ctrl = HeuristicController()
        action = ctrl.decide(_state(stop_count=2))
        assert action == Action.AUDITOR
        assert ctrl.decisions[-1].rule_fired == 4

    def test_rule5_refresher_on_turn_multiple_of_8(self) -> None:
        ctrl = HeuristicController()
        action = ctrl.decide(_state(turn_number=16))
        assert action == Action.REFRESHER
        assert ctrl.decisions[-1].rule_fired == 5

    def test_rule6_refresher_on_redirect(self) -> None:
        ctrl = HeuristicController()
        action = ctrl.decide(_state(followup_type=FollowUpType.REDIRECT))
        assert action == Action.REFRESHER
        assert ctrl.decisions[-1].rule_fired == 6

    def test_rule7_queue_guard_suppresses(self) -> None:
        ctrl = HeuristicController()
        # stop_count=1 would trigger rule 3 → AUDITOR, but queue blocks it
        action = ctrl.decide(_state(stop_count=1, queue_depth=3))
        assert action == Action.DO_NOTHING
        assert ctrl.decisions[-1].rule_fired == 7

    def test_rule8_stagger_guard_suppresses(self) -> None:
        ctrl = HeuristicController()
        # stop_count=1 → AUDITOR candidate; executing tool remaining 3000ms,
        # AUDITOR predicted at 3500ms → within 1000ms → suppressed
        action = ctrl.decide(
            _state(
                stop_count=1,
                executing_tools=(ExecutingTool("t1", 3000.0),),
                predicted_latencies={Action.AUDITOR.value: 3500.0},
            )
        )
        assert action == Action.DO_NOTHING
        assert ctrl.decisions[-1].rule_fired == 8

    def test_rule9_default_do_nothing(self) -> None:
        ctrl = HeuristicController()
        # turn=2, stop_count=0, no redirect → no candidate fires
        action = ctrl.decide(_state())
        assert action == Action.DO_NOTHING
        assert ctrl.decisions[-1].rule_fired == 9


# --- Guard tests ---


class TestGuards:
    """Queue and stagger guards work correctly."""

    def test_queue_guard_suppresses_all_tool_actions(self) -> None:
        ctrl = HeuristicController()
        # Rule 6 (redirect) should be suppressed by queue depth
        action = ctrl.decide(
            _state(followup_type=FollowUpType.REDIRECT, queue_depth=3)
        )
        assert action == Action.DO_NOTHING

    def test_stagger_guard_within_threshold_suppresses(self) -> None:
        ctrl = HeuristicController()
        # 3000ms remaining vs 3500ms predicted → diff 500ms ≤ 1000ms → suppressed
        action = ctrl.decide(
            _state(
                turn_number=1,
                executing_tools=(ExecutingTool("t1", 3000.0),),
                predicted_latencies={Action.SCANNER.value: 3500.0},
            )
        )
        assert action == Action.DO_NOTHING

    def test_stagger_guard_outside_threshold_allows(self) -> None:
        ctrl = HeuristicController()
        # 3000ms remaining vs 1000ms predicted → diff 2000ms > 1000ms → allowed
        action = ctrl.decide(
            _state(
                turn_number=1,
                executing_tools=(ExecutingTool("t1", 3000.0),),
                predicted_latencies={Action.SCANNER.value: 1000.0},
            )
        )
        assert action == Action.SCANNER


# --- Priority ordering ---


class TestPriorityOrdering:
    """When multiple rules match, highest-numbered fires."""

    def test_redirect_beats_stop_count(self) -> None:
        # Rule 6 (redirect) should fire over rule 3/4 (stop_count)
        ctrl = HeuristicController()
        action = ctrl.decide(
            _state(followup_type=FollowUpType.REDIRECT, stop_count=2)
        )
        assert action == Action.REFRESHER
        assert ctrl.decisions[-1].rule_fired == 6

    def test_turn_multiple_of_8_beats_5(self) -> None:
        # turn=40: divisible by both 8 and 5; rule 5 (mod 8) > rule 2 (mod 5)
        ctrl = HeuristicController()
        action = ctrl.decide(_state(turn_number=40))
        assert action == Action.REFRESHER
        assert ctrl.decisions[-1].rule_fired == 5

    def test_stop_count_beats_scanner(self) -> None:
        # turn=5, stop_count=1: rule 3 (stop) should fire over rule 2 (scanner)
        ctrl = HeuristicController()
        action = ctrl.decide(_state(turn_number=5, stop_count=1))
        assert action == Action.AUDITOR
        assert ctrl.decisions[-1].rule_fired == 3


# --- Decision logging ---


class TestDecisionLogging:
    """Decisions are logged with complete state."""

    def test_all_decisions_logged(self) -> None:
        ctrl = HeuristicController()
        states = [
            _state(turn_number=1),
            _state(turn_number=2, stop_count=1),
            _state(turn_number=3, followup_type=FollowUpType.REDIRECT),
            _state(turn_number=4, queue_depth=5, stop_count=2),
            _state(turn_number=7),
        ]
        for s in states:
            ctrl.decide(s)

        decisions = ctrl.decisions
        assert len(decisions) == 5

        for i, d in enumerate(decisions):
            assert d.state == states[i]
            assert d.action in Action
            assert 1 <= d.rule_fired <= 9
            assert d.timestamp_ms > 0


# --- Performance ---


class TestPerformance:
    """decide() completes well within 10ms."""

    def test_decide_under_10ms(self) -> None:
        ctrl = HeuristicController()
        s = _state(turn_number=1)
        start = time.perf_counter()
        for _ in range(1000):
            ctrl.decide(s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 1000
        assert avg_ms < 10, f"Average decide() took {avg_ms:.3f}ms"
