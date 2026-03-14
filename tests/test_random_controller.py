"""Tests for the RandomController."""

from __future__ import annotations

import time

from bicameral_agent.followup_classifier import FollowUpType
from bicameral_agent.heuristic_controller import Action, FullState
from bicameral_agent.random_controller import RandomController


def _make_state(
    turn: int = 1,
    queue_depth: int = 0,
) -> FullState:
    return FullState(
        turn_number=turn,
        stop_count=0,
        followup_type=FollowUpType.ELABORATION,
        queue_depth=queue_depth,
        executing_tools=(),
        predicted_latencies={},
    )


class TestInvocationRate:
    """Verify ~20% invocation rate over many decisions."""

    def test_action_rate_near_twenty_percent(self) -> None:
        ctrl = RandomController(seed=42)
        state = _make_state()
        n = 1000
        actions = [ctrl.decide(state) for _ in range(n)]
        action_count = sum(1 for a in actions if a != Action.DO_NOTHING)
        rate = action_count / n
        assert 0.17 <= rate <= 0.23, f"Action rate {rate:.3f} outside 17-23%"


class TestToolDistribution:
    """Verify roughly uniform distribution across the three tools."""

    def test_uniform_tool_selection(self) -> None:
        ctrl = RandomController(seed=123)
        state = _make_state()
        n = 3000
        actions = [ctrl.decide(state) for _ in range(n)]
        tool_actions = [a for a in actions if a != Action.DO_NOTHING]
        assert len(tool_actions) > 0

        counts = {
            Action.SCANNER: 0,
            Action.AUDITOR: 0,
            Action.REFRESHER: 0,
        }
        for a in tool_actions:
            counts[a] += 1

        total_tools = len(tool_actions)
        for tool, count in counts.items():
            frac = count / total_tools
            assert 0.25 <= frac <= 0.40, (
                f"{tool.value} fraction {frac:.3f} outside 25-40%"
            )


class TestQueueGuard:
    """Queue depth >= 3 must always yield DO_NOTHING."""

    def test_queue_depth_three_always_do_nothing(self) -> None:
        ctrl = RandomController(action_probability=1.0, seed=0)
        for depth in (3, 4, 10):
            state = _make_state(queue_depth=depth)
            for _ in range(50):
                assert ctrl.decide(state) == Action.DO_NOTHING

    def test_queue_depth_below_three_allows_actions(self) -> None:
        ctrl = RandomController(action_probability=1.0, seed=7)
        state = _make_state(queue_depth=2)
        actions = [ctrl.decide(state) for _ in range(20)]
        assert all(a != Action.DO_NOTHING for a in actions)


class TestDeterminism:
    """Same seed must produce identical decision sequences."""

    def test_same_seed_same_decisions(self) -> None:
        state = _make_state()
        n = 100
        ctrl_a = RandomController(seed=99)
        ctrl_b = RandomController(seed=99)
        for _ in range(n):
            assert ctrl_a.decide(state) == ctrl_b.decide(state)

    def test_different_seeds_differ(self) -> None:
        state = _make_state()
        n = 200
        ctrl_a = RandomController(seed=1)
        ctrl_b = RandomController(seed=2)
        seq_a = [ctrl_a.decide(state) for _ in range(n)]
        seq_b = [ctrl_b.decide(state) for _ in range(n)]
        assert seq_a != seq_b


class TestPerformance:
    """Single decision must complete in under 1ms."""

    def test_decision_under_one_ms(self) -> None:
        ctrl = RandomController(seed=0)
        state = _make_state()
        # Warm up
        ctrl.decide(state)
        start = time.perf_counter()
        ctrl.decide(state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1.0, f"Decision took {elapsed_ms:.3f}ms"


class TestDecisionLog:
    """Verify decisions are recorded correctly."""

    def test_decisions_property(self) -> None:
        ctrl = RandomController(seed=42)
        state = _make_state(turn=5)
        ctrl.decide(state)
        ctrl.decide(state)
        assert len(ctrl.decisions) == 2
        assert all(d.rule_fired == 0 for d in ctrl.decisions)
        assert all(d.state.turn_number == 5 for d in ctrl.decisions)

    def test_decisions_returns_copy(self) -> None:
        ctrl = RandomController(seed=42)
        state = _make_state()
        ctrl.decide(state)
        d1 = ctrl.decisions
        d2 = ctrl.decisions
        assert d1 == d2
        assert d1 is not d2
