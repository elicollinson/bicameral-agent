"""Rule-based heuristic controller for tool invocation decisions.

Serves as the baseline controller for MCTS comparison and the initial
data collection mechanism. Evaluates a fixed set of priority-ordered
rules against the current conversation state to decide which tool
(scanner, auditor, refresher) to invoke — or to do nothing.

Rules are evaluated 6→1 (highest-priority tool rule first) to find a
candidate action, then guard rules 7–8 can override to DO_NOTHING.
If no tool rule fires, rule 9 defaults to DO_NOTHING.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass

from bicameral_agent.followup_classifier import FollowUpType

logger = logging.getLogger(__name__)


class Action(str, enum.Enum):
    """Tool invocation actions the controller can select."""

    SCANNER = "SCANNER"
    AUDITOR = "AUDITOR"
    REFRESHER = "REFRESHER"
    DO_NOTHING = "DO_NOTHING"


@dataclass(frozen=True, slots=True)
class ExecutingTool:
    """A tool currently running with its predicted remaining time."""

    tool_id: str
    predicted_remaining_ms: float


@dataclass(frozen=True, slots=True)
class FullState:
    """Complete state snapshot for the heuristic controller."""

    turn_number: int
    stop_count: int
    followup_type: FollowUpType
    queue_depth: int
    executing_tools: tuple[ExecutingTool, ...]
    predicted_latencies: dict[str, float]


@dataclass(frozen=True, slots=True)
class DecisionLog:
    """Record of a single controller decision."""

    action: Action
    rule_fired: int
    state: FullState
    timestamp_ms: float


class HeuristicController:
    """Rule-based controller that decides when to invoke tools.

    Rules are evaluated in priority order (6→1) to find a candidate
    tool action, then guard rules (7–8) may suppress it.
    """

    def __init__(self) -> None:
        self._decisions: list[DecisionLog] = []

    def decide(self, state: FullState) -> Action:
        """Evaluate rules against state and return the chosen action."""
        action, rule = self._evaluate(state)

        log_entry = DecisionLog(
            action=action,
            rule_fired=rule,
            state=state,
            timestamp_ms=time.time() * 1000,
        )
        self._decisions.append(log_entry)
        logger.debug(
            "rule=%d action=%s turn=%d stop_count=%d queue=%d",
            rule,
            action.value,
            state.turn_number,
            state.stop_count,
            state.queue_depth,
        )
        return action

    @property
    def decisions(self) -> list[DecisionLog]:
        """Return all recorded decisions."""
        return list(self._decisions)

    @staticmethod
    def _evaluate(state: FullState) -> tuple[Action, int]:
        """Return (action, rule_number) after evaluating all rules."""
        # --- Candidate selection: rules 6→1 ---
        candidate: Action | None = None
        rule: int = 9

        if state.followup_type == FollowUpType.REDIRECT:
            candidate, rule = Action.REFRESHER, 6
        elif state.turn_number % 8 == 0:
            candidate, rule = Action.REFRESHER, 5
        elif state.stop_count >= 2:
            candidate, rule = Action.AUDITOR, 4
        elif state.stop_count >= 1:
            candidate, rule = Action.AUDITOR, 3
        elif state.turn_number % 5 == 0 and state.turn_number > 1:
            candidate, rule = Action.SCANNER, 2
        elif state.turn_number == 1:
            candidate, rule = Action.SCANNER, 1

        if candidate is None:
            return Action.DO_NOTHING, 9

        # --- Guard rules: override candidate to DO_NOTHING ---
        # Rule 7: queue depth guard
        if state.queue_depth >= 3:
            return Action.DO_NOTHING, 7

        # Rule 8: stagger guard
        candidate_latency = state.predicted_latencies.get(candidate.value, 0.0)
        for tool in state.executing_tools:
            if abs(tool.predicted_remaining_ms - candidate_latency) <= 1000:
                return Action.DO_NOTHING, 8

        return candidate, rule
