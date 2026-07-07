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

# Default rule thresholds. Single source of truth shared with
# ``config.HeuristicConfig``; override per-run via constructor kwargs.
DEFAULT_SCANNER_INTERVAL = 5
DEFAULT_REFRESHER_INTERVAL = 8
DEFAULT_AUDITOR_STOP_THRESHOLD = 1
DEFAULT_AUDITOR_HIGH_STOP_THRESHOLD = 2
DEFAULT_QUEUE_DEPTH_GUARD = 3
DEFAULT_STAGGER_TOLERANCE_MS = 1000.0


class Action(str, enum.Enum):
    """Tool invocation actions the controller can select."""

    SCANNER = "SCANNER"
    AUDITOR = "AUDITOR"
    REFRESHER = "REFRESHER"
    DO_NOTHING = "DO_NOTHING"


TOOL_IDS: dict[Action, str] = {
    Action.SCANNER: "research_gap_scanner",
    Action.AUDITOR: "assumption_auditor",
    Action.REFRESHER: "context_refresher",
}
"""Canonical mapping from Action enum to tool identifier string."""


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
    """Tools currently in flight. Always empty today: tools run synchronously
    within the turn, so nothing is executing when the controller decides.
    Reserved for async tool execution; until then the stagger guard
    (rule 8) never fires in production."""
    predicted_latencies: dict[str, float]
    """Predicted mean latency per tool, keyed by tool_id (``TOOL_IDS`` values)."""


@dataclass(frozen=True, slots=True)
class DecisionLog:
    """Record of a single controller decision."""

    action: Action
    rule_fired: int
    state: FullState
    timestamp_ms: float


class DecisionLoggingController:
    """Base for controllers: records DecisionLog entries and exposes copies.

    Shared by the heuristic, random, and no-subconscious controllers so the
    append-a-DecisionLog boilerplate lives in one place (issue #54).
    """

    def __init__(self) -> None:
        self._decisions: list[DecisionLog] = []

    def _record_decision(
        self, action: Action, rule_fired: int, state: FullState
    ) -> None:
        """Append a DecisionLog entry stamped with the current time."""
        self._decisions.append(
            DecisionLog(
                action=action,
                rule_fired=rule_fired,
                state=state,
                timestamp_ms=time.time() * 1000,
            )
        )

    @property
    def decisions(self) -> list[DecisionLog]:
        """Return a copy of all recorded decisions."""
        return list(self._decisions)


class HeuristicController(DecisionLoggingController):
    """Rule-based controller that decides when to invoke tools.

    Rules are evaluated in priority order (6→1) to find a candidate
    tool action, then guard rules (7–8) may suppress it.

    Thresholds are constructor parameters so they can be swept as
    hyperparameters (see ``HyperConfig.to_heuristic_controller``);
    defaults match the original hardcoded rule set.
    """

    def __init__(
        self,
        *,
        scanner_interval: int = DEFAULT_SCANNER_INTERVAL,
        refresher_interval: int = DEFAULT_REFRESHER_INTERVAL,
        auditor_stop_threshold: int = DEFAULT_AUDITOR_STOP_THRESHOLD,
        auditor_high_stop_threshold: int = DEFAULT_AUDITOR_HIGH_STOP_THRESHOLD,
        queue_depth_guard: int = DEFAULT_QUEUE_DEPTH_GUARD,
        stagger_tolerance_ms: float = DEFAULT_STAGGER_TOLERANCE_MS,
    ) -> None:
        super().__init__()
        self._scanner_interval = scanner_interval
        self._refresher_interval = refresher_interval
        self._auditor_stop_threshold = auditor_stop_threshold
        self._auditor_high_stop_threshold = auditor_high_stop_threshold
        self._queue_depth_guard = queue_depth_guard
        self._stagger_tolerance_ms = stagger_tolerance_ms

    def decide(self, state: FullState) -> Action:
        """Evaluate rules against state and return the chosen action."""
        action, rule = self._evaluate(state)

        self._record_decision(action, rule, state)
        logger.debug(
            "rule=%d action=%s turn=%d stop_count=%d queue=%d",
            rule,
            action.value,
            state.turn_number,
            state.stop_count,
            state.queue_depth,
        )
        return action

    def _evaluate(self, state: FullState) -> tuple[Action, int]:
        """Return (action, rule_number) after evaluating all rules."""
        # --- Candidate selection: rules 6→1 ---
        candidate: Action | None = None
        rule: int = 9

        if state.followup_type == FollowUpType.REDIRECT:
            candidate, rule = Action.REFRESHER, 6
        elif state.turn_number % self._refresher_interval == 0:
            candidate, rule = Action.REFRESHER, 5
        elif state.stop_count >= self._auditor_high_stop_threshold:
            candidate, rule = Action.AUDITOR, 4
        elif state.stop_count >= self._auditor_stop_threshold:
            candidate, rule = Action.AUDITOR, 3
        elif state.turn_number % self._scanner_interval == 0 and state.turn_number > 1:
            candidate, rule = Action.SCANNER, 2
        elif state.turn_number == 1:
            candidate, rule = Action.SCANNER, 1

        if candidate is None:
            return Action.DO_NOTHING, 9

        # --- Guard rules: override candidate to DO_NOTHING ---
        # Rule 7: queue depth guard
        if state.queue_depth >= self._queue_depth_guard:
            return Action.DO_NOTHING, 7

        # Rule 8: stagger guard. predicted_latencies is keyed by tool_id
        # (the TOOL_IDS convention shared with EpisodeRunner), not by the
        # Action enum value (issue #54).
        candidate_latency = state.predicted_latencies.get(TOOL_IDS[candidate], 0.0)
        for tool in state.executing_tools:
            if abs(tool.predicted_remaining_ms - candidate_latency) <= self._stagger_tolerance_ms:
                return Action.DO_NOTHING, 8

        return candidate, rule
