"""Null controller that never invokes any tools.

Serves as the lowest baseline: always returns DO_NOTHING regardless
of state, while logging every decision with full state for analysis.
"""

from __future__ import annotations

import logging
import time

from bicameral_agent.heuristic_controller import Action, DecisionLog, FullState

logger = logging.getLogger(__name__)


class NoSubconsciousController:
    """Controller that always returns DO_NOTHING.

    Provides the absolute-zero baseline — no tools are ever invoked.
    Every decision is logged with full state for comparison against
    controllers that do invoke tools.
    """

    def __init__(self) -> None:
        self._decisions: list[DecisionLog] = []

    def decide(self, state: FullState) -> Action:
        action = Action.DO_NOTHING

        self._decisions.append(
            DecisionLog(
                action=action,
                rule_fired=0,
                state=state,
                timestamp_ms=time.time() * 1000,
            )
        )
        logger.debug(
            "action=%s turn=%d queue=%d followup=%s stop_count=%d executing=%d",
            action.value,
            state.turn_number,
            state.queue_depth,
            state.followup_type.value,
            state.stop_count,
            len(state.executing_tools),
        )
        return action

    @property
    def decisions(self) -> list[DecisionLog]:
        return list(self._decisions)
