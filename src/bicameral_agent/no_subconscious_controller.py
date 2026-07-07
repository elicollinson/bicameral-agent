"""Null controller that never invokes any tools.

Serves as the lowest baseline: always returns DO_NOTHING regardless
of state, while logging every decision with full state for analysis.
"""

from __future__ import annotations

import logging

from bicameral_agent.heuristic_controller import (
    Action,
    DecisionLoggingController,
    FullState,
)

logger = logging.getLogger(__name__)


class NoSubconsciousController(DecisionLoggingController):
    """Controller that always returns DO_NOTHING.

    Provides the absolute-zero baseline — no tools are ever invoked.
    Every decision is logged with full state for comparison against
    controllers that do invoke tools.
    """

    def decide(self, state: FullState) -> Action:
        action = Action.DO_NOTHING

        self._record_decision(action, 0, state)
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
