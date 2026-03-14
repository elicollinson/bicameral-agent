"""Random invocation controller for baseline comparison.

Selects tool actions randomly with a configurable probability,
providing an unbiased baseline against which learned or heuristic
controllers can be evaluated.
"""

from __future__ import annotations

import logging
import random
import time

from bicameral_agent.heuristic_controller import Action, DecisionLog, FullState

logger = logging.getLogger(__name__)


_TOOL_ACTIONS = (Action.SCANNER, Action.AUDITOR, Action.REFRESHER)


class RandomController:
    """Controller that randomly selects tool actions.

    With probability ``action_probability``, picks uniformly from
    {SCANNER, AUDITOR, REFRESHER}. Respects queue depth guard (depth >= 3).
    Uses ``random.Random(seed)`` for reproducibility.
    """

    def __init__(
        self,
        action_probability: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self._action_probability = action_probability
        self._rng = random.Random(seed)
        self._decisions: list[DecisionLog] = []

    def decide(self, state: FullState) -> Action:
        # Queue depth guard (matches heuristic controller rule 7)
        if state.queue_depth >= 3:
            action = Action.DO_NOTHING
        elif self._rng.random() < self._action_probability:
            action = self._rng.choice(_TOOL_ACTIONS)
        else:
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
            "action=%s turn=%d queue=%d",
            action.value,
            state.turn_number,
            state.queue_depth,
        )
        return action

    @property
    def decisions(self) -> list[DecisionLog]:
        return list(self._decisions)
