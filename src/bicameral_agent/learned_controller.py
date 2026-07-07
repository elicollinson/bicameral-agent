"""Learned-policy controller for the episode runner (issue #29).

Wraps a policy/value network — optionally improved by
:class:`~bicameral_agent.mcts.MCTSEngine` search — as a
:class:`~bicameral_agent.episode_runner.Controller`, so the EpisodeRunner
can collect episodes with the current learned policy.

Encode-at-decision-time (train/serve consistency)
-------------------------------------------------
Training states are 108-dim vectors built by
:class:`~bicameral_agent.training_pipeline.TrainingDataPipeline` from
*replayed* episodes: each decision point is an assistant message, and its
state is everything logged strictly before that message. To guarantee the
network sees the same encoding at serve time, this controller does not
derive features from the (much thinner)
:class:`~bicameral_agent.heuristic_controller.FullState` the runner passes
to ``decide``. Instead the runner hands it a live episode snapshot via
:meth:`observe_episode` (``ConversationLogger.snapshot()``), and the
controller replays that snapshot with the *same*
:class:`~bicameral_agent.replay.EpisodeReplayer` +
``TrainingDataPipeline.build_state_vector`` code path the pipeline uses:
the snapshot's last decision point is exactly the decision being made now.

With the default BREAKPOINT injection mode this yields bit-identical
vectors to the ones the pipeline later builds from the finalized episode.
The SYNCHRONOUS/INTERRUPT modes can regenerate the assistant message after
the decision (``replace_last_message`` moves its timestamp past this
turn's tool events), which shifts the post-hoc cutoff; the serve-time
encoding then reflects the state actually seen at decision time.

Action selection
----------------
- Raw-policy mode (``mcts_engine=None``): the network's action
  probabilities are the decision distribution.
- MCTS mode: ``mcts_engine.search`` (with the configured simulation
  budget and optional root Dirichlet noise) produces the visit-count
  distribution.

Either distribution is then reduced to an action: greedy argmax when
``temperature == 0`` (evaluation), else a sample from
``dist ** (1 / temperature)`` (training-time exploration), drawn from a
private seeded numpy generator so decisions are reproducible.
"""

from __future__ import annotations

import logging

import numpy as np

from bicameral_agent.heuristic_controller import (
    Action,
    DecisionLoggingController,
    FullState,
)
from bicameral_agent.mcts import MCTSEngine, SupportsPolicyValue
from bicameral_agent.replay import DecisionPoint, EpisodeReplayer
from bicameral_agent.schema import Episode
from bicameral_agent.training_pipeline import (
    _ACTION_ORDER,  # cross-checked identical to policy_value_net.ACTION_ORDER
    STATE_DIM,
    TrainingDataPipeline,
)

logger = logging.getLogger(__name__)

LEARNED_RULE_ID: int = 0
"""``DecisionLog.rule_fired`` value for learned decisions (no rule fired;
matches the RandomController convention)."""


class LearnedPolicyController(DecisionLoggingController):
    """Policy-network controller (optionally MCTS-improved) for EpisodeRunner.

    Parameters
    ----------
    policy_value_net:
        Network with a ``predict(state) -> (probs, value)`` method over
        the 108-dim pipeline state (construct
        ``PolicyValueNetwork(input_dim=STATE_DIM)``). Used directly in
        raw-policy mode; in MCTS mode the engine holds its own reference,
        but this one is kept for diagnostics.
    mcts_engine:
        Optional :class:`MCTSEngine`. When given, decisions use the
        engine's visit-count distribution instead of the raw policy.
    num_simulations:
        Search budget per decision in MCTS mode.
    add_root_noise:
        Mix Dirichlet noise into the root priors in MCTS mode
        (training-time exploration; leave off for evaluation).
    temperature:
        0 (default) selects the argmax action; > 0 samples from
        ``dist ** (1 / temperature)``.
    pipeline:
        The :class:`TrainingDataPipeline` used to encode decision states.
        Pass the same encoder/latency-model/max_turns configuration used
        when processing episodes for training. Defaults to a fresh
        pipeline with default components (matching how episodes are
        processed by default).
    seed:
        Seed for the private sampling generator (only consumed when
        ``temperature > 0``).
    """

    def __init__(
        self,
        policy_value_net: SupportsPolicyValue,
        *,
        mcts_engine: MCTSEngine | None = None,
        num_simulations: int = 50,
        add_root_noise: bool = False,
        temperature: float = 0.0,
        pipeline: TrainingDataPipeline | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        input_dim = getattr(policy_value_net, "input_dim", None)
        if input_dim is not None and input_dim != STATE_DIM:
            msg = (
                f"policy_value_net.input_dim must be STATE_DIM ({STATE_DIM}) "
                f"to consume pipeline states, got {input_dim} "
                "(construct the network with input_dim=STATE_DIM)"
            )
            raise ValueError(msg)
        if num_simulations < 1:
            msg = f"num_simulations must be >= 1, got {num_simulations}"
            raise ValueError(msg)
        if temperature < 0.0:
            msg = f"temperature must be >= 0, got {temperature}"
            raise ValueError(msg)

        self._policy = policy_value_net
        self._engine = mcts_engine
        self._num_simulations = num_simulations
        self._add_root_noise = add_root_noise
        self._temperature = temperature
        self._pipeline = pipeline or TrainingDataPipeline()
        self._rng = np.random.default_rng(seed)
        self._snapshot: Episode | None = None
        self._encoded_states: list[np.ndarray] = []
        self._distributions: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # EpisodeRunner integration
    # ------------------------------------------------------------------

    def observe_episode(self, snapshot: Episode) -> None:
        """Receive the live episode snapshot for the next ``decide`` call.

        The EpisodeRunner calls this immediately before ``decide`` with
        ``ConversationLogger.snapshot()``; the snapshot must end with the
        assistant message of the decision being made.
        """
        self._snapshot = snapshot

    def decide(self, state: FullState) -> Action:
        """Encode the current decision point and select an action."""
        vec = self._encode_decision_point(state)
        distribution = self._action_distribution(vec)

        idx = self._select_index(distribution)
        action = _ACTION_ORDER[idx]

        self._encoded_states.append(vec)
        self._distributions.append(distribution.astype(np.float32))
        self._record_decision(action, LEARNED_RULE_ID, state)
        logger.debug(
            "learned action=%s turn=%d dist=%s",
            action.value,
            state.turn_number,
            np.round(distribution, 3),
        )
        return action

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def encoded_states(self) -> list[np.ndarray]:
        """Copies of the 108-dim state vectors encoded at each decision."""
        return [s.copy() for s in self._encoded_states]

    @property
    def distributions(self) -> list[np.ndarray]:
        """Copies of the action distribution used at each decision."""
        return [d.copy() for d in self._distributions]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode_decision_point(self, state: FullState) -> np.ndarray:
        """Build the pipeline state vector for the current decision."""
        if self._snapshot is None:
            msg = (
                "LearnedPolicyController.decide called without an episode "
                "snapshot; it must run under an EpisodeRunner (which calls "
                "observe_episode before each decision)"
            )
            raise RuntimeError(msg)

        dp = self._last_decision_point(self._snapshot)
        if dp is None:
            msg = "episode snapshot contains no assistant message to decide on"
            raise RuntimeError(msg)
        if dp.state.turn_number != state.turn_number:
            msg = (
                f"snapshot decision point is at turn {dp.state.turn_number} "
                f"but the runner reports turn {state.turn_number}; the "
                "snapshot is stale"
            )
            raise RuntimeError(msg)
        return self._pipeline.build_state_vector(dp.state, dp.action.timestamp_ms)

    @staticmethod
    def _last_decision_point(snapshot: Episode) -> DecisionPoint | None:
        last: DecisionPoint | None = None
        for dp in EpisodeReplayer(snapshot).iter_decision_points():
            last = dp
        return last

    def _action_distribution(self, vec: np.ndarray) -> np.ndarray:
        if self._engine is not None:
            result = self._engine.search(
                vec,
                num_simulations=self._num_simulations,
                add_root_noise=self._add_root_noise,
            )
            return np.asarray(result.action_distribution, dtype=np.float64)
        probs, _value = self._policy.predict(vec)
        return np.asarray(probs, dtype=np.float64)

    def _select_index(self, distribution: np.ndarray) -> int:
        if self._temperature == 0.0:
            return int(np.argmax(distribution))
        scaled = np.power(distribution, 1.0 / self._temperature)
        total = scaled.sum()
        if total <= 0.0 or not np.isfinite(total):  # degenerate distribution
            return int(np.argmax(distribution))
        return int(self._rng.choice(len(scaled), p=scaled / total))
