"""Monte Carlo Tree Search over learned models (issue #28).

Plans entirely inside the learned environment: the policy/value network
(issue #25, :class:`~bicameral_agent.policy_value_net.PolicyValueNetwork`)
provides action priors and leaf value estimates, and the transition model
(issue #27, :class:`~bicameral_agent.transition_model.TransitionModel`)
simulates ``(next_state, reward)`` for expansions — no real episodes are
run during search.

Algorithm
---------
Each simulation performs the four classic phases:

- **Selection**: descend from the root by PUCT —
  ``argmax_a Q(s, a) + c_puct * P(a) * sqrt(N(s)) / (1 + N(s, a))`` —
  until an unexpanded child is reached. ``Q(s, a)`` is the immediate
  predicted reward plus the discounted mean value of the child's subtree.
- **Expansion**: the transition model predicts ``(next_state, reward)``
  for the selected edge and the policy head supplies the new node's
  child priors.
- **Evaluation**: the value head estimates the new leaf's value; no
  rollouts are performed.
- **Backup**: the discounted leaf return is *averaged* into every node on
  the path (``Q = W / N`` — mean value backup, not minimax). This domain
  is single-agent control (choosing which subconscious tool to fire),
  not an adversarial game: there is no opponent whose best response
  should be propagated with a min step, so the expected return under
  continued search is the correct target.

Exploration noise
-----------------
For training-time self-play, Dirichlet noise can be mixed into the root
priors (``P' = (1 - eps) * P + eps * Dir(alpha)``) so search occasionally
explores actions the prior underweights; for evaluation the noise is
toggled off via ``search(..., add_root_noise=False)`` (the default).

Determinism
-----------
The models are used in inference mode only, and their forward passes are
deterministic — no torch RNG is consumed anywhere in the search. The only
stochastic step is the root Dirichlet noise, drawn from a private numpy
generator seeded by the ``seed`` constructor argument, so a fixed seed
(or noise disabled) yields bit-identical search results for the same
root state.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Protocol

import numpy as np

from bicameral_agent.training_pipeline import DISCOUNT_GAMMA

DEFAULT_C_PUCT: float = 1.5
"""PUCT exploration constant per the issue #28 spec."""

DEFAULT_DIRICHLET_ALPHA: float = 0.3
"""Concentration of the root Dirichlet noise (AlphaZero-style default)."""

DEFAULT_DIRICHLET_EPSILON: float = 0.25
"""Fraction of the root prior replaced by Dirichlet noise when enabled."""


class SupportsPolicyValue(Protocol):
    """Anything with a ``PolicyValueNetwork``-shaped ``predict``."""

    num_actions: int

    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Return ``(action_probs, value)`` for a single state."""
        ...


class SupportsTransition(Protocol):
    """Anything with a ``TransitionModel``-shaped ``predict``."""

    num_actions: int

    def predict(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float]:
        """Return ``(next_state, reward)`` for a single ``(state, action)``."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class MCTSResult:
    """Output of :meth:`MCTSEngine.search`.

    Attributes
    ----------
    action_distribution:
        Root visit counts normalized to a probability distribution,
        shape ``(num_actions,)``, float32, sums to 1. Indexed by
        :data:`~bicameral_agent.policy_value_net.ACTION_ORDER`.
    root_value:
        Mean backed-up value at the root (includes the value head's
        initial estimate of the root state as one sample).
    visit_counts:
        Raw root visit counts, shape ``(num_actions,)``, int64; sums to
        ``num_simulations``.
    """

    action_distribution: np.ndarray
    root_value: float
    visit_counts: np.ndarray


class _Node:
    """One tree node: statistics for the state reached via its edge."""

    __slots__ = ("prior", "state", "reward", "visit_count", "value_sum", "children")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.state: np.ndarray | None = None  # None until expanded
        self.reward = 0.0  # predicted reward on the edge into this node
        self.visit_count = 0
        self.value_sum = 0.0  # sum of backed-up returns from this state on
        self.children: list[_Node] = []

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTSEngine:
    """PUCT Monte Carlo Tree Search over learned policy/value/transition models.

    Parameters
    ----------
    policy_value_net:
        Provides action priors and leaf values; typically a
        :class:`~bicameral_agent.policy_value_net.PolicyValueNetwork`.
    transition_model:
        Predicts ``(next_state, reward)`` per edge; typically a
        :class:`~bicameral_agent.transition_model.TransitionModel`. Its
        state dimensionality must match the policy network's input.
    c_puct:
        PUCT exploration constant (default 1.5). Higher values weight the
        prior/visit-count exploration term over observed values.
    discount:
        Per-step discount applied during backup (default: the training
        pipeline's ``DISCOUNT_GAMMA`` so search returns match the returns
        the value head is trained on).
    dirichlet_alpha:
        Concentration of the root Dirichlet noise.
    dirichlet_epsilon:
        Mixing fraction of the noise into the root priors.
    seed:
        Seed for the private numpy generator that draws root noise. All
        other computation is deterministic, so a fixed seed makes
        :meth:`search` fully reproducible.
    """

    def __init__(
        self,
        policy_value_net: SupportsPolicyValue,
        transition_model: SupportsTransition,
        c_puct: float = DEFAULT_C_PUCT,
        discount: float = DISCOUNT_GAMMA,
        dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON,
        seed: int | None = None,
    ) -> None:
        if policy_value_net.num_actions != transition_model.num_actions:
            msg = (
                "policy_value_net and transition_model disagree on num_actions: "
                f"{policy_value_net.num_actions} != {transition_model.num_actions}"
            )
            raise ValueError(msg)
        # A default PolicyValueNetwork is 64-dim (encoder FEATURE_DIM) while a
        # default TransitionModel is 108-dim (pipeline STATE_DIM); catch that
        # mismatch here instead of as a shape error mid-search.
        policy_dim = getattr(policy_value_net, "input_dim", None)
        transition_dim = getattr(transition_model, "state_dim", None)
        if policy_dim is not None and transition_dim is not None and policy_dim != transition_dim:
            msg = (
                "policy_value_net and transition_model disagree on state dim: "
                f"input_dim={policy_dim} != state_dim={transition_dim} "
                "(construct the policy net with input_dim=STATE_DIM)"
            )
            raise ValueError(msg)
        if c_puct <= 0.0:
            msg = f"c_puct must be > 0, got {c_puct}"
            raise ValueError(msg)
        if not 0.0 <= discount <= 1.0:
            msg = f"discount must be in [0, 1], got {discount}"
            raise ValueError(msg)
        if dirichlet_alpha <= 0.0:
            msg = f"dirichlet_alpha must be > 0, got {dirichlet_alpha}"
            raise ValueError(msg)
        if not 0.0 <= dirichlet_epsilon <= 1.0:
            msg = f"dirichlet_epsilon must be in [0, 1], got {dirichlet_epsilon}"
            raise ValueError(msg)

        self._policy = policy_value_net
        self._transition = transition_model
        self.num_actions = policy_value_net.num_actions
        self.c_puct = c_puct
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self._rng = np.random.default_rng(seed)

    def search(
        self,
        root_state: np.ndarray,
        num_simulations: int = 50,
        add_root_noise: bool = False,
    ) -> MCTSResult:
        """Run MCTS from ``root_state`` and return the improved policy.

        Parameters
        ----------
        root_state:
            1-D state vector matching the models' state dimensionality.
            Any float dtype is accepted; it is cast to float32.
        num_simulations:
            Simulation budget (e.g. 10 / 50 / 200). Each simulation adds
            exactly one node to the tree.
        add_root_noise:
            Mix Dirichlet noise into the root priors (training-time
            exploration). Leave off for deterministic evaluation.

        Returns
        -------
        MCTSResult
            Normalized root visit counts and the root value estimate.

        Raises
        ------
        ValueError
            If ``num_simulations`` is not >= 1.
        """
        if num_simulations < 1:
            msg = f"num_simulations must be >= 1, got {num_simulations}"
            raise ValueError(msg)

        root = _Node(prior=1.0)
        root.state = np.asarray(root_state, dtype=np.float32)
        priors, root_raw_value = self._policy.predict(root.state)
        priors = np.asarray(priors, dtype=np.float64)
        if add_root_noise:
            noise = self._rng.dirichlet(np.full(self.num_actions, self.dirichlet_alpha))
            priors = (1.0 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise
        root.children = [_Node(prior=float(p)) for p in priors]
        # The root's own evaluation counts as its first visit (MuZero
        # convention): sqrt(N(root)) >= 1 on the first selection, and the
        # root value averages the raw estimate with the simulated returns.
        root.visit_count = 1
        root.value_sum = float(root_raw_value)

        for _ in range(num_simulations):
            self._simulate(root)

        visits = np.array([c.visit_count for c in root.children], dtype=np.int64)
        distribution = (visits / visits.sum()).astype(np.float32)
        return MCTSResult(
            action_distribution=distribution,
            root_value=root.value_sum / root.visit_count,
            visit_counts=visits,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _simulate(self, root: _Node) -> None:
        """One selection -> expansion -> evaluation -> backup pass."""
        node = root
        path = [root]
        while True:
            action, child = self._select_child(node)
            path.append(child)
            if child.state is None:
                # Expansion: simulate the edge with the transition model.
                next_state, reward = self._transition.predict(node.state, action)
                child.state = np.asarray(next_state, dtype=np.float32)
                child.reward = float(reward)
                # Evaluation: priors + value for the new leaf.
                priors, value = self._policy.predict(child.state)
                child.children = [_Node(prior=float(p)) for p in np.asarray(priors)]
                leaf_value = float(value)
                break
            node = child

        # Backup: average the discounted return into every node on the
        # path (mean backup — see module docstring).
        value = leaf_value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = n.reward + self.discount * value

    def _select_child(self, node: _Node) -> tuple[int, _Node]:
        """PUCT argmax over ``node``'s children (deterministic tie-break)."""
        sqrt_n = math.sqrt(node.visit_count)
        scores = np.empty(self.num_actions, dtype=np.float64)
        for i, child in enumerate(node.children):
            if child.visit_count:
                q = child.reward + self.discount * child.mean_value
            else:
                q = 0.0
            u = self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            scores[i] = q + u
        best = int(np.argmax(scores))
        return best, node.children[best]
