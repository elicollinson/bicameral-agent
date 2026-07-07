"""Tests for the MCTS engine (issue #28).

The module requires torch for the real-network tests, so it is skipped
wholesale when torch is unavailable (same pattern as
test_transition_model.py).

Synthetic construction for the search-quality ACs
-------------------------------------------------
Random-initialized networks give search nothing to discover: the value
landscape is flat noise, so MCTS visit counts simply track the prior and
the policy-improvement / budget-sensitivity ACs cannot be exercised.
Instead those tests use deterministic stub models implementing the same
``predict`` interfaces, built so the value landscape *contradicts* the
prior:

- ``_StubPolicyValue``: the prior always favors action 0 (probs
  ``[0.7, 0.1, 0.1, 0.1]``); the value head returns 0, so the search
  signal comes purely from predicted rewards.
- ``_StubTransition``: taking the state-dependent "best" action
  (``int(state[0] * num_actions)``) yields reward 1, every other action
  yields 0; dynamics are static (``next_state == state``) so the
  landscape is stationary down the tree.

Raw-policy argmax is therefore always action 0, while the rewarding
action is uniform over all 4 actions for uniformly random states — so a
search that reads the transition model's rewards should disagree with
the prior on ~75% of states (AC: >= 10%), concentrate more with a larger
budget, and shift meaningfully between 10 and 50 simulations.
"""

from __future__ import annotations

# ruff: noqa: E402  (imports below the importorskip guard are intentional)

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bicameral_agent.mcts import MCTSEngine, MCTSResult
from bicameral_agent.policy_value_net import NUM_ACTIONS, PolicyValueNetwork
from bicameral_agent.training_pipeline import STATE_DIM
from bicameral_agent.transition_model import TransitionModel

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _real_models(seed: int = 0) -> tuple[PolicyValueNetwork, TransitionModel]:
    """Random-initialized real networks on a shared STATE_DIM state space."""
    torch.manual_seed(seed)
    policy = PolicyValueNetwork(input_dim=STATE_DIM)
    transition = TransitionModel()
    policy.eval()
    transition.eval()
    return policy, transition


def _random_states(n: int, dim: int = STATE_DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, dim)).astype(np.float32)


_STUB_DIM = 8


def _best_action(state: np.ndarray) -> int:
    """State-dependent rewarding action, uniform over actions for U[0,1) states."""
    return min(int(float(state[0]) * NUM_ACTIONS), NUM_ACTIONS - 1)


class _StubPolicyValue:
    """Prior always favors action 0; value head contributes nothing.

    See the module docstring for the full construction rationale.
    """

    num_actions = NUM_ACTIONS

    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        probs = np.full(NUM_ACTIONS, 0.1, dtype=np.float32)
        probs[0] = 0.7
        return probs, 0.0


class _StubTransition:
    """Reward 1 for the state-dependent best action, 0 otherwise; static dynamics."""

    num_actions = NUM_ACTIONS

    def predict(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float]:
        reward = 1.0 if action == _best_action(state) else 0.0
        return np.asarray(state, dtype=np.float32), reward


def _stub_engine(seed: int = 0, **kwargs) -> MCTSEngine:
    return MCTSEngine(_StubPolicyValue(), _StubTransition(), seed=seed, **kwargs)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """KL(p || q) with epsilon smoothing so zero visit counts stay finite."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


class TestSearchContract:
    def test_result_shapes_and_dtypes(self) -> None:
        policy, transition = _real_models()
        engine = MCTSEngine(policy, transition, seed=0)
        state = _random_states(1)[0]
        result = engine.search(state, num_simulations=10)
        assert isinstance(result, MCTSResult)
        assert result.action_distribution.shape == (NUM_ACTIONS,)
        assert result.action_distribution.dtype == np.float32
        assert (result.action_distribution >= 0).all()
        assert abs(float(result.action_distribution.sum()) - 1.0) < 1e-5
        assert result.visit_counts.shape == (NUM_ACTIONS,)
        assert result.visit_counts.dtype == np.int64
        assert int(result.visit_counts.sum()) == 10
        assert isinstance(result.root_value, float)
        assert math.isfinite(result.root_value)

    def test_accepts_float64_state(self) -> None:
        policy, transition = _real_models()
        engine = MCTSEngine(policy, transition, seed=0)
        state64 = np.random.default_rng(0).random(STATE_DIM)  # float64
        result = engine.search(state64, num_simulations=5)
        assert int(result.visit_counts.sum()) == 5

    def test_rejects_zero_simulations(self) -> None:
        engine = _stub_engine()
        state = np.zeros(_STUB_DIM, dtype=np.float32)
        with pytest.raises(ValueError, match="num_simulations"):
            engine.search(state, num_simulations=0)

    def test_constructor_validation(self) -> None:
        policy, transition = _StubPolicyValue(), _StubTransition()
        with pytest.raises(ValueError, match="c_puct"):
            MCTSEngine(policy, transition, c_puct=0.0)
        with pytest.raises(ValueError, match="discount"):
            MCTSEngine(policy, transition, discount=1.5)
        with pytest.raises(ValueError, match="dirichlet_alpha"):
            MCTSEngine(policy, transition, dirichlet_alpha=0.0)
        with pytest.raises(ValueError, match="dirichlet_epsilon"):
            MCTSEngine(policy, transition, dirichlet_epsilon=2.0)

    def test_rejects_mismatched_num_actions(self) -> None:
        policy = _StubPolicyValue()
        transition = _StubTransition()
        transition.num_actions = NUM_ACTIONS + 1
        with pytest.raises(ValueError, match="num_actions"):
            MCTSEngine(policy, transition)

    def test_rejects_mismatched_state_dims(self) -> None:
        # Default PolicyValueNetwork is FEATURE_DIM (64) while the default
        # TransitionModel is STATE_DIM (108); the engine must refuse the pair.
        torch.manual_seed(0)
        with pytest.raises(ValueError, match="state dim"):
            MCTSEngine(PolicyValueNetwork(), TransitionModel())


# ---------------------------------------------------------------------------
# AC1: determinism — same state + same seed -> identical distributions
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_identical_distributions_with_noise(self) -> None:
        policy, transition = _real_models()
        states = _random_states(5)

        def run(seed: int) -> list[MCTSResult]:
            engine = MCTSEngine(policy, transition, seed=seed)
            return [engine.search(s, num_simulations=25, add_root_noise=True) for s in states]

        for r1, r2 in zip(run(7), run(7), strict=True):
            np.testing.assert_array_equal(r1.action_distribution, r2.action_distribution)
            np.testing.assert_array_equal(r1.visit_counts, r2.visit_counts)
            assert r1.root_value == r2.root_value

    def test_noise_off_is_seed_independent(self) -> None:
        policy, transition = _real_models()
        state = _random_states(1)[0]
        r1 = MCTSEngine(policy, transition, seed=1).search(state, num_simulations=25)
        r2 = MCTSEngine(policy, transition, seed=2).search(state, num_simulations=25)
        np.testing.assert_array_equal(r1.action_distribution, r2.action_distribution)
        assert r1.root_value == r2.root_value

    def test_root_noise_changes_search(self) -> None:
        """The noise toggle must actually perturb the root priors."""
        policy, transition = _real_models()
        states = _random_states(5, seed=3)
        engine_noisy = MCTSEngine(policy, transition, seed=0, dirichlet_epsilon=0.5)
        engine_clean = MCTSEngine(policy, transition, seed=0)
        differs = any(
            not np.array_equal(
                engine_noisy.search(s, num_simulations=25, add_root_noise=True).visit_counts,
                engine_clean.search(s, num_simulations=25).visit_counts,
            )
            for s in states
        )
        assert differs, "Dirichlet root noise never changed the visit counts"


# ---------------------------------------------------------------------------
# AC2: policy improvement — search disagrees with the raw prior argmax
# ---------------------------------------------------------------------------


class TestPolicyImprovement:
    def test_mcts_differs_from_raw_policy_on_at_least_10pct(self) -> None:
        engine = _stub_engine()
        states = _random_states(40, dim=_STUB_DIM, seed=1)
        raw_argmax = 0  # the stub prior always peaks at action 0
        n_differ = sum(
            int(np.argmax(engine.search(s, num_simulations=50).action_distribution))
            != raw_argmax
            for s in states
        )
        assert n_differ / len(states) >= 0.10, (
            f"MCTS argmax differed from the prior on only {n_differ}/{len(states)} states"
        )

    def test_mcts_finds_the_rewarding_action(self) -> None:
        """Stronger than the AC: search should recover the true best action."""
        engine = _stub_engine()
        states = _random_states(40, dim=_STUB_DIM, seed=1)
        for s in states:
            result = engine.search(s, num_simulations=50)
            assert int(np.argmax(result.action_distribution)) == _best_action(s)


# ---------------------------------------------------------------------------
# AC3: budget sensitivity — KL(10-sim || 50-sim) > 0.01 on average
# ---------------------------------------------------------------------------


class TestBudgetSensitivity:
    def test_kl_between_10_and_50_simulations(self) -> None:
        engine = _stub_engine()
        states = _random_states(20, dim=_STUB_DIM, seed=2)
        kls = []
        for s in states:
            d10 = engine.search(s, num_simulations=10).action_distribution
            d50 = engine.search(s, num_simulations=50).action_distribution
            kls.append(_kl(d10, d50))
        mean_kl = float(np.mean(kls))
        assert mean_kl > 0.01, f"mean KL(10 || 50) = {mean_kl:.4f} <= 0.01"

    # AC4: larger budgets concentrate the distribution.
    def test_200_sims_lower_entropy_than_10(self) -> None:
        engine = _stub_engine()
        states = _random_states(20, dim=_STUB_DIM, seed=2)
        h10 = np.mean([_entropy(engine.search(s, 10).action_distribution) for s in states])
        h200 = np.mean([_entropy(engine.search(s, 200).action_distribution) for s in states])
        assert h200 < h10, f"entropy did not drop with budget: h10={h10:.3f}, h200={h200:.3f}"


# ---------------------------------------------------------------------------
# AC5 / AC6: latency and root-value sanity on the real networks
# ---------------------------------------------------------------------------


class TestRealNetworkBehavior:
    def test_50_simulations_under_200ms(self) -> None:
        import time

        policy, transition = _real_models()
        engine = MCTSEngine(policy, transition, seed=0)
        state = _random_states(1)[0]
        engine.search(state, num_simulations=50)  # warm up
        times = []
        for _ in range(5):
            start = time.perf_counter()
            engine.search(state, num_simulations=50)
            times.append(time.perf_counter() - start)
        median_ms = sorted(times)[len(times) // 2] * 1000
        assert median_ms < 200, f"50-sim search median={median_ms:.1f}ms > 200ms"

    def test_root_value_near_raw_value_estimate(self) -> None:
        policy, transition = _real_models()
        engine = MCTSEngine(policy, transition, seed=0)
        states = _random_states(10, seed=4)
        diffs = []
        for s in states:
            _, raw_value = policy.predict(s)
            result = engine.search(s, num_simulations=50)
            diffs.append(abs(result.root_value - raw_value))
        mean_diff = float(np.mean(diffs))
        assert mean_diff < 0.5, f"mean |root_value - raw value| = {mean_diff:.3f} >= 0.5"
