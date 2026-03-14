"""Tests for the policy/value network."""

from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from bicameral_agent.encoder import FEATURE_DIM
from bicameral_agent.heuristic_controller import Action
from bicameral_agent.policy_value_net import (
    ACTION_ORDER,
    NUM_ACTIONS,
    PolicyValueNetwork,
)


@pytest.fixture
def net() -> PolicyValueNetwork:
    return PolicyValueNetwork()


@pytest.fixture
def state() -> np.ndarray:
    return np.random.default_rng(42).random(FEATURE_DIM).astype(np.float32)


# ---- AC1: single state → probs sum to 1.0, scalar value ----


class TestSingleState:
    def test_probs_sum_to_one(self, net: PolicyValueNetwork, state: np.ndarray) -> None:
        probs, value = net.predict(state)
        assert probs.shape == (NUM_ACTIONS,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_value_is_scalar(self, net: PolicyValueNetwork, state: np.ndarray) -> None:
        _, value = net.predict(state)
        assert isinstance(value, float)

    def test_probs_non_negative(self, net: PolicyValueNetwork, state: np.ndarray) -> None:
        probs, _ = net.predict(state)
        assert (probs >= 0).all()


# ---- AC2: batch of 64 → correct shapes ----


class TestBatch:
    def test_batch_shapes(self, net: PolicyValueNetwork) -> None:
        batch = torch.randn(64, FEATURE_DIM)
        probs, values = net(batch)
        assert probs.shape == (64, NUM_ACTIONS)
        assert values.shape == (64,)

    def test_batch_probs_sum_to_one(self, net: PolicyValueNetwork) -> None:
        batch = torch.randn(64, FEATURE_DIM)
        probs, _ = net(batch)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(64), atol=1e-5)


# ---- AC3: param count in 50-80K ----


class TestParamCount:
    def test_param_count_in_range(self, net: PolicyValueNetwork) -> None:
        count = net.param_count
        assert 50_000 <= count <= 80_000, f"param_count={count} outside 50-80K range"


# ---- AC4: latency <5ms single, <20ms batch-64 ----


class TestLatency:
    def test_single_latency(self, net: PolicyValueNetwork, state: np.ndarray) -> None:
        # Warm up
        net.predict(state)

        times = []
        for _ in range(20):
            start = time.perf_counter()
            net.predict(state)
            times.append(time.perf_counter() - start)

        median_ms = sorted(times)[len(times) // 2] * 1000
        assert median_ms < 5, f"single inference median={median_ms:.2f}ms > 5ms"

    def test_batch_latency(self, net: PolicyValueNetwork) -> None:
        batch = torch.randn(64, FEATURE_DIM)
        # Warm up
        net(batch)

        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                net(batch)
            times.append(time.perf_counter() - start)

        median_ms = sorted(times)[len(times) // 2] * 1000
        assert median_ms < 20, f"batch-64 inference median={median_ms:.2f}ms > 20ms"


# ---- AC5: gradients flow through both heads ----


class TestGradients:
    def test_gradients_flow(self, net: PolicyValueNetwork) -> None:
        x = torch.randn(8, FEATURE_DIM)
        probs, values = net(x)

        # Loss combining both heads
        policy_loss = -probs.log().mean()
        value_loss = values.mean()
        loss = policy_loss + value_loss
        loss.backward()

        # Check gradients on trunk, policy head, and value head
        for name, param in net.named_parameters():
            assert param.grad is not None, f"no gradient for {name}"
            assert param.grad.abs().sum() > 0, f"zero gradient for {name}"


# ---- AC6: save/load checkpoint roundtrip ----


class TestCheckpoint:
    def test_save_load_roundtrip(self, net: PolicyValueNetwork, state: np.ndarray) -> None:
        probs_before, value_before = net.predict(state)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            net.save(path)
            loaded = PolicyValueNetwork.load(path)

        probs_after, value_after = loaded.predict(state)

        np.testing.assert_allclose(probs_before, probs_after, atol=1e-6)
        assert abs(value_before - value_after) < 1e-6

    def test_save_load_custom_dims(self) -> None:
        net = PolicyValueNetwork(input_dim=32, hidden_dim=64, num_actions=3)
        state = np.random.default_rng(0).random(32).astype(np.float32)
        probs_before, value_before = net.predict(state)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "custom.pt"
            net.save(path)
            loaded = PolicyValueNetwork.load(path, input_dim=32, hidden_dim=64, num_actions=3)

        probs_after, value_after = loaded.predict(state)
        np.testing.assert_allclose(probs_before, probs_after, atol=1e-6)


# ---- sample_action and temperature behavior ----


class TestSampleAction:
    def test_returns_valid_action(self, net: PolicyValueNetwork, state: np.ndarray) -> None:
        action = net.sample_action(state)
        assert isinstance(action, Action)
        assert action in ACTION_ORDER

    def test_low_temperature_is_greedy(self, net: PolicyValueNetwork, state: np.ndarray) -> None:
        """Low temperature should consistently pick the highest-prob action."""
        actions = [net.sample_action(state, temperature=0.001) for _ in range(50)]
        # With very low temperature, the most common action should dominate
        most_common_count = max(Counter(actions).values())
        assert most_common_count >= 48, f"greedy action only appeared {most_common_count}/50 times"

    def test_high_temperature_explores(self, net: PolicyValueNetwork) -> None:
        """High temperature should produce more action diversity."""
        rng = np.random.default_rng(123)
        state = rng.random(FEATURE_DIM).astype(np.float32)
        counts: Counter[Action] = Counter()
        for _ in range(200):
            counts[net.sample_action(state, temperature=10.0)] += 1
        # With high temp, should see at least 3 distinct actions
        assert len(counts) >= 3, f"only {len(counts)} distinct actions at high temperature"


# ---- Action mapping ----


class TestActionMapping:
    def test_action_order_length(self) -> None:
        assert len(ACTION_ORDER) == NUM_ACTIONS

    def test_action_order_contains_all_actions(self) -> None:
        assert set(ACTION_ORDER) == set(Action)

    def test_num_actions_constant(self) -> None:
        assert NUM_ACTIONS == 4
