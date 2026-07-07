"""Learned transition model for MCTS environment simulation.

Predicts ``(next_state, reward)`` from ``(state, action)`` so MCTS can
simulate rollouts without running the real system (the MuZero-style
environment model, issue #27).

Architecture
------------
- Input: 108-dim state vector (:data:`~bicameral_agent.training_pipeline.STATE_DIM`)
  concatenated with a 4-dim action one-hot (:data:`~bicameral_agent.policy_value_net.ACTION_ORDER`)
- Trunk: 3 hidden layers x 128 units, ReLU activations
- State head: Linear -> 108-dim predicted next state
- Reward head: Linear -> scalar predicted reward

Training data
-------------
The fit/eval helpers consume ``list[TrainingExample]`` — produced either
by :class:`~bicameral_agent.training_pipeline.TrainingDataPipeline`
(directly from parquet episode files) or by
:meth:`~bicameral_agent.training_data_store.TrainingDataStore.load_examples`.
The store's torch ``Dataset`` loaders are not used here: transition-model
fits are small, one-shot reads (hundreds of examples), so materializing
the examples is simpler than memory-mapped streaming and keeps a single
code path for both data sources.

Terminal decision points (``done=True``) carry a zero *placeholder*
``next_state``, so the state head is trained and evaluated only on
non-terminal transitions (masked loss). The reward head trains on all
transitions, including terminal ones — terminal rewards carry the episode
quality score, the main signal MCTS needs.

States are already normalized to roughly [0, 1] per dimension by the
pipeline's cap-based normalization, so "per-dimension normalized MSE" is
computed directly on the raw vectors.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bicameral_agent.policy_value_net import NUM_ACTIONS
from bicameral_agent.training_pipeline import STATE_DIM, TrainingExample

DEFAULT_HIDDEN_DIM: int = 128
"""Hidden-layer width per the issue #27 spec."""

ROLLOUT_NORM_BOUND_FACTOR: float = 10.0
"""Rollout states are "bounded" if their L2 norm stays below
``ROLLOUT_NORM_BOUND_FACTOR * sqrt(state_dim)``. A well-formed state has
components in ~[0, 1], hence norm <= sqrt(state_dim); the factor leaves
generous headroom while still catching exponential divergence."""


class TransitionModel(nn.Module):
    """MLP predicting ``(next_state, reward)`` from ``(state, action one-hot)``.

    Parameters
    ----------
    state_dim:
        Dimensionality of the state vector (default: ``STATE_DIM``).
    num_actions:
        Number of discrete actions (default: ``NUM_ACTIONS``); the action
        input is a one-hot of this width, indexed by
        ``policy_value_net.ACTION_ORDER``.
    hidden_dim:
        Width of each of the 3 hidden layers (default: 128).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, states: torch.Tensor, action_onehots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next states and rewards for a batch.

        Parameters
        ----------
        states:
            Tensor of shape ``(batch, state_dim)``.
        action_onehots:
            Tensor of shape ``(batch, num_actions)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(next_states, rewards)`` with shapes ``(batch, state_dim)``
            and ``(batch,)``.
        """
        features = self.trunk(torch.cat([states, action_onehots], dim=-1))
        next_states = self.state_head(features)
        rewards = self.reward_head(features).squeeze(-1)
        return next_states, rewards

    @torch.no_grad()
    def predict(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float]:
        """Numpy-in, numpy-out inference for a single ``(state, action)``.

        Parameters
        ----------
        state:
            1-D array of shape ``(state_dim,)``. Any float dtype is
            accepted; the input is cast to float32.
        action:
            Categorical action index in ``[0, num_actions)`` matching
            ``policy_value_net.ACTION_ORDER``.

        Returns
        -------
        tuple[ndarray, float]
            ``(next_state, reward)`` where ``next_state`` is a float32
            array of shape ``(state_dim,)`` and ``reward`` a Python float.
        """
        if not 0 <= action < self.num_actions:
            msg = f"action must be in [0, {self.num_actions}), got {action}"
            raise ValueError(msg)
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        onehot = torch.zeros((1, self.num_actions), dtype=torch.float32)
        onehot[0, action] = 1.0
        next_state, reward = self.forward(x, onehot)
        return next_state.squeeze(0).numpy(), reward.item()

    @torch.no_grad()
    def rollout(
        self, state: np.ndarray, actions: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a multi-step rollout by feeding predictions back in.

        Parameters
        ----------
        state:
            Initial state, shape ``(state_dim,)``.
        actions:
            Action index per step; the rollout length is ``len(actions)``.

        Returns
        -------
        tuple[ndarray, ndarray]
            ``(states, rewards)``: predicted states of shape
            ``(len(actions), state_dim)`` (float32) and rewards of shape
            ``(len(actions),)`` (float32).
        """
        states = np.zeros((len(actions), self.state_dim), dtype=np.float32)
        rewards = np.zeros(len(actions), dtype=np.float32)
        current = np.asarray(state, dtype=np.float32)
        for i, action in enumerate(actions):
            current, reward = self.predict(current, action)
            states[i] = current
            rewards[i] = reward
        return states, rewards

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        """Save model checkpoint to *path*."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: int) -> TransitionModel:
        """Load a model checkpoint from *path*.

        Parameters
        ----------
        path:
            File path to a saved state dict.
        **kwargs:
            Constructor keyword arguments (``state_dim``, ``num_actions``,
            ``hidden_dim``) to recreate the architecture before loading
            weights.
        """
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Training harness
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TransitionTrainingConfig:
    """Hyperparameters for :func:`fit_transition_model`."""

    hidden_dim: int = DEFAULT_HIDDEN_DIM
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1e-3
    reward_loss_weight: float = 1.0
    train_ratio: float = 0.8
    seed: int = 0
    rollout_steps: int = 5


@dataclasses.dataclass(frozen=True, slots=True)
class TransitionFitResult:
    """Output of :func:`fit_transition_model`.

    ``epoch_losses`` holds one dict per epoch with keys ``total``,
    ``state`` and ``reward`` (mean training loss over the epoch's
    batches). ``metrics`` is the held-out evaluation from
    :func:`evaluate_transition_model`.
    """

    model: TransitionModel
    epoch_losses: list[dict[str, float]]
    metrics: dict
    n_train: int
    n_val: int
    train_seconds: float


def split_examples(
    examples: list[TrainingExample],
    train_ratio: float = 0.8,
    seed: int = 0,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Deterministic (train, val) split, grouped by episode.

    Whole episodes go to one side or the other so temporally correlated
    decision points from the same episode never straddle the split.
    Episodes are assigned to train (in a seed-permuted order over sorted
    episode ids) until at least ``train_ratio`` of the examples are
    covered. With >= 2 episodes both sides are guaranteed non-empty.
    """
    if not 0.0 < train_ratio < 1.0:
        msg = f"train_ratio must be in (0, 1), got {train_ratio}"
        raise ValueError(msg)
    episode_ids = sorted({e.episode_id for e in examples})
    perm = np.random.default_rng(seed).permutation(len(episode_ids))
    ordered = [episode_ids[i] for i in perm]

    counts = {eid: 0 for eid in episode_ids}
    for e in examples:
        counts[e.episode_id] += 1
    total = len(examples)

    train_ids: set[str] = set()
    covered = 0
    for eid in ordered:
        if covered >= train_ratio * total and len(train_ids) < len(episode_ids):
            break
        # Never put every episode in train when a val side is possible.
        if len(train_ids) == len(episode_ids) - 1:
            break
        train_ids.add(eid)
        covered += counts[eid]

    train = [e for e in examples if e.episode_id in train_ids]
    val = [e for e in examples if e.episode_id not in train_ids]
    return train, val


def _to_tensors(
    examples: list[TrainingExample], num_actions: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack examples into ``(states, action_onehots, next_states, rewards, dones)``."""
    states = torch.from_numpy(np.stack([e.state for e in examples])).float()
    next_states = torch.from_numpy(np.stack([e.next_state for e in examples])).float()
    actions = torch.tensor([e.action for e in examples], dtype=torch.long)
    onehots = torch.nn.functional.one_hot(actions, num_classes=num_actions).float()
    rewards = torch.tensor([e.reward for e in examples], dtype=torch.float32)
    dones = torch.tensor([e.done for e in examples], dtype=torch.bool)
    return states, onehots, next_states, rewards, dones


def _masked_losses(
    model: TransitionModel,
    states: torch.Tensor,
    onehots: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(state_loss, reward_loss) with terminal rows excluded from state loss."""
    pred_next, pred_reward = model(states, onehots)
    non_terminal = ~dones
    if bool(non_terminal.any()):
        state_loss = torch.mean((pred_next[non_terminal] - next_states[non_terminal]) ** 2)
    else:
        state_loss = torch.zeros((), dtype=torch.float32)
    reward_loss = torch.mean((pred_reward - rewards) ** 2)
    return state_loss, reward_loss


def fit_transition_model(
    examples: list[TrainingExample],
    config: TransitionTrainingConfig | None = None,
) -> TransitionFitResult:
    """Train a :class:`TransitionModel` on training examples.

    Deterministic for a fixed ``config.seed`` and example list: weight
    initialization, the episode-level split and batch shuffling are all
    seeded. MSE losses for both heads (state loss masked on terminal
    rows, see module docstring).

    Raises
    ------
    ValueError
        If ``examples`` is empty.
    """
    config = config or TransitionTrainingConfig()
    if not examples:
        msg = "cannot fit a transition model on an empty example list"
        raise ValueError(msg)

    torch.manual_seed(config.seed)
    model = TransitionModel(hidden_dim=config.hidden_dim)

    train, val = split_examples(examples, config.train_ratio, config.seed)
    tensors = _to_tensors(train, model.num_actions)
    states, onehots, next_states, rewards, dones = tensors

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    n = len(train)

    start = time.perf_counter()
    epoch_losses: list[dict[str, float]] = []
    for _ in range(config.epochs):
        perm = rng.permutation(n)
        sums = {"total": 0.0, "state": 0.0, "reward": 0.0}
        n_batches = 0
        for lo in range(0, n, config.batch_size):
            idx = torch.from_numpy(perm[lo : lo + config.batch_size])
            state_loss, reward_loss = _masked_losses(
                model,
                states[idx],
                onehots[idx],
                next_states[idx],
                rewards[idx],
                dones[idx],
            )
            loss = state_loss + config.reward_loss_weight * reward_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums["total"] += loss.item()
            sums["state"] += state_loss.item()
            sums["reward"] += reward_loss.item()
            n_batches += 1
        epoch_losses.append({k: v / n_batches for k, v in sums.items()})
    train_seconds = time.perf_counter() - start

    model.eval()
    metrics = evaluate_transition_model(model, val, rollout_steps=config.rollout_steps)
    return TransitionFitResult(
        model=model,
        epoch_losses=epoch_losses,
        metrics=metrics,
        n_train=len(train),
        n_val=len(val),
        train_seconds=train_seconds,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def measure_forward_latency(
    model: TransitionModel,
    n_warmup: int = 10,
    n_iters: int = 100,
    seed: int = 0,
) -> float:
    """Median single-state ``predict`` latency in milliseconds."""
    rng = np.random.default_rng(seed)
    state = rng.random(model.state_dim).astype(np.float32)
    for _ in range(n_warmup):
        model.predict(state, 0)
    times = []
    for i in range(n_iters):
        start = time.perf_counter()
        model.predict(state, i % model.num_actions)
        times.append(time.perf_counter() - start)
    return float(sorted(times)[len(times) // 2] * 1000)


@torch.no_grad()
def evaluate_transition_model(
    model: TransitionModel,
    examples: list[TrainingExample],
    rollout_steps: int = 5,
    max_rollout_starts: int = 32,
) -> dict:
    """Evaluate a transition model on held-out examples.

    Returns a JSON-serializable dict with:

    - ``state_mse_per_dim``: per-dimension MSE over non-terminal
      transitions (states are pipeline-normalized, so these are the
      "normalized" MSEs from the issue AC), plus ``state_mse_per_dim_mean``
      and ``state_mse_per_dim_max``.
    - ``reward_mse`` and ``reward_correlation`` (Pearson r over all
      transitions; *None* when undefined, e.g. constant rewards or a
      single example).
    - ``rollout``: ``rollout_steps``-step feedback rollouts from held-out
      states — ``max_state_norm``, the boundedness threshold and a
      ``bounded`` flag.
    - ``latency_ms_median``: single-state forward latency.
    """
    was_training = model.training
    model.eval()

    out: dict = {
        "n_examples": len(examples),
        "n_state_examples": 0,
        "state_mse_per_dim": None,
        "state_mse_per_dim_mean": None,
        "state_mse_per_dim_max": None,
        "reward_mse": None,
        "reward_correlation": None,
        "rollout": None,
        "latency_ms_median": measure_forward_latency(model),
    }

    if examples:
        states, onehots, next_states, rewards, dones = _to_tensors(
            examples, model.num_actions
        )
        pred_next, pred_reward = model(states, onehots)

        non_terminal = ~dones
        out["n_state_examples"] = int(non_terminal.sum())
        if out["n_state_examples"] > 0:
            sq_err = (pred_next[non_terminal] - next_states[non_terminal]) ** 2
            per_dim = sq_err.mean(dim=0).numpy()
            out["state_mse_per_dim"] = [float(v) for v in per_dim]
            out["state_mse_per_dim_mean"] = float(per_dim.mean())
            out["state_mse_per_dim_max"] = float(per_dim.max())

        out["reward_mse"] = float(torch.mean((pred_reward - rewards) ** 2))
        actual = rewards.numpy()
        predicted = pred_reward.numpy()
        if len(actual) >= 2 and actual.std() > 0 and predicted.std() > 0:
            out["reward_correlation"] = float(np.corrcoef(actual, predicted)[0, 1])

        # Multi-step rollout divergence: feed predictions back in for
        # rollout_steps, cycling through the action space, from a sample
        # of held-out states.
        bound = ROLLOUT_NORM_BOUND_FACTOR * float(np.sqrt(model.state_dim))
        start_states = states[non_terminal] if out["n_state_examples"] else states
        n_starts = min(max_rollout_starts, len(start_states))
        max_norm = 0.0
        for i in range(n_starts):
            actions = [(i + step) % model.num_actions for step in range(rollout_steps)]
            rollout_states, _ = model.rollout(start_states[i].numpy(), actions)
            max_norm = max(max_norm, float(np.linalg.norm(rollout_states, axis=1).max()))
        out["rollout"] = {
            "steps": rollout_steps,
            "n_starts": n_starts,
            "max_state_norm": max_norm,
            "norm_bound": bound,
            "bounded": bool(n_starts > 0 and max_norm <= bound),
        }

    if was_training:
        model.train()
    return out
