"""Supervised pre-training of the policy/value network (issue #26).

Trains :class:`~bicameral_agent.policy_value_net.PolicyValueNetwork` on
heuristic-controller episodes to imitate the heuristic's actions:

- Policy head: cross-entropy against the heuristic's action at each
  decision point.
- Value head: MSE against the discounted return
  (:attr:`~bicameral_agent.training_pipeline.TrainingExample.discounted_return`,
  γ = 0.95 per the pipeline).

This validates that the network can learn from the 108-dim state
representation before attempting RL.

Training data
-------------
The fit/eval helpers consume ``list[TrainingExample]`` — produced either
by :class:`~bicameral_agent.training_pipeline.TrainingDataPipeline`
(directly from parquet episode files) or by
:meth:`~bicameral_agent.training_data_store.TrainingDataStore.load_examples`
— mirroring :mod:`~bicameral_agent.transition_model`, whose
episode-grouped deterministic :func:`~bicameral_agent.transition_model.split_examples`
is reused for the 80/20 train/validation split.

Early stopping
--------------
Training runs until the validation loss plateaus: after ``min_epochs``
epochs, it stops once the best validation loss has not improved by more
than ``min_delta`` for ``patience`` consecutive epochs. The returned
model carries the weights from the best-validation-loss epoch.
"""

from __future__ import annotations

import copy
import dataclasses
import time

import numpy as np
import torch
import torch.nn.functional as F

from bicameral_agent.policy_value_net import NUM_ACTIONS, PolicyValueNetwork
from bicameral_agent.training_pipeline import STATE_DIM, TrainingExample
from bicameral_agent.transition_model import split_examples

DEFAULT_HIDDEN_DIM: int = 160
"""Hidden-layer width, matching the PolicyValueNetwork default."""

DEFAULT_VALUE_LOSS_WEIGHT: float = 0.5
"""Weight of the value-head MSE in the combined training loss."""


@dataclasses.dataclass(frozen=True, slots=True)
class PretrainConfig:
    """Hyperparameters for :func:`pretrain_policy_value`."""

    hidden_dim: int = DEFAULT_HIDDEN_DIM
    max_epochs: int = 300
    min_epochs: int = 10
    patience: int = 20
    min_delta: float = 1e-4
    batch_size: int = 64
    learning_rate: float = 1e-3
    value_loss_weight: float = DEFAULT_VALUE_LOSS_WEIGHT
    train_ratio: float = 0.8
    seed: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class PretrainResult:
    """Output of :func:`pretrain_policy_value`.

    ``history`` holds one dict per completed epoch with keys
    ``train_loss``, ``train_policy_loss``, ``train_value_loss``,
    ``val_loss``, ``val_policy_loss``, ``val_value_loss``,
    ``val_action_accuracy`` and ``val_value_correlation`` (*None* when
    undefined). ``best_epoch`` is the 1-indexed epoch whose weights the
    returned model carries. ``metrics`` is the held-out evaluation from
    :func:`evaluate_policy_value` at those weights.
    """

    model: PolicyValueNetwork
    history: list[dict[str, float | None]]
    best_epoch: int
    metrics: dict
    n_train: int
    n_val: int
    train_seconds: float


def _to_tensors(
    examples: list[TrainingExample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack examples into ``(states, actions, returns)`` tensors."""
    states = torch.from_numpy(np.stack([e.state for e in examples])).float()
    actions = torch.tensor([e.action for e in examples], dtype=torch.long)
    returns = torch.tensor([e.discounted_return for e in examples], dtype=torch.float32)
    return states, actions, returns


def _losses(
    model: PolicyValueNetwork,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    value_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total, policy CE, value MSE) for a batch.

    Uses the network's pre-softmax logits (``_shared_forward``) so the
    cross-entropy is computed in a numerically stable way instead of
    re-deriving log-probs from the softmax output.
    """
    logits, values = model._shared_forward(states)  # noqa: SLF001 — same package
    policy_loss = F.cross_entropy(logits, actions)
    value_loss = F.mse_loss(values, returns)
    return policy_loss + value_loss_weight * value_loss, policy_loss, value_loss


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, or *None* when undefined (constant inputs, n < 2)."""
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate_policy_value(
    model: PolicyValueNetwork,
    examples: list[TrainingExample],
    value_loss_weight: float = DEFAULT_VALUE_LOSS_WEIGHT,
) -> dict:
    """Evaluate policy/value predictions on held-out examples.

    Returns a JSON-serializable dict with:

    - ``action_accuracy``: fraction of examples where the policy head's
      argmax matches the heuristic's action.
    - ``majority_action_fraction``: frequency of the most common true
      action — the accuracy a constant predictor would achieve, for
      calibrating the 80%-accuracy AC.
    - ``value_correlation``: Pearson r between predicted values and
      discounted returns (*None* when undefined) plus ``value_mse``.
    - ``policy_loss`` / ``value_loss`` / ``loss``: the training losses.
    - ``true_action_counts`` / ``predicted_action_counts``: per-action
      index histograms (length ``NUM_ACTIONS``).
    """
    was_training = model.training
    model.eval()
    out: dict = {
        "n_examples": len(examples),
        "action_accuracy": None,
        "majority_action_fraction": None,
        "value_correlation": None,
        "value_mse": None,
        "policy_loss": None,
        "value_loss": None,
        "loss": None,
        "true_action_counts": None,
        "predicted_action_counts": None,
    }
    if examples:
        states, actions, returns = _to_tensors(examples)
        logits, values = model._shared_forward(states)  # noqa: SLF001 — same package
        predicted = logits.argmax(dim=-1)
        policy_loss = F.cross_entropy(logits, actions)
        value_loss = F.mse_loss(values, returns)
        true_counts = torch.bincount(actions, minlength=NUM_ACTIONS)
        out.update(
            action_accuracy=float((predicted == actions).float().mean()),
            majority_action_fraction=float(true_counts.max()) / len(examples),
            value_correlation=_pearson_r(values.numpy(), returns.numpy()),
            value_mse=float(value_loss),
            policy_loss=float(policy_loss),
            value_loss=float(value_loss),
            loss=float(policy_loss) + value_loss_weight * float(value_loss),
            true_action_counts=true_counts.tolist(),
            predicted_action_counts=torch.bincount(
                predicted, minlength=NUM_ACTIONS
            ).tolist(),
        )
    if was_training:
        model.train()
    return out


def pretrain_policy_value(
    examples: list[TrainingExample],
    config: PretrainConfig | None = None,
) -> PretrainResult:
    """Pre-train a :class:`PolicyValueNetwork` to imitate logged actions.

    Deterministic for a fixed ``config.seed`` and example list: weight
    initialization, the episode-grouped split and batch shuffling are
    all seeded.

    Raises
    ------
    ValueError
        If ``examples`` is empty or covers fewer than two distinct
        episodes (an episode-grouped train/val split needs at least one
        whole episode on each side).
    """
    config = config or PretrainConfig()
    if not examples:
        msg = "cannot pre-train on an empty example list"
        raise ValueError(msg)
    n_episodes = len({e.episode_id for e in examples})
    if n_episodes < 2:
        msg = (
            "pre-training needs examples from >= 2 episodes for an "
            f"episode-grouped train/val split, got {n_episodes}"
        )
        raise ValueError(msg)

    torch.manual_seed(config.seed)
    model = PolicyValueNetwork(input_dim=STATE_DIM, hidden_dim=config.hidden_dim)

    train, val = split_examples(examples, config.train_ratio, config.seed)
    train_states, train_actions, train_returns = _to_tensors(train)
    val_states, val_actions, val_returns = _to_tensors(val)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    n = len(train)

    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    epochs_since_best = 0

    start = time.perf_counter()
    history: list[dict[str, float | None]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        perm = rng.permutation(n)
        sums = {"total": 0.0, "policy": 0.0, "value": 0.0}
        n_batches = 0
        for lo in range(0, n, config.batch_size):
            idx = torch.from_numpy(perm[lo : lo + config.batch_size])
            loss, policy_loss, value_loss = _losses(
                model,
                train_states[idx],
                train_actions[idx],
                train_returns[idx],
                config.value_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums["total"] += loss.item()
            sums["policy"] += policy_loss.item()
            sums["value"] += value_loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_total, val_policy, val_value = _losses(
                model, val_states, val_actions, val_returns, config.value_loss_weight
            )
            val_logits, val_values = model._shared_forward(val_states)  # noqa: SLF001
            val_accuracy = float((val_logits.argmax(dim=-1) == val_actions).float().mean())
            val_corr = _pearson_r(val_values.numpy(), val_returns.numpy())

        val_loss = float(val_total)
        history.append(
            {
                "train_loss": sums["total"] / n_batches,
                "train_policy_loss": sums["policy"] / n_batches,
                "train_value_loss": sums["value"] / n_batches,
                "val_loss": val_loss,
                "val_policy_loss": float(val_policy),
                "val_value_loss": float(val_value),
                "val_action_accuracy": val_accuracy,
                "val_value_correlation": val_corr,
            }
        )

        if best_state is None or val_loss < best_val - config.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch >= config.min_epochs and epochs_since_best > config.patience:
            break
    train_seconds = time.perf_counter() - start

    assert best_state is not None  # at least one epoch always runs
    model.load_state_dict(best_state)
    model.eval()
    metrics = evaluate_policy_value(model, val, config.value_loss_weight)
    return PretrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        metrics=metrics,
        n_train=len(train),
        n_val=len(val),
        train_seconds=train_seconds,
    )
