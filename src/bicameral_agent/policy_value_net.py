"""Policy/value network for MCTS-based action selection.

Takes the 53-dim state vector from :mod:`~bicameral_agent.encoder` and outputs
action probabilities (policy head) plus a scalar value estimate (value head).

Architecture
------------
- Trunk: 3 hidden layers × 160 units, ReLU activations
- Policy head: Linear → softmax over 4 actions
- Value head: Linear → scalar
- ~61K parameters
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from bicameral_agent.encoder import FEATURE_DIM
from bicameral_agent.heuristic_controller import Action

NUM_ACTIONS: int = 4
"""Number of discrete actions the policy head selects from."""

ACTION_ORDER: tuple[Action, ...] = (
    Action.SCANNER,
    Action.AUDITOR,
    Action.REFRESHER,
    Action.DO_NOTHING,
)
"""Maps policy output index → Action enum value."""


class PolicyValueNetwork(nn.Module):
    """Neural network that produces action probabilities and a value estimate.

    Parameters
    ----------
    input_dim:
        Dimensionality of the state vector (default: ``FEATURE_DIM``).
    hidden_dim:
        Width of each hidden layer (default: 160).
    num_actions:
        Number of discrete actions (default: ``NUM_ACTIONS``).
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 160,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _shared_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk and return raw (policy_logits, value).

        This private method exposes logits before softmax, used internally
        for temperature-scaled sampling.
        """
        features = self.trunk(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard forward pass.

        Parameters
        ----------
        x:
            State tensor of shape ``(batch, input_dim)`` or ``(input_dim,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(action_probs, value)`` where ``action_probs`` has shape
            ``(batch, num_actions)`` and ``value`` has shape ``(batch,)``.
        """
        logits, value = self._shared_forward(x)
        probs = torch.softmax(logits, dim=-1)
        return probs, value

    @torch.no_grad()
    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Numpy-in, numpy-out inference for a single state.

        Parameters
        ----------
        state:
            1-D array of shape ``(input_dim,)``.

        Returns
        -------
        tuple[ndarray, float]
            ``(probs, value)`` where ``probs`` is shape ``(num_actions,)``
            and ``value`` is a Python float.
        """
        x = torch.from_numpy(state).unsqueeze(0)
        probs, value = self.forward(x)
        return probs.squeeze(0).numpy(), value.item()

    @torch.no_grad()
    def sample_action(self, state: np.ndarray, temperature: float = 1.0) -> Action:
        """Sample an action from the policy with temperature scaling.

        Parameters
        ----------
        state:
            1-D array of shape ``(input_dim,)``.
        temperature:
            Controls exploration. Values < 1 sharpen the distribution
            (more greedy), values > 1 flatten it (more exploratory).
            Must be positive.

        Returns
        -------
        Action
            The sampled action.
        """
        x = torch.from_numpy(state).unsqueeze(0)
        logits, _ = self._shared_forward(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        return ACTION_ORDER[idx]

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        """Save model checkpoint to *path*."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: int) -> PolicyValueNetwork:
        """Load a model checkpoint from *path*.

        Parameters
        ----------
        path:
            File path to a saved state dict.
        **kwargs:
            Constructor keyword arguments (``input_dim``, ``hidden_dim``,
            ``num_actions``) to recreate the architecture before loading
            weights.
        """
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model
