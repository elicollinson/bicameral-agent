"""Shared state-dict checkpoint save/load for torch models.

Extracted from the identical save/load pairs on
:class:`~bicameral_agent.policy_value_net.PolicyValueNetwork` and
:class:`~bicameral_agent.transition_model.TransitionModel`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import torch


class TorchCheckpointMixin:
    """State-dict ``save``/``load`` for ``nn.Module`` subclasses.

    Mix in ahead of ``nn.Module``. ``load`` is a classmethod, so ``cls``
    stays bound to the concrete subclass it is called on and returns an
    instance of that subclass.
    """

    def save(self, path: str | Path) -> None:
        """Save model checkpoint (state dict) to *path*."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: int) -> Self:
        """Load a model checkpoint from *path*.

        Parameters
        ----------
        path:
            File path to a saved state dict.
        **kwargs:
            Constructor keyword arguments to recreate the architecture
            before loading weights (e.g. ``input_dim`` / ``hidden_dim``).
        """
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model
