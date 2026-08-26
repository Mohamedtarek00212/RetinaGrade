"""HeadLoss interface, shared by the Classification and Ordinal losses.

Mirrors :class:`src.models.heads.base.PredictionHead`'s pattern (an
``nn.Module`` ABC with a ``describe()`` hook for manifest provenance) so the
loss package composes with the model package's existing conventions instead
of introducing a new one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

__all__ = ["HeadLoss"]


class HeadLoss(nn.Module, ABC):
    """A loss consuming one head's raw logits and the batch's integer labels."""

    @abstractmethod
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Return a scalar loss for one batch.

        Args:
            logits: Raw, unnormalized head output.
            labels: ``[B]`` integer grade labels (``0..K-1``).
        """
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        """Serialisable description, for run-manifest provenance."""
        return {"name": type(self).__name__}
