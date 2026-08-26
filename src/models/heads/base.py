"""PredictionHead interface, shared by the Classification and Ordinal heads."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

__all__ = ["PredictionHead"]


class PredictionHead(nn.Module, ABC):
    """A head consuming the shared embedding produced by the neck."""

    @abstractmethod
    def forward(self, shared_embedding: Tensor) -> Tensor:
        """Return this head's raw logits for one shared embedding batch."""
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        """Serialisable description, for future run-manifest provenance."""
        return {"name": type(self).__name__}
