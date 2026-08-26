"""Shared Feature Neck interface.

Paper-explicit (Figure 1, component 4): "a shared fully connected layer
feeds into two branches" -- i.e. both heads must receive an identical
embedding produced by *one* shared computation. That sharing guarantee is
the one thing this module unconditionally enforces (both heads receive the
literal same tensor, in :meth:`SharedFeatureNeck.forward`).

Paper Gap PG-11 (see ``docs/milestone_04_paper_gaps.md``): the pooling
strategy used to reduce PLKA's spatial feature map to a vector is
unspecified -- :class:`NeckPooling` is an abstract slot with **no concrete
subclass shipped** in this milestone.

Paper Gap PG-12a / PG-12b: the shared FC layer's hidden dimension, and
whether any activation/dropout follows it *at all*, are unspecified.
Both are exposed as required, disable-able (``activation="identity"``,
``dropout=0.0``) constructor arguments rather than assumed present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import Tensor, nn

__all__ = ["NeckPooling", "SharedFeatureNeck", "GlobalAveragePooling"]


class NeckPooling(nn.Module, ABC):
    """Abstract spatial-reduction strategy (PG-11)."""

    @abstractmethod
    def forward(self, feature_map: Tensor) -> Tensor:
        """Reduce ``[B, C, H, W]`` to ``[B, C]``."""
        raise NotImplementedError


class GlobalAveragePooling(NeckPooling):
    """Engineering default resolving PG-11 -- **not a paper claim**.

    Chosen only so the model is trainable end-to-end while PG-11 (the
    spatial-reduction strategy) remains textually unspecified: plain global
    average pooling, the most common and parameter-free choice. Replace
    this class the moment PG-11 is resolved from the paper.
    """

    def forward(self, feature_map: Tensor) -> Tensor:
        return feature_map.mean(dim=(2, 3))


class SharedFeatureNeck(nn.Module):
    """Pools PLKA's output and projects it through one shared FC layer.

    Args:
        pooling: The (paper-unspecified) spatial-reduction strategy (PG-11).
        in_channels: Channel width of the incoming feature map.
        hidden_dim: Shared FC layer output width (PG-12a).
        dropout: Dropout probability after the FC layer; ``0.0`` fully
            disables it (PG-12b).
        activation_factory: Zero-argument factory for the activation
            following the FC layer; pass a factory returning
            ``nn.Identity()`` to disable it entirely (PG-12b).
    """

    def __init__(
        self,
        pooling: NeckPooling,
        in_channels: int,
        hidden_dim: int,
        dropout: float,
        activation_factory: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.fc = nn.Linear(in_channels, hidden_dim)
        self.activation = activation_factory()
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_map: Tensor) -> Tensor:
        """Return the shared embedding both heads will consume identically.

        Args:
            feature_map: ``[B, C, H, W]`` PLKA output.

        Returns:
            ``[B, hidden_dim]`` shared embedding.
        """
        pooled = self.pooling(feature_map)
        return self.dropout(self.activation(self.fc(pooled)))
