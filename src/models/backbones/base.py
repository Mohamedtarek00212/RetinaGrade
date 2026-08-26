"""RetinalBackbone interface.

Declares the contract every visual backbone must satisfy so that
downstream modules (SPM, PLKA, the shared neck) depend only on this
interface and never on a specific backbone implementation (for example
``timm``). This is the "backbone abstraction" the Milestone 04 plan calls
for: swapping the Swin variant, or even the whole backbone family in a
future reproduction, touches only a new subclass of this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

__all__ = ["RetinalBackbone"]


class RetinalBackbone(nn.Module, ABC):
    """Hierarchical visual feature extractor.

    The Dual-SwinOrd paper states the backbone is a "hierarchical Swin
    Transformer" producing multi-scale features "across four stages"
    (Figure 1 caption) -- this interface encodes exactly that contract and
    nothing more; every concrete value (variant, resolution, channel
    widths) is left to the implementation and its configuration.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> list[Tensor]:
        """Return one feature map per stage, in increasing-depth order.

        Args:
            x: Input batch, ``[B, 3, H, W]``.

        Returns:
            A list of ``[B, C_i, H_i, W_i]`` tensors, one per stage.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def out_channels(self) -> list[int]:
        """Channel count of each stage's output feature map."""
        raise NotImplementedError

    @property
    @abstractmethod
    def out_strides(self) -> list[int]:
        """Total downsampling factor of each stage relative to the input."""
        raise NotImplementedError
