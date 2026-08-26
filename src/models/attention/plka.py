"""Parallel/Progressive Lesion-aware Kernel Attention (PLKA).

Paper-explicit (Figure 1 caption, quoted): three parallel convolutional
branches with dilation rates "standard" (r=1), r=2, and r=3, followed by
"an attention-based fusion mechanism". The dilation rates are hardcoded
below via :data:`PLKA_DILATION_RATES` because they are the one
quantitatively explicit detail in the paper's own figure caption.

Note the paper's own naming inconsistency, reported verbatim rather than
silently resolved: the abstract calls this module "Progressive
Lesion-aware Kernel Attention" while the figure caption calls it "Parallel
Large-Kernel Attention" -- both expand to the same acronym, PLKA.

Paper Gap PG-07 / PG-08 / PG-09 / PG-10 (see
``docs/milestone_04_paper_gaps.md``): branch activation, normalization,
the fusion mechanism's architecture, and which backbone stage feeds PLKA
are all unspecified. The fusion step is therefore an abstract slot with
**no concrete subclass shipped** in this milestone; activation and
normalization are injected as constructor factories (never hardcoded to a
"common practice" choice such as ReLU/BatchNorm).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import Tensor, nn

__all__ = ["PLKA_DILATION_RATES", "PLKAFusion", "PLKA", "DefaultPLKAFusion"]

#: The one quantitatively explicit detail in the paper's figure caption:
#: "parallel convolutional branches with different dilation rates
#: (standard, r = 2, r = 3)". NOT configurable -- see the module docstring.
PLKA_DILATION_RATES: tuple[int, int, int] = (1, 2, 3)


class PLKAFusion(nn.Module, ABC):
    """Abstract "attention-based fusion mechanism" (PG-09).

    Concrete subclasses must accept ``channels: int`` as their sole
    constructor argument (the shared channel width of all three PLKA
    branches) so :func:`~src.models.registry.Registry.build` can
    instantiate any registered fusion strategy uniformly.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    @abstractmethod
    def forward(self, branch_outputs: list[Tensor]) -> Tensor:
        """Fuse the three dilation-branch outputs into one feature map.

        Args:
            branch_outputs: Three ``[B, C, H, W]`` tensors, one per entry
                of :data:`PLKA_DILATION_RATES`, in that order.

        Returns:
            A single ``[B, C, H, W]`` fused feature map.
        """
        raise NotImplementedError


class PLKA(nn.Module):
    """Three fixed-dilation convolutional branches plus a pluggable fusion step.

    Args:
        channels: Input and output channel width (matches the backbone
            stage feeding this module).
        kernel_size: Convolution kernel size shared by all three branches
            (PG-07; required, no default).
        activation_factory: Zero-argument factory returning a fresh
            activation module per branch (PG-07; see
            :func:`src.models.factories.activation_factory`).
        normalization_factory: ``(channels) -> nn.Module`` factory for a
            fresh normalization layer per branch (PG-08; see
            :func:`src.models.factories.normalization_factory`).
        fusion: The (paper-unspecified) attention-based fusion module
            (PG-09).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation_factory: Callable[[], nn.Module],
        normalization_factory: Callable[[int], nn.Module],
        fusion: PLKAFusion,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size,
                        padding=dilation * (kernel_size - 1) // 2,
                        dilation=dilation,
                    ),
                    normalization_factory(channels),
                    activation_factory(),
                )
                for dilation in PLKA_DILATION_RATES
            ]
        )
        self.fusion = fusion

    def forward(self, x: Tensor) -> Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        return self.fusion(branch_outputs)


class DefaultPLKAFusion(PLKAFusion):
    """Engineering default resolving PG-09 -- **not a paper claim**.

    Chosen only so the model is trainable end-to-end while PG-09 (the
    "attention-based fusion mechanism" architecture) remains textually
    unspecified: a squeeze-and-excite-style channel attention computes one
    scalar weight per branch per channel from the branches' concatenated,
    globally-pooled statistics, a softmax normalizes the three branch
    weights against each other, and the branches are combined by that
    weighted sum. Replace this class the moment PG-09 is resolved from the
    paper.

    Args:
        channels: Shared channel width of all three PLKA branches.
        reduction: Squeeze ratio for the attention bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__(channels)
        num_branches = len(PLKA_DILATION_RATES)
        hidden = max(channels // reduction, 1)
        self.attention = nn.Sequential(
            nn.Linear(channels * num_branches, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * num_branches),
        )
        self.num_branches = num_branches

    def forward(self, branch_outputs: list[Tensor]) -> Tensor:
        batch_size, channels = branch_outputs[0].shape[:2]
        pooled = [branch.mean(dim=(2, 3)) for branch in branch_outputs]  # each [B, C]
        stacked = torch.cat(pooled, dim=1)  # [B, C * num_branches]
        logits = self.attention(stacked).view(batch_size, self.num_branches, channels)
        weights = torch.softmax(logits, dim=1)  # normalize across branches

        fused = torch.zeros_like(branch_outputs[0])
        for i, branch in enumerate(branch_outputs):
            branch_weight = weights[:, i, :].view(batch_size, channels, 1, 1)
            fused = fused + branch * branch_weight
        return fused
