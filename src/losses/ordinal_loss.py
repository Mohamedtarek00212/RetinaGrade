"""Ordinal Loss (Eq. 8) -- "Deep Progressive Enhancement".

Paper-explicit (Section 3, Eq. 8, quoted; resolves Paper Gap PG-13/PG-14 --
see ``docs/milestone_04_paper_gaps.md``): "a Deep Progressive Enhancement
strategy to enforce rank consistency... decompose the task into K-1 binary
sub-tasks", ``L_ord = -sum_{k=1}^{K-1} [v_k log(p_k^ord) + (1-v_k) log(1-p_k^ord)]``.

The dataset (:class:`src.data.datasets.base.BaseRetinalDataset`) emits a
single plain integer label per the project's sample contract -- it
deliberately does not emit per-threshold ordinal targets, since that
construction is loss-specific, not a dataset concern. This module is
therefore the one place ``v_k = 1[label > k]`` is built.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.losses.base import HeadLoss

__all__ = ["OrdinalLoss", "ordinal_targets"]


def ordinal_targets(labels: Tensor, num_thresholds: int) -> Tensor:
    """Build the ``K-1`` per-threshold binary targets ``v_k = 1[label > k]``.

    Args:
        labels: ``[B]`` integer grade labels (``0..K-1``).
        num_thresholds: ``K - 1``.

    Returns:
        ``[B, K-1]`` float tensor of ``0.0``/``1.0`` targets.

    Example:
        >>> ordinal_targets(torch.tensor([0, 2, 4]), num_thresholds=4)
        tensor([[0., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 1., 1.]])
    """
    thresholds = torch.arange(num_thresholds, device=labels.device, dtype=labels.dtype)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


class OrdinalLoss(HeadLoss):
    """Binary cross-entropy over ``K-1`` independent "> k" thresholds (Eq. 8).

    Args:
        num_classes: ``K``, sourced from ``DataConfig.classes.num_classes``.
        pos_weight: Optional ``[K-1]`` positive-class weight forwarded to
            :class:`torch.nn.BCEWithLogitsLoss`. ``None`` reproduces Eq. 8
            exactly (no weighting term).
    """

    def __init__(self, num_classes: int, pos_weight: Tensor | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Return the scalar BCE loss for one batch.

        Args:
            logits: ``[B, K-1]`` raw "> k" threshold logits.
            labels: ``[B]`` integer grade labels (``0..K-1``).
        """
        targets = ordinal_targets(labels, self.num_thresholds)
        return self.criterion(logits, targets)

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "num_thresholds": self.num_thresholds,
            "pos_weighted": self.pos_weight is not None,
        }
