"""Total Loss (Eq. 9).

Paper-explicit (Section 3, Eq. 9, quoted): ``L_total = lambda * L_cls +
(1 - lambda) * L_ord``, "In our experiments, we set lambda = 0.5" -- see
``docs/milestone_04_paper_gaps.md``.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.losses.classification_loss import ClassificationLoss
from src.losses.ordinal_loss import OrdinalLoss

__all__ = ["TotalLoss"]


class TotalLoss(nn.Module):
    """Weighted sum of the Classification and Ordinal losses (Eq. 9).

    Args:
        classification_loss: Eq. 7 loss instance.
        ordinal_loss: Eq. 8 loss instance (or :class:`~src.losses.carm_loss.CARMLoss`).
        lambda_cls: Weight on ``classification_loss`` (paper-confirmed
            default ``0.5``); ``(1 - lambda_cls)`` weights ``ordinal_loss``.
    """

    def __init__(
        self,
        classification_loss: ClassificationLoss,
        ordinal_loss: OrdinalLoss,
        lambda_cls: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= lambda_cls <= 1.0:
            raise ValueError(f"lambda_cls must be in [0, 1], got {lambda_cls}")
        self.classification_loss = classification_loss
        self.ordinal_loss = ordinal_loss
        self.lambda_cls = lambda_cls

    def forward(self, outputs: dict[str, Tensor], labels: Tensor) -> dict[str, Tensor]:
        """Return the total and per-head losses for one batch.

        Args:
            outputs: Model output dict containing ``"classification_logits"``
                (``[B, K]``) and ``"ordinal_logits"`` (``[B, K-1]``) -- the
                exact keys produced by :meth:`src.models.dual_swinord.DualSwinOrd.forward`.
            labels: ``[B]`` integer grade labels (``0..K-1``).

        Returns:
            ``{"total": scalar, "classification": scalar, "ordinal": scalar}``.
        """
        classification = self.classification_loss(outputs["classification_logits"], labels)
        ordinal = self.ordinal_loss(outputs["ordinal_logits"], labels)
        total = self.lambda_cls * classification + (1.0 - self.lambda_cls) * ordinal
        return {"total": total, "classification": classification, "ordinal": ordinal}

    def describe(self) -> dict[str, object]:
        """Serialisable description, for run-manifest provenance."""
        return {
            "lambda_cls": self.lambda_cls,
            "classification_loss": self.classification_loss.describe(),
            "ordinal_loss": self.ordinal_loss.describe(),
        }
