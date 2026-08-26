"""Dual-Head composition: shared embedding -> {classification, ordinal}.

Paper-explicit (Figure 1): "A shared fully connected layer feeds into two
branches" (the FC layer itself lives in
:class:`~src.models.neck.shared_feature_neck.SharedFeatureNeck`; this
module only composes the two head outputs that consume its result) and
"The final prediction is derived via arg max over the classification
output, refined by the ordinal constraints." That refinement rule is an
**inference-time** decision rule, not a network layer, and is deliberately
**not** implemented here -- it belongs to the future Evaluation milestone.
This module only guarantees both raw logit streams are produced from the
identical shared embedding.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.models.heads.base import PredictionHead

__all__ = ["DualHead"]


class DualHead(nn.Module):
    """Composes the Classification Head and Ordinal Head.

    Args:
        classification_head: Produces ``[B, K]`` class logits.
        ordinal_head: Produces ``[B, K-1]`` "> k" threshold logits.
    """

    def __init__(self, classification_head: PredictionHead, ordinal_head: PredictionHead) -> None:
        super().__init__()
        self.classification_head = classification_head
        self.ordinal_head = ordinal_head

    def forward(self, shared_embedding: Tensor) -> dict[str, Tensor]:
        """Return both heads' raw logits, keyed by name.

        Args:
            shared_embedding: ``[B, hidden_dim]``, identical input to both
                heads (see :class:`~src.models.neck.shared_feature_neck.SharedFeatureNeck`).

        Returns:
            ``{"classification_logits": [B, K], "ordinal_logits": [B, K-1]}``.
        """
        return {
            "classification_logits": self.classification_head(shared_embedding),
            "ordinal_logits": self.ordinal_head(shared_embedding),
        }
