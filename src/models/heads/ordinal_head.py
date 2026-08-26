"""Ordinal Head interface.

Paper-explicit (Figure 1 caption, quoted verbatim): "an auxiliary Ordinal
Head that enforces rank-consistent constraints (i.e., predicting whether
severity > k)". This fixes only the *semantic* output contract: ``K - 1``
logits, one per threshold ``k = 0 .. K - 2``, each intended to indicate
"severity > k".

Paper Gap PG-13 (see ``docs/milestone_04_paper_gaps.md``): the
parameterization that achieves this -- independent per-threshold linear
layers, a CORAL-style shared-weight/per-threshold-bias formulation, a
CONDOR-style conditional formulation, or something else -- is not
specified, so **no concrete subclass ships in this milestone**.

Paper Gap PG-14: the abstract additionally names this head "DPE"
("Ordinal Regression Head (DPE)") without ever expanding or defining the
acronym anywhere in the retrieved paper excerpts. This module does **not**
claim to implement "DPE" -- only the quoted "> k" semantic.
"""

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor, nn

from src.models.heads.base import PredictionHead

__all__ = ["OrdinalHead", "IndependentOrdinalHead"]


class OrdinalHead(PredictionHead):
    """Abstract head producing ``K - 1`` "> k" threshold logits.

    Args:
        num_classes: ``K``, sourced from ``DataConfig.classes.num_classes``.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

    @abstractmethod
    def forward(self, shared_embedding: Tensor) -> Tensor:
        """Return ``[B, num_thresholds]`` logits, one per "> k" threshold."""
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "num_thresholds": self.num_thresholds}


class IndependentOrdinalHead(OrdinalHead):
    """Engineering default resolving PG-13/PG-14 -- **not a paper claim**.

    Chosen only so the model is trainable end-to-end while PG-13/PG-14 (the
    threshold parameterization and the meaning of "DPE") remain textually
    unspecified: one independent linear layer produces all ``K - 1``
    threshold logits directly from the shared embedding (no CORAL-style
    weight tying, no explicit rank-consistency constraint beyond the one
    :class:`~src.losses.ordinal_loss.OrdinalLoss` already applies at the
    loss level). Replace this class the moment PG-13/PG-14 is resolved
    from the paper.

    Args:
        hidden_dim: Width of the shared embedding produced by
            :class:`~src.models.neck.shared_feature_neck.SharedFeatureNeck`.
        num_classes: ``K``, sourced from ``DataConfig.classes.num_classes``.
    """

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__(num_classes)
        self.linear = nn.Linear(hidden_dim, self.num_thresholds)

    def forward(self, shared_embedding: Tensor) -> Tensor:
        return self.linear(shared_embedding)
