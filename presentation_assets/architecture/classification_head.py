"""Classification Head.

Paper-explicit (Section 3, quoted): "maps the feature vector to K = 5
logits" via a standard multi-class classification formulation. ``K`` is
sourced from the already-validated ``DataConfig.classes.num_classes``
rather than hardcoded to 5, keeping this module dataset-agnostic while
remaining paper-faithful for APTOS 2019.

Cross-entropy with label smoothing is the paper's stated *loss* for this
head -- out of scope for this milestone (see the Milestone 04 scope
boundary in ``docs/milestone_04_paper_gaps.md``); only the linear
projection to ``K`` logits is architecture, and is implemented here.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.models.heads.base import PredictionHead

__all__ = ["ClassificationHead"]


class ClassificationHead(PredictionHead):
    """Linear projection from the shared embedding to ``K`` class logits.

    Args:
        hidden_dim: Width of the shared embedding (matches
            ``NeckConfig.hidden_dim``).
        num_classes: ``K``, sourced from ``DataConfig.classes.num_classes``.
    """

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, shared_embedding: Tensor) -> Tensor:
        """Return ``[B, num_classes]`` raw logits (softmax deferred to the loss)."""
        return self.linear(shared_embedding)

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "num_classes": self.num_classes}
