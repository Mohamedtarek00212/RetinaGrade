"""Classification Loss (Eq. 7).

Paper-explicit (Section 3, Eq. 7, quoted): "Cross-Entropy loss with Label
Smoothing", ``L_cls = -sum_{i=0}^{K-1} y_i log(p_i^cls)``. ``nn.CrossEntropyLoss``
already implements exactly this (softmax + negative log-likelihood, with an
optional label-smoothing term folded into the target distribution ``y_i``),
so it is reused directly rather than reimplemented.

Paper Gap PG-20 (see ``docs/milestone_04_paper_gaps.md``): the smoothing
epsilon's numeric value is never given, so ``label_smoothing`` is a required
constructor argument with no default.

Optional class weighting is exposed via ``class_weights`` but is **not** a
paper claim (Eq. 7 shows no weight term); when supplied, the caller must
have computed it with :func:`src.data.statistics.compute_class_weights`
(never recomputed here) -- see ``docs/milestone_04_paper_gaps.md``'s class
imbalance division of responsibility.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.losses.base import HeadLoss

__all__ = ["ClassificationLoss"]


class ClassificationLoss(HeadLoss):
    """Cross-entropy with label smoothing over the ``K`` classification logits.

    Args:
        label_smoothing: Smoothing epsilon in ``[0, 1)`` (PG-20) -- required.
        class_weights: Optional per-class weight tensor of shape ``[K]``,
            precomputed by :func:`src.data.statistics.compute_class_weights`.
            ``None`` disables weighting entirely (Eq. 7's literal formula).
    """

    def __init__(self, label_smoothing: float, class_weights: Tensor | None = None) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=self.class_weights)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Return the scalar cross-entropy loss for one batch.

        Args:
            logits: ``[B, K]`` raw classification logits.
            labels: ``[B]`` integer grade labels (``0..K-1``).
        """
        return self.criterion(logits, labels)

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "label_smoothing": self.label_smoothing,
            "class_weighted": self.class_weights is not None,
        }
