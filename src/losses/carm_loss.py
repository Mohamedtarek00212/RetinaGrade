"""CARM Loss -- "Cost-sensitive Adaptive Risk Minimization" (named in the
abstract; Eq. 8 is its only concrete, retrievable formulation).

Paper Gap PG-17 (see ``docs/milestone_04_paper_gaps.md``): the abstract
names this loss "cost-sensitive" and "adaptive", stating its purpose is "to
prevent bias toward majority classes", but Eq. 8 -- the only concrete
ordinal-loss equation found -- is a plain, unweighted per-threshold binary
cross-entropy with no visible cost matrix or adaptive weighting term.

This class is therefore a thin subclass of :class:`~src.losses.ordinal_loss.OrdinalLoss`
that reproduces Eq. 8 **exactly** by default. It exposes one optional,
off-by-default, clearly-labeled extension point (``pos_weight``) so a future
resolution of PG-17 has somewhere to attach without inventing a mechanism
now; using it must be an explicit opt-in, never silent.
"""

from __future__ import annotations

from torch import Tensor

from src.losses.ordinal_loss import OrdinalLoss

__all__ = ["CARMLoss"]


class CARMLoss(OrdinalLoss):
    """Eq. 8's per-threshold BCE, with an optional (disabled by default)
    cost-weighting hook for the unconfirmed "cost-sensitive" mechanism (PG-17).

    Args:
        num_classes: ``K``, sourced from ``DataConfig.classes.num_classes``.
        pos_weight: **Not paper-confirmed** (PG-17). ``None`` (the default)
            reproduces Eq. 8 exactly. Supplying a ``[K-1]`` tensor enables
            :class:`torch.nn.BCEWithLogitsLoss`'s positive-class weighting
            as a future-extension point, not a paper-cited mechanism.
    """

    def __init__(self, num_classes: int, pos_weight: Tensor | None = None) -> None:
        super().__init__(num_classes=num_classes, pos_weight=pos_weight)

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "paper_gap": "PG-17: cost-sensitive mechanism unconfirmed beyond Eq. 8's plain BCE",
        }
