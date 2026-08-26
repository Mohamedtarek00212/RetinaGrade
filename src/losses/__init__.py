"""Loss functions: Eq. 7 (Classification), Eq. 8 (Ordinal/CARM), Eq. 9 (Total).

Reuses :class:`src.models.registry.Registry` directly rather than
implementing a second registry engine.
"""

from __future__ import annotations

from torch import Tensor

from src.losses.base import HeadLoss
from src.losses.carm_loss import CARMLoss
from src.losses.classification_loss import ClassificationLoss
from src.losses.ordinal_loss import OrdinalLoss, ordinal_targets
from src.losses.total_loss import TotalLoss
from src.models.registry import Registry
from src.training.config import LossConfig

__all__ = [
    "HeadLoss",
    "ClassificationLoss",
    "OrdinalLoss",
    "ordinal_targets",
    "CARMLoss",
    "TotalLoss",
    "LOSS_REGISTRY",
    "build_total_loss",
]

#: Registered head-loss implementations.
LOSS_REGISTRY: Registry[HeadLoss] = Registry("loss")
LOSS_REGISTRY.register("classification")(ClassificationLoss)
LOSS_REGISTRY.register("ordinal")(OrdinalLoss)
LOSS_REGISTRY.register("carm")(CARMLoss)


def build_total_loss(
    config: LossConfig,
    num_classes: int,
    class_weights: Tensor | None = None,
) -> TotalLoss:
    """Assemble the Eq. 9 :class:`TotalLoss` from a validated :class:`LossConfig`.

    Args:
        config: Validated loss configuration.
        num_classes: ``K``, sourced from ``DataConfig.classes.num_classes``.
        class_weights: Optional per-class weight tensor of shape ``[K]``,
            precomputed exactly once by
            :func:`src.data.statistics.compute_class_weights` -- never
            recomputed here.

    Returns:
        The assembled :class:`TotalLoss`.
    """
    classification_loss = ClassificationLoss(
        label_smoothing=config.label_smoothing, class_weights=class_weights
    )
    # `config.carm_pos_weight_enabled` is a reserved, currently-inert flag: no
    # concrete cost-weighting formula is confirmed for PG-17, so there is no
    # `pos_weight` tensor to wire up yet. Flipping it on today would still
    # produce Eq. 8's exact behavior (`pos_weight=None`); it exists only so a
    # future resolution of PG-17 has a config surface to attach to.
    ordinal_loss = CARMLoss(num_classes=num_classes, pos_weight=None)
    return TotalLoss(classification_loss, ordinal_loss, lambda_cls=config.lambda_cls)
