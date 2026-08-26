"""Training and evaluation augmentation pipelines.

Public API:

``build_train_transforms(config)``
    The medically safe training policy, guard-checked against the forbidden
    list before it is returned.

``build_eval_transforms(config)``
    Deliberately empty. Validation and test data receive deterministic
    preprocessing only, so reported metrics describe the distribution the model
    will actually meet in deployment. No test-time augmentation is applied in
    this milestone.

The split asymmetry is enforced twice -- here, and again in
:meth:`src.data.preprocessing.pipeline.PreprocessingPipeline.build`, which
ignores augmentation for any split other than ``train``.
"""

from __future__ import annotations

import albumentations as A

from src.data.augmentation.guards import (
    DEFAULT_FORBIDDEN,
    ForbiddenAugmentationError,
    assert_no_forbidden_transforms,
)
from src.data.augmentation.policies import (
    PAPER_TRANSFORMS,
    TaggedTransform,
    build_policy,
    transforms_with_evidence,
)
from src.utils.config import DataConfig
from src.utils.logger import get_logger

__all__ = [
    "build_train_transforms",
    "build_eval_transforms",
    "describe_policy",
    "build_policy",
    "transforms_with_evidence",
    "TaggedTransform",
    "PAPER_TRANSFORMS",
    "assert_no_forbidden_transforms",
    "ForbiddenAugmentationError",
    "DEFAULT_FORBIDDEN",
]

logger = get_logger(__name__)


def build_train_transforms(config: DataConfig) -> list[A.BasicTransform]:
    """Build the training augmentation list.

    Args:
        config: Validated data configuration.

    Returns:
        The ordered transforms, or an empty list when augmentation is disabled.

    Raises:
        ForbiddenAugmentationError: If the resulting pipeline contains any
            transform on the configured forbidden list.
    """
    if not config.augmentation.enabled:
        logger.warning("augmentation is disabled; the training split will be deterministic")
        return []

    tagged = build_policy(config)
    transforms = [item.transform for item in tagged]
    assert_no_forbidden_transforms(transforms, config.augmentation.forbidden)
    logger.info(
        "training augmentation (%s profile): %s",
        config.profile,
        ", ".join(item.key for item in tagged) or "<none>",
    )
    return transforms


def build_eval_transforms(config: DataConfig) -> list[A.BasicTransform]:
    """Return the validation/test augmentation list, which is always empty.

    Args:
        config: Validated data configuration (unused; the signature mirrors
            :func:`build_train_transforms` so callers can treat them uniformly).

    Returns:
        An empty list.
    """
    del config
    return []


def describe_policy(config: DataConfig) -> list[dict[str, object]]:
    """Return a serialisable description of the training policy.

    Written into run manifests so any reported result can state exactly which
    augmentations, at which probabilities and evidence tier, produced it.

    Args:
        config: Validated data configuration.

    Returns:
        One dictionary per enabled transform.
    """
    return [item.describe() for item in build_policy(config)]
