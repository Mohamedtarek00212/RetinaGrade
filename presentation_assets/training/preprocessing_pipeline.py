"""Composition of the deterministic preprocessing pipeline.

Canonical order, enforced here rather than left to whoever edits the YAML:

1. black-border removal        (EDA, essential)
2. circular crop               (EDA, essential)
3. fixed resize                (both, essential)
4. CLAHE                       (EDA, recommended, off by default)
5. illumination correction     (EDA, optional, off by default)
6. **augmentation**            (training split only, injected by the caller)
7. per-channel normalization   (both, essential)
8. tensor conversion

Why this order
--------------
* Cropping precedes resizing so the fixed-size budget is spent on retina rather
  than on padding, and so the resize factor is not distorted by the black frame.
* Resizing precedes augmentation for two reasons: rotating a 4288x2848 image is
  an order of magnitude more expensive than rotating a 512x512 one, and the
  circular crop makes post-resize rotation geometrically harmless.
* Normalization is last so it sees post-augmentation pixel statistics; the Data
  Preparation report is explicit on this point.
* Steps 1-5 are identical for train, validation, and test. That invariance is a
  code guarantee, not a convention: :meth:`PreprocessingPipeline.deterministic`
  takes no split argument, so the three partitions cannot silently diverge.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np

from src.data.preprocessing.base import RetinaTransform
from src.data.preprocessing.normalization import build_normalization
from src.data.preprocessing.registry import build_transform
from src.data.statistics import NormalizationStats
from src.utils.config import DataConfig
from src.utils.logger import get_logger

__all__ = ["PreprocessingPipeline", "build_deterministic_transforms"]

logger = get_logger(__name__)


def build_deterministic_transforms(config: DataConfig) -> list[RetinaTransform]:
    """Build the enabled deterministic transforms in canonical order.

    Args:
        config: Validated data configuration.

    Returns:
        The transform list. Disabled steps are simply absent, so an empty list
        is a legitimate result (for example, a ``paper_faithful`` profile with
        resizing handled elsewhere).
    """
    settings = config.preprocessing
    transforms: list[RetinaTransform] = []

    if settings.black_border_removal.enabled:
        transforms.append(
            build_transform(
                "black_border_removal",
                threshold=settings.black_border_removal.threshold,
                blur_kernel=settings.black_border_removal.blur_kernel,
                min_area_ratio=settings.black_border_removal.min_area_ratio,
                padding=settings.black_border_removal.padding,
            )
        )
    if settings.circular_crop.enabled:
        transforms.append(
            build_transform(
                "circular_crop",
                margin_ratio=settings.circular_crop.margin_ratio,
                fill_value=settings.circular_crop.fill_value,
            )
        )
    if settings.resize.enabled:
        transforms.append(
            build_transform(
                "resize",
                size=settings.image_size,
                interpolation_down=settings.resize.interpolation_down,
                interpolation_up=settings.resize.interpolation_up,
                keep_aspect_ratio=settings.resize.keep_aspect_ratio,
            )
        )
    if settings.clahe.enabled:
        transforms.append(
            build_transform(
                "clahe",
                clip_limit=settings.clahe.clip_limit,
                tile_grid_size=tuple(settings.clahe.tile_grid_size),
                color_space=settings.clahe.color_space,
            )
        )
    if settings.illumination_correction.enabled:
        transforms.append(
            build_transform(
                "illumination_correction",
                sigma=settings.illumination_correction.sigma,
                weight=settings.illumination_correction.weight,
            )
        )

    return transforms


class PreprocessingPipeline:
    """Deterministic preprocessing, composable with an augmentation policy.

    The same instance serves three consumers, which is what keeps the pipeline
    honest: the normalization-statistics pass (deterministic steps only), the
    dataset (deterministic + augmentation + normalization), and the preview
    renderer (deterministic steps, no tensor).

    Args:
        config: Validated data configuration.

    Example:
        >>> pipeline = PreprocessingPipeline(config)            # doctest: +SKIP
        >>> processed = pipeline(raw_rgb_image)                 # doctest: +SKIP
        >>> compose = pipeline.build(split="train",             # doctest: +SKIP
        ...                          stats=stats,
        ...                          augmentation=train_augmentations)
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.transforms: list[RetinaTransform] = build_deterministic_transforms(config)
        self._deterministic = A.Compose(list(self.transforms)) if self.transforms else None
        logger.debug(
            "deterministic preprocessing: %s",
            [transform.describe()["name"] for transform in self.transforms] or ["<none>"],
        )

    # -- deterministic path ------------------------------------------------

    def deterministic(self) -> A.Compose:
        """Return the split-invariant transforms as a ``Compose``.

        Notably takes no ``split`` argument: preprocessing that differed
        between train and test would introduce its own distribution shift.
        """
        return self._deterministic or A.Compose([])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply only the deterministic steps to a ``uint8`` RGB image.

        This is the callable injected into
        :func:`src.data.statistics.compute_normalization_stats`, guaranteeing
        that the statistics describe exactly the pixels the network will see
        before augmentation.

        Args:
            image: ``H x W x 3`` ``uint8`` RGB image.

        Returns:
            The processed ``uint8`` image.
        """
        if self._deterministic is None:
            return image
        return self._deterministic(image=image)["image"]

    # -- full pipeline -----------------------------------------------------

    def build(
        self,
        split: str,
        stats: NormalizationStats,
        augmentation: list[A.BasicTransform] | None = None,
        to_tensor: bool = True,
    ) -> A.Compose:
        """Compose the full pipeline for a split.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            stats: Resolved normalization statistics.
            augmentation: Augmentation transforms; ignored for any split other
                than ``"train"``, so an accidental call can never augment the
                evaluation data.
            to_tensor: Append ``ToTensorV2``.

        Returns:
            The composed pipeline.

        Raises:
            ValueError: If ``split`` is unknown.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split {split!r}; expected train, val, or test")

        stages: list[A.BasicTransform] = list(self.transforms)
        if split == "train" and augmentation:
            stages.extend(augmentation)
        elif augmentation:
            logger.debug("augmentation ignored for split '%s' (evaluation must stay deterministic)", split)

        stages.extend(build_normalization(stats, to_tensor=to_tensor))
        return A.Compose(stages)

    # -- introspection -----------------------------------------------------

    def describe(self) -> list[dict[str, Any]]:
        """Return a serialisable description of every deterministic step."""
        return [transform.describe() for transform in self.transforms]

    @property
    def cache_key(self) -> str:
        """Hash identifying this geometry configuration.

        Used as the on-disk cache key for preprocessed images and for
        normalization statistics, so changing an augmentation probability does
        not invalidate an expensive geometric cache.
        """
        return self.config.preprocessing_hash
