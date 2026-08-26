"""Named augmentation policies and their evidence tiers.

A *policy* is the mapping from configuration to an ordered transform list. Two
policies matter for this project:

``paper_faithful``
    Exactly the augmentations the Dual-SwinOrd paper names for APTOS 2019:
    horizontal flip, vertical flip, random rotation, colour jitter. Nothing
    else. This is the tier used to reproduce the published numbers on equal
    terms.

``eda_driven`` (default)
    The paper recipe plus the EDA-motivated additions -- independent brightness
    and contrast jitter, optional scale jitter, conservative crop, gamma, mild
    blur, and mild noise -- each enabled individually in the configuration.

Every entry carries an ``evidence`` tag, and
:func:`transforms_with_evidence` exposes them, so a unit test can assert that
the ``paper_faithful`` pipeline contains no EDA-only transform. That turns the
reproduction claim into something verified on every test run rather than
asserted in prose.

Transform ordering within the training policy is fixed: geometry first, then
photometry, then corruption. Geometric operations resample the image, so
applying them after adding noise would filter the noise and make its
configured magnitude meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import albumentations as A

from src.data.augmentation.corruption import build_gaussian_blur, build_gaussian_noise
from src.data.augmentation.geometric import (
    build_conservative_crop,
    build_horizontal_flip,
    build_rotation,
    build_scale_jitter,
    build_vertical_flip,
)
from src.data.augmentation.photometric import (
    build_color_jitter,
    build_gamma,
    build_random_brightness,
    build_random_contrast,
)
from src.utils.config import DataConfig
from src.utils.logger import get_logger

__all__ = ["TaggedTransform", "PAPER_TRANSFORMS", "build_policy", "transforms_with_evidence"]

logger = get_logger(__name__)

#: Configuration keys whose transforms the reference paper explicitly describes.
PAPER_TRANSFORMS: frozenset[str] = frozenset(
    {"horizontal_flip", "vertical_flip", "rotation", "color_jitter"}
)


@dataclass(frozen=True)
class TaggedTransform:
    """An augmentation together with its provenance.

    Attributes:
        key: Configuration key.
        transform: The Albumentations transform.
        evidence: ``"paper"``, ``"eda"``, or ``"both"``.
        priority: ``"essential"``, ``"recommended"``, or ``"optional"``.
    """

    key: str
    transform: A.BasicTransform
    evidence: str
    priority: str

    def describe(self) -> dict[str, Any]:
        """Return a serialisable description for the run manifest."""
        return {
            "key": self.key,
            "name": type(self.transform).__name__,
            "evidence": self.evidence,
            "priority": self.priority,
            "p": float(getattr(self.transform, "p", 1.0)),
        }


def build_policy(config: DataConfig) -> list[TaggedTransform]:
    """Build the training augmentation policy from the configuration.

    Only transforms whose ``enabled`` flag is set are included, and the
    ``paper_faithful`` profile has already disabled the EDA-only ones through
    :data:`src.utils.config.PAPER_FAITHFUL_OVERRIDES`, so the profile logic
    lives in exactly one place.

    Args:
        config: Validated data configuration.

    Returns:
        The ordered list of tagged transforms.
    """
    settings = config.augmentation.train
    size = config.preprocessing.image_size
    tagged: list[TaggedTransform] = []

    def add(key: str, transform: A.BasicTransform, evidence: str, priority: str) -> None:
        tagged.append(TaggedTransform(key=key, transform=transform, evidence=evidence, priority=priority))

    # -- geometry first: resampling must precede noise injection ----------
    if settings.horizontal_flip.enabled:
        add("horizontal_flip", build_horizontal_flip(settings.horizontal_flip), "both", "essential")
    if settings.vertical_flip.enabled:
        add("vertical_flip", build_vertical_flip(settings.vertical_flip), "paper", "recommended")
    if settings.rotation.enabled:
        add("rotation", build_rotation(settings.rotation), "both", "essential")
    if settings.scale_jitter.enabled:
        add("scale_jitter", build_scale_jitter(settings.scale_jitter), "eda", "optional")
    if settings.conservative_crop.enabled:
        add(
            "conservative_crop",
            build_conservative_crop(settings.conservative_crop, size),
            "eda",
            "optional",
        )

    # -- photometry --------------------------------------------------------
    if settings.random_brightness.enabled:
        add("random_brightness", build_random_brightness(settings.random_brightness), "both", "essential")
    if settings.random_contrast.enabled:
        add("random_contrast", build_random_contrast(settings.random_contrast), "both", "recommended")
    if settings.color_jitter.enabled:
        add("color_jitter", build_color_jitter(settings.color_jitter), "both", "essential")
    if settings.gamma.enabled:
        add("gamma", build_gamma(settings.gamma), "eda", "optional")

    # -- corruption last: it must not be resampled away -------------------
    if settings.gaussian_blur.enabled:
        add("gaussian_blur", build_gaussian_blur(settings.gaussian_blur), "eda", "optional")
    if settings.gaussian_noise.enabled:
        add("gaussian_noise", build_gaussian_noise(settings.gaussian_noise), "eda", "optional")

    logger.debug("augmentation policy (%s): %s", config.profile, [t.key for t in tagged])
    return tagged


def transforms_with_evidence(tagged: list[TaggedTransform], evidence: str) -> list[str]:
    """Return the keys of transforms carrying a given evidence tag.

    Args:
        tagged: Policy produced by :func:`build_policy`.
        evidence: Tag to filter on.

    Returns:
        Matching configuration keys.

    Example:
        >>> transforms_with_evidence([], "eda")
        []
    """
    return [item.key for item in tagged if item.evidence == evidence]
