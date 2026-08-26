"""Geometric augmentations -- all label-preserving for DR grading.

The retina has no canonical orientation at fundus-camera scale, so flips and
rotations do not change what a grader would call the image. Every builder here
is a thin, validated wrapper over Albumentations: the value added is the
evidence tagging, the parameter bounds taken from the EDA, and the ordering
constraints documented below.

Evidence summary
----------------
============== ========= ===================================================
Transform      Evidence  Justification
============== ========= ===================================================
Horizontal fli both      Left/right eye makes mirroring anatomically plausible
Vertical flip  paper     Explicitly in the reference training recipe
Rotation       both      No fixed rotational frame; matches disc geometry
Scale jitter   eda       ~70x native pixel-count range across 17 resolutions
Cons. crop     eda       Optional; bounded so lesions cannot be cropped away
============== ========= ===================================================
"""

from __future__ import annotations

import albumentations as A
import cv2

from src.utils.config import (
    ConservativeCropConfig,
    FlipConfig,
    RotationConfig,
    ScaleJitterConfig,
)

__all__ = [
    "BORDER_MODES",
    "build_horizontal_flip",
    "build_vertical_flip",
    "build_rotation",
    "build_scale_jitter",
    "build_conservative_crop",
]

#: Configuration strings mapped to OpenCV border modes.
BORDER_MODES: dict[str, int] = {
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
}


def build_horizontal_flip(settings: FlipConfig) -> A.HorizontalFlip:
    """Mirror the image left-to-right.

    Fundus photographs come from either eye, so a mirrored image remains a
    physically realisable acquisition and no lesion morphology is distorted.

    Args:
        settings: Validated flip configuration.

    Returns:
        The configured transform.
    """
    return A.HorizontalFlip(p=settings.p)


def build_vertical_flip(settings: FlipConfig) -> A.VerticalFlip:
    """Mirror the image top-to-bottom.

    Slightly less anatomically natural than a horizontal mirror because it
    inverts the superior/inferior axis, but it is explicitly part of the
    reference paper's validated APTOS recipe and demonstrably did not harm the
    reported results.

    Args:
        settings: Validated flip configuration.

    Returns:
        The configured transform.
    """
    return A.VerticalFlip(p=settings.p)


def build_rotation(settings: RotationConfig) -> A.Rotate:
    """Rotate within the configured angular limits.

    Must run **after** the circular crop. Rotating a rectangular frame drags
    the black corners across the retina and can reintroduce or relocate the
    padding that preprocessing just removed; on a circularly cropped image the
    operation is geometrically benign.

    ``rotate_method="largest_box"`` with a constant fill keeps the output size
    fixed, so the fixed-resolution contract of the pipeline is preserved.

    Args:
        settings: Validated rotation configuration.

    Returns:
        The configured transform.
    """
    return A.Rotate(
        limit=tuple(settings.limit),
        border_mode=BORDER_MODES[settings.border_mode],
        fill=settings.fill_value,
        interpolation=cv2.INTER_LINEAR,
        crop_border=False,
        p=settings.p,
    )


def build_scale_jitter(settings: ScaleJitterConfig) -> A.Affine:
    """Apply mild isotropic scale jitter.

    Motivated by the EDA's resolution heterogeneity (17 distinct native
    resolutions spanning roughly a 70-fold pixel-count range). The range is
    kept narrow deliberately: aggressive scale jitter would amplify the very
    resolution-grade confound (r = 0.57) that the fixed resize exists to remove.

    Args:
        settings: Validated scale-jitter configuration.

    Returns:
        The configured transform.
    """
    return A.Affine(
        scale=tuple(settings.scale_limit),
        rotate=0.0,
        shear=0.0,
        translate_percent=None,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        p=settings.p,
    )


def build_conservative_crop(settings: ConservativeCropConfig, size: int) -> A.RandomResizedCrop:
    """Crop a large sub-region and resize it back to ``size``.

    Disabled by default. The lower scale bound is validated at 0.8 in
    :class:`~src.utils.config.ConservativeCropConfig`, because a more
    aggressive crop can remove the only lesion in a Grade 1 or Grade 3 image
    and thereby corrupt its label. Even at 0.85 this remains an ablation
    candidate rather than a default.

    Args:
        settings: Validated crop configuration.
        size: Output side length, matching ``preprocessing.image_size``.

    Returns:
        The configured transform.
    """
    return A.RandomResizedCrop(
        size=(size, size),
        scale=tuple(settings.scale),
        ratio=(0.95, 1.05),
        interpolation=cv2.INTER_LINEAR,
        p=settings.p,
    )
