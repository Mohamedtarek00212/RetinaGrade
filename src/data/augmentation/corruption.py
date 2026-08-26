"""Acquisition-corruption augmentations: mild blur and mild sensor noise.

Both are optional, both are off by default, and both are deliberately weak.

Blur is *inherently hostile* to this task: microaneurysms are a few pixels
across, and a Gaussian kernel wide enough to matter can erase them outright.
The justification for including it at all is empirical rather than
theoretical -- roughly one percent of the EDA's quality sample was flagged as
defocused, with visually confirmed blur, so mild blur reflects a genuine
acquisition condition rather than an invented one. The bounded sigma range
(0.5-1.5 px) and low probability keep it a realism cue rather than a
destructive one, and it must never be combined with aggressive downscaling.

Noise is safer but equally bounded: the EDA measured a modest mean Immerkaer
sigma of 0.436 with a long right tail (max 2.03), so a small amount of additive
Gaussian noise reproduces the noisier end of the corpus without swamping
low-contrast lesions.
"""

from __future__ import annotations

import math

import albumentations as A

from src.utils.config import BlurConfig, NoiseConfig

__all__ = ["build_gaussian_blur", "build_gaussian_noise", "variance_to_std_range"]

#: Maximum intensity value; used to convert intensity-domain variances into the
#: normalised standard deviations Albumentations 2.x expects.
MAX_PIXEL_VALUE: float = 255.0


def build_gaussian_blur(settings: BlurConfig) -> A.GaussianBlur:
    """Apply a mild Gaussian blur.

    ``blur_limit=0`` lets Albumentations derive the kernel size from sigma, so
    the configured sigma range is the single source of truth for blur strength.

    Args:
        settings: Validated blur configuration.

    Returns:
        The configured transform.
    """
    return A.GaussianBlur(blur_limit=0, sigma_limit=tuple(settings.sigma_limit), p=settings.p)


def variance_to_std_range(var_limit: tuple[float, float]) -> tuple[float, float]:
    """Convert an intensity-domain variance range to a normalised sigma range.

    The EDA and the Data Preparation report describe noise in intensity units
    (variance on a 0-255 scale), while Albumentations 2.x expects the standard
    deviation as a fraction of the dynamic range. Converting here keeps the
    configuration expressed in the units the clinical analysis used.

    Args:
        var_limit: ``(low, high)`` variance in squared intensity units.

    Returns:
        ``(low, high)`` standard deviations in ``[0, 1]``.

    Example:
        >>> low, high = variance_to_std_range((5.0, 25.0))
        >>> round(high, 4)
        0.0196
    """
    low, high = var_limit
    return (math.sqrt(max(low, 0.0)) / MAX_PIXEL_VALUE, math.sqrt(max(high, 0.0)) / MAX_PIXEL_VALUE)


def build_gaussian_noise(settings: NoiseConfig) -> A.GaussNoise:
    """Add mild zero-mean Gaussian noise.

    Args:
        settings: Validated noise configuration, expressed as an intensity
            variance range.

    Returns:
        The configured transform.
    """
    return A.GaussNoise(
        std_range=variance_to_std_range(tuple(settings.var_limit)),
        mean_range=(0.0, 0.0),
        per_channel=True,
        p=settings.p,
    )
