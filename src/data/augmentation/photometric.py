"""Photometric augmentations targeting the illumination confound.

The EDA established the clinical motivation precisely: brightness spans
17.7-121.4 across the corpus and its tercile is *non-linearly associated with
DR grade* (chi-square = 31.688, p = 0.0001). That is a measured shortcut -- a
network can partially predict severity from exposure alone. Photometric
augmentation attacks it directly by making exposure uninformative.

Bounds matter as much as the transforms themselves:

* Brightness and contrast are held to roughly +/-20-25%. Beyond that, pale hard
  exudates wash out and dark haemorrhages sink into the background -- the
  augmentation would then destroy the very features it is meant to expose.
* Hue jitter is deliberately tiny (~3%). Measured hue is tightly clustered
  (23.57 deg +/- 14.65); wider jitter produces images outside the
  physiologically plausible retinal gamut, which is not robustness but noise.
* Gamma is optional and mild; it overlaps with brightness/contrast and is
  reported as redundant unless ablated separately.
"""

from __future__ import annotations

import albumentations as A

from src.utils.config import ColorJitterConfig, GammaConfig, MagnitudeConfig

__all__ = [
    "build_random_brightness",
    "build_random_contrast",
    "build_color_jitter",
    "build_gamma",
]


def build_random_brightness(settings: MagnitudeConfig) -> A.RandomBrightnessContrast:
    """Jitter brightness only.

    Implemented with ``RandomBrightnessContrast`` and a zero contrast limit so
    brightness and contrast remain independently configurable and independently
    ablatable.

    Args:
        settings: Validated magnitude configuration.

    Returns:
        The configured transform.
    """
    return A.RandomBrightnessContrast(
        brightness_limit=(-settings.limit, settings.limit),
        contrast_limit=(0.0, 0.0),
        brightness_by_max=True,
        ensure_safe_range=True,
        p=settings.p,
    )


def build_random_contrast(settings: MagnitudeConfig) -> A.RandomBrightnessContrast:
    """Jitter contrast only.

    Motivated by the observed grayscale-contrast range (10.9-70.7). Kept modest
    because extreme contrast scaling obscures microaneurysms, which are defined
    by small intensity differences from surrounding tissue.

    Args:
        settings: Validated magnitude configuration.

    Returns:
        The configured transform.
    """
    return A.RandomBrightnessContrast(
        brightness_limit=(0.0, 0.0),
        contrast_limit=(-settings.limit, settings.limit),
        brightness_by_max=True,
        ensure_safe_range=True,
        p=settings.p,
    )


def build_color_jitter(settings: ColorJitterConfig) -> A.ColorJitter:
    """Jitter brightness, contrast, saturation, and hue jointly.

    This is the transform the reference paper names explicitly for APTOS, so it
    carries both paper and EDA support; the EDA contributes the magnitude
    bounds rather than the decision to use it.

    Albumentations expresses brightness/contrast/saturation as multiplicative
    ranges around 1.0 and hue as an additive range, so the symmetric
    configuration values are converted accordingly.

    Args:
        settings: Validated colour-jitter configuration.

    Returns:
        The configured transform.
    """
    return A.ColorJitter(
        brightness=(1.0 - settings.brightness, 1.0 + settings.brightness),
        contrast=(1.0 - settings.contrast, 1.0 + settings.contrast),
        saturation=(1.0 - settings.saturation, 1.0 + settings.saturation),
        hue=(-settings.hue, settings.hue),
        p=settings.p,
    )


def build_gamma(settings: GammaConfig) -> A.RandomGamma:
    """Apply mild gamma jitter.

    Optional and off by default: it models exposure non-linearities that a
    simple additive brightness shift cannot, but overlaps heavily with
    brightness/contrast jitter and is not paper-validated.

    Args:
        settings: Validated gamma configuration (percentage units, so
            ``(80, 120)`` means gamma in ``[0.8, 1.2]``).

    Returns:
        The configured transform.
    """
    return A.RandomGamma(gamma_limit=tuple(settings.gamma_limit), p=settings.p)
