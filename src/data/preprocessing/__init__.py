"""Deterministic retinal preprocessing.

Public API:

``PreprocessingPipeline``
    Composes the enabled deterministic steps in canonical order and, given
    normalization statistics and an optional augmentation policy, produces the
    Albumentations pipeline a dataset consumes.

Individual transforms (``BlackBorderRemoval``, ``CircularCrop``,
``FundusResize``, ``CLAHEEnhancement``, ``IlluminationCorrection``) are exported
for testing and for ad-hoc use in notebooks.

Every step is tagged with the evidence supporting it (``"paper"``, ``"eda"``,
or ``"both"``); see :mod:`src.data.preprocessing.base`.
"""

from src.data.preprocessing.base import RetinaTransform, TransformValidationError
from src.data.preprocessing.geometry import (
    INTERPOLATIONS,
    BlackBorderRemoval,
    CircularCrop,
    FundusResize,
)
from src.data.preprocessing.intensity import CLAHEEnhancement, IlluminationCorrection
from src.data.preprocessing.normalization import build_normalization
from src.data.preprocessing.pipeline import PreprocessingPipeline, build_deterministic_transforms
from src.data.preprocessing.registry import (
    available_transforms,
    build_transform,
    get_transform_class,
    register_transform,
)

__all__ = [
    "RetinaTransform",
    "TransformValidationError",
    "BlackBorderRemoval",
    "CircularCrop",
    "FundusResize",
    "INTERPOLATIONS",
    "CLAHEEnhancement",
    "IlluminationCorrection",
    "build_normalization",
    "PreprocessingPipeline",
    "build_deterministic_transforms",
    "register_transform",
    "build_transform",
    "get_transform_class",
    "available_transforms",
]
