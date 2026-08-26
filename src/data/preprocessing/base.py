"""Base class shared by every deterministic retinal preprocessing transform.

Each transform is an Albumentations :class:`~albumentations.ImageOnlyTransform`
so that preprocessing and augmentation compose in a single pipeline, and so
that a pipeline can be serialised for a run manifest.

Two project-specific additions are made on top of the Albumentations contract:

``evidence``
    ``"paper"``, ``"eda"``, or ``"both"`` -- where the justification for this
    step comes from. The reference paper describes only resize, normalization,
    flips, rotation, and colour jitter for APTOS; everything else in this
    package is EDA-derived. Tagging makes that distinction machine-readable, so
    the ``paper_faithful`` profile can be verified by a unit test instead of a
    promise in a README.

``priority``
    ``"essential"``, ``"recommended"``, or ``"optional"``, mirroring the verdicts
    in the Data Preparation report.

Input validation is centralised here: every transform receives an ``H x W`` or
``H x W x 3`` ``uint8`` array, so no individual transform has to re-check.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import albumentations as A
import numpy as np

from src.utils.logger import get_logger

__all__ = ["RetinaTransform", "TransformValidationError"]

logger = get_logger(__name__)


class TransformValidationError(ValueError):
    """Raised when a transform receives an image it cannot process."""


class RetinaTransform(A.ImageOnlyTransform):
    """Abstract base for deterministic, image-only retinal transforms.

    Subclasses implement :meth:`transform` and declare :attr:`evidence` and
    :attr:`priority`. The default probability is ``1.0``: these steps are part
    of preprocessing, not augmentation, and must be applied identically to
    train, validation, and test data or they introduce their own distribution
    shift.

    Args:
        p: Probability of application. Leave at ``1.0`` for preprocessing.
    """

    #: Source of evidence for this step.
    evidence: ClassVar[str] = "eda"

    #: Priority as assessed in the Data Preparation report.
    priority: ClassVar[str] = "optional"

    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p=p)

    # -- Albumentations hooks ---------------------------------------------

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Validate the input and delegate to :meth:`transform`.

        Args:
            img: Input image supplied by Albumentations.
            **params: Unused Albumentations parameters.

        Returns:
            The transformed image.
        """
        self.validate(img)
        return self.transform(img)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the constructor argument names used for serialisation."""
        return ()

    # -- project API -------------------------------------------------------

    @abstractmethod
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply the transform to a validated image.

        Args:
            image: ``H x W`` or ``H x W x 3`` ``uint8`` array.

        Returns:
            The transformed image.
        """

    @staticmethod
    def validate(image: np.ndarray) -> None:
        """Validate the shape and dtype of an input image.

        Args:
            image: Candidate input.

        Raises:
            TransformValidationError: If the array is not a 2D or 3-channel
                ``uint8`` image with a non-zero extent.
        """
        if not isinstance(image, np.ndarray):
            raise TransformValidationError(f"expected a numpy array, got {type(image).__name__}")
        if image.ndim not in (2, 3):
            raise TransformValidationError(f"expected a 2D or 3D array, got {image.ndim} dimensions")
        if image.ndim == 3 and image.shape[2] != 3:
            raise TransformValidationError(
                f"expected 3 channels (RGB), got {image.shape[2]}; "
                "images must be converted to RGB before preprocessing"
            )
        if image.dtype != np.uint8:
            raise TransformValidationError(
                f"expected a uint8 image, got {image.dtype}; normalization must run last"
            )
        if min(image.shape[:2]) == 0:
            raise TransformValidationError("image has a zero-sized dimension")

    def describe(self) -> dict[str, Any]:
        """Return a serialisable description used in run manifests.

        Returns:
            Name, evidence tier, priority, probability, and constructor
            arguments, so any figure can be traced back to the exact pipeline
            that produced it.
        """
        return {
            "name": type(self).__name__,
            "evidence": self.evidence,
            "priority": self.priority,
            "p": float(self.p),
            "args": {name: getattr(self, name, None) for name in self.get_transform_init_args_names()},
        }
