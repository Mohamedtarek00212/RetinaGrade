"""Optional intensity preprocessing: CLAHE and illumination correction.

Both steps in this module are **EDA-driven, not paper-driven**, and both are
**disabled by default**.

CLAHE
-----
The reference paper describes no contrast enhancement of any kind for APTOS --
no CLAHE, no Ben Graham preprocessing, no illumination correction. The EDA,
however, measured a wide grayscale-contrast range (10.9-70.7) and a
statistically significant non-linear association between brightness tercile and
DR grade (chi-square = 31.688, p = 0.0001). That association is a concrete
shortcut-learning risk: a network can reach a respectable score by reading
exposure instead of pathology.

CLAHE addresses it locally, which is what matters clinically -- microaneurysms
and small haemorrhages are defined by *local* intensity differences against
surrounding tissue, not by global exposure. Enhancement is applied to the
lightness channel only (LAB) so that chrominance is untouched: the redness of a
haemorrhage and the paleness of a hard exudate are diagnostic colour cues, and
distorting them would trade one problem for a worse one. A ``green``-channel
mode is offered as the classic fundus alternative, since the green channel
carries the highest vessel/lesion contrast.

Because CLAHE is unvalidated on this dataset by the reference work, it is
treated as an ablation candidate: enable it, measure quadratic-weighted kappa
against the disabled baseline, and report both.

Illumination correction
-----------------------
Local-average subtraction (the "Ben Graham" style transform) targets the same
illumination variability more aggressively. The Data Preparation report rates
it *optional -- worth an ablation, not to be assumed necessary*, since CLAHE
alone may already remove most of the confound. It is implemented so the
ablation costs one configuration line, and disabled so it never runs by
accident.
"""

from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np

from src.data.preprocessing.base import RetinaTransform
from src.utils.logger import get_logger

__all__ = ["CLAHEEnhancement", "IlluminationCorrection"]

logger = get_logger(__name__)


class CLAHEEnhancement(RetinaTransform):
    """Contrast Limited Adaptive Histogram Equalisation for fundus images.

    Args:
        clip_limit: Contrast-limiting threshold. Values above ~4 amplify noise
            into lesion-sized speckle, which is exactly the failure mode this
            dataset cannot afford.
        tile_grid_size: Number of tiles along each axis.
        color_space: ``"lab"`` equalises the L channel (chrominance preserved),
            ``"green"`` equalises the green channel only, ``"hsv"`` equalises V.
        p: Application probability; keep at ``1.0`` for preprocessing.

    Raises:
        ValueError: If the colour space is unknown or a parameter is invalid.

    Example:
        >>> import numpy as np
        >>> image = np.full((32, 32, 3), 40, dtype=np.uint8)
        >>> CLAHEEnhancement().transform(image).shape
        (32, 32, 3)
    """

    evidence: ClassVar[str] = "eda"
    priority: ClassVar[str] = "recommended"

    #: Supported colour spaces.
    COLOR_SPACES: ClassVar[tuple[str, ...]] = ("lab", "green", "hsv")

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        color_space: str = "lab",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {clip_limit}")
        if len(tile_grid_size) != 2 or any(t <= 0 for t in tile_grid_size):
            raise ValueError(f"tile_grid_size must be two positive integers, got {tile_grid_size}")
        if color_space not in self.COLOR_SPACES:
            raise ValueError(
                f"unknown color_space {color_space!r}; expected one of {list(self.COLOR_SPACES)}"
            )

        self.clip_limit = float(clip_limit)
        self.tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))
        self.color_space = color_space

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the constructor argument names."""
        return ("clip_limit", "tile_grid_size", "color_space")

    def _clahe(self) -> cv2.CLAHE:
        """Build the OpenCV CLAHE operator."""
        return cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE in the configured colour space."""
        operator = self._clahe()

        if image.ndim == 2:
            return operator.apply(image)

        if self.color_space == "lab":
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[..., 0] = operator.apply(lab[..., 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if self.color_space == "hsv":
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[..., 2] = operator.apply(hsv[..., 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Green channel only: the highest-contrast band for retinal vasculature
        # and lesions; red saturates and blue is heavily attenuated in fundus
        # photography (EDA colour analysis: R 108.3, G 57.1, B 18.3).
        output = image.copy()
        output[..., 1] = operator.apply(output[..., 1])
        return output


class IlluminationCorrection(RetinaTransform):
    """Subtract a local average to flatten uneven illumination.

    Computes ``weight * image - weight * blur(image) + 128``, the widely used
    fundus normalisation popularised in the 2015 Diabetic Retinopathy Kaggle
    competition. It strongly suppresses low-frequency exposure gradients while
    preserving high-frequency lesion structure.

    Disabled by default: the Data Preparation report rates it optional, and it
    is aggressive enough that it must be validated against a CLAHE-only
    baseline rather than stacked on top blindly.

    Args:
        sigma: Gaussian sigma of the local-average estimate, in pixels at the
            working resolution.
        weight: Amplification factor applied to the residual.
        p: Application probability; keep at ``1.0`` for preprocessing.
    """

    evidence: ClassVar[str] = "eda"
    priority: ClassVar[str] = "optional"

    def __init__(self, sigma: float = 10.0, weight: float = 4.0, p: float = 1.0) -> None:
        super().__init__(p=p)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = float(sigma)
        self.weight = float(weight)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the constructor argument names."""
        return ("sigma", "weight")

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply local-average subtraction."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        corrected = cv2.addWeighted(image, self.weight, blurred, -self.weight, 128)
        return np.clip(corrected, 0, 255).astype(np.uint8)
