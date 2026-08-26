"""Geometric preprocessing: black-border removal, circular crop, and resize.

Medical and empirical rationale
-------------------------------
**Black-border removal (essential, EDA-only).** A meaningful fraction of APTOS
images are wide-aspect (1.33-1.51) and retain black padding around the circular
fundus disc -- up to a third of the frame. That padding carries no diagnostic
information, wastes model capacity, and distorts the effective receptive field
of a patch-based backbone. The paper describes no such step.

**Circular crop (essential, EDA-only).** Normalising the disc into a canonical
circular framing removes the residual rectangular corners. This matters
specifically because rotation is part of the augmentation policy: rotating an
uncropped frame drags structured black wedges across the retina, teaching the
network an artefact. Cropping first makes rotation geometrically benign.

**Fixed resize (essential).** A fixed input size is required by the backbone,
and it simultaneously neutralises the resolution-diagnosis confound the EDA
measured (width r = 0.57, height r = 0.52 with grade) across a corpus spanning
474x358 to 4288x2848 -- roughly a 70x range in pixel count. ``INTER_AREA`` is
used for downscaling because naive bilinear sampling of a 4K image down to 512
aliases away microaneurysms, the few-pixel lesions that define Grade 1.
"""

from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np

from src.data.preprocessing.base import RetinaTransform
from src.utils.helpers import tissue_bounding_box
from src.utils.logger import get_logger

__all__ = ["BlackBorderRemoval", "CircularCrop", "FundusResize", "INTERPOLATIONS"]

logger = get_logger(__name__)

#: Mapping from configuration strings to OpenCV interpolation flags.
INTERPOLATIONS: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


class BlackBorderRemoval(RetinaTransform):
    """Crop the image to the bounding box of the fundus disc.

    The mask is obtained by thresholding a median-blurred grayscale copy and
    keeping the largest connected component. Median blurring the *mask source*
    (never the output) suppresses hot pixels without eroding the disc edge.

    A safety guard is central to this transform: if the proposed crop retains
    less than ``min_area_ratio`` of the frame, the original image is returned
    unchanged. Roughly one percent of APTOS images are genuinely
    under-exposed, and for those a naive threshold can collapse the bounding
    box to almost nothing. Those images are clinically valid and must survive
    preprocessing intact; a silent near-total crop would destroy them.

    Args:
        threshold: Intensity at or below which a pixel counts as background.
        blur_kernel: Odd kernel size used to smooth the mask source.
        min_area_ratio: Minimum retained area, as a fraction of the original.
        padding: Extra pixels kept around the detected box.
        p: Application probability; keep at ``1.0``.

    Example:
        >>> import numpy as np, cv2
        >>> frame = np.zeros((200, 300, 3), np.uint8)
        >>> _ = cv2.circle(frame, (150, 100), 80, (120, 60, 20), -1)
        >>> BlackBorderRemoval().transform(frame).shape[:2]
        (161, 161)
    """

    evidence: ClassVar[str] = "eda"
    priority: ClassVar[str] = "essential"

    def __init__(
        self,
        threshold: int = 10,
        blur_kernel: int = 5,
        min_area_ratio: float = 0.10,
        padding: int = 0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if not 0 <= threshold <= 255:
            raise ValueError(f"threshold must lie in [0, 255], got {threshold}")
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be a positive odd integer, got {blur_kernel}")
        if not 0.0 <= min_area_ratio <= 1.0:
            raise ValueError(f"min_area_ratio must lie in [0, 1], got {min_area_ratio}")
        if padding < 0:
            raise ValueError(f"padding must be >= 0, got {padding}")

        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.min_area_ratio = min_area_ratio
        self.padding = padding

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the constructor argument names."""
        return ("threshold", "blur_kernel", "min_area_ratio", "padding")

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Crop away the black border, or return the input if that is unsafe."""
        height, width = image.shape[:2]
        box = tissue_bounding_box(image, threshold=self.threshold, blur_kernel=self.blur_kernel)
        if box is None:
            logger.debug("no tissue detected; returning the image uncropped")
            return image

        x, y, box_width, box_height = box
        x0 = max(0, x - self.padding)
        y0 = max(0, y - self.padding)
        x1 = min(width, x + box_width + self.padding)
        y1 = min(height, y + box_height + self.padding)
        if x1 <= x0 or y1 <= y0:
            return image

        retained = ((x1 - x0) * (y1 - y0)) / float(width * height)
        if retained < self.min_area_ratio:
            # Almost certainly an under-exposed acquisition rather than a
            # heavily padded one. Keep the image whole: it has a valid label.
            logger.debug(
                "crop would retain only %.1f%% of the frame (< %.1f%%); keeping the original",
                retained * 100,
                self.min_area_ratio * 100,
            )
            return image

        return image[y0:y1, x0:x1]


class CircularCrop(RetinaTransform):
    """Mask everything outside the inscribed circle of the fundus disc.

    Applied after :class:`BlackBorderRemoval`, when the frame is already tight
    around the disc, so the inscribed circle coincides with the retinal field
    of view. Masking (rather than warping) preserves lesion geometry exactly:
    no resampling, no interpolation, no sub-pixel smearing of microaneurysms.

    Args:
        margin_ratio: Fraction of the radius trimmed from the circle, useful
            for shaving the bright rim some cameras leave at the field edge.
        fill_value: Intensity written outside the circle.
        p: Application probability; keep at ``1.0``.
    """

    evidence: ClassVar[str] = "eda"
    priority: ClassVar[str] = "essential"

    def __init__(self, margin_ratio: float = 0.0, fill_value: int = 0, p: float = 1.0) -> None:
        super().__init__(p=p)
        if not 0.0 <= margin_ratio < 1.0:
            raise ValueError(f"margin_ratio must lie in [0, 1), got {margin_ratio}")
        if not 0 <= fill_value <= 255:
            raise ValueError(f"fill_value must lie in [0, 255], got {fill_value}")
        self.margin_ratio = margin_ratio
        self.fill_value = fill_value

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the constructor argument names."""
        return ("margin_ratio", "fill_value")

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Zero out the corners outside the inscribed circle."""
        height, width = image.shape[:2]
        radius = int(round(min(height, width) / 2 * (1.0 - self.margin_ratio)))
        if radius <= 0:
            return image

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (width // 2, height // 2), radius, 255, thickness=-1)
        output = np.full_like(image, self.fill_value)
        return np.where(mask[..., None] > 0 if image.ndim == 3 else mask > 0, image, output)


class FundusResize(RetinaTransform):
    """Resize to a fixed square input.

    Interpolation is chosen per image: ``INTER_AREA`` when downscaling (correct
    anti-aliasing, essential for preserving few-pixel lesions when reducing a
    4K acquisition) and a configurable kernel when upscaling.

    Args:
        size: Target side length in pixels.
        interpolation_down: Kernel used when the image is larger than ``size``.
        interpolation_up: Kernel used when the image is smaller than ``size``.
        keep_aspect_ratio: When ``True``, the image is resized so its longest
            side matches ``size`` and then zero-padded to a square, preserving
            the disc's true proportions. When ``False`` (the default) the image
            is stretched to a square, matching the reference implementation.
        pad_value: Fill value used when padding.
        p: Application probability; keep at ``1.0``.
    """

    evidence: ClassVar[str] = "both"
    priority: ClassVar[str] = "essential"

    def __init__(
        self,
        size: int = 512,
        interpolation_down: str = "area",
        interpolation_up: str = "linear",
        keep_aspect_ratio: bool = False,
        pad_value: int = 0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        for name, value in (("interpolation_down", interpolation_down), ("interpolation_up", interpolation_up)):
            if value not in INTERPOLATIONS:
                raise ValueError(f"{name}: unknown interpolation {value!r}; expected one of {list(INTERPOLATIONS)}")

        self.size = size
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up
        self.keep_aspect_ratio = keep_aspect_ratio
        self.pad_value = pad_value

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the constructor argument names."""
        return ("size", "interpolation_down", "interpolation_up", "keep_aspect_ratio", "pad_value")

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Resize the image to ``size x size``."""
        height, width = image.shape[:2]
        downscaling = max(height, width) > self.size
        flag = INTERPOLATIONS[self.interpolation_down if downscaling else self.interpolation_up]

        if not self.keep_aspect_ratio:
            return cv2.resize(image, (self.size, self.size), interpolation=flag)

        scale = self.size / float(max(height, width))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (new_width, new_height), interpolation=flag)

        pad_x = self.size - new_width
        pad_y = self.size - new_height
        return cv2.copyMakeBorder(
            resized,
            pad_y // 2,
            pad_y - pad_y // 2,
            pad_x // 2,
            pad_x - pad_x // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[self.pad_value] * (3 if image.ndim == 3 else 1),
        )
