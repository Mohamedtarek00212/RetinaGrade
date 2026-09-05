"""Conservative image-quality checks for deployment uploads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class ImageQualityReport:
    width: int
    height: int
    brightness: float
    sharpness: float
    visible_area_ratio: float
    retinal_color_score: float
    acceptable: bool
    warnings: list[str]


def assess_image_quality(image_path: str | Path) -> ImageQualityReport:
    """Measure basic usability without claiming clinical image validation."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {image_path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    visible_mask = gray > 10
    visible_area_ratio = float(visible_mask.mean())
    visible_pixels = gray[visible_mask]
    brightness = float(visible_pixels.mean()) if visible_pixels.size else 0.0
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if visible_pixels.size:
        blue = float(image[:, :, 0][visible_mask].mean())
        green = float(image[:, :, 1][visible_mask].mean())
        red = float(image[:, :, 2][visible_mask].mean())
        retinal_color_score = red / max(green, blue, 1.0)
    else:
        retinal_color_score = 0.0

    blocking_issues: list[str] = []
    warnings: list[str] = []
    if min(width, height) < 224:
        blocking_issues.append("Image resolution is below the 224 px minimum")
    if visible_area_ratio < 0.15:
        blocking_issues.append("Too little visible image area was detected")
    if brightness < 20:
        blocking_issues.append("Image is too dark for reliable processing")
    elif brightness > 235:
        blocking_issues.append("Image is too bright for reliable processing")
    if sharpness < 2.0:
        warnings.append("Image may be blurred; interpret the result with extra caution")
    if retinal_color_score < 1.05:
        warnings.append("Color profile is atypical for a retinal fundus photograph")

    warnings = [*blocking_issues, *warnings]
    return ImageQualityReport(
        width=width,
        height=height,
        brightness=brightness,
        sharpness=sharpness,
        visible_area_ratio=visible_area_ratio,
        retinal_color_score=retinal_color_score,
        acceptable=not blocking_issues,
        warnings=warnings,
    )
