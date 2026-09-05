"""Tests for conservative deployment image-quality checks."""

from pathlib import Path

import cv2
import numpy as np

from deployment.quality import assess_image_quality


def _write_image(path: Path, image: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), image)
    return path


def test_quality_accepts_fundus_like_image(tmp_path: Path) -> None:
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(image, (256, 256), 220, (20, 80, 160), -1)

    report = assess_image_quality(_write_image(tmp_path / "fundus.png", image))

    assert report.acceptable is True
    assert report.width == 512
    assert report.retinal_color_score > 1.05


def test_quality_rejects_dark_image(tmp_path: Path) -> None:
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    report = assess_image_quality(_write_image(tmp_path / "dark.png", image))

    assert report.acceptable is False
    assert any("visible image area" in warning for warning in report.warnings)
