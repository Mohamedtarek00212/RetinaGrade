"""Unit tests for :mod:`src.data.preprocessing`.

Verifies the canonical transform order, split-invariance of the deterministic
pipeline (no ``split`` argument exists for the deterministic path), shape/
dtype contracts of individual geometric transforms, and that augmentation is
only ever spliced in for the ``train`` split.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocessing import (
    BlackBorderRemoval,
    CircularCrop,
    FundusResize,
    PreprocessingPipeline,
)
from src.data.statistics import NormalizationStats


def _synthetic_fundus(size: int = 256) -> np.ndarray:
    import cv2

    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 2 - 10, (120, 60, 30), -1)
    return image


class TestGeometricTransforms:
    def test_black_border_removal_reduces_or_preserves_area(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        image[50:150, 50:250] = 100
        transform = BlackBorderRemoval(threshold=10, blur_kernel=5, min_area_ratio=0.05, padding=0)
        result = transform(image=image)["image"]
        assert result.shape[0] * result.shape[1] <= image.shape[0] * image.shape[1]

    def test_black_border_removal_output_dtype(self):
        image = _synthetic_fundus()
        transform = BlackBorderRemoval(threshold=10, blur_kernel=5, min_area_ratio=0.05, padding=0)
        result = transform(image=image)["image"]
        assert result.dtype == np.uint8

    def test_circular_crop_masks_corners_to_fill_value(self):
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        transform = CircularCrop(margin_ratio=0.0, fill_value=0)
        result = transform(image=image)["image"]
        assert result[0, 0].tolist() == [0, 0, 0]

    def test_fundus_resize_produces_exact_square(self):
        image = _synthetic_fundus(400)
        transform = FundusResize(size=224)
        result = transform(image=image)["image"]
        assert result.shape[:2] == (224, 224)

    def test_fundus_resize_upscales_small_images(self):
        image = _synthetic_fundus(50)
        transform = FundusResize(size=224)
        result = transform(image=image)["image"]
        assert result.shape[:2] == (224, 224)


class TestPreprocessingPipelineOrder:
    def test_deterministic_transform_names_follow_canonical_order(self, data_config):
        pipeline = PreprocessingPipeline(data_config)
        names = [step["name"] for step in pipeline.describe()]
        canonical_priority = {
            "black_border_removal": 0,
            "circular_crop": 1,
            "resize": 2,
            "clahe": 3,
            "illumination_correction": 4,
        }
        priorities = [canonical_priority[name] for name in names if name in canonical_priority]
        assert priorities == sorted(priorities)

    def test_call_applies_only_deterministic_steps(self, data_config):
        pipeline = PreprocessingPipeline(data_config)
        image = _synthetic_fundus(300)
        result = pipeline(image)
        assert result.shape[:2] == (data_config.preprocessing.image_size,) * 2

    def test_deterministic_output_is_identical_across_splits(self, data_config):
        """The deterministic path takes no split argument -- verify it truly
        produces bit-identical output regardless of intended split."""
        pipeline = PreprocessingPipeline(data_config)
        image = _synthetic_fundus(300)
        first = pipeline(image.copy())
        second = pipeline(image.copy())
        np.testing.assert_array_equal(first, second)

    def test_build_ignores_augmentation_for_non_train_splits(self, data_config):
        import albumentations as A

        pipeline = PreprocessingPipeline(data_config)
        stats = NormalizationStats(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), source="config")
        fake_augmentation = [A.HorizontalFlip(p=1.0)]

        val_compose = pipeline.build(split="val", stats=stats, augmentation=fake_augmentation)
        train_compose = pipeline.build(split="train", stats=stats, augmentation=fake_augmentation)

        val_names = [type(t).__name__ for t in val_compose.transforms]
        train_names = [type(t).__name__ for t in train_compose.transforms]
        assert "HorizontalFlip" not in val_names
        assert "HorizontalFlip" in train_names

    def test_build_raises_for_unknown_split(self, data_config):
        pipeline = PreprocessingPipeline(data_config)
        stats = NormalizationStats(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), source="config")
        with pytest.raises(ValueError):
            pipeline.build(split="bogus", stats=stats)

    def test_cache_key_matches_preprocessing_hash(self, data_config):
        pipeline = PreprocessingPipeline(data_config)
        assert pipeline.cache_key == data_config.preprocessing_hash
