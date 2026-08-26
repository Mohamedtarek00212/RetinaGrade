"""Unit tests for :mod:`src.utils.helpers`.

Covers hashing (exactness, sensitivity, and the Hamming metric), image
quality proxies (their direction, not just their type), and the tissue
bounding box used by black-border removal.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.helpers import (
    black_padding_ratio,
    dhash,
    estimate_noise_sigma,
    hamming_distance,
    hex_to_bits,
    image_brightness,
    image_contrast,
    laplacian_variance,
    md5_bytes,
    md5_file,
    normalized_sharpness,
    phash,
    tissue_bounding_box,
)


class TestMD5:
    def test_deterministic(self, tmp_path):
        path = tmp_path / "a.bin"
        path.write_bytes(b"hello world")
        assert md5_file(path) == md5_file(path)

    def test_content_sensitive(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"hello world")
        b.write_bytes(b"hello world!")
        assert md5_file(a) != md5_file(b)

    def test_bytes_matches_file(self, tmp_path):
        path = tmp_path / "a.bin"
        payload = b"retina grade"
        path.write_bytes(payload)
        assert md5_file(path) == md5_bytes(payload)


class TestPerceptualHashes:
    def _solid(self, value: int) -> np.ndarray:
        return np.full((64, 64, 3), value, dtype=np.uint8)

    def _disc(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        image = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        return image

    def test_dhash_identical_for_identical_images(self):
        image = self._disc(1)
        assert dhash(image) == dhash(image.copy())

    def test_phash_identical_for_identical_images(self):
        image = self._disc(1)
        assert phash(image) == phash(image.copy())

    def test_dhash_differs_for_different_images(self):
        assert dhash(self._disc(1)) != dhash(self._disc(2))

    def test_hex_to_bits_round_trip_length(self):
        digest = dhash(self._disc(1))
        bits = hex_to_bits(digest)
        # 8x8 dHash gradient grid -> 64 bits, padded to a byte boundary.
        assert bits.size >= 64

    def test_hamming_distance_zero_for_identical(self):
        digest = dhash(self._disc(3))
        assert hamming_distance(digest, digest) == 0

    def test_hamming_distance_symmetric(self):
        a, b = dhash(self._disc(1)), dhash(self._disc(2))
        assert hamming_distance(a, b) == hamming_distance(b, a)

    def test_dhash_robust_to_mild_brightness_shift(self):
        base = self._disc(5).astype(np.int16)
        shifted = np.clip(base + 5, 0, 255).astype(np.uint8)
        distance = hamming_distance(dhash(base.astype(np.uint8)), dhash(shifted))
        # Not required to be zero, but should be small relative to hash length.
        assert distance <= 10


class TestQualityMetrics:
    def test_brightness_orders_correctly(self):
        dark = np.full((32, 32, 3), 20, dtype=np.uint8)
        bright = np.full((32, 32, 3), 200, dtype=np.uint8)
        assert image_brightness(dark) < image_brightness(bright)

    def test_contrast_zero_for_uniform_image(self):
        flat = np.full((32, 32, 3), 100, dtype=np.uint8)
        assert image_contrast(flat) == pytest.approx(0.0, abs=1e-6)

    def test_contrast_positive_for_textured_image(self):
        rng = np.random.default_rng(0)
        textured = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        assert image_contrast(textured) > 0

    def test_sharpness_higher_for_high_frequency_content(self):
        flat = np.full((256, 256, 3), 128, dtype=np.uint8)
        checkerboard = np.indices((256, 256)).sum(axis=0) % 2 * 255
        sharp = np.repeat(checkerboard[..., None], 3, axis=2).astype(np.uint8)
        assert normalized_sharpness(sharp) > normalized_sharpness(flat)

    def test_laplacian_variance_zero_for_flat_image(self):
        flat = np.full((32, 32, 3), 50, dtype=np.uint8)
        assert laplacian_variance(flat) == pytest.approx(0.0, abs=1e-6)

    def test_noise_sigma_higher_for_noisy_image(self):
        rng = np.random.default_rng(0)
        clean = np.full((128, 128, 3), 100, dtype=np.uint8)
        noisy = np.clip(100 + rng.normal(0, 15, (128, 128, 3)), 0, 255).astype(np.uint8)
        assert estimate_noise_sigma(noisy) > estimate_noise_sigma(clean)


class TestTissueBoundingBox:
    def test_detects_centered_disc(self):
        import cv2

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.circle(image, (150, 100), 80, (120, 60, 20), -1)
        box = tissue_bounding_box(image)
        assert box is not None
        x, y, width, height = box
        assert 60 <= x <= 80
        assert 10 <= y <= 30
        assert width > 140
        assert height > 140

    def test_returns_none_for_all_black_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert tissue_bounding_box(image) is None

    def test_black_padding_ratio_bounds(self):
        import cv2

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        radius = 90
        cv2.circle(image, (100, 100), radius, (100, 100, 100), -1)
        ratio = black_padding_ratio(image)
        assert 0.0 <= ratio <= 1.0
        expected_foreground = np.pi * radius**2 / (200 * 200)
        assert ratio == pytest.approx(1 - expected_foreground, abs=0.02)
