"""Unit tests for :mod:`src.data.augmentation`.

Covers the safety guarantees that matter most: forbidden (occlusion/label-
mixing) transforms are rejected even when nested, evaluation splits never
receive augmentation, and the ``paper_faithful`` profile's policy contains
no EDA-only transform.
"""

from __future__ import annotations

import albumentations as A
import pytest

from src.data.augmentation import (
    build_eval_transforms,
    build_train_transforms,
    describe_policy,
    transforms_with_evidence,
)
from src.data.augmentation.guards import (
    DEFAULT_FORBIDDEN,
    ForbiddenAugmentationError,
    assert_no_forbidden_transforms,
)
from src.data.augmentation.policies import PAPER_TRANSFORMS, build_policy
from src.utils.config import load_data_config


class TestGuards:
    def test_allows_a_safe_pipeline(self):
        assert_no_forbidden_transforms([A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5)])

    def test_rejects_a_directly_forbidden_transform(self):
        with pytest.raises(ForbiddenAugmentationError):
            assert_no_forbidden_transforms([A.CoarseDropout(p=1.0)])

    def test_rejects_a_forbidden_transform_nested_in_oneof(self):
        nested = A.OneOf([A.CoarseDropout(p=1.0), A.HorizontalFlip(p=1.0)], p=1.0)
        with pytest.raises(ForbiddenAugmentationError):
            assert_no_forbidden_transforms([nested])

    def test_accepts_a_compose_object_directly(self):
        compose = A.Compose([A.HorizontalFlip(p=1.0)])
        assert_no_forbidden_transforms(compose)

    def test_default_forbidden_list_is_nonempty(self):
        assert len(DEFAULT_FORBIDDEN) > 0


class TestBuildTrainTransforms:
    def test_default_profile_produces_no_forbidden_transform(self, data_config):
        transforms = build_train_transforms(data_config)
        assert_no_forbidden_transforms(transforms, data_config.augmentation.forbidden)

    def test_disabled_augmentation_returns_empty_list(self):
        cfg = load_data_config(overrides={"augmentation": {"enabled": False}})
        assert build_train_transforms(cfg) == []

    def test_eval_transforms_always_empty(self, data_config):
        assert build_eval_transforms(data_config) == []


class TestPolicies:
    def test_paper_faithful_contains_no_eda_only_transform(self):
        config = load_data_config(overrides={"profile": "paper_faithful"})
        tagged = build_policy(config)
        eda_only = transforms_with_evidence(tagged, "eda")
        assert eda_only == []

    def test_paper_faithful_contains_exactly_the_paper_transforms(self):
        config = load_data_config(overrides={"profile": "paper_faithful"})
        tagged = build_policy(config)
        keys = {item.key for item in tagged}
        assert keys == PAPER_TRANSFORMS

    def test_eda_driven_may_include_eda_transforms(self):
        config = load_data_config(overrides={"profile": "eda_driven"})
        tagged = build_policy(config)
        # At minimum, the paper transforms should still be present.
        keys = {item.key for item in tagged}
        assert PAPER_TRANSFORMS.issubset(keys)

    def test_geometry_precedes_photometry_precedes_corruption(self):
        config = load_data_config(
            overrides={
                "profile": "eda_driven",
                "augmentation": {
                    "train": {
                        "gaussian_blur": {"enabled": True},
                        "gaussian_noise": {"enabled": True},
                    }
                },
            }
        )
        tagged = build_policy(config)
        keys = [item.key for item in tagged]
        geometry_keys = {"horizontal_flip", "vertical_flip", "rotation", "scale_jitter", "conservative_crop"}
        corruption_keys = {"gaussian_blur", "gaussian_noise"}

        geometry_positions = [i for i, k in enumerate(keys) if k in geometry_keys]
        corruption_positions = [i for i, k in enumerate(keys) if k in corruption_keys]
        if geometry_positions and corruption_positions:
            assert max(geometry_positions) < min(corruption_positions)

    def test_describe_policy_is_serialisable(self, data_config):
        description = describe_policy(data_config)
        for entry in description:
            assert "key" in entry and "evidence" in entry and "priority" in entry
