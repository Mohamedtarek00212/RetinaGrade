"""Unit tests for :mod:`src.utils.config`.

Focuses on the guarantees the rest of the pipeline relies on: fail-fast
validation with an actionable message, deterministic hashing, dotted-override
parsing, and the immutability of the safety-critical flags (``delete: false``,
``report_only: true``).
"""

from __future__ import annotations

import pytest

from src.utils.config import (
    ConfigError,
    config_hash,
    deep_merge,
    load_data_config,
    parse_overrides,
)


class TestLoadDataConfig:
    def test_loads_default_config(self):
        config = load_data_config()
        assert config.dataset_name == "aptos2019"
        assert config.seed > 0

    def test_overrides_are_applied(self):
        config = load_data_config(overrides={"seed": 123})
        assert config.seed == 123

    def test_dotted_override_via_parse_overrides(self):
        overrides = parse_overrides(["seed=777", "preprocessing.image_size=256"])
        config = load_data_config(overrides=overrides)
        assert config.seed == 777
        assert config.preprocessing.image_size == 256

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ConfigError):
            load_data_config(overrides={"totally_unknown_section": {"x": 1}})

    def test_invalid_image_size_raises(self):
        with pytest.raises(ConfigError):
            load_data_config(overrides={"preprocessing": {"image_size": -1}})


class TestImmutableSafetyFlags:
    def test_quality_flags_delete_must_be_false(self):
        with pytest.raises(ConfigError):
            load_data_config(overrides={"cleaning": {"rules": {"quality_flags": {"delete": True}}}})

    def test_near_duplicates_report_only_must_be_true(self):
        with pytest.raises(ConfigError):
            load_data_config(
                overrides={"cleaning": {"rules": {"near_duplicates": {"report_only": False}}}}
            )

    def test_blur_percentile_out_of_range_raises(self):
        with pytest.raises(ConfigError):
            load_data_config(overrides={"cleaning": {"rules": {"quality_flags": {"blur_percentile": 150}}}})


class TestProfiles:
    def test_paper_faithful_disables_eda_only_augmentations(self):
        config = load_data_config(overrides={"profile": "paper_faithful"})
        train = config.augmentation.train
        assert train.scale_jitter.enabled is False
        assert train.conservative_crop.enabled is False
        assert train.gaussian_blur.enabled is False
        assert train.gaussian_noise.enabled is False
        assert train.gamma.enabled is False

    def test_paper_faithful_keeps_paper_augmentations(self):
        config = load_data_config(overrides={"profile": "paper_faithful"})
        train = config.augmentation.train
        assert train.horizontal_flip.enabled is True
        assert train.vertical_flip.enabled is True
        assert train.rotation.enabled is True
        assert train.color_jitter.enabled is True

    def test_eda_driven_is_the_default_profile(self):
        config = load_data_config()
        assert config.profile == "eda_driven"


class TestHashing:
    def test_config_hash_deterministic(self):
        payload = {"a": 1, "b": [1, 2, 3]}
        assert config_hash(payload) == config_hash(payload)

    def test_config_hash_ignores_key_order(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_config_hash_sensitive_to_value(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_preprocessing_hash_ignores_unrelated_sections(self):
        base = load_data_config()
        changed = load_data_config(overrides={"dataloader": {"batch_size": 999}})
        assert base.preprocessing_hash == changed.preprocessing_hash

    def test_preprocessing_hash_sensitive_to_image_size(self):
        base = load_data_config()
        changed = load_data_config(overrides={"preprocessing": {"image_size": 384}})
        assert base.preprocessing_hash != changed.preprocessing_hash


class TestDeepMerge:
    def test_nested_dicts_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3}}
        merged = deep_merge(base, override)
        assert merged == {"a": {"x": 1, "y": 3}}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}

    def test_lists_are_replaced_not_concatenated(self):
        base = {"a": [1, 2, 3]}
        override = {"a": [4]}
        assert deep_merge(base, override) == {"a": [4]}


class TestParseOverrides:
    def test_parses_scalars(self):
        parsed = parse_overrides(["a=1", "b=true", "c=0.5", "d=hello"])
        assert parsed == {"a": 1, "b": True, "c": 0.5, "d": "hello"}

    def test_parses_nested_dotted_path(self):
        parsed = parse_overrides(["a.b.c=1"])
        assert parsed == {"a": {"b": {"c": 1}}}

    def test_empty_list_returns_empty_dict(self):
        assert parse_overrides([]) == {}
        assert parse_overrides(None) == {}

    def test_malformed_override_raises(self):
        with pytest.raises(ConfigError):
            parse_overrides(["no_equals_sign"])
