"""Tests for `src.models.config`: the model-architecture configuration engine."""

from __future__ import annotations

import pytest

from src.models.config import ConfigError, ModelConfig, load_model_config


def test_real_config_loads_successfully() -> None:
    """`configs/model.yaml` now ships with runnable default values."""
    config = load_model_config("configs/model.yaml")
    assert config.model_name == "dual_swinord"
    assert config.backbone.name == "timm_swin"
    assert config.backbone.image_size == 512


def test_missing_backbone_section_raises() -> None:
    """Validation still rejects a model config without a backbone section."""
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "missing_backbone.yaml"
        path.write_text("model_name: 'dual_swinord'\n")
        with pytest.raises(ConfigError, match="missing required key"):
            load_model_config(str(path))


def test_non_paper_fixture_loads_into_a_valid_model_config(non_paper_model_config: ModelConfig) -> None:
    config = non_paper_model_config
    assert config.model_name == "dual_swinord"
    assert config.backbone.name == "timm_swin"
    assert config.backbone.variant == "swin_tiny_patch4_window7_224"
    assert config.spm.inject_at_stage == 0
    assert config.plka.input_stage == 0
    assert config.plka.fusion.name == "test_plka_fusion"
    assert config.neck.hidden_dim == 32
    assert config.heads.ordinal.name == "test_ordinal_head"


def test_model_config_hash_is_stable(non_paper_model_config: ModelConfig) -> None:
    assert non_paper_model_config.model_config_hash == non_paper_model_config.model_config_hash
    assert len(non_paper_model_config.model_config_hash) == 12


def test_unknown_key_is_rejected(tmp_path) -> None:
    bad_config = tmp_path / "bad_model.yaml"
    bad_config.write_text("model_name: 'dual_swinord'\nnot_a_real_section: {}\n")
    try:
        load_model_config(bad_config)
    except ConfigError as exc:
        assert "not_a_real_section" in str(exc)
    else:
        raise AssertionError("expected a ConfigError for an unknown top-level key")
