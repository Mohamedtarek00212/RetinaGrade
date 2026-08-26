"""Tests for `src.training.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.config import ConfigError, load_training_config


def test_configs_training_yaml_loads_successfully() -> None:
    """`configs/training.yaml` now fills the previously open gap fields."""
    config = load_training_config()
    assert config.optimizer.name == "adamw"
    assert config.loss.label_smoothing == pytest.approx(0.1)
    assert config.scheduler.eta_min == pytest.approx(0.0)


def test_missing_required_key_raises(tmp_path: Path) -> None:
    """Validation still rejects a training config missing a required field."""
    path = tmp_path / "bad_training.yaml"
    path.write_text("optimizer:\n  name: adamw\n")
    with pytest.raises(ConfigError, match="missing required key"):
        load_training_config(path=str(path))


def test_non_paper_training_config_loads(tmp_path: Path) -> None:
    fixtures_dir = Path(__file__).parent / "fixtures"
    config = load_training_config(path=fixtures_dir / "non_paper_training_config.yaml")

    assert config.epochs == 2
    assert config.optimizer.name == "adamw"
    assert config.optimizer.lr == pytest.approx(1e-4)
    assert config.scheduler.eta_min == pytest.approx(1e-5)
    assert config.scheduler_t_max == config.epochs
    assert config.loss.label_smoothing == pytest.approx(0.1)
    assert config.loss.lambda_cls == pytest.approx(0.5)


def test_scheduler_t_max_overridable(tmp_path: Path) -> None:
    fixtures_dir = Path(__file__).parent / "fixtures"
    config = load_training_config(
        path=fixtures_dir / "non_paper_training_config.yaml",
        overrides={"scheduler": {"t_max": 100}},
    )
    assert config.scheduler_t_max == 100


def test_training_config_hash_is_stable(tmp_path: Path) -> None:
    fixtures_dir = Path(__file__).parent / "fixtures"
    a = load_training_config(path=fixtures_dir / "non_paper_training_config.yaml")
    b = load_training_config(path=fixtures_dir / "non_paper_training_config.yaml")
    assert a.training_config_hash == b.training_config_hash

    c = load_training_config(
        path=fixtures_dir / "non_paper_training_config.yaml",
        overrides={"epochs": 5},
    )
    assert c.training_config_hash != a.training_config_hash


@pytest.mark.parametrize(
    ("bad_field", "bad_value"),
    [
        ("label_smoothing", 1.5),
        ("lambda_cls", -0.1),
    ],
)
def test_loss_config_rejects_out_of_range_values(bad_field: str, bad_value: float) -> None:
    fixtures_dir = Path(__file__).parent / "fixtures"
    with pytest.raises(ConfigError):
        load_training_config(
            path=fixtures_dir / "non_paper_training_config.yaml",
            overrides={"loss": {bad_field: bad_value}},
        )


def test_reproducibility_seed_required() -> None:
    fixtures_dir = Path(__file__).parent / "fixtures"
    raw_path = fixtures_dir / "non_paper_training_config.yaml"
    with pytest.raises(ConfigError):
        # Negative seed must fail __post_init__ validation.
        load_training_config(path=raw_path, overrides={"reproducibility": {"seed": -1}})
