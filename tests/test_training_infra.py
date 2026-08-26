"""Tests for checkpointing, CSV/TensorBoard logging, early stopping, and manifests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.training.callbacks import EarlyStopping
from src.training.checkpoint import CheckpointManager
from src.training.config import CheckpointConfig, EarlyStoppingConfig, LoggingConfig
from src.training.csv_logger import CSVEpochLogger
from src.training.manifest import build_run_manifest
from src.training.tensorboard_logger import TensorBoardLogger


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_checkpoint_manager_tracks_best_and_last(tmp_path: Path) -> None:
    config = CheckpointConfig(dir=str(tmp_path / "ckpt"), monitor_metric="val_qwk", monitor_mode="max")
    manager = CheckpointManager(config, project_root=tmp_path)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    written_0 = manager.save(0, model, optimizer, scheduler=None, metrics={"val_qwk": 0.5})
    assert "best" in written_0
    assert manager.best_value == pytest.approx(0.5)

    written_1 = manager.save(1, model, optimizer, scheduler=None, metrics={"val_qwk": 0.3})
    assert "best" not in written_1
    assert manager.best_value == pytest.approx(0.5)

    written_2 = manager.save(2, model, optimizer, scheduler=None, metrics={"val_qwk": 0.9})
    assert "best" in written_2
    assert manager.best_value == pytest.approx(0.9)

    payload = manager.load(manager.best_path)
    assert payload["epoch"] == 2
    assert payload["metrics"]["val_qwk"] == pytest.approx(0.9)


def test_checkpoint_manager_min_mode(tmp_path: Path) -> None:
    config = CheckpointConfig(dir=str(tmp_path / "ckpt"), monitor_metric="val_loss", monitor_mode="min")
    manager = CheckpointManager(config, project_root=tmp_path)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    manager.save(0, model, optimizer, scheduler=None, metrics={"val_loss": 1.0})
    written = manager.save(1, model, optimizer, scheduler=None, metrics={"val_loss": 0.5})
    assert "best" in written
    assert manager.best_value == pytest.approx(0.5)


def test_checkpoint_manager_resumes_best_state(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    config = CheckpointConfig(dir=str(ckpt_dir), monitor_metric="val_qwk", monitor_mode="max")
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    manager_a = CheckpointManager(config, project_root=tmp_path)
    manager_a.save(0, model, optimizer, scheduler=None, metrics={"val_qwk": 0.7})

    manager_b = CheckpointManager(config, project_root=tmp_path)
    assert manager_b.best_value == pytest.approx(0.7)


def test_csv_epoch_logger_writes_rows(tmp_path: Path) -> None:
    logger = CSVEpochLogger(log_dir="logs", filename="epochs.csv", project_root=tmp_path)
    logger.log(0, {"train_loss": 1.0, "val_qwk": 0.5})
    logger.log(1, {"train_loss": 0.8, "val_qwk": 0.6})

    assert logger.path.exists()
    content = logger.path.read_text()
    assert "train_loss" in content
    assert "0.5" in content and "0.6" in content
    assert len(logger.rows) == 2


def test_tensorboard_logger_noop_when_disabled(tmp_path: Path) -> None:
    config = LoggingConfig(tensorboard_enabled=False)
    tb_logger = TensorBoardLogger(config, project_root=tmp_path)
    assert not tb_logger.enabled
    tb_logger.log_scalars(0, {"loss": 1.0})  # must not raise
    tb_logger.close()


def test_early_stopping_disabled_never_stops() -> None:
    config = EarlyStoppingConfig(enabled=False, patience=1)
    stopper = EarlyStopping(config)
    assert stopper.step({"val_qwk": 0.1}) is False
    assert stopper.step({"val_qwk": 0.05}) is False


def test_early_stopping_triggers_after_patience() -> None:
    config = EarlyStoppingConfig(enabled=True, patience=2, monitor_metric="val_qwk", mode="max")
    stopper = EarlyStopping(config)
    assert stopper.step({"val_qwk": 0.5}) is False  # improvement (first value)
    assert stopper.step({"val_qwk": 0.4}) is False  # bad epoch 1
    assert stopper.step({"val_qwk": 0.3}) is True  # bad epoch 2 -> stop


def test_early_stopping_resets_on_improvement() -> None:
    config = EarlyStoppingConfig(enabled=True, patience=2, monitor_metric="val_qwk", mode="max")
    stopper = EarlyStopping(config)
    stopper.step({"val_qwk": 0.5})
    stopper.step({"val_qwk": 0.4})  # bad epoch 1
    assert stopper.step({"val_qwk": 0.6}) is False  # improvement resets counter
    assert stopper.step({"val_qwk": 0.55}) is False  # bad epoch 1 again, not yet stopped


def test_build_run_manifest_contains_hashes(data_config, non_paper_model_config) -> None:
    from src.training.config import load_training_config

    fixtures_dir = Path(__file__).parent / "fixtures"
    training_config = load_training_config(path=fixtures_dir / "non_paper_training_config.yaml")

    manifest = build_run_manifest(data_config, non_paper_model_config, training_config)
    assert manifest["hashes"]["data"] == data_config.config_hash
    assert manifest["hashes"]["model"] == non_paper_model_config.model_config_hash
    assert manifest["hashes"]["training"] == training_config.training_config_hash
    assert manifest["seed"] == training_config.reproducibility.seed
