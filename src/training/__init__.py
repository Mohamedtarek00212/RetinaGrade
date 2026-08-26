"""Training package: config, optimizer/scheduler factories, loop, and utilities."""

from __future__ import annotations

from src.training.config import TrainingConfig, load_training_config
from src.training.trainer import FitResult, Trainer

__all__ = ["TrainingConfig", "load_training_config", "Trainer", "FitResult"]
