"""Run-manifest provenance.

Combines the three independently-hashed configuration layers -- data
(:attr:`~src.utils.config.DataConfig.config_hash`), model architecture
(:attr:`~src.models.config.ModelConfig.model_config_hash`), and training
(:attr:`~src.training.config.TrainingConfig.training_config_hash`) -- into
one experiment-identifying manifest, written once at the start of a run.
Reuses :func:`src.utils.helpers.write_json` rather than reimplementing
serialisation.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.models.config import ModelConfig
from src.training.config import TrainingConfig
from src.utils.config import DataConfig
from src.utils.helpers import write_json

__all__ = ["build_run_manifest", "write_run_manifest"]


def build_run_manifest(
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance dict for one training run.

    Args:
        data_config: Validated data-preparation configuration.
        model_config: Validated model-architecture configuration.
        training_config: Validated training configuration.
        extra: Additional caller-supplied provenance (for example git commit
            hash or CLI arguments).

    Returns:
        A JSON-serialisable manifest dict.
    """
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hashes": {
            "data": data_config.config_hash,
            "model": model_config.model_config_hash,
            "training": training_config.training_config_hash,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "platform": platform.platform(),
        },
        "seed": training_config.reproducibility.seed,
        "extra": extra or {},
    }


def write_run_manifest(
    path: str | Path,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Build and atomically write the run manifest to ``path``.

    Args:
        path: Destination JSON file path.
        data_config: Validated data-preparation configuration.
        model_config: Validated model-architecture configuration.
        training_config: Validated training configuration.
        extra: Additional caller-supplied provenance.

    Returns:
        The destination path.
    """
    manifest = build_run_manifest(data_config, model_config, training_config, extra)
    return write_json(path, manifest)
