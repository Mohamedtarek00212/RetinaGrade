"""Checkpoint saving, loading, and best-model selection.

Reuses :func:`src.utils.helpers.ensure_dir` and :func:`src.utils.helpers.write_json`
rather than reimplementing directory creation or JSON serialisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

from src.training.config import CheckpointConfig
from src.utils.helpers import ensure_dir, read_json, write_json
from src.utils.logger import get_logger

__all__ = ["CheckpointManager"]

logger = get_logger(__name__)


class CheckpointManager:
    """Saves/loads model+optimizer+scheduler state and tracks the best epoch.

    Args:
        config: Validated checkpoint configuration.
        project_root: Root used to resolve ``config.dir`` if it is relative.
    """

    def __init__(self, config: CheckpointConfig, project_root: Path) -> None:
        self.config = config
        self.directory = ensure_dir(
            Path(config.dir) if Path(config.dir).is_absolute() else project_root / config.dir
        )
        self._best_value: float | None = None
        self._best_path: Path | None = None
        self._last_path: Path | None = None
        state_path = self.directory / "best_state.json"
        if state_path.exists():
            state = read_json(state_path)
            self._best_value = state.get("best_value")
            self._best_path = Path(state["best_path"]) if state.get("best_path") else None
        last_path = self.directory / "last.pt"
        if last_path.exists():
            self._last_path = last_path

    @property
    def best_path(self) -> Path | None:
        """Path to the current best checkpoint, if one has been saved."""
        return self._best_path

    @property
    def last_path(self) -> Path | None:
        """Path to the most recent "last" checkpoint, if one has been saved."""
        return self._last_path

    def _is_better(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.config.monitor_mode == "max":
            return value > self._best_value
        return value < self._best_value

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Save a checkpoint for one epoch, updating "best" and "last" pointers.

        Args:
            epoch: 0-based epoch index.
            model: Model whose ``state_dict()`` is saved.
            optimizer: Optimizer whose ``state_dict()`` is saved.
            scheduler: Scheduler whose ``state_dict()`` is saved (``None``
                permitted, for callers without one).
            metrics: Epoch metrics dict; must contain
                ``self.config.monitor_metric`` if best-model tracking is used.
            extra: Additional JSON-serialisable provenance to embed.

        Returns:
            Mapping of the paths written this call (``{"epoch": ..., possibly
            "best": ..., possibly "last": ...}``).
        """
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "extra": extra or {},
        }
        written: dict[str, Path] = {}

        epoch_path = self.directory / f"epoch_{epoch:04d}.pt"
        torch.save(payload, epoch_path)
        written["epoch"] = epoch_path
        logger.info("saved checkpoint: %s", epoch_path)

        if self.config.save_last:
            last_path = self.directory / "last.pt"
            torch.save(payload, last_path)
            self._last_path = last_path
            written["last"] = last_path

        monitor_value = metrics.get(self.config.monitor_metric)
        if monitor_value is not None and self._is_better(monitor_value):
            best_path = self.directory / "best.pt"
            torch.save(payload, best_path)
            self._best_value = monitor_value
            self._best_path = best_path
            write_json(
                self.directory / "best_state.json",
                {"best_value": monitor_value, "best_path": str(best_path), "epoch": epoch},
            )
            written["best"] = best_path
            logger.info("new best checkpoint: %s=%.6f -> %s", self.config.monitor_metric, monitor_value, best_path)

        return written

    def load(self, path: str | Path, map_location: str | None = None) -> dict[str, Any]:
        """Load a raw checkpoint payload.

        Args:
            path: Checkpoint file path.
            map_location: Forwarded to :func:`torch.load`.

        Returns:
            The saved payload dict (see :meth:`save`'s ``payload``).
        """
        return torch.load(path, map_location=map_location, weights_only=False)

    @property
    def best_path(self) -> Path | None:
        """Path to the best checkpoint seen so far, if any."""
        return self._best_path

    @property
    def best_value(self) -> float | None:
        """Monitor-metric value of the best checkpoint seen so far, if any."""
        return self._best_value
