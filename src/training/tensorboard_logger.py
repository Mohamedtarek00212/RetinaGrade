"""Optional TensorBoard scalar logging.

``tensorboard`` is **not** a pinned dependency in ``pyproject.toml``.
Enabling :attr:`~src.training.config.LoggingConfig.tensorboard_enabled`
without it installed logs a warning once and silently no-ops for the rest
of the run, rather than crashing training over an optional convenience
feature.
"""

from __future__ import annotations

from pathlib import Path

from src.training.config import LoggingConfig
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

__all__ = ["TensorBoardLogger"]

logger = get_logger(__name__)


class TensorBoardLogger:
    """Thin wrapper around ``torch.utils.tensorboard.SummaryWriter``.

    Args:
        config: Validated logging configuration.
        project_root: Root used to resolve ``config.log_dir`` if relative.
    """

    def __init__(self, config: LoggingConfig, project_root: Path) -> None:
        self._writer = None
        if not config.tensorboard_enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning(
                "logging.tensorboard_enabled=True but the 'tensorboard' package is "
                "not installed; TensorBoard logging is disabled for this run. "
                "Install it with `pip install tensorboard` to enable it."
            )
            return

        directory = Path(config.log_dir) if Path(config.log_dir).is_absolute() else project_root / config.log_dir
        self._writer = SummaryWriter(log_dir=str(ensure_dir(directory) / "tensorboard"))

    @property
    def enabled(self) -> bool:
        """Whether a real ``SummaryWriter`` was successfully constructed."""
        return self._writer is not None

    def log_scalars(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log one epoch's metrics as scalars; a no-op when disabled.

        Args:
            epoch: 0-based epoch index, used as the scalar step.
            metrics: Flat mapping of metric name to value.
        """
        if self._writer is None:
            return
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, epoch)

    def close(self) -> None:
        """Flush and close the underlying writer, if any."""
        if self._writer is not None:
            self._writer.close()
