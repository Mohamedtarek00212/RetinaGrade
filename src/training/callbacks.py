"""Training callbacks.

Model checkpointing lives in :mod:`src.training.checkpoint` (it needs the
optimizer/scheduler state, not just a metric stream) and CSV/TensorBoard
logging live in :mod:`src.training.csv_logger`/:mod:`src.training.tensorboard_logger`.
This module holds the one callback that is purely a function of the metric
stream: early stopping.

The paper trains a fixed 50-epoch budget with no mention of early stopping
(see ``docs/milestone_04_paper_gaps.md``); :class:`EarlyStopping` is
therefore disabled by default (:attr:`~src.training.config.EarlyStoppingConfig.enabled`)
and is an opt-in engineering addition, not a paper-reported mechanism.
"""

from __future__ import annotations

from src.training.config import EarlyStoppingConfig
from src.utils.logger import get_logger

__all__ = ["EarlyStopping"]

logger = get_logger(__name__)


class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Args:
        config: Validated early-stopping configuration.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config
        self._best_value: float | None = None
        self._num_bad_epochs = 0
        self._should_stop = False

    def _is_better(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.config.mode == "max":
            return value > self._best_value
        return value < self._best_value

    def step(self, metrics: dict[str, float]) -> bool:
        """Update state with one epoch's metrics.

        Args:
            metrics: Flat mapping of metric name to value; must contain
                ``self.config.monitor_metric`` when early stopping is
                enabled.

        Returns:
            ``self.should_stop`` -- ``False`` unconditionally when
            ``config.enabled`` is ``False``.
        """
        if not self.config.enabled:
            return False

        value = metrics.get(self.config.monitor_metric)
        if value is None:
            logger.warning(
                "early_stopping.monitor_metric=%r not found in epoch metrics %s; skipping check",
                self.config.monitor_metric,
                sorted(metrics),
            )
            return self.should_stop

        if self._is_better(value):
            self._best_value = value
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1

        if self._num_bad_epochs >= self.config.patience:
            self._should_stop = True
            logger.info(
                "early stopping triggered: %s has not improved for %d epoch(s)",
                self.config.monitor_metric,
                self._num_bad_epochs,
            )
        return self.should_stop

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def num_bad_epochs(self) -> int:
        """Number of consecutive epochs without improvement."""
        return self._num_bad_epochs
