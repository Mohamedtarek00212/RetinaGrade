"""Automatic mixed precision helper.

Not mentioned anywhere in the retrieved paper excerpts -- see
``docs/milestone_04_paper_gaps.md``. Disabled by default
(:attr:`~src.training.config.AMPConfig.enabled`); this module exists so
enabling it is a one-line, reversible engineering opt-in rather than a
scattered set of ``if enabled`` checks throughout :mod:`src.training.trainer`.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch
from torch import Tensor

from src.training.config import AMPConfig
from src.utils.logger import get_logger

__all__ = ["AMPContext"]

logger = get_logger(__name__)


class AMPContext:
    """Bundles the autocast context manager and gradient scaler.

    Args:
        config: Validated AMP configuration.
        device_type: ``"cuda"`` or ``"cpu"``. AMP is only meaningfully
            beneficial on CUDA; the scaler is a no-op on CPU regardless of
            ``config.enabled``.
    """

    def __init__(self, config: AMPConfig, device_type: str) -> None:
        self.enabled = config.enabled and device_type == "cuda"
        if config.enabled and device_type != "cuda":
            logger.warning("amp.enabled=True has no effect on device_type=%r; ignored", device_type)
        self.device_type = device_type
        self.scaler = torch.amp.GradScaler(device_type, enabled=self.enabled)

    def autocast(self) -> AbstractContextManager:
        """Return the autocast context manager (a no-op context if disabled)."""
        if not self.enabled:
            return nullcontext()
        return torch.amp.autocast(device_type=self.device_type)

    def backward(self, loss: Tensor) -> None:
        """Scale-aware backward pass."""
        self.scaler.scale(loss).backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Scale-aware optimizer step, followed by scaler update."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients in-place, needed before gradient clipping."""
        self.scaler.unscale_(optimizer)
