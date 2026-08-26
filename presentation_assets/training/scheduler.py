"""Learning-rate scheduler factory.

Paper-explicit (Section 4, quoted): "A cosine annealing scheduler was
employed to dynamically adjust the learning rate." -- see
``docs/milestone_04_paper_gaps.md``.

Paper Gap PG-18: neither ``T_max`` nor ``eta_min`` is given a value anywhere
in the retrieved excerpts. ``T_max`` defaults to the run's total ``epochs``
(:attr:`~src.training.config.TrainingConfig.scheduler_t_max`) -- a
derivation from a paper-confirmed value, not an invention -- while
``eta_min`` is a required field with no default (see
:class:`~src.training.config.SchedulerConfig`).
"""

from __future__ import annotations

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from src.training.config import SchedulerConfig
from src.utils.logger import get_logger

__all__ = ["SCHEDULER_REGISTRY", "build_scheduler"]

logger = get_logger(__name__)

#: Registered scheduler constructors, keyed by ``SchedulerConfig.name``.
SCHEDULER_REGISTRY: dict[str, type[LRScheduler]] = {
    "cosine_annealing": CosineAnnealingLR,
}


def build_scheduler(optimizer: optim.Optimizer, config: SchedulerConfig, epochs: int) -> LRScheduler:
    """Assemble the cosine-annealing scheduler from a validated :class:`SchedulerConfig`.

    Args:
        optimizer: The optimizer whose learning rate is scheduled.
        config: Validated scheduler configuration.
        epochs: The run's total epoch budget; used as ``T_max`` when
            ``config.t_max`` is unset.

    Returns:
        The configured scheduler.

    Raises:
        KeyError: If ``config.name`` is not registered in
            :data:`SCHEDULER_REGISTRY`.
    """
    t_max = config.t_max if config.t_max is not None else epochs
    try:
        scheduler_cls = SCHEDULER_REGISTRY[config.name]
    except KeyError as exc:
        raise KeyError(f"unknown scheduler {config.name!r}; registered: {sorted(SCHEDULER_REGISTRY)}") from exc

    scheduler = scheduler_cls(optimizer, T_max=t_max, eta_min=config.eta_min)
    logger.info("built scheduler %s: T_max=%d eta_min=%s", config.name, t_max, config.eta_min)
    return scheduler
