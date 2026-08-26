"""Optimizer factory.

Paper-explicit (Section 4, quoted): "optimized using the AdamW algorithm...
learning rate = 1e-4... weight decay = 1e-4" -- see
``docs/milestone_04_paper_gaps.md``.

Splitting parameters into a weight-decay and a no-weight-decay group (norm
and bias parameters) is standard AdamW practice, not a paper claim -- see
``TrainingConfig.optimizer.no_decay_patterns``.

Paper Gap PG-19: layer-wise LR decay is never mentioned by the paper;
``layerwise_lr_decay`` is an optional, off-by-default hook.
"""

from __future__ import annotations

from torch import nn, optim

from src.training.config import OptimizerConfig
from src.utils.logger import get_logger

__all__ = ["OPTIMIZER_REGISTRY", "build_optimizer", "apply_frozen_patterns", "split_param_groups"]

logger = get_logger(__name__)

#: Registered optimizer constructors, keyed by ``OptimizerConfig.name``.
OPTIMIZER_REGISTRY: dict[str, type[optim.Optimizer]] = {
    "adamw": optim.AdamW,
}


def apply_frozen_patterns(model: nn.Module, frozen_patterns: list[str]) -> int:
    """Set ``requires_grad = False`` on every parameter matching a pattern.

    Args:
        model: The model whose parameters may be frozen.
        frozen_patterns: Substrings matched against each parameter's
            dotted name. Empty by default (nothing frozen).

    Returns:
        The number of parameters frozen, for logging.
    """
    if not frozen_patterns:
        return 0
    frozen = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in frozen_patterns):
            param.requires_grad = False
            frozen += 1
    logger.info("froze %d parameter(s) matching %s", frozen, frozen_patterns)
    return frozen


def split_param_groups(
    model: nn.Module, weight_decay: float, no_decay_patterns: list[str]
) -> list[dict[str, object]]:
    """Split trainable parameters into weight-decay and no-weight-decay groups.

    Args:
        model: The model to draw parameters from. Only parameters with
            ``requires_grad = True`` are included (frozen parameters, if
            any, are excluded from both groups).
        weight_decay: Weight decay applied to the "decay" group.
        no_decay_patterns: Substrings matched against each parameter's
            dotted name; a match routes the parameter to the "no decay"
            group (standard practice for norm/bias parameters).

    Returns:
        A two-element list of parameter-group dicts suitable for an
        optimizer's ``params`` argument.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(pattern in name for pattern in no_decay_patterns):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> optim.Optimizer:
    """Assemble the AdamW optimizer from a validated :class:`OptimizerConfig`.

    Args:
        model: The model to optimize.
        config: Validated optimizer configuration.

    Returns:
        The configured optimizer.

    Raises:
        KeyError: If ``config.name`` is not registered in
            :data:`OPTIMIZER_REGISTRY`.
    """
    apply_frozen_patterns(model, config.frozen_patterns)
    param_groups = split_param_groups(model, config.weight_decay, config.no_decay_patterns)

    if config.layerwise_lr_decay is not None:
        logger.warning(
            "optimizer.layerwise_lr_decay=%s is set but not wired into any layer "
            "grouping yet (PG-19: never mentioned by the paper); ignored",
            config.layerwise_lr_decay,
        )

    try:
        optimizer_cls = OPTIMIZER_REGISTRY[config.name]
    except KeyError as exc:
        raise KeyError(f"unknown optimizer {config.name!r}; registered: {sorted(OPTIMIZER_REGISTRY)}") from exc

    optimizer = optimizer_cls(param_groups, lr=config.lr)
    logger.info(
        "built optimizer %s: lr=%s weight_decay=%s no_decay_patterns=%s frozen_patterns=%s",
        config.name,
        config.lr,
        config.weight_decay,
        config.no_decay_patterns,
        config.frozen_patterns,
    )
    return optimizer
