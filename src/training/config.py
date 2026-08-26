"""Training configuration.

Mirrors :mod:`src.models.config`'s pattern (YAML -> validated frozen
dataclasses) and **reuses** :mod:`src.utils.config`'s generic,
dataclass-schema-agnostic coercion engine (``_from_mapping``, ``ConfigError``,
``deep_merge``, ``config_hash``, ``load_yaml``, ``PROJECT_ROOT``) rather than
reimplementing configuration parsing or validation.

Unlike most of :mod:`src.models.config`, several fields below **do** carry a
default value. This is deliberate and different from the model-architecture
policy: those defaults are not invented architectural guesses, they are the
Dual-SwinOrd paper's own reported experimental hyperparameters (AdamW,
lr=1e-4, weight_decay=1e-4, 50 epochs, cosine annealing, lambda=0.5-- see
``docs/milestone_04_paper_gaps.md``'s "What is explicitly supported" table,
Milestone 05 rows). Every field that has **no** default corresponds to an
open Paper Gap (PG-17 through PG-20) in that same document; its docstring
cites the Gap ID so a validation error can be traced straight to its
justification, exactly as :mod:`src.models.config` already does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT, ConfigError, _from_mapping, config_hash, deep_merge, load_yaml

__all__ = [
    "ConfigError",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "AMPConfig",
    "GradientConfig",
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "LoggingConfig",
    "ReproducibilityConfig",
    "TrainingConfig",
    "load_training_config",
]


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW optimizer settings.

    ``name``, ``lr``, and ``weight_decay`` are paper-confirmed (Section 4):
    "optimized using the AdamW algorithm... learning rate = 1e-4... weight
    decay = 1e-4". ``betas``/``eps`` are intentionally **not** fields here;
    when unset, :func:`src.training.optim.build_optimizer` lets
    :class:`torch.optim.AdamW` apply its own library defaults, which is a
    PyTorch default, not a Dual-SwinOrd paper claim.

    Paper Gap PG-19: layer-wise LR decay is never mentioned by the paper;
    ``layerwise_lr_decay`` stays ``None`` (disabled) unless a caller
    explicitly opts in.

    Attributes:
        name: Optimizer registry key. Only ``"adamw"`` is paper-confirmed.
        lr: Base learning rate (paper-confirmed: ``1e-4``).
        weight_decay: Weight decay (paper-confirmed: ``1e-4``).
        no_decay_patterns: Parameter-name substrings excluded from weight
            decay (standard AdamW practice for norm/bias parameters, not a
            paper claim).
        frozen_patterns: Parameter-name substrings whose ``requires_grad``
            is set to ``False``. Empty by default (nothing is frozen unless
            explicitly configured).
        layerwise_lr_decay: Optional per-layer LR decay factor (PG-19).
            ``None`` disables the feature entirely.
    """

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    no_decay_patterns: list[str] = field(default_factory=lambda: ["bias", "norm"])
    frozen_patterns: list[str] = field(default_factory=list)
    layerwise_lr_decay: float | None = None


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate scheduler settings.

    ``name`` is paper-confirmed (Section 4): "A cosine annealing scheduler
    was employed to dynamically adjust the learning rate."

    Paper Gap PG-18: neither ``t_max`` nor ``eta_min`` is given a value
    anywhere in the retrieved excerpts. ``t_max`` defaults to the run's
    total ``epochs`` (a derivation from a paper-confirmed value, not an
    invention) but remains overridable; ``eta_min`` has no default and must
    be supplied explicitly.

    Attributes:
        name: Scheduler registry key (paper-confirmed: ``"cosine_annealing"``).
        t_max: Cosine period, in epochs. ``None`` resolves to ``epochs``.
        eta_min: Minimum learning rate reached at the end of the cosine
            cycle (PG-18) -- required, no default.
    """

    eta_min: float
    name: str = "cosine_annealing"
    t_max: int | None = None


@dataclass(frozen=True)
class LossConfig:
    """Loss-package configuration (Eq. 7, Eq. 8, Eq. 9).

    Paper Gap PG-20: Eq. 7 names "Cross-Entropy loss with Label Smoothing"
    but never gives the smoothing epsilon's numeric value -- ``label_smoothing``
    is therefore required, with no default.

    ``lambda_cls`` is paper-confirmed (Eq. 9): "In our experiments, we set
    lambda = 0.5".

    Paper Gap PG-17: CARM's "cost-sensitive adaptive" mechanism is
    unconfirmed beyond Eq. 8's plain per-threshold BCE.
    ``carm_pos_weight_enabled`` is an optional, off-by-default hook -- see
    ``src/losses/carm_loss.py``.

    Attributes:
        label_smoothing: Epsilon for :class:`~src.losses.classification_loss.ClassificationLoss`
            (PG-20) -- required, no default.
        class_weight_strategy: Optional strategy name forwarded to
            :func:`src.data.statistics.compute_class_weights` for the
            Classification Loss. ``None`` disables class weighting
            entirely, matching Eq. 7's unweighted formula.
        lambda_cls: Weight on the Classification Loss in the total
            objective (Eq. 9; paper-confirmed default ``0.5``).
        carm_pos_weight_enabled: Opt-in, non-paper-confirmed cost hook for
            :class:`~src.losses.carm_loss.CARMLoss` (PG-17). Disabled by
            default.
    """

    label_smoothing: float
    class_weight_strategy: str | None = None
    lambda_cls: float = 0.5
    carm_pos_weight_enabled: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigError(f"loss.label_smoothing: expected a value in [0, 1), got {self.label_smoothing}")
        if not 0.0 <= self.lambda_cls <= 1.0:
            raise ConfigError(f"loss.lambda_cls: expected a value in [0, 1], got {self.lambda_cls}")


@dataclass(frozen=True)
class AMPConfig:
    """Automatic mixed precision.

    Not mentioned anywhere in the retrieved paper excerpts (the paper only
    states the GPU model, an A100); disabled by default as an opt-in
    engineering feature.
    """

    enabled: bool = False


@dataclass(frozen=True)
class GradientConfig:
    """Gradient accumulation and clipping.

    Neither mechanism is mentioned by the paper; both are opt-in
    engineering features, disabled by default.

    Attributes:
        accumulation_steps: Number of batches accumulated before an
            optimizer step. ``1`` disables accumulation.
        clip_norm: Max gradient norm for :func:`torch.nn.utils.clip_grad_norm_`.
            ``None`` disables clipping entirely.
    """

    accumulation_steps: int = 1
    clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.accumulation_steps < 1:
            raise ConfigError(f"gradient.accumulation_steps: expected >= 1, got {self.accumulation_steps}")
        if self.clip_norm is not None and self.clip_norm <= 0:
            raise ConfigError(f"gradient.clip_norm: expected a positive value or null, got {self.clip_norm}")


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpointing and best-model selection.

    ``monitor_metric`` defaults to ``"val_qwk"`` -- Quadratic Weighted Kappa
    is the paper's own headline metric (0.9370 on APTOS 2019), so selecting
    the best checkpoint by it is the most paper-aligned choice available,
    not an arbitrary engineering default.
    """

    dir: str = "outputs/checkpoints/training"
    monitor_metric: str = "val_qwk"
    monitor_mode: str = "max"
    save_last: bool = True

    def __post_init__(self) -> None:
        if self.monitor_mode not in ("max", "min"):
            raise ConfigError(f"checkpoint.monitor_mode: expected 'max' or 'min', got {self.monitor_mode!r}")


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping.

    The paper trains a fixed 50-epoch budget with no mention of early
    stopping; this feature is therefore disabled by default and is an
    opt-in engineering addition, not a paper-reported mechanism.
    """

    enabled: bool = False
    patience: int = 10
    monitor_metric: str = "val_qwk"
    mode: str = "max"

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ConfigError(f"early_stopping.patience: expected >= 1, got {self.patience}")
        if self.mode not in ("max", "min"):
            raise ConfigError(f"early_stopping.mode: expected 'max' or 'min', got {self.mode!r}")


@dataclass(frozen=True)
class LoggingConfig:
    """Progress bars, CSV logging, and TensorBoard.

    ``tensorboard_enabled`` defaults to ``False``: ``tensorboard`` is not a
    pinned dependency in ``pyproject.toml``. Setting it ``True`` without
    installing the package logs a warning and no-ops rather than crashing
    training -- see ``src/training/tensorboard_logger.py``.
    """

    log_dir: str = "outputs/logs/training"
    csv_filename: str = "epoch_log.csv"
    tensorboard_enabled: bool = False
    progress_bar: bool = True


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Seeding and determinism.

    ``seed`` has no default, mirroring ``DataConfig``'s own seed field: a
    silently-defaulted seed is exactly the kind of "it worked on my
    machine" hazard this project's reproducibility policy exists to
    prevent.
    """

    seed: int
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ConfigError(f"reproducibility.seed: expected a non-negative integer, got {self.seed}")


@dataclass(frozen=True)
class TrainingConfig:
    """Fully validated Training-milestone configuration.

    Reuses the exact validation engine :class:`~src.utils.config.DataConfig`
    and :class:`~src.models.config.ModelConfig` use; introduces no parallel
    config system.

    Attributes:
        epochs: Total training epochs (paper-confirmed: ``50``).
    """

    epochs: int
    reproducibility: ReproducibilityConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    project_root: Path = field(default=PROJECT_ROOT, compare=False, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError(f"epochs: expected >= 1, got {self.epochs}")

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve ``path`` against the project root if it is relative."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.project_root / candidate)

    @property
    def scheduler_t_max(self) -> int:
        """Effective cosine period: ``scheduler.t_max`` or ``epochs``."""
        return self.scheduler.t_max if self.scheduler.t_max is not None else self.epochs

    @property
    def training_config_hash(self) -> str:
        """Stable hash over the training-defining configuration.

        Intended to be combined with ``DataConfig.config_hash`` and
        ``ModelConfig.model_config_hash`` into one experiment hash -- see
        ``src/training/manifest.py``.
        """
        return config_hash(self.raw)


def load_training_config(
    path: str | Path = "configs/training.yaml",
    overrides: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> TrainingConfig:
    """Load, merge, and validate the training configuration.

    Args:
        path: Path to the YAML configuration file.
        overrides: Nested mapping of overrides, typically from
            :func:`~src.utils.config.parse_overrides`.
        project_root: Root used to resolve relative paths; defaults to the
            repository root.

    Returns:
        A validated :class:`TrainingConfig`.

    Raises:
        ConfigError: If the configuration is missing, malformed, or omits a
            gap-driven required field -- see
            ``docs/milestone_04_paper_gaps.md`` (PG-17 through PG-20).
    """
    root = project_root or PROJECT_ROOT
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path

    raw = load_yaml(config_path)
    if overrides:
        raw = deep_merge(raw, overrides)

    return _from_mapping(TrainingConfig, {**raw, "project_root": root, "raw": raw})
