"""Model architecture configuration.

Mirrors :mod:`src.utils.config`'s pattern (YAML -> validated frozen
dataclasses) and **reuses its generic, dataclass-schema-agnostic coercion
engine directly** (``_from_mapping``, ``ConfigError``, ``deep_merge``,
``config_hash``, ``load_yaml``, ``PROJECT_ROOT``) rather than
reimplementing configuration parsing or validation.

Unlike :class:`~src.utils.config.DataConfig`, most fields below have **no**
dataclass default. This is deliberate: the generic engine already raises
:class:`ConfigError` naming the missing key path when a required field is
absent from the YAML, which is exactly the "fail fast, never silently
default" behaviour the model architecture's paper-fidelity policy demands.
Every field without a default corresponds to an open entry in
``docs/milestone_04_paper_gaps.md``; each field's docstring cites the Gap
ID so a validation error can be traced straight to its justification.

Example
-------
>>> from src.models.config import load_model_config
>>> load_model_config("configs/model.yaml")  # doctest: +SKIP
Traceback (most recent call last):
    ...
src.utils.config.ConfigError: <root>: missing required key 'backbone'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT, ConfigError, _from_mapping, config_hash, deep_merge, load_yaml

__all__ = [
    "ConfigError",
    "BackboneConfig",
    "SPMConfig",
    "PLKAFusionConfig",
    "PLKAConfig",
    "NeckConfig",
    "OrdinalHeadConfig",
    "HeadsConfig",
    "ModelConfig",
    "load_model_config",
]


@dataclass(frozen=True)
class BackboneConfig:
    """Swin Transformer backbone.

    Paper Gap PG-01: variant, patch size, and window size unspecified.
    Paper Gap PG-02: input resolution not paper-confirmed.

    No field below has a default; omitting any of them from
    ``configs/model.yaml`` raises a ``ConfigError`` naming the missing key.

    Attributes:
        name: Registry key for the backbone implementation. Currently only
            ``"timm_swin"`` (:class:`~src.models.backbones.swin.SwinBackboneAdapter`)
            is registered -- using ``timm`` introduces no new dependency,
            since it was already pinned in ``pyproject.toml``.
        variant: ``timm`` model name (PG-01). Not chosen by this project.
        pretrained: Whether to load pretrained weights (PG-01).
        image_size: Expected square input resolution (PG-02).
    """

    name: str
    variant: str
    pretrained: bool
    image_size: int


@dataclass(frozen=True)
class SPMConfig:
    """Semantic Prior Modulation.

    Paper Gap PG-03: text-adapter embedding dimensionality is
    implementation-defined; no concrete :class:`TextAdapter` ships in this
    milestone regardless (PG-03/PG-04).
    Paper Gap PG-05/PG-05b: which single backbone stage receives the gate
    is unspecified; a single stage index is the most conservative reading
    of Figure 1 and is what this configuration supports.
    Paper Gap PG-06: the fusion-matrix algebra and gate/feature combination
    rule are unspecified; only the sigmoid nonlinearity is paper-fixed
    (enforced in :meth:`src.models.semantic_prior.spm.SemanticPriorModulation.forward`).

    Attributes:
        name: Registry key for the SPM implementation (PG-06). No
            production implementation is registered in this milestone.
        inject_at_stage: Index (0-based) into the backbone's stage outputs
            that receives the gate (PG-05).
        text_embedding_dim: Dimensionality expected from the (not-yet-
            implemented) text adapter (PG-03).
    """

    name: str
    inject_at_stage: int
    text_embedding_dim: int


@dataclass(frozen=True)
class PLKAFusionConfig:
    """Paper Gap PG-09: the "attention-based fusion mechanism" architecture.

    Attributes:
        name: Registry key for the fusion implementation. No production
            implementation is registered in this milestone.
    """

    name: str


@dataclass(frozen=True)
class PLKAConfig:
    """Parallel/Progressive Lesion-aware Kernel Attention.

    Dilation rates are **not** configurable: the paper explicitly fixes
    them at ``(1, 2, 3)`` (Figure 1 caption: "standard, r = 2, r = 3") --
    see :data:`src.models.attention.plka.PLKA_DILATION_RATES`. Making an
    explicit paper fact configurable would misrepresent it as a gap.

    Paper Gap PG-07: branch activation function and convolution kernel
    size unspecified.
    Paper Gap PG-08: branch normalization layer unspecified.
    Paper Gap PG-09: fusion mechanism unspecified (see :class:`PLKAFusionConfig`).
    Paper Gap PG-10: which single backbone stage feeds PLKA is unspecified.

    Attributes:
        input_stage: Index (0-based) into the backbone's stage outputs
            that feeds PLKA (PG-10).
        kernel_size: Convolution kernel size shared by all three branches
            (PG-07).
        activation: Registry key resolved via
            :func:`src.models.factories.activation_factory` (PG-07).
        normalization: Registry key resolved via
            :func:`src.models.factories.normalization_factory` (PG-08).
        fusion: Fusion-mechanism sub-configuration (PG-09).
    """

    input_stage: int
    kernel_size: int
    activation: str
    normalization: str
    fusion: PLKAFusionConfig


@dataclass(frozen=True)
class NeckConfig:
    """Shared Feature Neck.

    Paper-explicit: a single shared fully-connected layer feeds both heads
    (Figure 1). ``hidden_dim`` is that layer's output width.

    Paper Gap PG-11: pooling strategy (reducing PLKA's spatial output to a
    vector before the FC layer) unspecified.
    Paper Gap PG-12a: hidden_dim value unspecified.
    Paper Gap PG-12b: whether any activation/dropout follows the FC layer
    at all is unspecified; both are exposed as required, disable-able
    (``activation="identity"``, ``dropout=0.0``) fields rather than assumed
    present, so their existence in this schema is a configurability
    affordance, not a claim that the paper uses them.

    Attributes:
        pooling: Registry key for the spatial-reduction strategy (PG-11).
            No production implementation is registered in this milestone.
        hidden_dim: Shared FC layer output width (PG-12a).
        dropout: Dropout probability after the FC layer; ``0.0`` disables
            it entirely (PG-12b).
        activation: Registry key resolved via
            :func:`src.models.factories.activation_factory`; ``"identity"``
            disables it entirely (PG-12b).
    """

    pooling: str
    hidden_dim: int
    dropout: float
    activation: str


@dataclass(frozen=True)
class OrdinalHeadConfig:
    """Paper Gap PG-13/PG-14: parameterization and "DPE" meaning unresolved.

    Attributes:
        name: Registry key for the ordinal-head parameterization. No
            production implementation is registered in this milestone.
    """

    name: str


@dataclass(frozen=True)
class HeadsConfig:
    """Container for head-specific configuration.

    The Classification Head needs no gap-driven configuration of its own
    (``K`` comes from ``DataConfig.classes.num_classes``, and its hidden
    input width comes from :class:`NeckConfig`), so only the Ordinal Head
    appears here.
    """

    ordinal: OrdinalHeadConfig


@dataclass(frozen=True)
class ModelConfig:
    """Fully validated model-architecture configuration.

    Reuses the exact validation engine :class:`~src.utils.config.DataConfig`
    uses; introduces no parallel config system.

    Attributes:
        model_name: Registry key for the assembled model. Currently only
            ``"dual_swinord"`` is meaningful.
    """

    model_name: str
    backbone: BackboneConfig
    spm: SPMConfig
    plka: PLKAConfig
    neck: NeckConfig
    heads: HeadsConfig
    project_root: Path = field(default=PROJECT_ROOT, compare=False, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve ``path`` against the project root if it is relative."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.project_root / candidate)

    @property
    def model_config_hash(self) -> str:
        """Stable hash over the architecture-defining configuration.

        Intended, alongside ``DataConfig.preprocessing_hash``, to become
        part of the Training milestone's checkpoint/run-manifest
        provenance key -- not consumed by anything in this milestone.
        """
        return config_hash(self.raw)


def load_model_config(
    path: str | Path = "configs/model.yaml",
    overrides: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> ModelConfig:
    """Load, merge, and validate the model-architecture configuration.

    Unlike :func:`~src.utils.config.load_data_config`, there is no profile
    or per-dataset override system here: the policy for this milestone is
    that exactly one paper-faithful configuration should exist once every
    Paper Gap is resolved, so there is nothing to select between yet.

    Args:
        path: Path to the YAML configuration file.
        overrides: Nested mapping of overrides, typically from
            :func:`~src.utils.config.parse_overrides`.
        project_root: Root used to resolve relative paths; defaults to the
            repository root.

    Returns:
        A validated :class:`ModelConfig`.

    Raises:
        ConfigError: If the configuration is missing, malformed, or (most
            commonly at this stage of the project) omits a gap-driven
            required field -- see ``docs/milestone_04_paper_gaps.md``.
    """
    root = project_root or PROJECT_ROOT
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path

    raw = load_yaml(config_path)
    if overrides:
        raw = deep_merge(raw, overrides)

    return _from_mapping(ModelConfig, {**raw, "project_root": root, "raw": raw})
