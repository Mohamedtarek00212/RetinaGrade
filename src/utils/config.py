"""Configuration loading, validation, and provenance hashing.

This module is the single entry point for reading ``configs/data.yaml`` into a
validated, strongly-typed object graph. It is intentionally dependency-free
(standard library + PyYAML only) so that the reproducibility surface of the
project stays small: the exact semantics of every validation rule live in this
file rather than in a third-party schema library whose defaults may drift
between versions.

Responsibilities (orchestration only -- no domain logic):

* load YAML,
* deep-merge profile presets and per-dataset overrides,
* coerce the resulting mapping into frozen dataclasses,
* validate eagerly and fail fast with the offending key path,
* resolve relative paths against the project root,
* compute a stable ``config_hash`` used for cache invalidation and for
  stamping provenance into every generated report.

Design notes
------------
Defaults live in ``configs/data.yaml``, not in the dataclass field defaults.
Duplicating defaults in two places is a classic source of silent divergence,
so dataclass defaults are only provided for genuinely optional fields.

Example
-------
>>> from src.utils.config import load_data_config
>>> cfg = load_data_config("configs/data.yaml")
>>> cfg.preprocessing.image_size
512
>>> cfg.resolve_path(cfg.paths.raw_dir).name
'raw'
"""

from __future__ import annotations

import copy
import hashlib
import json
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import yaml

__all__ = [
    "ConfigError",
    "DataConfig",
    "load_data_config",
    "parse_overrides",
    "deep_merge",
    "config_hash",
    "PROJECT_ROOT",
    "PAPER_FAITHFUL_OVERRIDES",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Repository root, derived from this file's location (src/utils/config.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Valid evidence profiles. See ``configs/data.yaml`` for the full rationale.
VALID_PROFILES: tuple[str, ...] = ("paper_faithful", "eda_driven")

#: Overrides applied when ``profile == "paper_faithful"``.
#:
#: The Dual-SwinOrd paper describes only a random 80/20 split plus horizontal
#: flip, vertical flip, random rotation, and color jittering. Every step below
#: is EDA-derived and therefore disabled in the reproduction tier so that the
#: paper-faithful numbers are genuinely paper-faithful.
PAPER_FAITHFUL_OVERRIDES: dict[str, Any] = {
    "preprocessing": {
        "black_border_removal": {"enabled": False},
        "circular_crop": {"enabled": False},
        "clahe": {"enabled": False},
        "illumination_correction": {"enabled": False},
    },
    "augmentation": {
        "train": {
            "random_brightness": {"enabled": False},
            "random_contrast": {"enabled": False},
            "scale_jitter": {"enabled": False},
            "conservative_crop": {"enabled": False},
            "gamma": {"enabled": False},
            "gaussian_blur": {"enabled": False},
            "gaussian_noise": {"enabled": False},
        }
    },
}


class ConfigError(ValueError):
    """Raised when a configuration file is malformed, incomplete, or invalid.

    The message always contains the dotted key path of the offending entry so
    that a typo in a 350-line YAML file is a one-second fix rather than a hunt.
    """


# ---------------------------------------------------------------------------
# Generic mapping -> dataclass coercion
# ---------------------------------------------------------------------------


def _fail(path: str, message: str) -> None:
    """Raise a :class:`ConfigError` annotated with the offending key path.

    Args:
        path: Dotted key path (for example ``"augmentation.train.rotation.p"``).
        message: Human-readable description of the problem.

    Raises:
        ConfigError: Always.
    """
    location = path or "<root>"
    raise ConfigError(f"{location}: {message}")


def _type_name(tp: Any) -> str:
    """Return a readable name for a typing construct or class."""
    return getattr(tp, "__name__", str(tp))


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """Coerce and validate a raw YAML value against an annotated type.

    Supports nested dataclasses, ``list[T]``, fixed- and variable-length
    ``tuple``, ``dict[K, V]``, ``Optional[T]``/``T | None``, ``pathlib.Path``,
    and the primitive scalar types.

    Args:
        value: Raw value parsed from YAML.
        tp: Target type annotation.
        path: Dotted key path used for error reporting.

    Returns:
        The coerced value.

    Raises:
        ConfigError: If the value cannot be coerced to ``tp``.
    """
    if tp is Any:
        return value

    origin = get_origin(tp)
    args = get_args(tp)

    # Optional[T] / Union[...] -- try each member, ``None`` short-circuits.
    if origin is Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        for candidate in (a for a in args if a is not type(None)):
            try:
                return _coerce(value, candidate, path)
            except ConfigError:
                continue
        _fail(path, f"expected one of {[_type_name(a) for a in args]}, got {value!r}")

    # Nested dataclass section.
    if is_dataclass(tp):
        if not isinstance(value, Mapping):
            _fail(path, f"expected a mapping for section '{_type_name(tp)}', got {type(value).__name__}")
        return _from_mapping(tp, value, path)

    if origin in (list, Sequence):
        if not isinstance(value, list):
            _fail(path, f"expected a list, got {type(value).__name__}")
        item_type = args[0] if args else Any
        return [_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            _fail(path, f"expected a sequence, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            _fail(path, f"expected exactly {len(args)} items, got {len(value)}")
        return tuple(_coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args)))

    if origin is dict or origin is Mapping:
        if not isinstance(value, Mapping):
            _fail(path, f"expected a mapping, got {type(value).__name__}")
        key_type = args[0] if args else Any
        val_type = args[1] if len(args) > 1 else Any
        return {
            _coerce(k, key_type, f"{path}.<key>"): _coerce(v, val_type, f"{path}.{k}")
            for k, v in value.items()
        }

    if tp is Path:
        if not isinstance(value, (str, Path)):
            _fail(path, f"expected a path string, got {type(value).__name__}")
        return Path(value)

    if tp is bool:
        if not isinstance(value, bool):
            _fail(path, f"expected a boolean, got {type(value).__name__} ({value!r})")
        return value

    if tp is int:
        # ``bool`` is a subclass of ``int``; reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(path, f"expected an integer, got {type(value).__name__} ({value!r})")
        return value

    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(path, f"expected a number, got {type(value).__name__} ({value!r})")
        return float(value)

    if tp is str:
        if not isinstance(value, str):
            _fail(path, f"expected a string, got {type(value).__name__} ({value!r})")
        return value

    return value


def _from_mapping(cls: type, data: Mapping[str, Any], path: str = "") -> Any:
    """Build a dataclass instance from a mapping, rejecting unknown keys.

    Args:
        cls: Target dataclass type.
        data: Mapping parsed from YAML.
        path: Dotted key path prefix used for error reporting.

    Returns:
        An instance of ``cls``.

    Raises:
        ConfigError: On unknown keys, missing required keys, or type errors.
    """
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls) if f.init}
    unknown = set(data) - known
    if unknown:
        _fail(path, f"unknown key(s) {sorted(unknown)}; expected any of {sorted(known)}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if not f.init:
            continue
        child_path = f"{path}.{f.name}" if path else f.name
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], hints[f.name], child_path)
        elif f.default is not MISSING or f.default_factory is not MISSING:
            continue  # a dataclass default applies
        else:
            _fail(path, f"missing required key '{f.name}'")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Validation helpers (shared by the dataclasses below)
# ---------------------------------------------------------------------------


def _check_probability(name: str, value: float) -> None:
    """Validate that ``value`` is a probability in ``[0, 1]``."""
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name}: probability must lie in [0, 1], got {value}")


def _check_positive(name: str, value: float) -> None:
    """Validate that ``value`` is strictly positive."""
    if value <= 0:
        raise ConfigError(f"{name}: expected a positive value, got {value}")


def _check_choice(name: str, value: str, choices: Sequence[str]) -> None:
    """Validate that ``value`` is one of ``choices``."""
    if value not in choices:
        raise ConfigError(f"{name}: expected one of {list(choices)}, got {value!r}")


def _check_range(name: str, value: Sequence[float]) -> None:
    """Validate that ``value`` is an ascending two-element range."""
    if len(value) != 2 or value[0] > value[1]:
        raise ConfigError(f"{name}: expected an ascending [low, high] range, got {list(value)}")


def _check_odd(name: str, value: int) -> None:
    """Validate that ``value`` is a positive odd integer (OpenCV kernel size)."""
    if value <= 0 or value % 2 == 0:
        raise ConfigError(f"{name}: expected a positive odd kernel size, got {value}")


# ---------------------------------------------------------------------------
# Dataset identity / paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathsConfig:
    """Top-level dataset directories (relative to the project root)."""

    base_dir: str
    raw_dir: str
    splits_dir: str
    processed_dir: str


@dataclass(frozen=True)
class SplitPathsConfig:
    """Manifest CSV and image directory for a single split."""

    csv: str
    image_dir: str


@dataclass(frozen=True)
class SplitsConfig:
    """Per-split file locations for train, validation, and test."""

    train: SplitPathsConfig
    val: SplitPathsConfig
    test: SplitPathsConfig

    def as_dict(self) -> dict[str, SplitPathsConfig]:
        """Return the splits keyed by canonical split name."""
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class ImageConfig:
    """Image file conventions."""

    extension: str

    def __post_init__(self) -> None:
        if not self.extension.startswith("."):
            raise ConfigError(f"image.extension: expected a leading dot, got {self.extension!r}")


@dataclass(frozen=True)
class ColumnsConfig:
    """Candidate column names used to locate the id and label columns.

    Candidate lists (rather than fixed names) keep the pipeline usable across
    retinal datasets that spell these columns differently.
    """

    id_candidates: list[str]
    label_candidates: list[str]

    def __post_init__(self) -> None:
        if not self.id_candidates:
            raise ConfigError("columns.id_candidates: must not be empty")
        if not self.label_candidates:
            raise ConfigError("columns.label_candidates: must not be empty")


@dataclass(frozen=True)
class ClassesConfig:
    """Ordinal class definition for the grading task."""

    num_classes: int
    order: list[int]
    names: dict[int, str]

    def __post_init__(self) -> None:
        _check_positive("classes.num_classes", self.num_classes)
        if len(self.order) != self.num_classes:
            raise ConfigError(
                f"classes.order: expected {self.num_classes} entries, got {len(self.order)}"
            )
        missing = set(self.order) - set(self.names)
        if missing:
            raise ConfigError(f"classes.names: missing display names for grades {sorted(missing)}")


@dataclass(frozen=True)
class OutputsConfig:
    """Locations of every artefact produced by the data-preparation pipeline."""

    root: str
    audit_manifest: str
    audit_report: str
    clean_manifest: str
    cleaning_report: str
    quarantine_manifest: str
    split_report: str
    statistics_report: str
    class_distribution: str
    preview_dir: str
    log_dir: str


# ---------------------------------------------------------------------------
# Stage 1 -- audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptualHashConfig:
    """Parameters of the internally implemented dHash/pHash primitives."""

    dhash_size: int
    phash_size: int
    phash_image_size: int

    def __post_init__(self) -> None:
        _check_positive("audit.perceptual_hash.dhash_size", self.dhash_size)
        _check_positive("audit.perceptual_hash.phash_size", self.phash_size)
        if self.phash_image_size < self.phash_size:
            raise ConfigError(
                "audit.perceptual_hash.phash_image_size: must be >= phash_size "
                f"({self.phash_image_size} < {self.phash_size})"
            )


@dataclass(frozen=True)
class AuditConfig:
    """Stage 1 -- full-corpus, read-only measurement pass."""

    enabled: bool
    full_decode: bool
    metric_long_side: int
    compute_md5: bool
    compute_perceptual_hash: bool
    perceptual_hash: PerceptualHashConfig
    compute_quality_metrics: bool
    tissue_threshold: int
    num_workers: int
    chunk_size: int
    resume: bool

    def __post_init__(self) -> None:
        _check_positive("audit.metric_long_side", self.metric_long_side)
        _check_positive("audit.chunk_size", self.chunk_size)
        if self.num_workers < 0:
            raise ConfigError(f"audit.num_workers: must be >= 0, got {self.num_workers}")
        if not 0 <= self.tissue_threshold <= 255:
            raise ConfigError(
                f"audit.tissue_threshold: must lie in [0, 255], got {self.tissue_threshold}"
            )


# ---------------------------------------------------------------------------
# Stage 2 -- cleaning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactDuplicatesConfig:
    """MD5-based exact-duplicate policy (the authoritative duplicate detector)."""

    enabled: bool
    exclude_cross_split: bool
    keep_split_priority: list[str]
    exclude_within_split: bool

    def __post_init__(self) -> None:
        valid = {"train", "val", "test"}
        unknown = set(self.keep_split_priority) - valid
        if unknown:
            raise ConfigError(
                f"cleaning.rules.exact_duplicates.keep_split_priority: unknown split(s) {sorted(unknown)}"
            )


@dataclass(frozen=True)
class NearDuplicatesConfig:
    """Perceptual-hash policy.

    ``report_only`` is validated to be ``True``: perceptual hashing is an
    investigation and clustering tool only and may never justify an automatic
    exclusion. The cleaning rule enforces this independently, so violating the
    contract requires defeating both this check and a dedicated unit test.
    """

    enabled: bool
    hamming_threshold: int
    report_only: bool

    def __post_init__(self) -> None:
        if self.hamming_threshold < 0:
            raise ConfigError(
                f"cleaning.rules.near_duplicates.hamming_threshold: must be >= 0, got {self.hamming_threshold}"
            )
        if not self.report_only:
            raise ConfigError(
                "cleaning.rules.near_duplicates.report_only: perceptual hashing is "
                "investigation-only and can never drive automatic exclusion; this "
                "flag must remain true. Use MD5 (exact_duplicates) for removals."
            )


@dataclass(frozen=True)
class IntegrityRuleConfig:
    """Integrity-check policy (unreadable, zero-byte, or missing files)."""

    enabled: bool
    exclude_unreadable: bool


@dataclass(frozen=True)
class LabelRuleConfig:
    """Label-consistency policy."""

    enabled: bool
    exclude_missing_label: bool
    exclude_out_of_range: bool
    exclude_duplicate_ids: bool
    exclude_missing_file: bool


@dataclass(frozen=True)
class QualityFlagsConfig:
    """Quality-outlier policy.

    ``delete`` is validated to be ``False``. Dark, bright, blurry, noisy, and
    low-contrast fundus images are genuine clinical acquisitions with valid
    labels; deleting them would only worsen minority-class scarcity. They are
    flagged for monitoring and test-time uncertainty analysis instead.
    """

    enabled: bool
    delete: bool
    dark_brightness_max: float
    bright_brightness_min: float
    low_contrast_std_max: float
    blur_percentile: float
    noise_percentile: float

    def __post_init__(self) -> None:
        if self.delete:
            raise ConfigError(
                "cleaning.rules.quality_flags.delete: quality outliers must never be "
                "deleted (their labels remain clinically valid); this flag must stay false."
            )
        if self.dark_brightness_max >= self.bright_brightness_min:
            raise ConfigError(
                "cleaning.rules.quality_flags: dark_brightness_max must be < bright_brightness_min "
                f"({self.dark_brightness_max} >= {self.bright_brightness_min})"
            )
        for name, value in (("blur_percentile", self.blur_percentile), ("noise_percentile", self.noise_percentile)):
            if not 0.0 <= value <= 100.0:
                raise ConfigError(
                    f"cleaning.rules.quality_flags.{name} must lie in [0, 100], got {value}. "
                    "Sharpness and noise are unitless and scale-dependent, so they are flagged "
                    "by percentile rather than by an absolute cutoff (set to 0 to disable)."
                )


@dataclass(frozen=True)
class CleaningRulesConfig:
    """Container for the individual cleaning rules."""

    exact_duplicates: ExactDuplicatesConfig
    near_duplicates: NearDuplicatesConfig
    integrity: IntegrityRuleConfig
    labels: LabelRuleConfig
    quality_flags: QualityFlagsConfig


@dataclass(frozen=True)
class CleaningConfig:
    """Stage 2 -- decision-only cleaning (never mutates ``data/raw``)."""

    enabled: bool
    rules: CleaningRulesConfig


# ---------------------------------------------------------------------------
# Stage 3 -- split verification / regeneration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitVerifyConfig:
    """Always-on, read-only split verification checks."""

    enabled: bool
    check_id_overlap: bool
    check_md5_overlap: bool
    check_near_duplicate_overlap: bool
    check_class_homogeneity: bool
    min_images_per_class: int
    strict: bool

    def __post_init__(self) -> None:
        if self.min_images_per_class < 0:
            raise ConfigError(
                f"splits_policy.verify.min_images_per_class: must be >= 0, got {self.min_images_per_class}"
            )


@dataclass(frozen=True)
class SplitRegenerateConfig:
    """Opt-in, group-aware split regeneration (off by default)."""

    enabled: bool
    output_dir: str
    ratios: dict[str, float]
    group_by: str
    stratify_by: str
    overwrite: bool

    def __post_init__(self) -> None:
        _check_choice("splits_policy.regenerate.group_by", self.group_by, ["content_cluster", "md5", "none"])
        _check_choice("splits_policy.regenerate.stratify_by", self.stratify_by, ["label", "none"])
        expected = {"train", "val", "test"}
        if set(self.ratios) != expected:
            raise ConfigError(
                f"splits_policy.regenerate.ratios: expected keys {sorted(expected)}, got {sorted(self.ratios)}"
            )
        total = sum(self.ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"splits_policy.regenerate.ratios: must sum to 1.0, got {total}")


@dataclass(frozen=True)
class SplitsPolicyConfig:
    """Stage 3 configuration."""

    verify: SplitVerifyConfig
    regenerate: SplitRegenerateConfig


# ---------------------------------------------------------------------------
# Stage 4 -- statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImbalanceConfig:
    """Class-imbalance analysis settings.

    Analysis only: this milestone deliberately implements no sampler, no
    oversampling, and no class balancing. Weighting strategies are exposed as
    pure functions for the Training milestone to consume.
    """

    enabled: bool
    weight_strategies: list[str]
    effective_number_beta: float
    save_plot: bool

    def __post_init__(self) -> None:
        valid = {"inverse", "inverse_sqrt", "effective_number", "balanced"}
        unknown = set(self.weight_strategies) - valid
        if unknown:
            raise ConfigError(
                f"statistics.imbalance.weight_strategies: unknown strategy/strategies {sorted(unknown)}"
            )
        if not 0.0 < self.effective_number_beta < 1.0:
            raise ConfigError(
                "statistics.imbalance.effective_number_beta: must lie in (0, 1), "
                f"got {self.effective_number_beta}"
            )


@dataclass(frozen=True)
class NormalizationConfig:
    """Per-channel normalization statistics policy.

    ``auto`` computes statistics from the training split only, after the
    deterministic geometric preprocessing and before augmentation, then caches
    them keyed by the preprocessing-config hash. Statistics must not be taken
    from the EDA's raw-image measurements because black-border removal changes
    the pixel population and biases those values low.
    """

    mode: str
    cache_path: str
    max_images: int | None
    fallback_mean: list[float]
    fallback_std: list[float]
    imagenet_mean: list[float]
    imagenet_std: list[float]

    def __post_init__(self) -> None:
        _check_choice("statistics.normalization.mode", self.mode, ["auto", "config", "imagenet"])
        if self.max_images is not None:
            _check_positive("statistics.normalization.max_images", self.max_images)
        for name, values in (
            ("fallback_mean", self.fallback_mean),
            ("fallback_std", self.fallback_std),
            ("imagenet_mean", self.imagenet_mean),
            ("imagenet_std", self.imagenet_std),
        ):
            if len(values) != 3:
                raise ConfigError(
                    f"statistics.normalization.{name}: expected 3 channel values, got {len(values)}"
                )
        if any(s <= 0 for s in (*self.fallback_std, *self.imagenet_std)):
            raise ConfigError("statistics.normalization: standard deviations must be strictly positive")


@dataclass(frozen=True)
class StatisticsConfig:
    """Stage 4 configuration."""

    enabled: bool
    imbalance: ImbalanceConfig
    normalization: NormalizationConfig


# ---------------------------------------------------------------------------
# Stage 5 -- preprocessing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlackBorderRemovalConfig:
    """Tissue-bounding-box crop that removes camera black padding."""

    enabled: bool
    threshold: int
    blur_kernel: int
    min_area_ratio: float
    padding: int

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 255:
            raise ConfigError(
                f"preprocessing.black_border_removal.threshold: must lie in [0, 255], got {self.threshold}"
            )
        _check_odd("preprocessing.black_border_removal.blur_kernel", self.blur_kernel)
        _check_probability("preprocessing.black_border_removal.min_area_ratio", self.min_area_ratio)
        if self.padding < 0:
            raise ConfigError(
                f"preprocessing.black_border_removal.padding: must be >= 0, got {self.padding}"
            )


@dataclass(frozen=True)
class CircularCropConfig:
    """Circular mask matching the fundus disc geometry."""

    enabled: bool
    margin_ratio: float
    fill_value: int

    def __post_init__(self) -> None:
        _check_probability("preprocessing.circular_crop.margin_ratio", self.margin_ratio)
        if not 0 <= self.fill_value <= 255:
            raise ConfigError(
                f"preprocessing.circular_crop.fill_value: must lie in [0, 255], got {self.fill_value}"
            )


@dataclass(frozen=True)
class ResizeConfig:
    """Fixed-size resize policy.

    Separate interpolation choices for down- and up-scaling: ``area`` is the
    only correct anti-aliasing choice when reducing 4K acquisitions to a few
    hundred pixels, because naive bilinear downsampling aliases away
    microaneurysm-scale lesions.
    """

    enabled: bool
    interpolation_down: str
    interpolation_up: str
    keep_aspect_ratio: bool

    def __post_init__(self) -> None:
        valid = ["nearest", "linear", "cubic", "area", "lanczos"]
        _check_choice("preprocessing.resize.interpolation_down", self.interpolation_down, valid)
        _check_choice("preprocessing.resize.interpolation_up", self.interpolation_up, valid)


@dataclass(frozen=True)
class ClaheConfig:
    """Contrast Limited Adaptive Histogram Equalisation.

    EDA-driven, not paper-driven: the reference paper never mentions CLAHE.
    Disabled by default and intended to be justified by an A/B ablation.
    """

    enabled: bool
    clip_limit: float
    tile_grid_size: tuple[int, int]
    color_space: str

    def __post_init__(self) -> None:
        _check_positive("preprocessing.clahe.clip_limit", self.clip_limit)
        if any(t <= 0 for t in self.tile_grid_size):
            raise ConfigError(
                f"preprocessing.clahe.tile_grid_size: must be positive, got {list(self.tile_grid_size)}"
            )
        _check_choice("preprocessing.clahe.color_space", self.color_space, ["lab", "green", "hsv"])


@dataclass(frozen=True)
class IlluminationCorrectionConfig:
    """Local-average subtraction; ablation-only, disabled by default."""

    enabled: bool
    sigma: float
    weight: float

    def __post_init__(self) -> None:
        _check_positive("preprocessing.illumination_correction.sigma", self.sigma)


@dataclass(frozen=True)
class PreprocessingCacheConfig:
    """On-disk cache of the deterministic crop+resize result."""

    enabled: bool
    dir: str
    format: str

    def __post_init__(self) -> None:
        _check_choice("preprocessing.cache.format", self.format, ["png", "npy", "jpg"])


@dataclass(frozen=True)
class PreprocessingConfig:
    """Stage 5 configuration (deterministic, identical across all splits)."""

    image_size: int
    black_border_removal: BlackBorderRemovalConfig
    circular_crop: CircularCropConfig
    resize: ResizeConfig
    clahe: ClaheConfig
    illumination_correction: IlluminationCorrectionConfig
    cache: PreprocessingCacheConfig

    def __post_init__(self) -> None:
        _check_positive("preprocessing.image_size", self.image_size)
        if self.image_size < 224:
            raise ConfigError(
                "preprocessing.image_size: values below 224 px erase few-pixel Grade-1 "
                f"microaneurysms and are not supported, got {self.image_size}"
            )


# ---------------------------------------------------------------------------
# Stage 6 -- augmentation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProbabilisticTransformConfig:
    """Base for any augmentation with an ``enabled`` flag and a probability."""

    enabled: bool
    p: float

    def __post_init__(self) -> None:
        _check_probability(f"{type(self).__name__}.p", self.p)


@dataclass(frozen=True)
class FlipConfig(_ProbabilisticTransformConfig):
    """Horizontal or vertical flip."""


@dataclass(frozen=True)
class RotationConfig(_ProbabilisticTransformConfig):
    """Random rotation, applied after the circular crop."""

    limit: tuple[float, float]
    border_mode: str
    fill_value: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_range("augmentation.train.rotation.limit", self.limit)
        _check_choice(
            "augmentation.train.rotation.border_mode",
            self.border_mode,
            ["constant", "reflect", "replicate", "wrap"],
        )


@dataclass(frozen=True)
class MagnitudeConfig(_ProbabilisticTransformConfig):
    """Symmetric-magnitude photometric transform (brightness or contrast)."""

    limit: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_probability("augmentation.train.<photometric>.limit", self.limit)


@dataclass(frozen=True)
class ColorJitterConfig(_ProbabilisticTransformConfig):
    """Combined brightness/contrast/saturation/hue jitter.

    The hue range is deliberately narrow: measured hue is tightly clustered
    (23.57 deg +/- 14.65), so wide jitter leaves the physiologically plausible
    retinal colour gamut.
    """

    brightness: float
    contrast: float
    saturation: float
    hue: float

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("brightness", "contrast", "saturation", "hue"):
            _check_probability(f"augmentation.train.color_jitter.{name}", getattr(self, name))
        if self.hue > 0.1:
            raise ConfigError(
                f"augmentation.train.color_jitter.hue: values above 0.1 push images outside "
                f"the physiological retinal colour gamut, got {self.hue}"
            )


@dataclass(frozen=True)
class ScaleJitterConfig(_ProbabilisticTransformConfig):
    """Mild scale jitter applied before the fixed resize."""

    scale_limit: tuple[float, float]

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_range("augmentation.train.scale_jitter.scale_limit", self.scale_limit)


@dataclass(frozen=True)
class ConservativeCropConfig(_ProbabilisticTransformConfig):
    """Area-preserving random crop.

    The lower scale bound is validated at 0.8: anything more aggressive can
    remove the sole lesion evidence in Grade 1/3 images.
    """

    scale: tuple[float, float]

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_range("augmentation.train.conservative_crop.scale", self.scale)
        if self.scale[0] < 0.8:
            raise ConfigError(
                "augmentation.train.conservative_crop.scale: the lower bound must be >= 0.8 to "
                f"avoid discarding sparse lesion evidence, got {self.scale[0]}"
            )


@dataclass(frozen=True)
class GammaConfig(_ProbabilisticTransformConfig):
    """Gamma jitter expressed in Albumentations' percentage units."""

    gamma_limit: tuple[int, int]

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_range("augmentation.train.gamma.gamma_limit", self.gamma_limit)


@dataclass(frozen=True)
class BlurConfig(_ProbabilisticTransformConfig):
    """Mild Gaussian blur; kept infrequent because blur is hostile to
    few-pixel lesions."""

    sigma_limit: tuple[float, float]

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_range("augmentation.train.gaussian_blur.sigma_limit", self.sigma_limit)


@dataclass(frozen=True)
class NoiseConfig(_ProbabilisticTransformConfig):
    """Mild additive Gaussian noise matching measured sensor noise."""

    var_limit: tuple[float, float]

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_range("augmentation.train.gaussian_noise.var_limit", self.var_limit)


@dataclass(frozen=True)
class TrainAugmentationConfig:
    """Training-time augmentation policy (never applied to val/test)."""

    horizontal_flip: FlipConfig
    vertical_flip: FlipConfig
    rotation: RotationConfig
    random_brightness: MagnitudeConfig
    random_contrast: MagnitudeConfig
    color_jitter: ColorJitterConfig
    scale_jitter: ScaleJitterConfig
    conservative_crop: ConservativeCropConfig
    gamma: GammaConfig
    gaussian_blur: BlurConfig
    gaussian_noise: NoiseConfig


@dataclass(frozen=True)
class AugmentationConfig:
    """Stage 6 configuration."""

    enabled: bool
    train: TrainAugmentationConfig
    forbidden: list[str]

    def __post_init__(self) -> None:
        if not self.forbidden:
            raise ConfigError(
                "augmentation.forbidden: the forbidden list must not be empty; it is the "
                "machine-enforced guard against MixUp/CutMix/CutOut/RandomErasing."
            )


# ---------------------------------------------------------------------------
# Stage 7 -- dataloader / reproducibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShuffleConfig:
    """Per-split shuffling policy."""

    train: bool
    val: bool
    test: bool

    def get(self, split: str) -> bool:
        """Return the shuffle flag for ``split``."""
        try:
            return getattr(self, split)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ConfigError(f"dataloader.shuffle: unknown split {split!r}") from exc


@dataclass(frozen=True)
class DataLoaderConfig:
    """DataLoader factory parameters.

    Deliberately excludes samplers, class balancing, and distributed options:
    those are Training-milestone concerns.
    """

    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    drop_last: bool
    shuffle: ShuffleConfig

    def __post_init__(self) -> None:
        _check_positive("dataloader.batch_size", self.batch_size)
        _check_positive("dataloader.prefetch_factor", self.prefetch_factor)
        if self.num_workers < 0:
            raise ConfigError(f"dataloader.num_workers: must be >= 0, got {self.num_workers}")
        if self.num_workers == 0 and self.persistent_workers:
            raise ConfigError(
                "dataloader.persistent_workers: requires num_workers > 0 (PyTorch raises otherwise)"
            )


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Determinism switches applied by :mod:`src.utils.seed`."""

    deterministic: bool
    cudnn_benchmark: bool

    def __post_init__(self) -> None:
        if self.deterministic and self.cudnn_benchmark:
            raise ConfigError(
                "reproducibility: cudnn_benchmark must be false when deterministic is true "
                "(benchmark autotuning is nondeterministic)"
            )


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Fully validated data-preparation configuration.

    Attributes:
        profile: Evidence tier, ``"paper_faithful"`` or ``"eda_driven"``.
        seed: Global random seed shared by every stochastic component.
        dataset_name: Key selecting the active per-dataset override block.
    """

    paths: PathsConfig
    splits: SplitsConfig
    image: ImageConfig
    columns: ColumnsConfig
    classes: ClassesConfig
    seed: int
    profile: str
    outputs: OutputsConfig
    audit: AuditConfig
    cleaning: CleaningConfig
    splits_policy: SplitsPolicyConfig
    statistics: StatisticsConfig
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig
    dataloader: DataLoaderConfig
    reproducibility: ReproducibilityConfig
    dataset_name: str
    datasets: dict[str, Any] = field(default_factory=dict)
    project_root: Path = field(default=PROJECT_ROOT, compare=False, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        _check_choice("profile", self.profile, VALID_PROFILES)
        if self.seed < 0:
            raise ConfigError(f"seed: must be >= 0, got {self.seed}")
        if self.dataset_name not in self.datasets:
            raise ConfigError(
                f"dataset_name: {self.dataset_name!r} has no entry under 'datasets' "
                f"(available: {sorted(self.datasets)})"
            )

    # -- convenience -------------------------------------------------------

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve ``path`` against the project root if it is relative.

        Args:
            path: Absolute or repository-relative path.

        Returns:
            An absolute :class:`~pathlib.Path`.
        """
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.project_root / candidate)

    @property
    def config_hash(self) -> str:
        """Stable SHA-256 hash of the fully merged configuration."""
        return config_hash(self.raw)

    @property
    def preprocessing_hash(self) -> str:
        """Hash of the deterministic preprocessing section only.

        Used as the cache key for preprocessed images and for normalization
        statistics, so that changing an augmentation probability does not
        needlessly invalidate an expensive geometric cache.
        """
        subset = {
            "image_size": self.raw.get("preprocessing", {}).get("image_size"),
            "black_border_removal": self.raw.get("preprocessing", {}).get("black_border_removal"),
            "circular_crop": self.raw.get("preprocessing", {}).get("circular_crop"),
            "resize": self.raw.get("preprocessing", {}).get("resize"),
            "clahe": self.raw.get("preprocessing", {}).get("clahe"),
            "illumination_correction": self.raw.get("preprocessing", {}).get("illumination_correction"),
        }
        return config_hash(subset)

    def split_names(self) -> tuple[str, ...]:
        """Return the canonical split names in a fixed order."""
        return ("train", "val", "test")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either.

    Nested mappings are merged key-by-key; every other type (including lists)
    is replaced wholesale, which keeps override semantics predictable.

    Args:
        base: Base mapping.
        override: Mapping whose values take precedence.

    Returns:
        A new merged dictionary.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_overrides(overrides: Sequence[str] | None) -> dict[str, Any]:
    """Convert ``key.path=value`` CLI strings into a nested mapping.

    Values are parsed with the YAML scalar parser, so ``true``, ``3``, ``0.5``,
    ``null``, and ``[1, 2]`` all yield the expected Python types.

    Args:
        overrides: Sequence of ``dotted.key=value`` strings, or ``None``.

    Returns:
        A nested dictionary suitable for :func:`deep_merge`.

    Raises:
        ConfigError: If an entry does not contain ``=``.

    Example:
        >>> parse_overrides(["preprocessing.image_size=384"])
        {'preprocessing': {'image_size': 384}}
    """
    result: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r}: expected the form 'dotted.key=value'")
        key, _, raw_value = item.partition("=")
        value = yaml.safe_load(raw_value)
        cursor = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def config_hash(payload: Any, length: int = 12) -> str:
    """Return a short, stable SHA-256 hash of a JSON-serialisable payload.

    Keys are sorted so the hash depends on content only, never on YAML key
    order. The result is embedded in every generated report and used as a cache
    key, making stale artefacts detectable rather than silently reused.

    Args:
        payload: Any JSON-serialisable object.
        length: Number of leading hex characters to return.

    Returns:
        The truncated hexadecimal digest.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping.

    Raises:
        ConfigError: If the file is missing or does not contain a mapping.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"configuration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{file_path}: expected a YAML mapping at the document root")
    return dict(data)


def load_data_config(
    path: str | Path = "configs/data.yaml",
    overrides: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> DataConfig:
    """Load, merge, and validate the data-preparation configuration.

    Merge order (later wins):

    1. the YAML file,
    2. the profile preset (only for ``paper_faithful``),
    3. the ``datasets.<dataset_name>`` override block,
    4. explicit programmatic/CLI overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Nested mapping of overrides, typically from
            :func:`parse_overrides`.
        project_root: Root used to resolve relative paths; defaults to the
            repository root inferred from this file's location.

    Returns:
        A validated :class:`DataConfig`.

    Raises:
        ConfigError: If the configuration is missing, malformed, or invalid.
    """
    root = project_root or PROJECT_ROOT
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path

    raw = load_yaml(config_path)

    # The effective profile and dataset may themselves be selected by an
    # override, so resolve them from a provisional merge before deciding which
    # preset and dataset block to apply.
    provisional = deep_merge(raw, overrides or {})
    profile = provisional.get("profile", "eda_driven")
    dataset_name = provisional.get("dataset_name")

    # 2. Profile preset. Applied before dataset overrides so a dataset block can
    #    still opt back into a step deliberately, and before CLI overrides so the
    #    command line always has the final word.
    if profile == "paper_faithful":
        raw = deep_merge(raw, PAPER_FAITHFUL_OVERRIDES)

    # 3. Per-dataset override block.
    dataset_overrides = (raw.get("datasets") or {}).get(dataset_name) or {}
    if dataset_overrides:
        raw = deep_merge(raw, dataset_overrides)

    # 4. Explicit overrides.
    if overrides:
        raw = deep_merge(raw, overrides)

    config = _from_mapping(DataConfig, {**raw, "project_root": root, "raw": raw})
    return typing.cast(DataConfig, config)
