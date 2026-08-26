"""Stage 4 -- Dataset statistics.

Two products, deliberately kept in one module because both are descriptive
summaries of the cleaned manifest:

1. **Class-imbalance analysis.** Counts, proportions, majority-to-minority
   ratio, Shannon entropy, a normalized imbalance index, and a chi-square test
   of split homogeneity. *Analysis only* -- this milestone implements no
   sampler, no oversampling, and no class balancing. Weighting strategies are
   exposed as pure functions plus an :class:`ImbalanceStrategy` protocol so the
   Training milestone can consume them without any change here. Resampling is
   entangled with the loss function and the ordinal objective, so it belongs
   where it can be validated against quadratic-weighted kappa.

2. **Normalization statistics.** Per-channel mean and standard deviation
   computed with a streaming Welford accumulator from the **training split
   only**, over **included** images, **after** the deterministic geometric
   preprocessing and **before** augmentation.

Why normalization statistics are computed rather than hardcoded
---------------------------------------------------------------
The EDA measured RGB means of 108.26 / 57.09 / 18.30 on *raw* images that still
contain black padding. Black-border removal and circular cropping change the
pixel population materially, so those numbers become biased low the moment the
geometric steps are enabled: they are a fallback, never the default. Computing
on the training split alone also avoids leaking evaluation-set statistics into
the training-time scaling. Results are cached under the preprocessing-config
hash, so they are recomputed exactly when -- and only when -- the geometry
changes. An ``imagenet`` mode is retained because the Swin backbone will be
initialised from ImageNet-pretrained weights, making input-distribution
matching a legitimate competing hypothesis worth ablating.

This module never imports the preprocessing package: the transform is injected
as a callable, which keeps the dependency graph acyclic and makes the routine
testable with a trivial identity transform.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.utils.config import DataConfig
from src.utils.helpers import ensure_dir, read_image_rgb, read_json, write_json
from src.utils.logger import get_logger, log_duration, log_section

__all__ = [
    "ChannelStatsAccumulator",
    "NormalizationStats",
    "ImbalanceStrategy",
    "class_distribution",
    "imbalance_metrics",
    "chi_square_homogeneity",
    "compute_class_weights",
    "compute_normalization_stats",
    "resolve_normalization_stats",
    "DatasetStatistics",
]

logger = get_logger(__name__)

#: Weighting strategies exposed for the Training milestone. None are applied here.
WEIGHT_STRATEGIES: tuple[str, ...] = ("inverse", "inverse_sqrt", "effective_number", "balanced")


# ---------------------------------------------------------------------------
# Interfaces reserved for the Training milestone
# ---------------------------------------------------------------------------


@runtime_checkable
class ImbalanceStrategy(Protocol):
    """Interface a future sampling or weighting strategy must satisfy.

    Declaring the contract here -- without implementing it -- lets the Training
    milestone add a ``WeightedRandomSampler``, a class-balanced batch sampler,
    or an LDAM-style loss weighting without touching any data-preparation code.

    Nothing in this milestone instantiates or calls an implementation.
    """

    def sample_weights(self, labels: Sequence[int]) -> np.ndarray:
        """Return a per-sample weight for each label."""
        ...


# ---------------------------------------------------------------------------
# Class distribution and imbalance
# ---------------------------------------------------------------------------


def class_distribution(frame: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    """Build a per-split class-count table.

    Args:
        frame: Manifest containing ``split`` and ``label`` columns.
        num_classes: Number of ordinal grades; missing grades are reported as
            zero rather than omitted, so downstream tables are always aligned.

    Returns:
        A DataFrame indexed by split with one column per grade plus ``total``.
    """
    grades = list(range(num_classes))
    if frame.empty:
        return pd.DataFrame(columns=[*grades, "total"])

    labelled = frame[frame["label"].notna()].copy()
    labelled["label"] = labelled["label"].astype(int)
    table = (
        labelled.pivot_table(index="split", columns="label", values="id_code", aggfunc="count")
        .reindex(columns=grades, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    table["total"] = table[grades].sum(axis=1)
    overall = table[grades].sum(axis=0)
    table.loc["overall"] = [*overall.tolist(), int(overall.sum())]
    return table


def imbalance_metrics(counts: Sequence[int]) -> dict[str, float]:
    """Quantify how imbalanced a label distribution is.

    Three complementary measures are reported because each answers a different
    question: the ratio describes the worst case, entropy describes the whole
    distribution, and the normalized index makes datasets comparable.

    Args:
        counts: Per-class counts.

    Returns:
        Mapping with ``majority_minority_ratio``, ``entropy_bits``,
        ``max_entropy_bits``, ``normalized_imbalance_index`` (0 = balanced,
        1 = maximally imbalanced), ``gini``, and ``effective_num_classes``.

    Example:
        >>> metrics = imbalance_metrics([1805, 370, 999, 193, 295])
        >>> round(metrics["majority_minority_ratio"], 2)
        9.35
    """
    values = np.asarray(list(counts), dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return {}

    non_zero = values[values > 0]
    proportions = non_zero / total
    entropy = float(-(proportions * np.log2(proportions)).sum())
    max_entropy = float(math.log2(len(values))) if len(values) > 1 else 0.0
    minority = float(non_zero.min())
    majority = float(values.max())

    return {
        "majority_minority_ratio": majority / minority if minority else float("inf"),
        "entropy_bits": entropy,
        "max_entropy_bits": max_entropy,
        "normalized_imbalance_index": 1.0 - (entropy / max_entropy) if max_entropy else 0.0,
        "gini": float(1.0 - np.square(values / total).sum()),
        "effective_num_classes": float(2.0**entropy),
    }


def chi_square_homogeneity(table: pd.DataFrame) -> dict[str, Any]:
    """Test whether class proportions are homogeneous across splits.

    A non-significant result means the imbalance is a property of the dataset
    rather than an artefact of partitioning -- the EDA reported
    ``chi2 = 7.083, dof = 8, p = 0.5277`` for the shipped split, and this test
    re-checks that property after cleaning has removed rows.

    Args:
        table: Contingency table of counts (splits as rows, grades as columns).

    Returns:
        Mapping with ``statistic``, ``dof``, ``p_value``, and ``interpretation``.
        ``p_value`` is ``None`` when SciPy is unavailable; the statistic is
        still computed from NumPy so the check never silently disappears.
    """
    observed = np.asarray(table.values, dtype=np.float64)
    if observed.ndim != 2 or observed.size == 0 or observed.sum() == 0:
        return {}

    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    statistic = float(terms.sum())
    dof = int((observed.shape[0] - 1) * (observed.shape[1] - 1))

    p_value: float | None
    try:
        from scipy import stats

        p_value = float(stats.chi2.sf(statistic, dof)) if dof > 0 else None
    except ImportError:  # pragma: no cover - SciPy ships with scikit-learn
        p_value = None
        logger.warning("SciPy unavailable: chi-square p-value not computed")

    interpretation = "unknown"
    if p_value is not None:
        interpretation = (
            "class proportions differ significantly across splits (p < 0.05)"
            if p_value < 0.05
            else "class proportions are homogeneous across splits (p >= 0.05)"
        )

    return {"statistic": statistic, "dof": dof, "p_value": p_value, "interpretation": interpretation}


def compute_class_weights(
    counts: Sequence[int], strategy: str = "inverse", beta: float = 0.9999
) -> np.ndarray:
    """Compute class weights **without applying them anywhere**.

    Provided as a pure function so the Training milestone can pass the result to
    a loss function or a sampler. Nothing in the data-preparation pipeline calls
    this function on the model's behalf.

    Strategies:
        ``inverse``
            ``total / (num_classes * count)`` -- the classic reweighting.
        ``inverse_sqrt``
            Square-root damped inverse frequency; less aggressive, which suits
            an ordinal task where extreme weights destabilise the ordinal head.
        ``effective_number``
            Cui et al. (2019): ``(1 - beta) / (1 - beta**count)``, accounting for
            information overlap among near-duplicate samples -- relevant here,
            since the EDA found a substantial near-duplicate signal.
        ``balanced``
            scikit-learn's ``class_weight="balanced"`` convention.

    Args:
        counts: Per-class counts, ordered by grade.
        strategy: One of :data:`WEIGHT_STRATEGIES`.
        beta: Hyper-parameter of the ``effective_number`` strategy.

    Returns:
        Weights normalized to a mean of 1.0 so that changing strategy does not
        implicitly rescale the learning rate.

    Raises:
        ValueError: If the strategy is unknown, counts are empty, or beta is
            outside ``(0, 1)``.

    Example:
        >>> weights = compute_class_weights([100, 10], strategy="inverse")
        >>> bool(weights[1] > weights[0])
        True
    """
    values = np.asarray(list(counts), dtype=np.float64)
    if values.size == 0:
        raise ValueError("counts must not be empty")
    if strategy not in WEIGHT_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {list(WEIGHT_STRATEGIES)}")
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must lie in (0, 1), got {beta}")

    safe = np.where(values > 0, values, np.nan)
    total = float(np.nansum(safe))
    num_classes = float(values.size)

    if strategy == "inverse":
        weights = total / (num_classes * safe)
    elif strategy == "inverse_sqrt":
        weights = np.sqrt(total / (num_classes * safe))
    elif strategy == "effective_number":
        effective = (1.0 - np.power(beta, safe)) / (1.0 - beta)
        weights = 1.0 / effective
    else:  # "balanced"
        weights = total / (num_classes * safe)

    # Absent classes receive zero weight rather than infinity.
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0)
    mean = float(weights[weights > 0].mean()) if np.any(weights > 0) else 1.0
    return weights / mean


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------


class ChannelStatsAccumulator:
    """Streaming per-channel mean/variance accumulator (Welford / Chan et al.).

    Constant memory regardless of dataset size and numerically stable for the
    millions of pixels a full training split contributes -- a naive
    sum-of-squares accumulator loses precision well before that point.

    Example:
        >>> accumulator = ChannelStatsAccumulator(channels=3)
        >>> _ = accumulator.update(np.zeros((4, 4, 3), dtype=np.uint8))
        >>> accumulator.mean.tolist()
        [0.0, 0.0, 0.0]
    """

    def __init__(self, channels: int = 3) -> None:
        self.channels = channels
        self._count = 0.0
        self._mean = np.zeros(channels, dtype=np.float64)
        self._m2 = np.zeros(channels, dtype=np.float64)
        self.images = 0

    def update(self, image: np.ndarray) -> ChannelStatsAccumulator:
        """Fold one image into the running statistics.

        Args:
            image: ``H x W x C`` array. ``uint8`` inputs are scaled to ``[0, 1]``
                so cached statistics are independent of the input dtype.

        Returns:
            ``self``, to allow chaining.

        Raises:
            ValueError: If the channel count does not match.
        """
        if image.ndim != 3 or image.shape[2] != self.channels:
            raise ValueError(f"expected an H x W x {self.channels} image, got shape {image.shape}")

        pixels = image.reshape(-1, self.channels).astype(np.float64)
        if image.dtype == np.uint8:
            pixels /= 255.0

        batch_count = float(pixels.shape[0])
        batch_mean = pixels.mean(axis=0)
        batch_m2 = ((pixels - batch_mean) ** 2).sum(axis=0)

        delta = batch_mean - self._mean
        total = self._count + batch_count
        self._mean += delta * (batch_count / total)
        self._m2 += batch_m2 + (delta**2) * (self._count * batch_count / total)
        self._count = total
        self.images += 1
        return self

    @property
    def mean(self) -> np.ndarray:
        """Per-channel mean in ``[0, 1]``."""
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        """Per-channel population standard deviation in ``[0, 1]``."""
        if self._count <= 1:
            return np.zeros(self.channels, dtype=np.float64)
        return np.sqrt(self._m2 / self._count)

    @property
    def pixel_count(self) -> float:
        """Number of pixels folded in so far."""
        return self._count


@dataclass(frozen=True)
class NormalizationStats:
    """Per-channel normalization statistics plus their provenance."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    source: str
    images: int = 0
    pixels: float = 0.0
    preprocessing_hash: str = ""
    split: str = "train"
    computed_at_utc: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Write the statistics to JSON."""
        return write_json(path, self.as_dict())

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        """Read statistics previously written by :meth:`save`."""
        payload = read_json(path)
        return cls(
            mean=tuple(float(v) for v in payload["mean"]),  # type: ignore[arg-type]
            std=tuple(float(v) for v in payload["std"]),  # type: ignore[arg-type]
            source=str(payload.get("source", "cache")),
            images=int(payload.get("images", 0)),
            pixels=float(payload.get("pixels", 0.0)),
            preprocessing_hash=str(payload.get("preprocessing_hash", "")),
            split=str(payload.get("split", "train")),
            computed_at_utc=str(payload.get("computed_at_utc", "")),
        )


def compute_normalization_stats(
    paths: Iterable[str | Path],
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    preprocessing_hash: str = "",
    max_images: int | None = None,
    rng: np.random.Generator | None = None,
    split: str = "train",
) -> NormalizationStats:
    """Compute per-channel statistics over a set of images.

    Args:
        paths: Image paths (already filtered to the training split and to
            included rows by the caller).
        transform: Deterministic geometric preprocessing applied before
            measurement. Injected rather than imported so this module stays
            independent of the preprocessing package.
        preprocessing_hash: Hash of the geometry configuration, stored for
            cache invalidation.
        max_images: Optional cap; a reproducible random subset is used when the
            cap is smaller than the corpus.
        rng: Generator used for subsampling; pass a seeded one for reproducibility.
        split: Name of the split the statistics were computed on.

    Returns:
        The computed :class:`NormalizationStats`.

    Raises:
        ValueError: If no image could be read.
    """
    candidates = [str(path) for path in paths]
    if max_images is not None and 0 < max_images < len(candidates):
        generator = rng or np.random.default_rng(0)
        selection = generator.choice(len(candidates), size=max_images, replace=False)
        candidates = [candidates[int(index)] for index in sorted(selection)]

    accumulator = ChannelStatsAccumulator(channels=3)
    failures = 0
    for path in candidates:
        image = read_image_rgb(path)
        if image is None:
            failures += 1
            continue
        if transform is not None:
            image = transform(image)
        accumulator.update(image)

    if accumulator.images == 0:
        raise ValueError("normalization statistics require at least one readable image")
    if failures:
        logger.warning("%d image(s) could not be read while computing statistics", failures)

    return NormalizationStats(
        mean=tuple(float(v) for v in accumulator.mean),  # type: ignore[arg-type]
        std=tuple(float(v) for v in accumulator.std),  # type: ignore[arg-type]
        source="computed",
        images=accumulator.images,
        pixels=accumulator.pixel_count,
        preprocessing_hash=preprocessing_hash,
        split=split,
        computed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def resolve_normalization_stats(
    config: DataConfig,
    manifest: pd.DataFrame | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    force: bool = False,
) -> NormalizationStats:
    """Return the statistics dictated by ``statistics.normalization.mode``.

    ``auto`` uses the cache when its ``preprocessing_hash`` matches the current
    geometry configuration, otherwise recomputes from the training split and
    rewrites the cache. ``config`` and ``imagenet`` short-circuit to fixed
    values and never touch the filesystem.

    Args:
        config: Validated data configuration.
        manifest: Clean manifest; required for ``auto`` when no valid cache
            exists.
        transform: Deterministic preprocessing callable (see
            :func:`compute_normalization_stats`).
        force: Ignore an existing cache.

    Returns:
        The resolved :class:`NormalizationStats`.

    Raises:
        ValueError: If ``auto`` is requested without a manifest and without a
            valid cache.
    """
    settings = config.statistics.normalization

    if settings.mode == "imagenet":
        return NormalizationStats(
            mean=tuple(settings.imagenet_mean),  # type: ignore[arg-type]
            std=tuple(settings.imagenet_std),  # type: ignore[arg-type]
            source="imagenet",
        )
    if settings.mode == "config":
        return NormalizationStats(
            mean=tuple(settings.fallback_mean),  # type: ignore[arg-type]
            std=tuple(settings.fallback_std),  # type: ignore[arg-type]
            source="config",
        )

    cache_path = config.resolve_path(settings.cache_path)
    expected_hash = config.preprocessing_hash
    if not force and cache_path.is_file():
        cached = NormalizationStats.load(cache_path)
        if cached.preprocessing_hash == expected_hash:
            logger.info("using cached normalization statistics from %s", cache_path)
            return cached
        logger.info(
            "cached normalization statistics are stale (%s != %s); recomputing",
            cached.preprocessing_hash,
            expected_hash,
        )

    if manifest is None:
        raise ValueError(
            "normalization mode 'auto' requires the clean manifest when no valid cache exists"
        )

    training = manifest[
        (manifest["split"] == "train") & manifest["included"] & manifest["readable"]
    ]
    if training.empty:
        raise ValueError("no included, readable training images available for normalization statistics")

    from src.utils.seed import make_generator

    with log_duration(logger, f"computing normalization statistics from {len(training)} training images"):
        stats = compute_normalization_stats(
            training["path"].tolist(),
            transform=transform,
            preprocessing_hash=expected_hash,
            max_images=settings.max_images,
            rng=make_generator(config.seed, "normalization_stats"),
            split="train",
        )

    ensure_dir(cache_path.parent)
    stats.save(cache_path)
    logger.info("normalization statistics cached at %s", cache_path)
    return stats


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class StatisticsResult:
    """Class distribution table, normalization statistics, and the report."""

    distribution: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)
    normalization: NormalizationStats | None = None

    def save(self, report_path: str | Path, distribution_path: str | Path) -> None:
        """Persist the report (JSON) and the class-distribution table (CSV)."""
        ensure_dir(Path(report_path).parent)
        self.distribution.to_csv(distribution_path, index=True)
        write_json(report_path, self.report)
        logger.info("statistics report written to %s", report_path)
        logger.info("class distribution written to %s", distribution_path)


class DatasetStatistics:
    """Produce the imbalance analysis and (optionally) normalization statistics.

    Args:
        config: Validated data configuration.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def run(
        self,
        manifest: pd.DataFrame,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        compute_normalization: bool = True,
        force: bool = False,
    ) -> StatisticsResult:
        """Compute and persist dataset statistics.

        Args:
            manifest: Clean manifest from :mod:`src.data.cleaning`.
            transform: Deterministic preprocessing callable used for
                normalization statistics.
            compute_normalization: Set to ``False`` to skip the pixel pass.
            force: Recompute normalization statistics even if cached.

        Returns:
            The :class:`StatisticsResult`, already written to disk.
        """
        log_section(logger, "Stage 4 / Dataset statistics (analysis only, no balancing)")
        started = dt.datetime.now(dt.timezone.utc)

        included = manifest[manifest["included"]] if "included" in manifest else manifest
        num_classes = self.config.classes.num_classes
        table = class_distribution(included, num_classes)

        grades = list(range(num_classes))
        overall_counts = [int(table.loc["overall", grade]) for grade in grades] if not table.empty else []
        split_table = table.drop(index="overall", errors="ignore")[grades] if not table.empty else table

        report: dict[str, Any] = {
            "stage": "statistics",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_hash": self.config.config_hash,
            "profile": self.config.profile,
            "policy": {
                "balancing_applied": False,
                "sampler_implemented": False,
                "note": (
                    "class imbalance is analysed and reported here; weighting and sampling "
                    "are deferred to the Training milestone, where they can be validated "
                    "against quadratic-weighted kappa"
                ),
            },
            "counts": {
                "per_split": {
                    str(split): {str(grade): int(row[grade]) for grade in grades}
                    for split, row in split_table.iterrows()
                },
                "overall": {str(grade): count for grade, count in zip(grades, overall_counts)},
                "total_images": int(sum(overall_counts)),
            },
            "class_names": {str(k): v for k, v in self.config.classes.names.items()},
            "imbalance": imbalance_metrics(overall_counts) if overall_counts else {},
            "split_homogeneity_chi_square": chi_square_homogeneity(split_table)
            if len(split_table) > 1
            else {},
            "reference_class_weights": self._reference_weights(overall_counts),
        }

        normalization: NormalizationStats | None = None
        if compute_normalization:
            try:
                normalization = resolve_normalization_stats(
                    self.config, manifest=manifest, transform=transform, force=force
                )
                report["normalization"] = normalization.as_dict()
            except (ValueError, OSError) as exc:
                logger.warning("normalization statistics unavailable: %s", exc)
                report["normalization"] = {"error": str(exc)}

        report["duration_seconds"] = round(
            (dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 2
        )

        result = StatisticsResult(distribution=table, report=report, normalization=normalization)
        result.save(
            self.config.resolve_path(self.config.outputs.statistics_report),
            self.config.resolve_path(self.config.outputs.class_distribution),
        )
        if self.config.statistics.imbalance.save_plot:
            self._save_plot(table, grades)
        self._log_summary(report)
        return result

    # -- helpers -----------------------------------------------------------

    def _reference_weights(self, counts: Sequence[int]) -> dict[str, list[float]]:
        """Compute reference weights for every configured strategy.

        These are reported for transparency and future reuse; nothing in this
        pipeline consumes them.
        """
        if not counts:
            return {}
        beta = self.config.statistics.imbalance.effective_number_beta
        weights: dict[str, list[float]] = {}
        for strategy in self.config.statistics.imbalance.weight_strategies:
            weights[strategy] = [
                round(float(w), 6) for w in compute_class_weights(counts, strategy=strategy, beta=beta)
            ]
        return weights

    def _save_plot(self, table: pd.DataFrame, grades: Sequence[int]) -> None:
        """Render a per-split class-distribution bar chart."""
        if table.empty:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless: the pipeline must run on a server
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - matplotlib is a declared dependency
            logger.warning("matplotlib unavailable; skipping the class-distribution plot")
            return

        plot_table = table.drop(index="overall", errors="ignore")[list(grades)]
        axis = plot_table.T.plot(kind="bar", figsize=(9, 5))
        axis.set_xlabel("DR grade")
        axis.set_ylabel("images")
        axis.set_title("Class distribution per split (after cleaning)")
        axis.legend(title="split")
        figure = axis.get_figure()
        figure.tight_layout()

        destination = self.config.resolve_path(self.config.outputs.root) / "class_distribution.png"
        ensure_dir(destination.parent)
        figure.savefig(destination, dpi=150)
        plt.close(figure)
        logger.info("class-distribution plot written to %s", destination)

    @staticmethod
    def _log_summary(report: Mapping[str, Any]) -> None:
        """Log the headline imbalance numbers."""
        imbalance = report.get("imbalance", {})
        if imbalance:
            logger.info(
                "imbalance: ratio %.2f:1 | entropy %.3f/%.3f bits | index %.3f",
                imbalance.get("majority_minority_ratio", float("nan")),
                imbalance.get("entropy_bits", float("nan")),
                imbalance.get("max_entropy_bits", float("nan")),
                imbalance.get("normalized_imbalance_index", float("nan")),
            )
        chi = report.get("split_homogeneity_chi_square", {})
        if chi:
            logger.info(
                "split homogeneity: chi2 %.3f (dof %s, p %s)",
                chi.get("statistic", float("nan")),
                chi.get("dof"),
                chi.get("p_value"),
            )
