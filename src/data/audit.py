"""Stage 1 -- Dataset Audit.

A full-corpus, **read-only** measurement pass that produces one authoritative
manifest. Every later stage (cleaning, split verification, statistics) consumes
that manifest instead of re-reading pixels, which is what makes the rest of the
pipeline cheap enough to iterate on.

Why the audit exists as a separate stage
----------------------------------------
* **Measurement is separated from judgement.** The audit states what *is*
  (dimensions, MD5, brightness); the cleaning stage decides what to *do*.
  Changing a cleaning threshold then costs seconds, not a full re-decode of
  3,662 images, several of which are 4288x2848.
* **The EDA sampled; the audit does not.** The EDA's duplicate, quality, and
  colour findings come from samples of 1,500 / 3,000 / 600 / 400 images. The
  Data Preparation report explicitly requires cross-split duplicates to be
  identified across the *full* dataset before any exclusion decision is made.
* **Provenance.** The manifest is a diffable artefact stamped with the config
  hash, so two runs that disagree can be compared instead of argued about.

What is measured per image
--------------------------
Existence, byte size, decodability, container format and colour mode, width,
height, aspect ratio, MD5 of the raw bytes (authoritative duplicate key),
dHash/pHash (investigation only), and quality metrics: brightness, contrast,
raw and resolution-normalized sharpness, Immerkaer noise sigma, black-padding
ratio, and the fundus-disc bounding box.

Nothing is written to ``data/raw`` and no file is ever modified or deleted.

Example
-------
>>> from src.utils.config import load_data_config
>>> from src.data.audit import DatasetAuditor
>>> config = load_data_config()                       # doctest: +SKIP
>>> result = DatasetAuditor(config).run()             # doctest: +SKIP
>>> result.report["totals"]["images"]                 # doctest: +SKIP
3662
"""

from __future__ import annotations

import concurrent.futures as futures
import datetime as dt
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import DataConfig
from src.utils.helpers import (
    black_padding_ratio,
    dhash,
    downscale_long_side,
    estimate_noise_sigma,
    ensure_dir,
    image_brightness,
    image_contrast,
    laplacian_variance,
    md5_file,
    normalized_sharpness,
    phash,
    read_image_rgb,
    resolve_column,
    tissue_bounding_box,
    write_json,
)
from src.utils.logger import get_logger, log_duration, log_section

__all__ = [
    "AuditRecord",
    "AuditOptions",
    "AuditResult",
    "DatasetAuditor",
    "load_split_manifests",
    "measure_image",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditOptions:
    """Picklable subset of the configuration needed by a worker process.

    Passing a small frozen dataclass (rather than the whole ``DataConfig``)
    keeps the per-task pickling cost negligible when the audit fans out over a
    process pool.
    """

    full_decode: bool
    metric_long_side: int
    compute_md5: bool
    compute_perceptual_hash: bool
    compute_quality_metrics: bool
    tissue_threshold: int
    dhash_size: int
    phash_size: int
    phash_image_size: int

    @classmethod
    def from_config(cls, config: DataConfig) -> AuditOptions:
        """Build options from a validated :class:`~src.utils.config.DataConfig`."""
        audit = config.audit
        return cls(
            full_decode=audit.full_decode,
            metric_long_side=audit.metric_long_side,
            compute_md5=audit.compute_md5,
            compute_perceptual_hash=audit.compute_perceptual_hash,
            compute_quality_metrics=audit.compute_quality_metrics,
            tissue_threshold=audit.tissue_threshold,
            dhash_size=audit.perceptual_hash.dhash_size,
            phash_size=audit.perceptual_hash.phash_size,
            phash_image_size=audit.perceptual_hash.phash_image_size,
        )


@dataclass
class AuditRecord:
    """One row of the audit manifest: everything measured about one image.

    Attributes are deliberately flat and JSON/CSV friendly so the manifest can
    be inspected with any tool, including a spreadsheet, without a custom
    reader.
    """

    # -- identity -------------------------------------------------------
    split: str
    id_code: str
    path: str
    label: int | None = None

    # -- integrity ------------------------------------------------------
    exists: bool = False
    size_bytes: int = 0
    mtime_ns: int = 0
    readable: bool = False
    error: str = ""

    # -- container ------------------------------------------------------
    image_format: str = ""
    color_mode: str = ""
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0

    # -- hashes ---------------------------------------------------------
    md5: str = ""
    dhash: str = ""
    phash: str = ""

    # -- quality --------------------------------------------------------
    brightness: float = float("nan")
    contrast: float = float("nan")
    laplacian_var: float = float("nan")
    sharpness_norm: float = float("nan")
    noise_sigma: float = float("nan")
    padding_ratio: float = float("nan")
    disc_x: int = -1
    disc_y: int = -1
    disc_w: int = -1
    disc_h: int = -1

    # -- label consistency ---------------------------------------------
    label_present: bool = False
    label_in_range: bool = False

    @classmethod
    def column_names(cls) -> list[str]:
        """Return the manifest column order."""
        return [f.name for f in fields(cls)]

    def to_row(self) -> dict[str, Any]:
        """Return the record as a plain dictionary suitable for CSV writing."""
        return asdict(self)


@dataclass
class AuditResult:
    """Outcome of an audit run: the per-image records plus a summary report."""

    records: list[AuditRecord]
    report: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return the records as a :class:`pandas.DataFrame`."""
        frame = pd.DataFrame([record.to_row() for record in self.records])
        if frame.empty:
            frame = pd.DataFrame(columns=AuditRecord.column_names())
        return frame[AuditRecord.column_names()]

    def save(self, manifest_path: str | Path, report_path: str | Path) -> None:
        """Persist the manifest (CSV) and the summary report (JSON).

        Args:
            manifest_path: Destination CSV path.
            report_path: Destination JSON path.
        """
        ensure_dir(Path(manifest_path).parent)
        # ``to_csv`` is used rather than the generic helper because the manifest
        # is a DataFrame-shaped artefact that pandas round-trips exactly.
        self.to_frame().to_csv(manifest_path, index=False)
        write_json(report_path, self.report)
        logger.info("audit manifest written to %s", manifest_path)
        logger.info("audit report written to %s", report_path)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_split_manifests(config: DataConfig) -> pd.DataFrame:
    """Read the split CSVs into a single long-format table.

    Args:
        config: Validated data configuration.

    Returns:
        A DataFrame with columns ``split``, ``id_code``, ``label``, ``path``.
        Missing or unparsable labels are represented as ``NaN`` rather than
        being dropped, because a missing label is a finding the audit must
        report, not an error it should hide.

    Raises:
        FileNotFoundError: If a split CSV is missing.
        KeyError: If the id column cannot be resolved from the candidates.
    """
    frames: list[pd.DataFrame] = []
    for split, paths in config.splits.as_dict().items():
        csv_path = config.resolve_path(paths.csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"split CSV not found for '{split}': {csv_path}")

        table = pd.read_csv(csv_path)
        id_column = resolve_column(list(table.columns), config.columns.id_candidates, "id")
        try:
            label_column = resolve_column(list(table.columns), config.columns.label_candidates, "label")
        except KeyError:
            # A label-free split (for example a Kaggle submission set) is a
            # valid input; the audit records the absence instead of failing.
            label_column = None
            logger.warning("split '%s' has no label column; labels will be recorded as missing", split)

        image_dir = config.resolve_path(paths.image_dir)
        extension = config.image.extension
        frame = pd.DataFrame(
            {
                "split": split,
                "id_code": table[id_column].astype(str).str.strip(),
                "label": (
                    pd.to_numeric(table[label_column], errors="coerce")
                    if label_column is not None
                    else np.nan
                ),
            }
        )
        frame["path"] = [str(image_dir / f"{code}{extension}") for code in frame["id_code"]]
        frames.append(frame)
        logger.info("loaded %d rows for split '%s' from %s", len(frame), split, csv_path.name)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-image measurement (module level so it is picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def measure_image(path: str, options: AuditOptions) -> dict[str, Any]:
    """Measure a single image file.

    Never raises for data problems: an unreadable or corrupted file yields a
    record with ``readable=False`` and an ``error`` message, because a corrupt
    file is a finding to be reported, not a reason to abort a two-hour audit.

    Args:
        path: Absolute path to the image file.
        options: Measurement options.

    Returns:
        A dictionary of measured fields, ready to be merged into an
        :class:`AuditRecord`.
    """
    result: dict[str, Any] = {"path": path}
    file_path = Path(path)

    if not file_path.is_file():
        result.update(exists=False, readable=False, error="file not found")
        return result

    stat = file_path.stat()
    result.update(exists=True, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if stat.st_size == 0:
        result.update(readable=False, error="zero-byte file")
        return result

    # Container metadata comes from a header-only read; it is cheap and tells us
    # the declared format/mode even when full decoding later fails.
    try:
        from PIL import Image

        with Image.open(file_path) as handle:
            result["image_format"] = handle.format or ""
            result["color_mode"] = handle.mode or ""
            width, height = handle.size
            result["width"], result["height"] = int(width), int(height)
    except Exception as exc:  # noqa: BLE001 - any decoder error is a finding
        result.update(readable=False, error=f"header read failed: {exc}")
        return result

    if options.compute_md5:
        try:
            result["md5"] = md5_file(file_path)
        except OSError as exc:
            result.update(readable=False, error=f"md5 failed: {exc}")
            return result

    if not options.full_decode:
        result["readable"] = True
        result["aspect_ratio"] = _aspect_ratio(result["width"], result["height"])
        return result

    image = read_image_rgb(file_path)
    if image is None:
        result.update(readable=False, error="decode failed")
        return result

    result["readable"] = True
    result["height"], result["width"] = int(image.shape[0]), int(image.shape[1])
    result["aspect_ratio"] = _aspect_ratio(result["width"], result["height"])

    # Metrics are computed on a bounded proxy: this keeps a full-corpus pass
    # tractable while preserving the relative ordering of every metric.
    proxy = downscale_long_side(image, options.metric_long_side)

    if options.compute_perceptual_hash:
        result["dhash"] = dhash(proxy, hash_size=options.dhash_size)
        result["phash"] = phash(proxy, hash_size=options.phash_size, image_size=options.phash_image_size)

    if options.compute_quality_metrics:
        result["brightness"] = image_brightness(proxy)
        result["contrast"] = image_contrast(proxy)
        result["laplacian_var"] = laplacian_variance(proxy)
        # Resolution-normalized: raw Laplacian variance correlates r ~= -0.80
        # with width, and width correlates with grade, so the raw value must
        # never be thresholded.
        result["sharpness_norm"] = normalized_sharpness(proxy)
        result["noise_sigma"] = estimate_noise_sigma(proxy)
        result["padding_ratio"] = black_padding_ratio(proxy, threshold=options.tissue_threshold)
        box = tissue_bounding_box(proxy, threshold=options.tissue_threshold)
        if box is not None:
            scale = max(image.shape[:2]) / max(proxy.shape[:2])
            result["disc_x"] = int(round(box[0] * scale))
            result["disc_y"] = int(round(box[1] * scale))
            result["disc_w"] = int(round(box[2] * scale))
            result["disc_h"] = int(round(box[3] * scale))

    return result


def _aspect_ratio(width: int, height: int) -> float:
    """Return ``width / height``, or ``0.0`` for a degenerate image."""
    return float(width) / float(height) if height else 0.0


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


class DatasetAuditor:
    """Run the read-only, full-corpus dataset audit.

    Args:
        config: Validated data configuration.

    Example:
        >>> auditor = DatasetAuditor(config)          # doctest: +SKIP
        >>> result = auditor.run(force=False)         # doctest: +SKIP
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.options = AuditOptions.from_config(config)

    # -- public API --------------------------------------------------------

    def run(self, force: bool = False) -> AuditResult:
        """Execute the audit and persist its artefacts.

        Args:
            force: Ignore any cached manifest and re-measure every image.

        Returns:
            The :class:`AuditResult`, already written to disk.
        """
        log_section(logger, "Stage 1 / Dataset audit (read-only)")
        started = dt.datetime.now(dt.timezone.utc)

        table = load_split_manifests(self.config)
        cache = {} if force else self._load_cache()
        if cache:
            logger.info("resume enabled: %d cached measurements available", len(cache))

        with log_duration(logger, f"measuring {len(table)} images"):
            measurements = self._measure_all(table["path"].tolist(), cache)

        records = self._build_records(table, measurements)
        extra_files = self._find_unreferenced_files(table)
        report = self._build_report(records, extra_files, started)

        result = AuditResult(records=records, report=report)
        result.save(
            self.config.resolve_path(self.config.outputs.audit_manifest),
            self.config.resolve_path(self.config.outputs.audit_report),
        )
        self._log_summary(report)
        return result

    # -- measurement -------------------------------------------------------

    def _measure_all(self, paths: Sequence[str], cache: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Measure every path, reusing cached rows whose signature is unchanged.

        Args:
            paths: Absolute image paths.
            cache: Previously measured rows keyed by path.

        Returns:
            Mapping of path to measurement dictionary.
        """
        pending: list[str] = []
        measurements: dict[str, dict[str, Any]] = {}

        for path in paths:
            cached = cache.get(path)
            if cached is not None and self._cache_is_valid(path, cached):
                measurements[path] = cached
            else:
                pending.append(path)

        if not pending:
            logger.info("all %d measurements served from cache", len(measurements))
            return measurements

        logger.info("measuring %d image(s); %d served from cache", len(pending), len(measurements))
        workers = self.config.audit.num_workers
        if workers <= 1:
            for path in pending:
                measurements[path] = measure_image(path, self.options)
            return measurements

        # Image decoding is CPU-bound and releases no GIL time worth sharing,
        # so processes (not threads) are the correct parallelism primitive.
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for path, measurement in zip(
                pending,
                pool.map(measure_image, pending, [self.options] * len(pending), chunksize=self.config.audit.chunk_size),
            ):
                measurements[path] = measurement
        return measurements

    def _cache_is_valid(self, path: str, cached: dict[str, Any]) -> bool:
        """Return ``True`` when a cached row still matches the file on disk."""
        if not self.config.audit.resume:
            return False
        try:
            stat = Path(path).stat()
        except OSError:
            return False
        return (
            int(cached.get("size_bytes", -1)) == stat.st_size
            and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
        )

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        """Load the previous manifest, if one exists, keyed by path."""
        manifest_path = self.config.resolve_path(self.config.outputs.audit_manifest)
        if not self.config.audit.resume or not manifest_path.is_file():
            return {}
        try:
            frame = pd.read_csv(manifest_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("could not read the cached manifest (%s); re-measuring", exc)
            return {}
        if "path" not in frame.columns:
            return {}
        frame = frame.where(pd.notna(frame), None)
        return {str(row["path"]): dict(row) for _, row in frame.iterrows()}

    # -- assembly ----------------------------------------------------------

    def _build_records(self, table: pd.DataFrame, measurements: dict[str, dict[str, Any]]) -> list[AuditRecord]:
        """Combine manifest rows with their measurements into audit records."""
        valid_labels = set(self.config.classes.order)
        known_fields = set(AuditRecord.column_names())
        records: list[AuditRecord] = []

        for row in table.itertuples(index=False):
            label_value = getattr(row, "label")
            has_label = label_value is not None and not (
                isinstance(label_value, float) and math.isnan(label_value)
            )
            label_int = int(label_value) if has_label and float(label_value).is_integer() else None

            record = AuditRecord(
                split=str(row.split),
                id_code=str(row.id_code),
                path=str(row.path),
                label=label_int,
                label_present=has_label,
                label_in_range=label_int in valid_labels if label_int is not None else False,
            )
            for key, value in measurements.get(str(row.path), {}).items():
                if key in known_fields and key != "path" and value is not None:
                    setattr(record, key, value)
            records.append(record)

        return records

    def _find_unreferenced_files(self, table: pd.DataFrame) -> dict[str, list[str]]:
        """List image files on disk that no split CSV references.

        Unreferenced files are reported rather than acted upon: they may be a
        stale download, an extra split, or a genuine manifest error, and only a
        human can tell which.
        """
        referenced = set(table["path"])
        extras: dict[str, list[str]] = {}
        extension = self.config.image.extension
        for split, paths in self.config.splits.as_dict().items():
            image_dir = self.config.resolve_path(paths.image_dir)
            if not image_dir.is_dir():
                extras[split] = []
                continue
            found = {str(p) for p in image_dir.glob(f"*{extension}")}
            extras[split] = sorted(found - referenced)
        return extras

    # -- reporting ---------------------------------------------------------

    def _build_report(
        self,
        records: Sequence[AuditRecord],
        extra_files: dict[str, list[str]],
        started: dt.datetime,
    ) -> dict[str, Any]:
        """Assemble the JSON summary report."""
        frame = pd.DataFrame([record.to_row() for record in records])
        finished = dt.datetime.now(dt.timezone.utc)

        readable = frame["readable"] if not frame.empty else pd.Series(dtype=bool)
        duplicate_ids = (
            frame.groupby("split")["id_code"].apply(lambda s: int(s.duplicated().sum())).to_dict()
            if not frame.empty
            else {}
        )

        report: dict[str, Any] = {
            "stage": "audit",
            "generated_at_utc": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 2),
            "config_hash": self.config.config_hash,
            "profile": self.config.profile,
            "dataset": self.config.dataset_name,
            "totals": {
                "images": int(len(frame)),
                "readable": int(readable.sum()) if len(frame) else 0,
                "missing_files": int((~frame["exists"]).sum()) if len(frame) else 0,
                "zero_byte_files": int((frame["size_bytes"] == 0).sum()) if len(frame) else 0,
                "unreadable_files": int((~readable).sum()) if len(frame) else 0,
                "total_bytes": int(frame["size_bytes"].sum()) if len(frame) else 0,
            },
            "per_split": self._per_split_summary(frame),
            "labels": {
                "missing": int((~frame["label_present"]).sum()) if len(frame) else 0,
                "out_of_range": int((frame["label_present"] & ~frame["label_in_range"]).sum())
                if len(frame)
                else 0,
                "duplicate_id_codes_per_split": duplicate_ids,
                "class_counts": self._class_counts(frame),
            },
            "formats": self._value_counts(frame, "image_format"),
            "color_modes": self._value_counts(frame, "color_mode"),
            "resolution": self._resolution_summary(frame),
            "quality": self._quality_summary(frame),
            "hashes": {
                "md5_computed": int((frame["md5"].astype(str) != "").sum()) if len(frame) else 0,
                "unique_md5": int(frame.loc[frame["md5"].astype(str) != "", "md5"].nunique())
                if len(frame)
                else 0,
                "note": (
                    "MD5 is the authoritative exact-duplicate key. Perceptual hashes are "
                    "recorded for investigation and clustering only and never justify exclusion."
                ),
            },
            "unreferenced_files": {split: len(paths) for split, paths in extra_files.items()},
            "unreferenced_file_examples": {
                split: paths[:10] for split, paths in extra_files.items() if paths
            },
            "errors": self._error_examples(frame),
        }
        return report

    @staticmethod
    def _per_split_summary(frame: pd.DataFrame) -> dict[str, Any]:
        """Per-split counts of rows, readable images, and missing files."""
        if frame.empty:
            return {}
        grouped = frame.groupby("split")
        return {
            str(split): {
                "images": int(len(group)),
                "readable": int(group["readable"].sum()),
                "missing_files": int((~group["exists"]).sum()),
            }
            for split, group in grouped
        }

    @staticmethod
    def _class_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Raw label counts per split (analysis only; no balancing is applied)."""
        if frame.empty:
            return {}
        labelled = frame[frame["label_present"]]
        counts: dict[str, dict[str, int]] = {}
        for split, group in labelled.groupby("split"):
            counts[str(split)] = {
                str(int(label)): int(count) for label, count in group["label"].value_counts().items()
            }
        return counts

    @staticmethod
    def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
        """Value counts for a categorical manifest column."""
        if frame.empty or column not in frame:
            return {}
        series = frame[column].astype(str).replace("", "<unknown>")
        return {str(key): int(value) for key, value in series.value_counts().items()}

    @staticmethod
    def _resolution_summary(frame: pd.DataFrame) -> dict[str, Any]:
        """Summarise width, height, aspect ratio, and distinct resolutions."""
        readable = frame[frame["readable"]] if not frame.empty else frame
        if readable.empty:
            return {}
        pairs = readable.groupby(["width", "height"]).size().sort_values(ascending=False)
        return {
            "width": _describe(readable["width"]),
            "height": _describe(readable["height"]),
            "aspect_ratio": _describe(readable["aspect_ratio"]),
            "distinct_resolutions": int(len(pairs)),
            "most_common_resolutions": [
                {"width": int(w), "height": int(h), "count": int(c)} for (w, h), c in pairs.head(10).items()
            ],
        }

    @staticmethod
    def _quality_summary(frame: pd.DataFrame) -> dict[str, Any]:
        """Summarise the quality metrics that later drive flags (never deletions)."""
        readable = frame[frame["readable"]] if not frame.empty else frame
        if readable.empty:
            return {}
        return {
            column: _describe(readable[column])
            for column in (
                "brightness",
                "contrast",
                "laplacian_var",
                "sharpness_norm",
                "noise_sigma",
                "padding_ratio",
            )
            if column in readable
        }

    @staticmethod
    def _error_examples(frame: pd.DataFrame, limit: int = 20) -> list[dict[str, str]]:
        """Return up to ``limit`` failing rows with their error messages."""
        if frame.empty:
            return []
        failing = frame[frame["error"].astype(str) != ""]
        return [
            {"split": str(row.split), "id_code": str(row.id_code), "error": str(row.error)}
            for row in failing.head(limit).itertuples(index=False)
        ]

    @staticmethod
    def _log_summary(report: dict[str, Any]) -> None:
        """Log the headline numbers so a run is self-documenting in the console."""
        totals = report.get("totals", {})
        labels = report.get("labels", {})
        logger.info(
            "audit: %s images | readable %s | missing %s | unreadable %s",
            totals.get("images"),
            totals.get("readable"),
            totals.get("missing_files"),
            totals.get("unreadable_files"),
        )
        logger.info(
            "labels: missing %s | out of range %s | unique MD5 %s",
            labels.get("missing"),
            labels.get("out_of_range"),
            report.get("hashes", {}).get("unique_md5"),
        )


def _describe(series: Iterable[float]) -> dict[str, float]:
    """Return a compact numeric summary of a series.

    Args:
        series: Numeric values (NaNs are ignored).

    Returns:
        Mean, standard deviation, minimum, quartiles, and maximum.
    """
    values = pd.Series(list(series), dtype="float64").dropna()
    if values.empty:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }
