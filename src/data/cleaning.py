"""Stage 2 -- Cleaning.

Cleaning is a **pure decision function over the audit manifest**. It never
reads pixels, never writes to ``data/raw``, and never deletes a file. Its only
output is an annotated manifest plus a report: an ``included`` flag, an
``exclusion_reason``, ``quality_flags``, and cluster identifiers.

Why decisions instead of deletions
----------------------------------
Deletion is irreversible, silently invalidates the completed EDA, and -- for
this corpus specifically -- would remove clinically valid images. The Data
Preparation report is explicit: dark, bright, blurry, noisy, and low-contrast
fundus photographs are genuine acquisitions with valid labels, and discarding
them would only worsen the scarcity of Grades 1, 3, and 4. A manifest-based
exclusion is reversible, diffable, and auditable; ``git diff`` on a CSV shows
exactly what a threshold change did.

Duplicate policy
----------------
* **MD5 is authoritative.** Exact byte-level duplicates are the only evidence
  strong enough to justify dropping an image. The EDA confirmed nine
  cross-split duplicate groups in a 1,500-image sample; those are leakage and
  are resolved by keeping one copy (train by default) and excluding the rest.
* **Perceptual hashing is investigation-only.** :class:`NearDuplicateRule` is
  hard-wired so that it *cannot* emit an exclusion -- see ``can_exclude``. Two
  eyes of one patient, or two acquisitions of one eye, are near-duplicates yet
  remain legitimately distinct samples; only a human can adjudicate that.

Example
-------
>>> from src.data.cleaning import DatasetCleaner
>>> result = DatasetCleaner(config).run(audit_frame)       # doctest: +SKIP
>>> int(result.frame["included"].sum())                    # doctest: +SKIP
3653
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.utils.config import DataConfig
from src.utils.helpers import ensure_dir, hex_to_bits, write_json
from src.utils.logger import get_logger, log_section

__all__ = [
    "Decision",
    "RuleResult",
    "CleaningRule",
    "IntegrityRule",
    "LabelConsistencyRule",
    "ExactDuplicateRule",
    "NearDuplicateRule",
    "QualityFlagRule",
    "CleaningResult",
    "DatasetCleaner",
    "cluster_by_hash",
    "cluster_by_dual_hash",
    "merge_group_keys",
    "UnionFind",
]

logger = get_logger(__name__)

#: Columns appended to the audit manifest by this stage.
CLEANING_COLUMNS: Final[tuple[str, ...]] = (
    "included",
    "exclusion_reason",
    "exclusion_rule",
    "quality_flags",
    "md5_group",
    "md5_group_size",
    "md5_cross_split",
    "content_cluster",
    "content_cluster_size",
    "content_cluster_cross_split",
)


class Decision(str, Enum):
    """Per-image outcome of a cleaning rule.

    ``FLAG`` annotates without changing inclusion; ``EXCLUDE`` removes the row
    from the usable dataset while leaving the file on disk untouched.
    """

    KEEP = "keep"
    FLAG = "flag"
    EXCLUDE = "exclude"


@dataclass
class RuleResult:
    """Outcome of applying one cleaning rule to the manifest.

    Attributes:
        rule: Rule name.
        exclusions: Mapping of manifest index to exclusion reason.
        flags: Mapping of manifest index to the list of flags added.
        columns: Extra columns the rule contributes to the manifest.
        details: Rule-specific detail included verbatim in the report.
    """

    rule: str
    exclusions: dict[int, str] = field(default_factory=dict)
    flags: dict[int, list[str]] = field(default_factory=dict)
    columns: dict[str, pd.Series] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class CleaningRule(ABC):
    """Base class for cleaning rules.

    Subclasses declare whether they are even *allowed* to exclude
    (``can_exclude``) and where their evidence comes from (``evidence``), so the
    provenance of every removal is machine-readable rather than a code comment.
    """

    #: Stable rule identifier used in reports and manifest columns.
    name: str = "rule"

    #: ``"eda"``, ``"paper"``, or ``"both"``.
    evidence: str = "eda"

    #: Whether the rule may ever return an exclusion. Rules with ``False`` are
    #: validated at runtime; a violation raises rather than silently deleting.
    can_exclude: bool = True

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    @abstractmethod
    def apply(self, frame: pd.DataFrame) -> RuleResult:
        """Evaluate the rule against the manifest.

        Args:
            frame: Audit manifest (one row per image).

        Returns:
            The :class:`RuleResult`.
        """

    def __call__(self, frame: pd.DataFrame) -> RuleResult:
        """Apply the rule and enforce the ``can_exclude`` contract."""
        result = self.apply(frame)
        if result.exclusions and not self.can_exclude:
            raise RuntimeError(
                f"rule '{self.name}' returned {len(result.exclusions)} exclusion(s) but is "
                "declared report-only; this is a programming error, not a configuration issue"
            )
        return result


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class IntegrityRule(CleaningRule):
    """Exclude files that cannot be used at all.

    The EDA reported zero corrupted and zero missing files, so a nonzero count
    here means the local copy of the dataset differs from the audited one --
    which is exactly the kind of silent drift this rule exists to surface.
    """

    name = "integrity"
    evidence = "eda"

    def apply(self, frame: pd.DataFrame) -> RuleResult:
        """Flag or exclude unreadable, missing, and zero-byte files."""
        settings = self.config.cleaning.rules.integrity
        result = RuleResult(rule=self.name)
        if not settings.enabled:
            return result

        missing = ~frame["exists"].astype(bool)
        zero_byte = frame["size_bytes"].fillna(0).astype("int64") == 0
        unreadable = ~frame["readable"].astype(bool)

        for index in frame.index[missing]:
            self._record(result, index, "file not found", settings.exclude_unreadable)
        for index in frame.index[zero_byte & ~missing]:
            self._record(result, index, "zero-byte file", settings.exclude_unreadable)
        for index in frame.index[unreadable & ~missing & ~zero_byte]:
            reason = str(frame.at[index, "error"]) or "undecodable image"
            self._record(result, index, reason, settings.exclude_unreadable)

        result.details = {
            "missing_files": int(missing.sum()),
            "zero_byte_files": int(zero_byte.sum()),
            "undecodable_files": int((unreadable & ~missing & ~zero_byte).sum()),
        }
        return result

    @staticmethod
    def _record(result: RuleResult, index: int, reason: str, exclude: bool) -> None:
        """Attach an exclusion or a flag depending on the configured policy."""
        if exclude:
            result.exclusions[index] = reason
        else:
            result.flags.setdefault(index, []).append("integrity_issue")


class LabelConsistencyRule(CleaningRule):
    """Validate the label column against the configured class definition."""

    name = "labels"
    evidence = "eda"

    def apply(self, frame: pd.DataFrame) -> RuleResult:
        """Exclude unusable label rows and flag duplicate identifiers."""
        settings = self.config.cleaning.rules.labels
        result = RuleResult(rule=self.name)
        if not settings.enabled:
            return result

        missing_label = ~frame["label_present"].astype(bool)
        out_of_range = frame["label_present"].astype(bool) & ~frame["label_in_range"].astype(bool)
        duplicate_ids = frame.duplicated(subset=["split", "id_code"], keep="first")

        for index in frame.index[missing_label]:
            if settings.exclude_missing_label:
                result.exclusions[index] = "missing label"
            else:
                result.flags.setdefault(index, []).append("missing_label")
        for index in frame.index[out_of_range]:
            if settings.exclude_out_of_range:
                result.exclusions[index] = "label outside the configured class set"
            else:
                result.flags.setdefault(index, []).append("label_out_of_range")
        for index in frame.index[duplicate_ids]:
            if settings.exclude_duplicate_ids:
                result.exclusions[index] = "duplicate id_code within split"
            else:
                result.flags.setdefault(index, []).append("duplicate_id_code")

        result.details = {
            "missing_labels": int(missing_label.sum()),
            "out_of_range_labels": int(out_of_range.sum()),
            "duplicate_id_codes": int(duplicate_ids.sum()),
        }
        return result


class ExactDuplicateRule(CleaningRule):
    """Resolve byte-identical duplicates using MD5.

    This is the only rule permitted to remove an image on the grounds of
    redundancy, because MD5 equality is exact: no threshold, no tuning, no
    dependence on a decoder version.

    Cross-split groups are true leakage and are resolved by keeping the copy in
    the highest-priority split (train by default, so the training set retains
    the sample and the evaluation sets stay clean) and excluding the others.
    Within-split groups are mere redundancy and are flagged by default.
    """

    name = "exact_duplicates"
    evidence = "eda"

    def apply(self, frame: pd.DataFrame) -> RuleResult:
        """Group by MD5 and resolve cross-split and within-split duplicates."""
        settings = self.config.cleaning.rules.exact_duplicates
        result = RuleResult(rule=self.name)

        group_ids = pd.Series("", index=frame.index, dtype=object)
        group_sizes = pd.Series(1, index=frame.index, dtype="int64")
        cross_split = pd.Series(False, index=frame.index, dtype=bool)
        result.columns = {
            "md5_group": group_ids,
            "md5_group_size": group_sizes,
            "md5_cross_split": cross_split,
        }
        if not settings.enabled:
            return result

        hashed = frame[frame["md5"].astype(str) != ""]
        priority = {split: rank for rank, split in enumerate(settings.keep_split_priority)}
        cross_split_groups: list[dict[str, Any]] = []
        within_split_groups = 0

        for digest, group in hashed.groupby("md5"):
            if len(group) < 2:
                continue
            group_ids.loc[group.index] = str(digest)
            group_sizes.loc[group.index] = int(len(group))
            splits = sorted(set(group["split"]))
            is_cross_split = len(splits) > 1
            cross_split.loc[group.index] = is_cross_split

            if is_cross_split:
                keeper = self._choose_keeper(group, priority)
                cross_split_groups.append(
                    {
                        "md5": str(digest),
                        "size": int(len(group)),
                        "splits": splits,
                        "kept": {"split": str(group.at[keeper, "split"]), "id_code": str(group.at[keeper, "id_code"])},
                        "members": [
                            {"split": str(row.split), "id_code": str(row.id_code)}
                            for row in group.itertuples(index=False)
                        ],
                    }
                )
                for index in group.index:
                    if index == keeper:
                        result.flags.setdefault(index, []).append("duplicate_kept")
                        continue
                    if settings.exclude_cross_split:
                        result.exclusions[index] = f"cross-split MD5 duplicate of {digest[:12]}"
                    else:
                        result.flags.setdefault(index, []).append("cross_split_duplicate")
            else:
                within_split_groups += 1
                keeper = self._choose_keeper(group, priority)
                for index in group.index:
                    if index == keeper:
                        result.flags.setdefault(index, []).append("duplicate_kept")
                        continue
                    if settings.exclude_within_split:
                        result.exclusions[index] = f"within-split MD5 duplicate of {digest[:12]}"
                    else:
                        result.flags.setdefault(index, []).append("within_split_duplicate")

        result.details = {
            "duplicate_groups": int(len(cross_split_groups) + within_split_groups),
            "cross_split_groups": int(len(cross_split_groups)),
            "within_split_groups": int(within_split_groups),
            "cross_split_group_details": cross_split_groups,
            "policy": {
                "exclude_cross_split": settings.exclude_cross_split,
                "exclude_within_split": settings.exclude_within_split,
                "keep_split_priority": list(settings.keep_split_priority),
            },
        }
        return result

    @staticmethod
    def _choose_keeper(group: pd.DataFrame, priority: dict[str, int]) -> int:
        """Pick which member of a duplicate group to retain.

        Preference order: readable before unreadable, then the configured split
        priority, then ``id_code`` so the choice is deterministic across runs
        and machines.
        """
        ordered = group.assign(
            _readable_rank=(~group["readable"].astype(bool)).astype(int),
            _split_rank=[priority.get(str(split), len(priority)) for split in group["split"]],
        ).sort_values(["_readable_rank", "_split_rank", "id_code"], kind="mergesort")
        return int(ordered.index[0])


class NearDuplicateRule(CleaningRule):
    """Cluster visually similar images from perceptual hashes.

    **Report-only by construction.** ``can_exclude`` is ``False`` and the base
    class raises if this rule ever returns an exclusion, so perceptual
    similarity cannot remove an image no matter how the configuration is
    edited. The clusters serve two legitimate purposes:

    1. quantifying content-level leakage risk beyond exact duplicates, and
    2. providing a grouping key for the opt-in, group-aware split regeneration
       in :mod:`src.data.splits` (APTOS ships no patient identifier).
    """

    name = "near_duplicates"
    evidence = "eda"
    can_exclude = False

    def apply(self, frame: pd.DataFrame) -> RuleResult:
        """Cluster by dHash Hamming distance and flag every multi-image cluster."""
        settings = self.config.cleaning.rules.near_duplicates
        result = RuleResult(rule=self.name)

        clusters = pd.Series(range(len(frame)), index=frame.index, dtype="int64")
        sizes = pd.Series(1, index=frame.index, dtype="int64")
        cross_split = pd.Series(False, index=frame.index, dtype=bool)
        result.columns = {
            "content_cluster": clusters,
            "content_cluster_size": sizes,
            "content_cluster_cross_split": cross_split,
        }
        if not settings.enabled:
            return result

        usable = frame[frame["dhash"].astype(str) != ""]
        if usable.empty:
            logger.warning("no perceptual hashes present; near-duplicate analysis skipped")
            return result

        has_phash = "phash" in usable and (usable["phash"].astype(str) != "").all()
        if has_phash:
            labels = cluster_by_dual_hash(
                usable["dhash"].astype(str).tolist(),
                usable["phash"].astype(str).tolist(),
                settings.hamming_threshold,
            )
        else:
            logger.warning("pHash unavailable; falling back to dHash-only clustering (more chaining)")
            labels = cluster_by_hash(usable["dhash"].astype(str).tolist(), settings.hamming_threshold)
        clusters.loc[usable.index] = labels

        cross_split_clusters = 0
        multi_image_clusters = 0
        for label, group in frame.groupby(clusters):
            size = int(len(group))
            sizes.loc[group.index] = size
            if size < 2:
                continue
            multi_image_clusters += 1
            spans_splits = group["split"].nunique() > 1
            cross_split.loc[group.index] = spans_splits
            if spans_splits:
                cross_split_clusters += 1
            for index in group.index:
                result.flags.setdefault(index, []).append(
                    "near_duplicate_cross_split" if spans_splits else "near_duplicate"
                )
            del label

        result.details = {
            "hamming_threshold": settings.hamming_threshold,
            "clustering": "dhash AND phash agreement" if has_phash else "dhash only",
            "clusters_with_multiple_images": multi_image_clusters,
            "cross_split_clusters": cross_split_clusters,
            "policy": (
                "investigation only: perceptual similarity never excludes an image; "
                "clusters are reported and reused as split-regeneration groups"
            ),
        }
        return result


class QualityFlagRule(CleaningRule):
    """Flag -- never remove -- quality outliers.

    Blur detection uses the resolution-normalized sharpness measured by the
    audit. The raw Laplacian variance correlates r ~= -0.80 with image width in
    this corpus, and width itself correlates with DR grade (r = 0.57), so a raw
    threshold would preferentially flag high-grade images: a bias, not a
    quality signal.

    Sharpness and noise are flagged by **percentile**, not by an absolute
    threshold. Both are unitless quantities whose scale depends on the metric
    proxy resolution and on the camera mix, so any fixed cut-off is
    dataset-specific and silently wrong elsewhere: an absolute sharpness
    threshold of 0.15 flagged 100% of APTOS, whose measured values span
    0.004-0.078. Percentiles are self-calibrating and reproduce the ~1% flag
    rate the EDA reported.

    Brightness and contrast keep absolute thresholds, because those are in
    interpretable intensity units and the EDA anchored them directly.
    """

    name = "quality_flags"
    evidence = "eda"
    can_exclude = False

    def apply(self, frame: pd.DataFrame) -> RuleResult:
        """Attach dark/bright/low-contrast/blurry/noisy flags."""
        settings = self.config.cleaning.rules.quality_flags
        result = RuleResult(rule=self.name)
        if not settings.enabled:
            return result

        readable = frame["readable"].astype(bool)
        blur_cutoff = self._percentile(frame.loc[readable, "sharpness_norm"], settings.blur_percentile)
        noise_cutoff = self._percentile(frame.loc[readable, "noise_sigma"], settings.noise_percentile)

        conditions = {
            "too_dark": readable & (frame["brightness"] < settings.dark_brightness_max),
            "too_bright": readable & (frame["brightness"] > settings.bright_brightness_min),
            "low_contrast": readable & (frame["contrast"] < settings.low_contrast_std_max),
            "blurry": readable & (frame["sharpness_norm"] < blur_cutoff)
            if blur_cutoff is not None
            else readable & False,
            "noisy": readable & (frame["noise_sigma"] > noise_cutoff)
            if noise_cutoff is not None
            else readable & False,
        }

        counts: dict[str, int] = {}
        for flag, mask in conditions.items():
            mask = mask.fillna(False)
            counts[flag] = int(mask.sum())
            for index in frame.index[mask]:
                result.flags.setdefault(index, []).append(flag)

        result.details = {
            "counts": counts,
            "thresholds": {
                "dark_brightness_max": settings.dark_brightness_max,
                "bright_brightness_min": settings.bright_brightness_min,
                "low_contrast_std_max": settings.low_contrast_std_max,
                "blur_percentile": settings.blur_percentile,
                "blur_cutoff_value": blur_cutoff,
                "noise_percentile": settings.noise_percentile,
                "noise_cutoff_value": noise_cutoff,
            },
            "policy": (
                "flag only: these are genuine clinical acquisitions with valid labels; "
                "deleting them would worsen minority-class scarcity"
            ),
        }
        return result

    @staticmethod
    def _percentile(values: pd.Series, percentile: float) -> float | None:
        """Return the requested percentile of a metric, or ``None`` if unusable.

        Args:
            values: Metric values for readable images.
            percentile: Percentile in ``[0, 100]``. A value of ``0`` disables
                the corresponding flag entirely.

        Returns:
            The cut-off value, or ``None`` when the flag is disabled or no data
            is available.
        """
        clean = values.dropna()
        if percentile <= 0 or percentile >= 100 or clean.empty:
            return None
        return float(np.percentile(clean.to_numpy(), percentile))


# ---------------------------------------------------------------------------
# Clustering utility
# ---------------------------------------------------------------------------


class UnionFind:
    """Minimal disjoint-set structure with path compression.

    Shared by the near-duplicate clustering here and by the group-aware split
    regeneration in :mod:`src.data.splits`, so the two never drift apart.
    """

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        """Return the representative of ``item``'s set."""
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        """Merge the sets containing ``left`` and ``right``."""
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


def cluster_by_hash(digests: Sequence[str], threshold: int, chunk_size: int = 512) -> np.ndarray:
    """Cluster hexadecimal digests by Hamming distance.

    Pairwise distances are computed as a chunked matrix product over the
    unpacked bit matrix (``d = |a| + |b| - 2 a.b``), which turns an O(n^2)
    Python loop into a handful of BLAS calls. Chunking bounds peak memory so
    the routine scales to datasets an order of magnitude larger than APTOS.

    Args:
        digests: Hexadecimal digests of equal length.
        threshold: Maximum Hamming distance for two items to share a cluster.
        chunk_size: Number of rows compared per block.

    Returns:
        An integer array of cluster labels, one per input digest.

    Raises:
        ValueError: If ``threshold`` is negative.

    Example:
        >>> labels = cluster_by_hash(["00", "01", "ff"], threshold=1)
        >>> labels[0] == labels[1] and labels[0] != labels[2]
        np.True_
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    count = len(digests)
    if count == 0:
        return np.zeros(0, dtype=np.int64)

    bits = np.stack([hex_to_bits(digest) for digest in digests]).astype(np.float32)
    popcount = bits.sum(axis=1)
    union_find = UnionFind(count)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        block = bits[start:stop]
        # Only the upper triangle is needed; comparing each block against the
        # full matrix and masking is simpler and still BLAS-bound.
        distances = popcount[start:stop, None] + popcount[None, :] - 2.0 * (block @ bits.T)
        close = np.argwhere(distances <= threshold)
        for local_row, column in close:
            row = start + int(local_row)
            column = int(column)
            if row < column:
                union_find.union(row, column)

    return np.array([union_find.find(index) for index in range(count)], dtype=np.int64)


def cluster_by_dual_hash(
    dhashes: Sequence[str], phashes: Sequence[str], threshold: int, chunk_size: int = 512
) -> np.ndarray:
    """Cluster images that are close under **both** perceptual hashes.

    Single-hash clustering collapses on this corpus. Every fundus photograph is
    a bright disc on a black field, so dHash neighbourhoods overlap and
    single-linkage transitive closure chains them together: on the full APTOS
    corpus a dHash-only threshold of 6 merges 3,087 of 3,662 images into one
    cluster, which is meaningless as a near-duplicate signal.

    Requiring agreement between the gradient hash (dHash) and the
    low-frequency DCT hash (pHash) removes the chaining, because the two
    encode different properties and rarely produce the same spurious
    neighbour. On the same corpus, agreement at threshold 2 yields 195
    multi-image clusters covering 492 images, with a largest cluster of 35.

    Args:
        dhashes: Gradient-hash digests.
        phashes: DCT-hash digests, aligned with ``dhashes``.
        threshold: Maximum Hamming distance, applied to each hash separately.
        chunk_size: Block size passed to :func:`cluster_by_hash`.

    Returns:
        Integer cluster labels.

    Raises:
        ValueError: If the two digest sequences have different lengths.
    """
    if len(dhashes) != len(phashes):
        raise ValueError(f"digest count mismatch: {len(dhashes)} dHashes vs {len(phashes)} pHashes")
    if not dhashes:
        return np.zeros(0, dtype=np.int64)

    dhash_labels = cluster_by_hash(dhashes, threshold, chunk_size=chunk_size)
    phash_labels = cluster_by_hash(phashes, threshold, chunk_size=chunk_size)

    # Intersection of the two partitions: two images share a cluster only when
    # both hashes place them together.
    pairs = {}
    labels = np.empty(len(dhashes), dtype=np.int64)
    for position, key in enumerate(zip(dhash_labels.tolist(), phash_labels.tolist())):
        labels[position] = pairs.setdefault(key, len(pairs))
    return labels


def merge_group_keys(*key_series: pd.Series) -> pd.Series:
    """Merge several grouping keys into one transitive-closure grouping.

    Two rows land in the same output group when they share a value in *any*
    input key. This is what makes content clustering safe for split
    regeneration: an MD5 group and a perceptual cluster that overlap by a single
    image must not be separable, or exact duplicates could still straddle the
    partition boundary.

    Args:
        *key_series: Aligned key columns. Empty strings are treated as "no key"
            and never merge rows.

    Returns:
        A string Series of merged group identifiers, aligned to the inputs.

    Raises:
        ValueError: If no series is supplied or their indices differ.

    Example:
        >>> import pandas as pd
        >>> a = pd.Series(["x", "x", "y"])
        >>> b = pd.Series(["1", "2", "2"])
        >>> merge_group_keys(a, b).nunique()
        1
    """
    if not key_series:
        raise ValueError("at least one key series is required")
    reference = key_series[0].index
    if any(not series.index.equals(reference) for series in key_series):
        raise ValueError("all key series must share the same index")

    count = len(reference)
    union_find = UnionFind(count)
    for series in key_series:
        first_seen: dict[str, int] = {}
        for position, value in enumerate(series.astype(str)):
            if value == "":
                continue
            anchor = first_seen.setdefault(value, position)
            union_find.union(anchor, position)

    roots = [union_find.find(position) for position in range(count)]
    return pd.Series([f"g{root}" for root in roots], index=reference, dtype=object)


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------


@dataclass
class CleaningResult:
    """Annotated manifest plus the cleaning report."""

    frame: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def included(self) -> pd.DataFrame:
        """Rows that survived cleaning."""
        return self.frame[self.frame["included"]]

    @property
    def excluded(self) -> pd.DataFrame:
        """Rows excluded by a rule (the quarantine manifest)."""
        return self.frame[~self.frame["included"]]

    def save(self, manifest_path: str | Path, report_path: str | Path, quarantine_path: str | Path) -> None:
        """Persist the clean manifest, the quarantine manifest, and the report."""
        ensure_dir(Path(manifest_path).parent)
        self.frame.to_csv(manifest_path, index=False)
        self.excluded.to_csv(quarantine_path, index=False)
        write_json(report_path, self.report)
        logger.info("clean manifest written to %s", manifest_path)
        logger.info("quarantine manifest written to %s (%d rows)", quarantine_path, len(self.excluded))
        logger.info("cleaning report written to %s", report_path)


class DatasetCleaner:
    """Apply the configured cleaning rules to an audit manifest.

    The cleaner itself owns no cleaning logic: it orders rules, merges their
    results, records before/after class distributions, and writes artefacts.
    Adding a rule therefore requires no change to this class.

    Args:
        config: Validated data configuration.
        rules: Optional explicit rule list (used by tests); defaults to the
            standard ordering below.
    """

    def __init__(self, config: DataConfig, rules: Sequence[CleaningRule] | None = None) -> None:
        self.config = config
        # Order matters only for reporting clarity: integrity and label problems
        # are more fundamental than redundancy, which is more actionable than a
        # quality flag.
        self.rules: list[CleaningRule] = list(
            rules
            if rules is not None
            else [
                IntegrityRule(config),
                LabelConsistencyRule(config),
                ExactDuplicateRule(config),
                NearDuplicateRule(config),
                QualityFlagRule(config),
            ]
        )

    def run(self, audit_frame: pd.DataFrame) -> CleaningResult:
        """Run every rule and produce the annotated manifest.

        Args:
            audit_frame: Manifest produced by :class:`~src.data.audit.DatasetAuditor`.

        Returns:
            The :class:`CleaningResult`, already written to disk.
        """
        log_section(logger, "Stage 2 / Cleaning (decisions only, no deletions)")
        started = dt.datetime.now(dt.timezone.utc)

        frame = audit_frame.reset_index(drop=True).copy()
        before = self._class_distribution(frame)

        frame["included"] = True
        frame["exclusion_reason"] = ""
        frame["exclusion_rule"] = ""
        flags: dict[int, list[str]] = {}
        rule_reports: dict[str, Any] = {}

        for rule in self.rules:
            result = rule(frame)
            for index, reason in result.exclusions.items():
                # The first rule to exclude a row owns the reason: integrity
                # problems should not be masked by a later duplicate finding.
                if frame.at[index, "included"]:
                    frame.at[index, "included"] = False
                    frame.at[index, "exclusion_reason"] = reason
                    frame.at[index, "exclusion_rule"] = rule.name
            for index, added in result.flags.items():
                flags.setdefault(index, []).extend(added)
            for column, series in result.columns.items():
                frame[column] = series
            rule_reports[rule.name] = {
                "evidence": rule.evidence,
                "can_exclude": rule.can_exclude,
                "excluded": len(result.exclusions),
                "flagged": len(result.flags),
                **result.details,
            }
            logger.info(
                "rule '%s': %d excluded, %d flagged", rule.name, len(result.exclusions), len(result.flags)
            )

        frame["quality_flags"] = [
            "|".join(sorted(set(flags.get(index, [])))) for index in range(len(frame))
        ]
        for column in CLEANING_COLUMNS:
            if column not in frame:
                frame[column] = "" if column in ("quality_flags", "md5_group") else 0

        after = self._class_distribution(frame[frame["included"]])
        report = self._build_report(frame, rule_reports, before, after, started)

        result = CleaningResult(frame=frame, report=report)
        result.save(
            self.config.resolve_path(self.config.outputs.clean_manifest),
            self.config.resolve_path(self.config.outputs.cleaning_report),
            self.config.resolve_path(self.config.outputs.quarantine_manifest),
        )
        self._log_summary(report)
        return result

    # -- reporting ---------------------------------------------------------

    @staticmethod
    def _class_distribution(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Per-split label counts, used to prove cleaning did not skew classes."""
        if frame.empty or "label" not in frame:
            return {}
        labelled = frame[frame["label"].notna()]
        return {
            str(split): {
                str(int(label)): int(count) for label, count in group["label"].value_counts().items()
            }
            for split, group in labelled.groupby("split")
        }

    def _build_report(
        self,
        frame: pd.DataFrame,
        rule_reports: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
        started: dt.datetime,
    ) -> dict[str, Any]:
        """Assemble the JSON cleaning report."""
        finished = dt.datetime.now(dt.timezone.utc)
        excluded = frame[~frame["included"]]
        flag_counts: dict[str, int] = {}
        for entry in frame["quality_flags"]:
            for flag in filter(None, str(entry).split("|")):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        return {
            "stage": "cleaning",
            "generated_at_utc": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 2),
            "config_hash": self.config.config_hash,
            "profile": self.config.profile,
            "policy": {
                "files_deleted": 0,
                "raw_data_modified": False,
                "authoritative_duplicate_detector": "md5",
                "perceptual_hash_role": "investigation and clustering only",
                "quality_outliers": "flagged, never deleted",
            },
            "totals": {
                "images": int(len(frame)),
                "included": int(frame["included"].sum()),
                "excluded": int(len(excluded)),
                "flagged": int((frame["quality_flags"].astype(str) != "").sum()),
            },
            "exclusions_by_rule": {
                str(rule): int(count) for rule, count in excluded["exclusion_rule"].value_counts().items()
            },
            "exclusions_by_split": {
                str(split): int(count) for split, count in excluded["split"].value_counts().items()
            },
            "flag_counts": flag_counts,
            "rules": rule_reports,
            "class_distribution_before": before,
            "class_distribution_after": after,
        }

    @staticmethod
    def _log_summary(report: dict[str, Any]) -> None:
        """Log the headline cleaning numbers."""
        totals = report["totals"]
        logger.info(
            "cleaning: %d images | included %d | excluded %d | flagged %d | files deleted 0",
            totals["images"],
            totals["included"],
            totals["excluded"],
            totals["flagged"],
        )
