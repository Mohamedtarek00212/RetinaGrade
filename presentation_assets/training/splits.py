"""Stage 3 -- Split verification and (opt-in) split regeneration.

Two strictly separated responsibilities:

:class:`SplitVerifier`
    Always runs, always read-only. Answers one question: *is this partition
    actually independent?* Identifier overlap is the weak check that the EDA
    already passed (0 shared ``id_code`` values); MD5 overlap is the real one,
    and it failed -- nine byte-identical groups straddle the train/test
    boundary. Perceptual-hash cluster overlap is reported as *risk*, never as
    failure, because visual similarity is not proof of identity.

:class:`SplitRegenerator`
    Off by default. Regenerating the partition changes every downstream number
    and breaks comparability with the paper's 80/20 protocol, so it must be an
    explicit, logged, versioned act rather than a side effect. It writes to a
    new directory and never overwrites the shipped CSVs.

Grouping caveat
---------------
APTOS 2019 ships no patient or eye identifier. Content clusters (MD5 groups
merged with perceptual-hash clusters) are therefore the best available proxy
for "the same eye appears twice". This is an approximation and is reported as
such: it prevents *content* leakage, and it must not be described as
patient-level independence.

This module owns split policy only. Distribution and chi-square mathematics are
imported from :mod:`src.data.statistics`; hashing and clustering come from the
cleaning stage's manifest. Nothing is recomputed here.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.cleaning import merge_group_keys
from src.data.statistics import chi_square_homogeneity, class_distribution
from src.utils.config import DataConfig
from src.utils.helpers import ensure_dir, write_json
from src.utils.logger import get_logger, log_section
from src.utils.seed import make_generator

__all__ = [
    "SplitLeakageError",
    "SplitVerificationReport",
    "SplitVerifier",
    "SplitPlan",
    "SplitRegenerator",
]

logger = get_logger(__name__)

#: Severity levels used by the verification checks.
PASS, WARN, FAIL = "pass", "warn", "fail"


class SplitLeakageError(RuntimeError):
    """Raised in strict mode when exact-duplicate leakage survives cleaning."""


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class SplitVerificationReport:
    """Result of verifying a partition."""

    status: str
    checks: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """``True`` when no check failed."""
        return self.status != FAIL

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {"status": self.status, "summary": self.summary, "checks": self.checks}

    def save(self, path: str | Path) -> Path:
        """Write the report to JSON."""
        ensure_dir(Path(path).parent)
        return write_json(path, self.as_dict())


class SplitVerifier:
    """Read-only verification of an existing partition.

    Args:
        config: Validated data configuration.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def verify(self, manifest: pd.DataFrame, save: bool = True) -> SplitVerificationReport:
        """Run every enabled check against the cleaned manifest.

        Args:
            manifest: Clean manifest from :mod:`src.data.cleaning`. Only rows
                with ``included == True`` participate: an excluded duplicate is
                precisely how leakage gets resolved, so counting it again would
                report a problem that has already been fixed.
            save: Persist the report to the configured location.

        Returns:
            The :class:`SplitVerificationReport`.

        Raises:
            SplitLeakageError: If ``strict`` is enabled and residual exact
                duplicate leakage is detected.
        """
        log_section(logger, "Stage 3 / Split verification")
        settings = self.config.splits_policy.verify
        included = manifest[manifest["included"]] if "included" in manifest else manifest

        checks: dict[str, Any] = {}
        if settings.enabled:
            if settings.check_id_overlap:
                checks["id_overlap"] = self._check_key_overlap(included, "id_code", severity=FAIL)
            if settings.check_md5_overlap:
                checks["md5_overlap"] = self._check_key_overlap(included, "md5", severity=FAIL)
            if settings.check_near_duplicate_overlap:
                checks["near_duplicate_overlap"] = self._check_cluster_overlap(included)
            if settings.check_class_homogeneity:
                checks["class_homogeneity"] = self._check_class_homogeneity(included)
            checks["class_coverage"] = self._check_class_coverage(included, settings.min_images_per_class)
        else:
            logger.warning("split verification is disabled in the configuration")

        status = PASS
        if any(check.get("status") == FAIL for check in checks.values()):
            status = FAIL
        elif any(check.get("status") == WARN for check in checks.values()):
            status = WARN

        report = SplitVerificationReport(
            status=status,
            checks=checks,
            summary={
                "stage": "split_verification",
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "config_hash": self.config.config_hash,
                "profile": self.config.profile,
                "images_considered": int(len(included)),
                "per_split_counts": {
                    str(split): int(count) for split, count in included["split"].value_counts().items()
                },
                "regeneration_enabled": self.config.splits_policy.regenerate.enabled,
            },
        )

        if save:
            report.save(self.config.resolve_path(self.config.outputs.split_report))
        self._log_summary(report)

        if settings.strict and checks.get("md5_overlap", {}).get("status") == FAIL:
            raise SplitLeakageError(
                "exact-duplicate leakage remains after cleaning; enable "
                "cleaning.rules.exact_duplicates.exclude_cross_split or regenerate the splits"
            )
        return report

    # -- individual checks -------------------------------------------------

    @staticmethod
    def _check_key_overlap(frame: pd.DataFrame, column: str, severity: str) -> dict[str, Any]:
        """Detect values of ``column`` that appear in more than one split."""
        if column not in frame or frame.empty:
            return {"status": PASS, "detail": f"column '{column}' unavailable", "overlaps": 0}

        usable = frame[frame[column].astype(str) != ""]
        spread = usable.groupby(column)["split"].nunique()
        offenders = spread[spread > 1]
        examples = [
            {
                "key": str(key),
                "splits": sorted(set(usable.loc[usable[column] == key, "split"].astype(str))),
            }
            for key in offenders.index[:20]
        ]
        return {
            "status": severity if len(offenders) else PASS,
            "overlaps": int(len(offenders)),
            "affected_images": int(usable[usable[column].isin(offenders.index)].shape[0]),
            "examples": examples,
            "meaning": (
                "byte-identical images shared between splits are unambiguous data leakage"
                if column == "md5"
                else "the same identifier appears in more than one split"
            ),
        }

    @staticmethod
    def _check_cluster_overlap(frame: pd.DataFrame) -> dict[str, Any]:
        """Report perceptual-hash clusters that straddle splits.

        Always a warning, never a failure: near-duplicates may be two eyes of
        one patient or two acquisitions of one eye, which is a genuine risk to
        flag but not proof of leakage.
        """
        if "content_cluster" not in frame or frame.empty:
            return {"status": PASS, "detail": "no perceptual clusters available", "clusters": 0}

        spread = frame.groupby("content_cluster")["split"].nunique()
        offenders = spread[spread > 1]
        affected = int(frame[frame["content_cluster"].isin(offenders.index)].shape[0])
        return {
            "status": WARN if len(offenders) else PASS,
            "clusters": int(len(offenders)),
            "affected_images": affected,
            "meaning": (
                "visually similar images span splits; this is a content-level leakage RISK, "
                "reported for investigation only and never acted on automatically"
            ),
        }

    def _check_class_homogeneity(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Chi-square test that class proportions match across splits."""
        grades = list(range(self.config.classes.num_classes))
        table = class_distribution(frame, self.config.classes.num_classes)
        if table.empty:
            return {"status": PASS, "detail": "no labels available"}
        split_table = table.drop(index="overall", errors="ignore")[grades]
        result = chi_square_homogeneity(split_table)
        p_value = result.get("p_value")
        status = WARN if (p_value is not None and p_value < 0.05) else PASS
        return {
            "status": status,
            **result,
            "per_split_proportions": {
                str(split): {
                    str(grade): round(float(row[grade]) / float(row.sum()), 4) if row.sum() else 0.0
                    for grade in grades
                }
                for split, row in split_table.iterrows()
            },
        }

    def _check_class_coverage(self, frame: pd.DataFrame, minimum: int) -> dict[str, Any]:
        """Ensure every split contains at least ``minimum`` images per grade."""
        grades = list(range(self.config.classes.num_classes))
        table = class_distribution(frame, self.config.classes.num_classes)
        if table.empty:
            return {"status": PASS, "detail": "no labels available"}
        split_table = table.drop(index="overall", errors="ignore")[grades]
        shortfalls = [
            {"split": str(split), "grade": int(grade), "count": int(row[grade])}
            for split, row in split_table.iterrows()
            for grade in grades
            if int(row[grade]) < minimum
        ]
        return {
            "status": FAIL if shortfalls else PASS,
            "minimum_required": minimum,
            "shortfalls": shortfalls,
        }

    @staticmethod
    def _log_summary(report: SplitVerificationReport) -> None:
        """Log one line per check plus the overall status."""
        for name, check in report.checks.items():
            logger.info("check '%s': %s", name, check.get("status", "unknown"))
        md5 = report.checks.get("md5_overlap", {})
        if md5.get("overlaps"):
            logger.error(
                "LEAKAGE: %d MD5 group(s) span splits, affecting %d image(s)",
                md5["overlaps"],
                md5.get("affected_images", 0),
            )
        logger.info("split verification status: %s", report.status.upper())


# ---------------------------------------------------------------------------
# Regeneration (opt-in)
# ---------------------------------------------------------------------------


@dataclass
class SplitPlan:
    """Proposed partition produced by :class:`SplitRegenerator`."""

    assignments: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)
    written_files: list[str] = field(default_factory=list)


class SplitRegenerator:
    """Group-aware, stratified split regeneration -- disabled by default.

    Algorithm: greedy group assignment. Groups (content clusters) are visited
    largest-first and each is placed in the split whose per-class deficit
    relative to its target is largest. Exact stratification under grouping
    constraints is NP-hard, so a deterministic greedy heuristic is both
    standard and honest; the resulting distributions are reported so the
    approximation is visible rather than assumed.

    A fixed, seeded shuffle breaks size ties, making the output reproducible
    across machines while avoiding the systematic bias of alphabetical order.

    Args:
        config: Validated data configuration.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def regenerate(self, manifest: pd.DataFrame, write: bool | None = None) -> SplitPlan:
        """Propose (and optionally write) a leakage-free partition.

        Args:
            manifest: Clean manifest from :mod:`src.data.cleaning`.
            write: Override the configured ``enabled`` flag. When ``None``, the
                configuration decides; when ``False``, the plan is computed and
                reported but nothing is written.

        Returns:
            The :class:`SplitPlan`.

        Raises:
            FileExistsError: If output CSVs exist and ``overwrite`` is ``False``.
        """
        settings = self.config.splits_policy.regenerate
        should_write = settings.enabled if write is None else write
        log_section(
            logger,
            f"Stage 3b / Split regeneration ({'writing' if should_write else 'dry run'})",
        )
        if not settings.enabled and write is None:
            logger.info(
                "split regeneration is disabled; computing a dry-run plan only. "
                "Enable splits_policy.regenerate.enabled to write new CSVs."
            )

        included = manifest[manifest["included"]].copy() if "included" in manifest else manifest.copy()
        included = included[included["label"].notna()]
        if included.empty:
            raise ValueError("cannot regenerate splits: no included, labelled images")

        included["group"] = self._group_keys(included, settings.group_by)
        assignments = self._assign(included, settings.ratios)
        plan_frame = included.assign(new_split=assignments)

        report = self._build_report(plan_frame, settings.ratios)
        written: list[str] = []
        if should_write:
            written = self._write(plan_frame, settings.output_dir, settings.overwrite)
            report["written_files"] = written

        self._log_summary(report)
        return SplitPlan(assignments=plan_frame, report=report, written_files=written)

    # -- grouping ----------------------------------------------------------

    @staticmethod
    def _group_keys(frame: pd.DataFrame, group_by: str) -> pd.Series:
        """Derive the grouping key.

        ``content_cluster`` takes the transitive closure of the perceptual
        clusters and the MD5 groups, so both exact and near duplicates land in
        the same partition. Taking the closure matters: an MD5 group and a
        perceptual cluster that overlap in a single image would otherwise be
        separable, letting byte-identical images straddle the boundary again.
        ``md5`` restricts grouping to byte-identical images, and ``none``
        degenerates to a plain stratified split (each image its own group).
        """
        if group_by == "none":
            return pd.Series([f"row_{i}" for i in range(len(frame))], index=frame.index, dtype=object)

        digests = frame["md5"].astype(str) if "md5" in frame else pd.Series("", index=frame.index)
        digests = digests.where(digests != "", "")
        if group_by == "md5":
            fallback = pd.Series([f"row_{i}" for i in range(len(frame))], index=frame.index)
            return digests.where(digests != "", fallback).astype(str)

        clusters = (
            frame["content_cluster"].astype(str)
            if "content_cluster" in frame
            else pd.Series([f"row_{i}" for i in range(len(frame))], index=frame.index, dtype=object)
        )
        return merge_group_keys(clusters, digests)

    # -- assignment --------------------------------------------------------

    def _assign(self, frame: pd.DataFrame, ratios: Mapping[str, float]) -> list[str]:
        """Assign every group to a split, balancing per-class targets."""
        splits = list(ratios)
        labels = sorted(frame["label"].astype(int).unique())
        total_per_label = frame["label"].astype(int).value_counts().to_dict()
        targets = {
            split: {label: ratios[split] * total_per_label.get(label, 0) for label in labels}
            for split in splits
        }
        current: dict[str, dict[int, float]] = {
            split: dict.fromkeys(labels, 0.0) for split in splits
        }

        groups: dict[str, list[int]] = defaultdict(list)
        for group, label in zip(frame["group"], frame["label"].astype(int)):
            groups[str(group)].append(label)

        # Largest groups first: they constrain the solution most. A seeded
        # shuffle precedes the stable size sort so that equally sized groups are
        # not systematically ordered alphabetically, while the result stays
        # reproducible for a given seed.
        rng = make_generator(self.config.seed, "split_regeneration")
        keys = sorted(groups)
        order = [keys[int(i)] for i in rng.permutation(len(keys))]
        order.sort(key=lambda key: -len(groups[key]))

        assignment: dict[str, str] = {}
        for group in order:
            counts = groups[group]
            best_split = max(
                splits,
                key=lambda split: sum(
                    targets[split][label] - current[split][label] for label in counts
                )
                / max(ratios[split], 1e-9),
            )
            assignment[group] = best_split
            for label in counts:
                current[best_split][label] += 1.0

        return [assignment[str(group)] for group in frame["group"]]

    # -- output ------------------------------------------------------------

    def _write(self, frame: pd.DataFrame, output_dir: str, overwrite: bool) -> list[str]:
        """Write the regenerated split CSVs to a versioned directory."""
        directory = ensure_dir(self.config.resolve_path(output_dir))
        id_column = self.config.columns.id_candidates[0]
        label_column = self.config.columns.label_candidates[0]
        file_names = {"train": "train.csv", "val": "valid.csv", "test": "test.csv"}

        written: list[str] = []
        for split, file_name in file_names.items():
            destination = directory / file_name
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"{destination} already exists; set splits_policy.regenerate.overwrite to true "
                    "or choose a different output_dir"
                )
            subset = frame[frame["new_split"] == split]
            table = pd.DataFrame(
                {
                    id_column: subset["id_code"].astype(str),
                    label_column: subset["label"].astype(int),
                }
            ).sort_values(id_column)
            table.to_csv(destination, index=False)
            written.append(str(destination))
            logger.info("wrote %d rows to %s", len(table), destination)

        # A provenance sidecar makes a regenerated split self-describing: which
        # config, which seed, which grouping produced it.
        write_json(
            directory / "regeneration_provenance.json",
            {
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "config_hash": self.config.config_hash,
                "seed": self.config.seed,
                "group_by": self.config.splits_policy.regenerate.group_by,
                "ratios": dict(self.config.splits_policy.regenerate.ratios),
                "caveat": (
                    "grouping uses content clusters as a proxy for patient/eye identity; "
                    "APTOS 2019 ships no patient identifier, so this prevents content "
                    "leakage but is not patient-level independence"
                ),
            },
        )
        return written

    def _build_report(self, frame: pd.DataFrame, ratios: Mapping[str, float]) -> dict[str, Any]:
        """Summarise the proposed partition and verify its group disjointness."""
        grades = list(range(self.config.classes.num_classes))
        proposed = frame.rename(columns={"split": "old_split", "new_split": "split"})
        table = class_distribution(proposed, self.config.classes.num_classes)
        split_table = table.drop(index="overall", errors="ignore")[grades]

        group_spread = proposed.groupby("group")["split"].nunique()
        md5_spread = (
            proposed[proposed["md5"].astype(str) != ""].groupby("md5")["split"].nunique()
            if "md5" in proposed
            else pd.Series(dtype=int)
        )

        return {
            "stage": "split_regeneration",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_hash": self.config.config_hash,
            "seed": self.config.seed,
            "enabled": self.config.splits_policy.regenerate.enabled,
            "group_by": self.config.splits_policy.regenerate.group_by,
            "target_ratios": dict(ratios),
            "achieved_ratios": {
                str(split): round(float(count) / float(len(proposed)), 4)
                for split, count in proposed["split"].value_counts().items()
            },
            "counts_per_split": {
                str(split): {str(grade): int(row[grade]) for grade in grades}
                for split, row in split_table.iterrows()
            },
            "groups": int(proposed["group"].nunique()),
            "groups_spanning_splits": int((group_spread > 1).sum()),
            "md5_groups_spanning_splits": int((md5_spread > 1).sum()) if len(md5_spread) else 0,
            "class_homogeneity_chi_square": chi_square_homogeneity(split_table),
            "caveat": (
                "content clusters proxy for patient/eye identity; APTOS ships no patient id"
            ),
        }

    @staticmethod
    def _log_summary(report: Mapping[str, Any]) -> None:
        """Log the achieved ratios and the disjointness guarantees."""
        logger.info("regeneration target ratios: %s", report.get("target_ratios"))
        logger.info("regeneration achieved ratios: %s", report.get("achieved_ratios"))
        logger.info(
            "groups %s | groups spanning splits %s | MD5 groups spanning splits %s",
            report.get("groups"),
            report.get("groups_spanning_splits"),
            report.get("md5_groups_spanning_splits"),
        )
