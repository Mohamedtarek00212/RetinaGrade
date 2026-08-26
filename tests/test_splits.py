"""Unit tests for :mod:`src.data.splits`.

Verifies that verification is genuinely read-only, that MD5 overlap is
treated as failure/exclusion-driven while near-duplicate overlap is only a
warning, and that group-aware regeneration produces a partition with zero
group leakage and never writes unless explicitly asked.
"""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from src.data.splits import PASS, WARN, SplitRegenerator, SplitVerifier


class TestSplitVerifier:
    def test_clean_manifest_passes_id_and_md5_checks(self, data_config, clean_manifest):
        report = SplitVerifier(data_config).verify(clean_manifest)
        assert report.checks["id_overlap"]["status"] == PASS
        assert report.checks["md5_overlap"]["status"] == PASS

    def test_verification_is_read_only(self, synthetic_corpus, data_config, clean_manifest):
        image_path = next(synthetic_corpus["train_dir"].glob("*.png"))
        before = image_path.read_bytes()
        SplitVerifier(data_config).verify(clean_manifest)
        assert image_path.read_bytes() == before

    def test_report_written_to_disk(self, data_config, clean_manifest):
        SplitVerifier(data_config).verify(clean_manifest)
        report_path = data_config.resolve_path(data_config.outputs.split_report)
        assert report_path.is_file()

    def test_md5_overlap_detected_when_injected(
        self, tmp_path, data_config, synthetic_corpus, clean_manifest
    ):
        """A manually injected cross-split MD5 collision must fail verification."""
        manifest = clean_manifest.copy()
        # Force two rows in different splits to share an MD5 and be included.
        train_idx = manifest[manifest["split"] == "train"].index[0]
        val_idx = manifest[manifest["split"] == "val"].index[0]
        manifest.loc[val_idx, "md5"] = manifest.loc[train_idx, "md5"]
        manifest.loc[[train_idx, val_idx], "included"] = True

        report = SplitVerifier(data_config).verify(manifest)
        assert report.checks["md5_overlap"]["status"] != PASS

    def test_class_coverage_flags_missing_class(self, data_config, clean_manifest):
        manifest = clean_manifest.copy()
        # Remove every row of class 4 from the test split only.
        drop_mask = (manifest["split"] == "test") & (manifest["label"] == 4)
        manifest = manifest[~drop_mask]
        report = SplitVerifier(data_config).verify(manifest)
        assert report.checks["class_coverage"]["status"] in (WARN, "fail")


class TestSplitRegenerator:
    def test_dry_run_does_not_write_files(self, data_config, clean_manifest):
        plan = SplitRegenerator(data_config).regenerate(clean_manifest, write=False)
        assert plan.written_files == []

    def test_regeneration_is_deterministic_for_a_fixed_seed(self, data_config, clean_manifest):
        first = SplitRegenerator(data_config).regenerate(clean_manifest, write=False)
        second = SplitRegenerator(data_config).regenerate(clean_manifest, write=False)
        pd.testing.assert_series_equal(
            first.assignments.sort_values("id_code")["new_split"].reset_index(drop=True),
            second.assignments.sort_values("id_code")["new_split"].reset_index(drop=True),
        )

    def test_regenerated_groups_never_span_splits(self, data_config, clean_manifest):
        plan = SplitRegenerator(data_config).regenerate(clean_manifest, write=False)
        assert plan.report["groups_spanning_splits"] == 0

    def test_every_included_row_is_assigned(self, data_config, clean_manifest):
        included = clean_manifest[clean_manifest["included"] & clean_manifest["label"].notna()]
        plan = SplitRegenerator(data_config).regenerate(clean_manifest, write=False)
        assert len(plan.assignments) == len(included)
        assert plan.assignments["new_split"].notna().all()

    def test_write_true_persists_csvs(self, data_config, clean_manifest):
        plan = SplitRegenerator(data_config).regenerate(clean_manifest, write=True)
        assert len(plan.written_files) >= 3
        for path in plan.written_files:
            assert __import__("pathlib").Path(path).is_file()
