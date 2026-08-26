"""Unit tests for :mod:`src.data.audit`.

Verifies the audit stage's core contract: it is read-only (no file is ever
modified or deleted), every configured split is represented, per-image
metrics are populated, and re-running with caching enabled is idempotent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.audit import AuditRecord, DatasetAuditor


class TestDatasetAuditor:
    def test_manifest_has_one_row_per_image(self, data_config, audit_manifest):
        assert len(audit_manifest) == 16 + 6 + 6

    def test_manifest_contains_required_columns(self, audit_manifest):
        missing = [c for c in AuditRecord.column_names() if c not in audit_manifest.columns]
        assert not missing

    def test_all_splits_represented(self, audit_manifest):
        assert set(audit_manifest["split"]) == {"train", "val", "test"}

    def test_synthetic_images_are_all_readable(self, audit_manifest):
        assert bool(audit_manifest["readable"].all())

    def test_md5_present_and_nonempty(self, audit_manifest):
        assert (audit_manifest["md5"].astype(str).str.len() > 0).all()

    def test_perceptual_hashes_present(self, audit_manifest):
        assert (audit_manifest["dhash"].astype(str).str.len() > 0).all()
        assert (audit_manifest["phash"].astype(str).str.len() > 0).all()

    def test_quality_metrics_are_finite(self, audit_manifest):
        for column in ("brightness", "contrast", "sharpness_norm", "noise_sigma"):
            assert audit_manifest[column].notna().all()

    def test_is_read_only(self, synthetic_corpus, data_config):
        """Auditing must never modify or delete source images."""
        image_path = next((synthetic_corpus["train_dir"]).glob("*.png"))
        before_bytes = image_path.read_bytes()
        DatasetAuditor(data_config).run(force=True)
        after_bytes = image_path.read_bytes()
        assert before_bytes == after_bytes

    def test_rerun_with_cache_matches_forced_run(self, data_config):
        first = DatasetAuditor(data_config).run(force=True).to_frame()
        second = DatasetAuditor(data_config).run(force=False).to_frame()
        # "error" round-trips as "" in memory but NaN through the CSV cache;
        # normalize before comparing since this is a serialisation artifact.
        first["error"] = first["error"].mask(first["error"] == "", np.nan)
        second["error"] = second["error"].mask(second["error"] == "", np.nan)
        pd.testing.assert_frame_equal(
            first.sort_values("id_code").reset_index(drop=True),
            second.sort_values("id_code").reset_index(drop=True),
            check_dtype=False,
        )

    def test_report_written_to_disk(self, data_config, audit_manifest):
        report_path = data_config.resolve_path(data_config.outputs.audit_report)
        manifest_path = data_config.resolve_path(data_config.outputs.audit_manifest)
        assert report_path.is_file()
        assert manifest_path.is_file()
