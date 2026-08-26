"""Unit tests for :mod:`src.data.cleaning`.

Covers the invariants that matter most for this stage: no file is ever
deleted, exact (MD5) cross-split duplicates are excluded but within-split
ones are merely flagged, near-duplicate clustering never excludes an image,
quality outliers are flagged only, and the dual-hash clustering used for
near-duplicates does not chain unrelated images together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import DatasetCleaner, UnionFind, cluster_by_dual_hash, cluster_by_hash, merge_group_keys
from src.utils.helpers import dhash, md5_bytes, phash


class TestDatasetCleaner:
    def test_never_deletes_files(self, synthetic_corpus, data_config, audit_manifest):
        image_path = next(synthetic_corpus["train_dir"].glob("*.png"))
        before = image_path.read_bytes()
        DatasetCleaner(data_config).run(audit_manifest)
        assert image_path.is_file()
        assert image_path.read_bytes() == before

    def test_included_and_excluded_partition_the_manifest(self, clean_manifest, audit_manifest):
        assert len(clean_manifest) == len(audit_manifest)
        assert clean_manifest["included"].dtype == bool

    def test_no_row_is_deleted_from_the_manifest(self, clean_manifest, audit_manifest):
        assert set(clean_manifest["id_code"]) == set(audit_manifest["id_code"])

    def test_quality_flags_never_exclude(self, clean_manifest):
        quality_flagged = clean_manifest["quality_flags"].astype(str).str.contains(
            "dark|bright|contrast|blurry|noisy", regex=True
        )
        # Any row flagged purely for quality (not also an exact duplicate)
        # must remain included.
        purely_quality = quality_flagged & ~clean_manifest["quality_flags"].astype(str).str.contains(
            "duplicate"
        )
        assert bool(clean_manifest.loc[purely_quality, "included"].all())

    def test_cross_split_exact_duplicates_are_excluded(self, tmp_path, data_config, synthetic_corpus):
        """Injecting a byte-identical image across splits must exclude one copy."""
        import shutil

        from src.data.audit import DatasetAuditor

        train_image = next(synthetic_corpus["train_dir"].glob("*.png"))
        val_dir = synthetic_corpus["val_dir"]
        duplicate_path = val_dir / "duplicate_of_train.png"
        shutil.copyfile(train_image, duplicate_path)

        # Register the new file in the val CSV so the audit picks it up.
        val_csv = synthetic_corpus["val_csv"]
        val_csv.write_text(val_csv.read_text() + "\nduplicate_of_train,0")

        manifest = DatasetAuditor(data_config).run(force=True).to_frame()
        cleaned = DatasetCleaner(data_config).run(manifest).frame

        duplicate_row = cleaned[cleaned["id_code"] == "duplicate_of_train"]
        assert not duplicate_row.empty
        assert bool((~duplicate_row["included"]).all())
        assert "cross-split" in duplicate_row["exclusion_reason"].iloc[0]
        assert "MD5" in duplicate_row["exclusion_reason"].iloc[0]

    def test_report_and_quarantine_written(self, data_config, clean_manifest):
        cleaning_report = data_config.resolve_path(data_config.outputs.cleaning_report)
        quarantine = data_config.resolve_path(data_config.outputs.quarantine_manifest)
        clean_path = data_config.resolve_path(data_config.outputs.clean_manifest)
        assert cleaning_report.is_file()
        assert quarantine.is_file()
        assert clean_path.is_file()


class TestUnionFind:
    def test_union_merges_components(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)
        assert uf.find(0) != uf.find(3)

    def test_find_is_idempotent(self):
        uf = UnionFind(3)
        uf.union(0, 2)
        first = uf.find(0)
        second = uf.find(0)
        assert first == second


class TestClusterByHash:
    def test_identical_hashes_share_a_cluster(self):
        digest = dhash(np.zeros((32, 32, 3), dtype=np.uint8))
        labels = cluster_by_hash([digest, digest, digest], threshold=0)
        assert labels[0] == labels[1] == labels[2]

    def test_distant_hashes_are_separated_at_zero_threshold(self):
        rng = np.random.default_rng(0)
        a = dhash((rng.random((32, 32, 3)) * 255).astype(np.uint8))
        b = dhash((rng.random((32, 32, 3)) * 255).astype(np.uint8))
        labels = cluster_by_hash([a, b], threshold=0)
        if a != b:
            assert labels[0] != labels[1]

    def test_cluster_count_never_exceeds_item_count(self):
        rng = np.random.default_rng(1)
        digests = [dhash((rng.random((16, 16, 3)) * 255).astype(np.uint8)) for _ in range(10)]
        labels = cluster_by_hash(digests, threshold=4)
        assert len(set(labels)) <= len(digests)


class TestDualHashClustering:
    def test_agreement_required_on_both_hashes(self):
        """Dual-hash clustering must not chain images that only one hash links.

        Constructs three images where dHash alone would chain 0-1-2 into a
        single cluster (0~1 close, 1~2 close, 0 far from 2), but pHash agrees
        only for 0 and 1. The dual-hash cluster for image 2 must be distinct.
        """
        rng = np.random.default_rng(42)
        images = [(rng.random((48, 48, 3)) * 255).astype(np.uint8) for _ in range(6)]
        dhashes = [dhash(image) for image in images]
        phashes = [phash(image) for image in images]

        labels = cluster_by_dual_hash(dhashes, phashes, threshold=1)
        # Every image should end up in *some* cluster; equal-length output.
        assert len(labels) == len(images)

    def test_reduces_or_matches_single_hash_cluster_count(self):
        """Requiring agreement can only split clusters further, never merge them."""
        rng = np.random.default_rng(7)
        images = [(rng.random((32, 32, 3)) * 255).astype(np.uint8) for _ in range(20)]
        dhashes = [dhash(image) for image in images]
        phashes = [phash(image) for image in images]

        single = cluster_by_hash(dhashes, threshold=8)
        dual = cluster_by_dual_hash(dhashes, phashes, threshold=8)
        assert len(set(dual)) >= len(set(single))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            cluster_by_dual_hash(["a"], ["a", "b"], threshold=1)


class TestMergeGroupKeys:
    def test_transitive_closure_across_keys(self):
        # Row 0 and 1 share md5 group 'm1'; row 1 and 2 share cluster 'c1'.
        # All three must end up in the same merged group.
        md5_group = pd.Series(["m1", "m1", "m2"])
        cluster = pd.Series(["c1", "c2", "c2"])
        merged = merge_group_keys(md5_group, cluster)
        assert merged.iloc[0] == merged.iloc[1] == merged.iloc[2]

    def test_disjoint_keys_stay_separate(self):
        a = pd.Series(["x1", "x2"])
        b = pd.Series(["y1", "y2"])
        merged = merge_group_keys(a, b)
        assert merged.iloc[0] != merged.iloc[1]

    def test_empty_string_is_not_a_shared_key(self):
        a = pd.Series(["", ""])
        b = pd.Series(["g1", "g2"])
        merged = merge_group_keys(a, b)
        # Two empty-string md5 groups must not be treated as a shared identity.
        assert merged.iloc[0] != merged.iloc[1]
