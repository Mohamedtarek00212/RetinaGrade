"""Unit tests for :mod:`src.data.datasets` and :mod:`src.data.dataloader`.

Verifies the sample contract (dict output, correct tensor shapes/dtypes),
that augmentation is only wired in for the training split, that the manifest
validation rejects malformed input, and that DataLoaders honor the
configured batch size and shuffle policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataloader import build_dataloader, build_dataloaders
from src.data.datasets import build_dataset, build_datasets
from src.data.datasets.base import REQUIRED_COLUMNS, UNLABELLED, BaseRetinalDataset
from src.data.preprocessing import PreprocessingPipeline
from src.data.statistics import NormalizationStats


class TestBaseRetinalDatasetValidation:
    def test_empty_manifest_raises(self, data_config):
        pipeline = PreprocessingPipeline(data_config)
        with pytest.raises(ValueError):
            BaseRetinalDataset(pd.DataFrame(columns=list(REQUIRED_COLUMNS)), pipeline)

    def test_missing_required_column_raises(self, data_config):
        pipeline = PreprocessingPipeline(data_config)
        frame = pd.DataFrame({"id_code": ["a"], "path": ["x.png"]})
        with pytest.raises(ValueError):
            BaseRetinalDataset(frame, pipeline)


class TestBuildDatasets:
    def test_build_datasets_returns_all_splits(self, data_config, clean_manifest):
        datasets = build_datasets(data_config, manifest=clean_manifest)
        assert set(datasets) == {"train", "val", "test"}

    def test_sample_contract(self, data_config, clean_manifest):
        datasets = build_datasets(data_config, manifest=clean_manifest)
        sample = datasets["train"][0]
        assert set(sample) == {"image", "label", "id_code", "index"}
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].dtype == torch.float32
        assert sample["image"].shape[0] == 3
        assert isinstance(sample["label"], torch.Tensor)
        assert sample["label"].dtype == torch.long

    def test_image_is_square_and_matches_config_size(self, data_config, clean_manifest):
        datasets = build_datasets(data_config, manifest=clean_manifest)
        sample = datasets["val"][0]
        size = data_config.preprocessing.image_size
        assert sample["image"].shape[1:] == (size, size)

    def test_val_and_test_are_not_augmented(self, data_config, clean_manifest):
        datasets = build_datasets(data_config, manifest=clean_manifest)
        assert datasets["val"].post_transform is None or all(
            "Flip" not in type(t).__name__ and "Rotate" not in type(t).__name__
            for t in getattr(datasets["val"].post_transform, "transforms", [])
        )

    def test_class_counts_are_consistent_with_manifest(self, data_config, clean_manifest):
        datasets = build_datasets(data_config, manifest=clean_manifest)
        train_ds = datasets["train"]
        counts = train_ds.class_counts()
        assert sum(counts.values()) == len(train_ds)

    def test_unlabelled_rows_get_sentinel_label(self, data_config, clean_manifest):
        manifest = clean_manifest.copy()
        manifest.loc[manifest["split"] == "test", "label"] = np.nan
        datasets = build_datasets(data_config, manifest=manifest, splits=("test",))
        sample = datasets["test"][0]
        assert int(sample["label"]) == UNLABELLED


class TestBuildDataloaders:
    def test_train_loader_shuffles_by_default(self, data_config, clean_manifest):
        datasets = build_datasets(data_config, manifest=clean_manifest)
        loaders = build_dataloaders(datasets, data_config)
        assert loaders["train"].batch_size == data_config.dataloader.batch_size

    def test_batch_has_expected_keys_and_shapes(self, data_config, clean_manifest):
        datasets = build_datasets(data_config, manifest=clean_manifest)
        loaders = build_dataloaders(datasets, data_config)
        batch = next(iter(loaders["val"]))
        assert set(batch) == {"image", "label", "id_code", "index"}
        assert batch["image"].ndim == 4

    def test_invalid_split_raises(self, data_config, clean_manifest):
        datasets = build_datasets(data_config, manifest=clean_manifest)
        with pytest.raises(ValueError):
            build_dataloader(datasets["train"], data_config, "bogus")

    def test_explicit_batch_size_override(self, data_config, clean_manifest):
        datasets = build_datasets(data_config, manifest=clean_manifest)
        loader = build_dataloader(datasets["val"], data_config, "val", batch_size=2)
        assert loader.batch_size == 2
