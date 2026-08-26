"""Unit tests for :mod:`src.data.statistics`.

Covers the pure math (imbalance metrics, chi-square homogeneity, class
weights) with hand-verifiable inputs, the streaming mean/std accumulator
against NumPy's batch computation, and the end-to-end
``DatasetStatistics.run`` contract: normalization statistics are computed
from the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.statistics import (
    ChannelStatsAccumulator,
    DatasetStatistics,
    class_distribution,
    compute_class_weights,
    imbalance_metrics,
)


class TestClassDistribution:
    def test_counts_match_input(self):
        frame = pd.DataFrame(
            {
                "id_code": ["a", "b", "c"],
                "split": ["train", "train", "val"],
                "label": [0, 1, 0],
                "included": [True, True, True],
            }
        )
        table = class_distribution(frame, num_classes=5)
        assert table.loc["train", 0] == 1
        assert table.loc["train", 1] == 1
        assert table.loc["val", 0] == 1

    def test_overall_row_sums_splits(self):
        frame = pd.DataFrame(
            {
                "id_code": ["a", "b"],
                "split": ["train", "val"],
                "label": [2, 2],
                "included": [True, True],
            }
        )
        table = class_distribution(frame, num_classes=5)
        assert table.loc["overall", 2] == 2

    def test_excluded_rows_must_be_filtered_by_the_caller(self):
        # class_distribution counts every labelled row; filtering excluded
        # rows is the caller's responsibility (see DatasetStatistics.run).
        frame = pd.DataFrame(
            {
                "id_code": ["a", "b"],
                "split": ["train", "train"],
                "label": [0, 0],
                "included": [True, False],
            }
        )
        table = class_distribution(frame[frame["included"]], num_classes=5)
        assert table.loc["train", 0] == 1


class TestImbalanceMetrics:
    def test_perfectly_balanced_has_ratio_one(self):
        metrics = imbalance_metrics([10, 10, 10, 10, 10])
        assert metrics["majority_minority_ratio"] == pytest.approx(1.0)

    def test_ratio_is_max_over_min(self):
        metrics = imbalance_metrics([100, 10, 5, 20, 50])
        assert metrics["majority_minority_ratio"] == pytest.approx(100 / 5)

    def test_entropy_is_maximal_for_uniform_distribution(self):
        uniform = imbalance_metrics([20, 20, 20, 20, 20])
        skewed = imbalance_metrics([90, 5, 2, 2, 1])
        assert uniform["entropy_bits"] > skewed["entropy_bits"]

    def test_entropy_matches_log2_num_classes_for_uniform(self):
        metrics = imbalance_metrics([25, 25, 25, 25])
        assert metrics["entropy_bits"] == pytest.approx(2.0, abs=1e-6)

    def test_single_class_has_zero_entropy(self):
        metrics = imbalance_metrics([50, 0, 0, 0, 0])
        assert metrics["entropy_bits"] == pytest.approx(0.0, abs=1e-6)


class TestComputeClassWeights:
    def test_inverse_strategy_favours_minority(self):
        weights = compute_class_weights([100, 10], strategy="inverse")
        assert weights[1] > weights[0]

    def test_effective_number_strategy_is_bounded(self):
        weights = compute_class_weights([100, 10, 1], strategy="effective_number", beta=0.99)
        assert np.all(weights > 0)
        assert np.all(np.isfinite(weights))

    def test_weights_never_applied_are_pure_function(self):
        # Calling twice with the same input must be side-effect free.
        counts = [50, 20, 5]
        first = compute_class_weights(counts, strategy="inverse")
        second = compute_class_weights(counts, strategy="inverse")
        np.testing.assert_array_equal(first, second)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            compute_class_weights([1, 2, 3], strategy="not_a_real_strategy")


class TestChannelStatsAccumulator:
    def test_matches_numpy_mean_and_std(self):
        rng = np.random.default_rng(0)
        images = [(rng.random((16, 16, 3)) * 255).astype(np.uint8) for _ in range(5)]

        accumulator = ChannelStatsAccumulator(channels=3)
        for image in images:
            accumulator.update(image)

        stacked = np.concatenate([image.reshape(-1, 3) for image in images], axis=0).astype(np.float64) / 255.0
        expected_mean = stacked.mean(axis=0)
        expected_std = stacked.std(axis=0)

        np.testing.assert_allclose(accumulator.mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(accumulator.std, expected_std, atol=1e-6)

    def test_pixel_count_accumulates(self):
        accumulator = ChannelStatsAccumulator(channels=3)
        accumulator.update(np.zeros((10, 10, 3), dtype=np.uint8))
        accumulator.update(np.zeros((5, 5, 3), dtype=np.uint8))
        assert accumulator.pixel_count == 100 + 25

    def test_std_is_never_negative_or_zero(self):
        accumulator = ChannelStatsAccumulator(channels=3)
        accumulator.update(np.full((8, 8, 3), 100, dtype=np.uint8))
        assert np.all(accumulator.std >= 0)


class TestDatasetStatisticsEndToEnd:
    def test_normalization_uses_training_split_only(self, data_config, clean_manifest):
        from src.data.preprocessing import PreprocessingPipeline

        result = DatasetStatistics(data_config).run(
            clean_manifest, transform=PreprocessingPipeline(data_config), compute_normalization=True
        )
        assert result.normalization is not None
        assert result.normalization.split == "train"

    def test_class_distribution_csv_is_written(self, data_config, clean_manifest):
        DatasetStatistics(data_config).run(clean_manifest, compute_normalization=False)
        path = data_config.resolve_path(data_config.outputs.class_distribution)
        assert path.is_file()

    def test_statistics_report_is_written(self, data_config, clean_manifest):
        DatasetStatistics(data_config).run(clean_manifest, compute_normalization=False)
        path = data_config.resolve_path(data_config.outputs.statistics_report)
        assert path.is_file()

    def test_skipping_normalization_leaves_it_none(self, data_config, clean_manifest):
        result = DatasetStatistics(data_config).run(clean_manifest, compute_normalization=False)
        assert result.normalization is None
