"""Shared fixtures for the data-preparation test suite.

Every fixture builds a small synthetic corpus on disk rather than depending on
the real APTOS dataset, so the suite runs anywhere (CI included) without the
~10 GB download and finishes in seconds rather than minutes. Synthetic images
are non-trivial: each has a distinct fundus-like disc, distinct noise, and
distinct brightness, so hash-based and statistic-based logic exercises real
code paths instead of degenerate all-identical inputs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.models.config import ModelConfig, load_model_config
from src.utils.config import DataConfig, load_data_config


def _make_fundus(width: int, height: int, seed: int, radius_ratio: float = 0.42) -> np.ndarray:
    """Synthesize a plausible fundus photograph: a disc on a black field."""
    rng = np.random.default_rng(seed)
    image = (rng.random((height, width, 3)) * 8).astype(np.uint8)
    center = (width // 2, height // 2)
    radius = int(min(width, height) * radius_ratio)
    color = (
        int(90 + 40 * rng.random()),
        int(40 + 30 * rng.random()),
        int(15 + 15 * rng.random()),
    )
    cv2.circle(image, center, radius, color, thickness=-1)
    noise = (rng.random((height, width, 3)) * 10).astype(np.uint8)
    return cv2.add(image, noise)


@pytest.fixture()
def synthetic_corpus(tmp_path: Path) -> dict[str, Path]:
    """Write a small, distinct-content APTOS-style corpus with CSV manifests.

    Returns:
        Mapping with ``root``, per-split CSV paths, and per-split image dirs.
    """
    root = tmp_path / "corpus"
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True)

    layout = {"train": 16, "val": 6, "test": 6}
    paths: dict[str, Path] = {"root": root}

    for split, count in layout.items():
        image_dir = root / "raw" / split
        image_dir.mkdir(parents=True)
        rows = ["id_code,diagnosis"]
        for i in range(count):
            image = _make_fundus(360 + 4 * i, 260 + 3 * i, seed=hash((split, i)) % (2**31))
            cv2.imwrite(str(image_dir / f"{split}_{i}.png"), image)
            rows.append(f"{split}_{i},{i % 5}")
        csv_name = "valid.csv" if split == "val" else f"{split}.csv"
        csv_path = splits_dir / csv_name
        csv_path.write_text("\n".join(rows))
        paths[f"{split}_csv"] = csv_path
        paths[f"{split}_dir"] = image_dir

    return paths


@pytest.fixture()
def data_config(synthetic_corpus: dict[str, Path], tmp_path: Path) -> DataConfig:
    """A validated :class:`DataConfig` pointed at the synthetic corpus."""
    root = synthetic_corpus["root"]
    outputs = tmp_path / "outputs"

    def out(name: str, ext: str) -> str:
        return str(outputs / f"{name}{ext}")

    overrides = {
        "splits": {
            split: {
                "csv": str(synthetic_corpus[f"{split}_csv"]),
                "image_dir": str(synthetic_corpus[f"{split}_dir"]),
            }
            for split in ("train", "val", "test")
        },
        "outputs": {
            "root": str(outputs),
            "audit_manifest": out("audit_manifest", ".csv"),
            "audit_report": out("audit_report", ".json"),
            "clean_manifest": out("clean_manifest", ".csv"),
            "cleaning_report": out("cleaning_report", ".json"),
            "quarantine_manifest": out("quarantine_manifest", ".csv"),
            "split_report": out("split_report", ".json"),
            "statistics_report": out("statistics_report", ".json"),
            "class_distribution": out("class_distribution", ".csv"),
            "preview_dir": str(outputs / "preview"),
            "log_dir": str(tmp_path / "logs"),
        },
        "audit": {"num_workers": 1},
        "preprocessing": {
            "image_size": 224,
            "cache": {"enabled": False},
        },
        "statistics": {
            "normalization": {"cache_path": str(tmp_path / "norm_stats.json")},
            "imbalance": {"save_plot": False},
        },
        "cleaning": {
            "rules": {"near_duplicates": {"hamming_threshold": 4}},
        },
        "splits_policy": {
            "regenerate": {
                "output_dir": str(tmp_path / "splits_v2"),
                "overwrite": True,
            },
        },
        "dataloader": {
            "batch_size": 4,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
        },
    }
    return load_data_config(overrides=overrides)


@pytest.fixture()
def non_paper_model_config() -> ModelConfig:
    """A fully-specified, NOT paper-faithful :class:`ModelConfig`.

    Loaded from ``tests/fixtures/non_paper_test_config.yaml`` -- see that
    file and ``docs/milestone_04_paper_gaps.md`` before relying on any of
    its values outside a test.
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    return load_model_config(path=fixtures_dir / "non_paper_test_config.yaml")


@pytest.fixture()
def audit_manifest(data_config: DataConfig) -> pd.DataFrame:
    """Run Stage 1 once and return the resulting manifest."""
    from src.data.audit import DatasetAuditor

    return DatasetAuditor(data_config).run(force=True).to_frame()


@pytest.fixture()
def clean_manifest(data_config: DataConfig, audit_manifest: pd.DataFrame) -> pd.DataFrame:
    """Run Stage 2 once and return the resulting manifest."""
    from src.data.cleaning import DatasetCleaner

    return DatasetCleaner(data_config).run(audit_manifest).frame
