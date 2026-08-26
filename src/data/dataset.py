"""Stable import surface for the dataset classes.

Intentionally contains **no logic**. The implementations live in
:mod:`src.data.datasets`, where each corpus gets its own adapter; this module
exists so that the layout documented in the README (``src/data/dataset.py``)
keeps working and so that existing imports do not break when a new dataset is
added.

Example:
    >>> from src.data.dataset import APTOS2019Dataset, build_dataset
"""

from src.data.datasets import (
    DATASET_REGISTRY,
    UNLABELLED,
    APTOS2019Dataset,
    BaseRetinalDataset,
    build_dataset,
    build_datasets,
    load_manifest_for_split,
)

__all__ = [
    "APTOS2019Dataset",
    "BaseRetinalDataset",
    "DATASET_REGISTRY",
    "UNLABELLED",
    "build_dataset",
    "build_datasets",
    "load_manifest_for_split",
]
