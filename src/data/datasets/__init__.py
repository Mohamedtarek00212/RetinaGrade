"""Per-dataset adapters and the dataset factory.

Adding a new retinal corpus means adding one module here and one entry in
:data:`DATASET_REGISTRY`. No stage of the pipeline -- audit, cleaning,
statistics, splits, preprocessing, augmentation -- needs to change.
"""

from __future__ import annotations

import pandas as pd

from src.data.datasets.aptos2019 import APTOS2019Dataset, load_manifest_for_split
from src.data.datasets.base import UNLABELLED, BaseRetinalDataset
from src.utils.config import DataConfig
from src.utils.logger import get_logger

__all__ = [
    "BaseRetinalDataset",
    "APTOS2019Dataset",
    "load_manifest_for_split",
    "DATASET_REGISTRY",
    "build_dataset",
    "build_datasets",
    "UNLABELLED",
]

logger = get_logger(__name__)

#: Dataset name (``dataset_name`` in the configuration) -> adapter class.
DATASET_REGISTRY: dict[str, type[BaseRetinalDataset]] = {
    "aptos2019": APTOS2019Dataset,
}


def build_dataset(config: DataConfig, split: str, **kwargs: object) -> BaseRetinalDataset:
    """Build the dataset for one split using the configured adapter.

    Args:
        config: Validated data configuration.
        split: ``"train"``, ``"val"``, or ``"test"``.
        **kwargs: Forwarded to the adapter's ``from_config``.

    Returns:
        The dataset.

    Raises:
        KeyError: If ``dataset_name`` has no registered adapter.
    """
    try:
        adapter = DATASET_REGISTRY[config.dataset_name]
    except KeyError as exc:
        raise KeyError(
            f"no dataset adapter registered for {config.dataset_name!r}; "
            f"available: {sorted(DATASET_REGISTRY)}"
        ) from exc
    return adapter.from_config(config, split, **kwargs)  # type: ignore[attr-defined]


def build_datasets(
    config: DataConfig,
    manifest: pd.DataFrame | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, BaseRetinalDataset]:
    """Build every split at once, sharing the pipeline and statistics.

    Sharing matters: resolving normalization statistics can trigger a pass over
    the training images, and it must happen once per process, not once per
    split.

    Args:
        config: Validated data configuration.
        manifest: Pre-loaded clean manifest.
        splits: Splits to build.

    Returns:
        Mapping of split name to dataset. Splits that cannot be built (for
        example an absent test set) are logged and omitted rather than raising,
        so a partial dataset directory remains usable.
    """
    from src.data.augmentation import build_train_transforms
    from src.data.preprocessing.pipeline import PreprocessingPipeline
    from src.data.statistics import resolve_normalization_stats

    pipeline = PreprocessingPipeline(config)
    stats = resolve_normalization_stats(config, manifest=manifest, transform=pipeline)
    augmentation = build_train_transforms(config)

    datasets: dict[str, BaseRetinalDataset] = {}
    for split in splits:
        try:
            datasets[split] = build_dataset(
                config,
                split,
                manifest=manifest,
                pipeline=pipeline,
                augmentation=augmentation if split == "train" else [],
                stats=stats,
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("skipping split '%s': %s", split, exc)
    return datasets
