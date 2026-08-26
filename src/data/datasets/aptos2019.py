"""APTOS 2019 dataset adapter.

The only dataset-specific knowledge in the pipeline lives here: how to obtain a
manifest for a split. Two sources are supported, in priority order:

1. **The clean manifest** produced by :mod:`src.data.cleaning`. This is the
   correct source for any experiment, because it carries the ``included`` flag
   that resolves the cross-split MD5 duplicates the EDA found. Using it means
   the dataset cannot silently re-introduce leakage that cleaning removed.
2. **The raw split CSVs**, via :func:`src.data.audit.load_split_manifests`.
   A fallback for quick inspection before the pipeline has been run; it emits a
   warning, because those rows have not been cleaned.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.datasets.base import BaseRetinalDataset
from src.utils.config import DataConfig
from src.utils.logger import get_logger

__all__ = ["APTOS2019Dataset", "load_manifest_for_split"]

logger = get_logger(__name__)


def load_manifest_for_split(
    config: DataConfig,
    split: str,
    manifest: pd.DataFrame | None = None,
    include_only: bool = True,
) -> pd.DataFrame:
    """Return the manifest rows belonging to one split.

    Args:
        config: Validated data configuration.
        split: ``"train"``, ``"val"``, or ``"test"``.
        manifest: Pre-loaded clean manifest. When ``None``, the configured
            clean-manifest CSV is read; if that does not exist either, the raw
            split CSVs are used with a warning.
        include_only: Keep only rows cleaning marked as included.

    Returns:
        The filtered manifest with ``id_code``, ``path``, and ``label`` columns.

    Raises:
        ValueError: If the split has no usable rows.
    """
    frame = manifest
    if frame is None:
        clean_path = config.resolve_path(config.outputs.clean_manifest)
        if clean_path.is_file():
            frame = pd.read_csv(clean_path)
            logger.debug("loaded clean manifest from %s", clean_path)
        else:
            from src.data.audit import load_split_manifests

            logger.warning(
                "clean manifest not found at %s; falling back to the raw split CSVs. "
                "These rows have NOT been cleaned and may contain cross-split duplicates.",
                clean_path,
            )
            frame = load_split_manifests(config)
            frame["included"] = True

    subset = frame[frame["split"] == split]
    if include_only and "included" in subset.columns:
        subset = subset[subset["included"].astype(bool)]
    if "readable" in subset.columns:
        subset = subset[subset["readable"].astype(bool)]

    if subset.empty:
        raise ValueError(f"no usable rows for split '{split}' after filtering")

    columns = [c for c in ("id_code", "path", "label", "split", "quality_flags") if c in subset.columns]
    return subset[columns].reset_index(drop=True)


class APTOS2019Dataset(BaseRetinalDataset):
    """APTOS 2019 Blindness Detection dataset.

    Use :meth:`from_config` rather than the constructor: it wires the
    preprocessing pipeline, the augmentation policy, and the normalization
    statistics together in the one place where that wiring belongs.
    """

    #: Human-readable dataset identifier used in logs and manifests.
    dataset_name = "aptos2019"

    @classmethod
    def from_config(
        cls,
        config: DataConfig,
        split: str,
        manifest: pd.DataFrame | None = None,
        pipeline: "PreprocessingPipeline | None" = None,  # noqa: F821 - forward reference
        augmentation: list | None = None,
        stats: "NormalizationStats | None" = None,  # noqa: F821 - forward reference
        to_tensor: bool = True,
    ) -> APTOS2019Dataset:
        """Build a dataset for one split from the configuration.

        Args:
            config: Validated data configuration.
            split: ``"train"``, ``"val"``, or ``"test"``.
            manifest: Pre-loaded clean manifest.
            pipeline: Deterministic preprocessing pipeline; built from the
                configuration when omitted.
            augmentation: Training augmentations; built from the configuration
                when omitted. Ignored for non-training splits.
            stats: Normalization statistics; resolved from the configuration
                when omitted.
            to_tensor: Append tensor conversion. Set ``False`` for previews.

        Returns:
            The configured dataset.
        """
        # Imported here to keep the module importable without torch-side deps
        # and to avoid a circular import through the preprocessing package.
        from src.data.augmentation import build_train_transforms
        from src.data.preprocessing.normalization import build_normalization
        from src.data.preprocessing.pipeline import PreprocessingPipeline
        from src.data.statistics import resolve_normalization_stats

        rows = load_manifest_for_split(config, split, manifest=manifest)
        pipeline = pipeline or PreprocessingPipeline(config)
        if augmentation is None:
            augmentation = build_train_transforms(config) if split == "train" else []
        if stats is None:
            stats = resolve_normalization_stats(config, manifest=manifest, transform=pipeline)

        cache_dir: Path | None = None
        if config.preprocessing.cache.enabled:
            cache_dir = config.resolve_path(config.preprocessing.cache.dir) / config.preprocessing_hash

        return cls(
            manifest=rows,
            pipeline=pipeline,
            augmentation=augmentation,
            normalization=build_normalization(stats, to_tensor=to_tensor),
            split=split,
            cache_dir=cache_dir,
            cache_format=config.preprocessing.cache.format,
        )
