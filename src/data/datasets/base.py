"""Dataset-agnostic base class for retinal grading datasets.

The base class owns everything that is *not* dataset-specific: manifest
validation, image loading, the optional preprocessed-image cache, transform
application, and the sample contract. A new corpus (EyePACS, DDR, Messidor)
therefore needs only a small adapter that knows how to build a manifest -- the
audit, cleaning, statistics, preprocessing, and augmentation stages stay
untouched.

Sample contract
---------------
``__getitem__`` returns a dictionary, not a tuple::

    {
        "image":   FloatTensor[C, H, W],
        "label":   LongTensor[],       # -1 when the split is unlabelled
        "id_code": str,
        "index":   int,
    }

A dictionary lets the Training milestone add ordinal targets, soft labels, or
per-sample weights without breaking every call site -- a tuple cannot be
extended without changing every consumer.

Caching
-------
When enabled, the deterministic geometric result is cached on disk under the
preprocessing-config hash. Decoding a 4288x2848 PNG dominates per-epoch cost by
one to two orders of magnitude over the augmentation itself, so this is the
single largest throughput lever available. The cache key includes the geometry
hash, so a stale cache can never be silently reused after a configuration
change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing.pipeline import PreprocessingPipeline
from src.utils.helpers import ensure_dir, read_image_rgb
from src.utils.logger import get_logger

__all__ = ["BaseRetinalDataset", "REQUIRED_COLUMNS", "UNLABELLED"]

logger = get_logger(__name__)

#: Columns every dataset manifest must provide.
REQUIRED_COLUMNS: tuple[str, ...] = ("id_code", "path", "label")

#: Label value used when a split carries no annotations.
UNLABELLED: int = -1


class BaseRetinalDataset(Dataset):
    """Manifest-driven dataset applying a preprocessing/augmentation pipeline.

    Args:
        manifest: One row per image with at least ``id_code``, ``path``, and
            ``label`` columns. Rows are used in the order given.
        pipeline: Deterministic preprocessing pipeline.
        augmentation: Augmentation transforms; applied only when ``split`` is
            ``"train"``.
        normalization: Terminal transforms (normalize + tensor conversion).
        split: Split name, used to decide whether augmentation applies.
        cache_dir: Directory for cached deterministic results, or ``None`` to
            disable caching.
        cache_format: ``"png"`` or ``"npy"``.

    Raises:
        ValueError: If the manifest is empty or missing a required column.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        pipeline: PreprocessingPipeline,
        augmentation: list[A.BasicTransform] | None = None,
        normalization: list[A.BasicTransform] | None = None,
        split: str = "train",
        cache_dir: str | Path | None = None,
        cache_format: str = "png",
    ) -> None:
        self._validate_manifest(manifest)
        self.manifest = manifest.reset_index(drop=True)
        self.split = split
        self.pipeline = pipeline

        stages: list[A.BasicTransform] = []
        if split == "train" and augmentation:
            stages.extend(augmentation)
        stages.extend(normalization or [])
        # Augmentation and normalization are composed separately from the
        # deterministic steps so a cached image can skip straight to this stage.
        self.post_transform = A.Compose(stages) if stages else None

        self.cache_dir = Path(cache_dir) / split if cache_dir else None
        self.cache_format = cache_format
        if self.cache_dir is not None:
            ensure_dir(self.cache_dir)

        logger.info(
            "%s[%s]: %d samples | augmentation %s | cache %s",
            type(self).__name__,
            split,
            len(self.manifest),
            "on" if (split == "train" and augmentation) else "off",
            self.cache_dir or "off",
        )

    # -- Dataset protocol --------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load, preprocess, and return one sample.

        Args:
            index: Row index into the manifest.

        Returns:
            The sample dictionary described in the module docstring.

        Raises:
            FileNotFoundError: If the image cannot be decoded. Cleaning is
                responsible for excluding unreadable files, so reaching this
                point means the manifest and the filesystem disagree -- a
                condition that must fail loudly rather than yield a black image.
        """
        row = self.manifest.iloc[index]
        image = self._load_processed(str(row["path"]), str(row["id_code"]))

        if self.post_transform is not None:
            image = self.post_transform(image=image)["image"]
        if not isinstance(image, torch.Tensor):
            # No ToTensorV2 in the pipeline (preview mode); hand back CHW anyway
            # so the contract holds regardless of configuration.
            image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

        label = row["label"]
        label_value = UNLABELLED if pd.isna(label) else int(label)

        return {
            "image": image,
            "label": torch.tensor(label_value, dtype=torch.long),
            "id_code": str(row["id_code"]),
            "index": int(index),
        }

    # -- loading -----------------------------------------------------------

    def _load_processed(self, path: str, id_code: str) -> np.ndarray:
        """Return the deterministically preprocessed image, using the cache."""
        cache_path = self._cache_path(id_code)
        if cache_path is not None and cache_path.is_file():
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        image = read_image_rgb(path)
        if image is None:
            raise FileNotFoundError(
                f"could not decode '{path}'; the manifest and the filesystem disagree. "
                "Re-run the audit and cleaning stages."
            )
        processed = self.pipeline(image)

        if cache_path is not None:
            self._write_cache(cache_path, processed)
        return processed

    def _cache_path(self, id_code: str) -> Path | None:
        """Return the cache file path for an id, or ``None`` when disabled."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{id_code}.{self.cache_format}"

    def _read_cache(self, path: Path) -> np.ndarray | None:
        """Read a cached image, returning ``None`` if it is unusable."""
        try:
            if self.cache_format == "npy":
                return np.load(path)
            return read_image_rgb(path)
        except (OSError, ValueError) as exc:
            logger.warning("ignoring unreadable cache entry %s: %s", path, exc)
            return None

    def _write_cache(self, path: Path, image: np.ndarray) -> None:
        """Write a preprocessed image to the cache, tolerating failures."""
        try:
            if self.cache_format == "npy":
                np.save(path, image)
                return
            import cv2

            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        except (OSError, ValueError) as exc:  # pragma: no cover - disk issues
            logger.warning("could not write cache entry %s: %s", path, exc)

    # -- validation --------------------------------------------------------

    @staticmethod
    def _validate_manifest(manifest: pd.DataFrame) -> None:
        """Check that the manifest can drive a dataset.

        Args:
            manifest: Candidate manifest.

        Raises:
            ValueError: If it is empty or missing a required column.
        """
        if manifest is None or len(manifest) == 0:
            raise ValueError("dataset manifest is empty; check the cleaning stage output")
        missing = [column for column in REQUIRED_COLUMNS if column not in manifest.columns]
        if missing:
            raise ValueError(f"dataset manifest is missing required column(s): {missing}")

    # -- introspection -----------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """Labels as an integer array, with ``-1`` for unlabelled rows.

        Exposed so the Training milestone can build class weights or a sampler
        from the dataset without re-reading the manifest. Nothing in this
        milestone consumes it.
        """
        return self.manifest["label"].fillna(UNLABELLED).astype(int).to_numpy()

    def class_counts(self) -> dict[int, int]:
        """Return the label histogram for this split."""
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}
