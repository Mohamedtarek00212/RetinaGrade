"""DataLoader factory.

A thin, configurable wrapper over :class:`torch.utils.data.DataLoader`. It
exposes only plumbing parameters -- batch size, worker count, pinning,
persistence, prefetching, drop-last, shuffling, and seeding.

Deliberately **not** implemented here:

* ``WeightedRandomSampler`` and any other sampler,
* oversampling, undersampling, or class-balanced batching,
* distributed samplers or any multi-process training logic.

Those are Training-milestone concerns. Resampling changes the effective
definition of an epoch and is entangled with the loss function and the ordinal
objective, so it must be introduced where it can be validated against
quadratic-weighted kappa -- not silently baked into data preparation, where it
would also make the augmentation ablations uninterpretable.

Reproducibility is handled by two mechanisms working together: a seeded
``torch.Generator`` fixes the shuffle order, and ``worker_init_fn`` seeds the
NumPy and Python RNGs inside each worker (PyTorch seeds neither, yet
Albumentations relies on both).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import DataConfig
from src.utils.logger import get_logger
from src.utils.seed import seed_worker

__all__ = ["build_dataloader", "build_dataloaders"]

logger = get_logger(__name__)


def build_dataloader(
    dataset: Dataset,
    config: DataConfig,
    split: str,
    shuffle: bool | None = None,
    batch_size: int | None = None,
    **overrides: Any,
) -> DataLoader:
    """Create a DataLoader for one split.

    Args:
        dataset: The dataset to wrap.
        config: Validated data configuration.
        split: ``"train"``, ``"val"``, or ``"test"``; selects the configured
            shuffling policy.
        shuffle: Explicit override of the configured shuffle flag.
        batch_size: Explicit override of the configured batch size.
        **overrides: Any other :class:`~torch.utils.data.DataLoader` keyword,
            forwarded verbatim.

    Returns:
        The configured DataLoader.

    Raises:
        ValueError: If ``split`` is unknown.

    Example:
        >>> loader = build_dataloader(dataset, config, "val")   # doctest: +SKIP
        >>> batch = next(iter(loader))                          # doctest: +SKIP
        >>> batch["image"].shape                                # doctest: +SKIP
        torch.Size([16, 3, 512, 512])
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}; expected train, val, or test")

    settings = config.dataloader
    workers = int(overrides.pop("num_workers", settings.num_workers))

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size if batch_size is not None else settings.batch_size),
        "shuffle": bool(settings.shuffle.get(split) if shuffle is None else shuffle),
        "num_workers": workers,
        "pin_memory": settings.pin_memory,
        "drop_last": settings.drop_last and split == "train",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    # Both options are invalid when the data is loaded in the main process.
    if workers > 0:
        kwargs["persistent_workers"] = settings.persistent_workers
        kwargs["prefetch_factor"] = settings.prefetch_factor
    kwargs.update(overrides)

    logger.info(
        "dataloader[%s]: batch_size=%s shuffle=%s num_workers=%s drop_last=%s",
        split,
        kwargs["batch_size"],
        kwargs["shuffle"],
        kwargs["num_workers"],
        kwargs["drop_last"],
    )
    return DataLoader(dataset, **kwargs)


def build_dataloaders(
    datasets: dict[str, Dataset], config: DataConfig, **overrides: Any
) -> dict[str, DataLoader]:
    """Create one DataLoader per available split.

    Args:
        datasets: Mapping of split name to dataset.
        config: Validated data configuration.
        **overrides: Forwarded to :func:`build_dataloader`.

    Returns:
        Mapping of split name to DataLoader.
    """
    return {
        split: build_dataloader(dataset, config, split, **overrides)
        for split, dataset in datasets.items()
    }
