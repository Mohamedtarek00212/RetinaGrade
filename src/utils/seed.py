"""Random seed control and determinism switches.

Reproducibility in a data pipeline has three distinct scopes, and conflating
them is the usual cause of "why did the same command give different numbers":

1. **Process-global seeding** (:func:`set_seed`) - Python, NumPy, and PyTorch
   (CPU and CUDA). Applied once at the start of every entry point.
2. **Per-worker seeding** (:func:`seed_worker`) - PyTorch DataLoader workers
   are forked/spawned copies that would otherwise share (or randomly differ in)
   their NumPy and Python RNG state. Derived deterministically from PyTorch's
   own per-worker base seed, so augmentation streams are reproducible *and*
   distinct across workers.
3. **Local, non-global randomness** (:func:`derive_seed`,
   :func:`temporary_numpy_seed`) - stage-scoped generators for sampling and
   split regeneration, so calling one stage never perturbs another stage's
   stream.

Torch is imported lazily: the audit, cleaning, statistics, and split stages
must run on machines where torch is unavailable or unnecessary.

Example
-------
>>> from src.utils.seed import set_seed, derive_seed
>>> set_seed(42)                                   # doctest: +SKIP
>>> derive_seed(42, "normalization_stats")         # deterministic, stage-scoped
2280279116
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from src.utils.logger import get_logger

__all__ = [
    "MAX_SEED",
    "set_seed",
    "seed_worker",
    "derive_seed",
    "make_generator",
    "temporary_numpy_seed",
]

logger = get_logger(__name__)

#: Upper bound for seeds accepted by NumPy's legacy ``RandomState`` (2**32).
MAX_SEED: int = 2**32


def set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> int:
    """Seed every RNG used by the project and optionally enforce determinism.

    Args:
        seed: Base seed. Must be non-negative.
        deterministic: If ``True``, request deterministic cuDNN/algorithm
            selection and set ``CUBLAS_WORKSPACE_CONFIG`` (required by CUDA
            >= 10.2 for deterministic matmuls). Slower, but mandatory for
            results that will be reported.
        cudnn_benchmark: cuDNN autotuning. Ignored (forced ``False``) when
            ``deterministic`` is ``True``, because autotuning selects different
            kernels across runs.

    Returns:
        The seed that was applied, for logging and manifest provenance.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % MAX_SEED)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch-free stages must still work
        logger.debug("torch is not installed; seeded Python and NumPy only")
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required for deterministic cuBLAS reductions; must be set before the
        # first CUDA context is created, hence the early call in every entry point.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError) as exc:  # pragma: no cover
            logger.warning("could not enable deterministic algorithms: %s", exc)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = cudnn_benchmark

    logger.debug("seeded all RNGs with %d (deterministic=%s)", seed, deterministic)
    return seed


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by PyTorch
    """Seed NumPy and Python RNGs inside a DataLoader worker.

    Intended to be passed as ``worker_init_fn``. PyTorch already gives each
    worker a distinct ``initial_seed()`` derived from the loader's generator;
    this propagates that value to the RNGs Albumentations and NumPy actually
    use, which PyTorch does not seed itself.

    Args:
        worker_id: Worker index supplied by PyTorch (unused; the base seed
            already encodes it).
    """
    import torch

    worker_seed = torch.initial_seed() % MAX_SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def derive_seed(base_seed: int, tag: str) -> int:
    """Derive a stable, stage-scoped seed from a base seed and a label.

    Using a derived seed per stage means adding, removing, or reordering a
    stage cannot shift the random stream of any other stage - a subtle but
    real reproducibility hazard when a pipeline grows.

    Args:
        base_seed: The global seed from the configuration.
        tag: Stable stage identifier, for example ``"split_regeneration"``.

    Returns:
        A deterministic seed in ``[0, 2**32)``.

    Example:
        >>> derive_seed(42, "a") == derive_seed(42, "a")
        True
        >>> derive_seed(42, "a") == derive_seed(42, "b")
        False
    """
    digest = hashlib.sha256(f"{base_seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def make_generator(base_seed: int, tag: str) -> np.random.Generator:
    """Create an isolated NumPy generator for a stage.

    Args:
        base_seed: The global seed from the configuration.
        tag: Stable stage identifier.

    Returns:
        A seeded :class:`numpy.random.Generator` that does not touch, and is
        not affected by, the global NumPy random state.
    """
    return np.random.default_rng(derive_seed(base_seed, tag))


@contextmanager
def temporary_numpy_seed(seed: int) -> Iterator[None]:
    """Temporarily set the global NumPy seed and restore the previous state.

    Useful for reproducible previews and plots without disturbing the RNG
    stream of the surrounding pipeline.

    Args:
        seed: Seed to apply inside the block.

    Yields:
        ``None``.
    """
    state = np.random.get_state()
    np.random.seed(seed % MAX_SEED)
    try:
        yield
    finally:
        np.random.set_state(state)
