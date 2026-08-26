"""Final normalization and tensor conversion.

These two steps always run **last**, after augmentation, for a specific reason:
augmentation must operate on natural-range pixel values (brightness and
contrast jitter are defined in those units), and the normalization the network
sees must reflect the statistics of the pixels it is actually fed. Normalising
first and augmenting afterwards would silently shift the input distribution
away from the statistics that were computed.

Where the statistics come from is decided in :mod:`src.data.statistics`; this
module only consumes a resolved :class:`~src.data.statistics.NormalizationStats`
instance. That keeps the "how do we measure" question separate from the "how do
we apply" question, and lets the dataset be built from cached statistics
without recomputing anything.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.statistics import NormalizationStats
from src.utils.logger import get_logger

__all__ = ["build_normalization", "MAX_PIXEL_VALUE"]

logger = get_logger(__name__)

#: Albumentations scales by this value before applying mean/std, so statistics
#: expressed in ``[0, 1]`` map directly onto ``uint8`` inputs.
MAX_PIXEL_VALUE: float = 255.0


def build_normalization(stats: NormalizationStats, to_tensor: bool = True) -> list[A.BasicTransform]:
    """Build the terminal transforms of a pipeline.

    Args:
        stats: Resolved per-channel statistics in ``[0, 1]``.
        to_tensor: Append :class:`~albumentations.pytorch.ToTensorV2`, which
            converts ``H x W x C`` ``float32`` into a ``C x H x W`` tensor.
            Set to ``False`` when the pipeline output is a NumPy array (for
            previews, statistics, or caching).

    Returns:
        The transform list, ready to be appended to a ``Compose``.

    Raises:
        ValueError: If any standard deviation is non-positive, which would
            produce infinities on the first batch.

    Example:
        >>> from src.data.statistics import NormalizationStats
        >>> stats = NormalizationStats(mean=(0.4, 0.2, 0.1), std=(0.2, 0.1, 0.1), source="test")
        >>> len(build_normalization(stats))
        2
    """
    if any(value <= 0 for value in stats.std):
        raise ValueError(f"normalization std must be strictly positive, got {stats.std}")

    logger.debug(
        "normalization: mean=%s std=%s (source=%s)",
        [round(v, 4) for v in stats.mean],
        [round(v, 4) for v in stats.std],
        stats.source,
    )
    transforms: list[A.BasicTransform] = [
        A.Normalize(mean=tuple(stats.mean), std=tuple(stats.std), max_pixel_value=MAX_PIXEL_VALUE, p=1.0)
    ]
    if to_tensor:
        transforms.append(ToTensorV2())
    return transforms
