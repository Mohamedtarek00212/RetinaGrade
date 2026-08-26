"""Swin Transformer backbone, sourced from ``timm``.

``timm>=1.0`` was already a pinned dependency in ``pyproject.toml`` before
this milestone began (it predates any model-architecture code), so using
``timm.create_model`` here introduces **no new dependency** -- see the
Dependency Policy in the Milestone 04 plan.

Paper Gap PG-01 / PG-02 (see ``docs/milestone_04_paper_gaps.md``): the
exact Swin variant, patch size, window size, pretrained-weight source, and
input resolution are not specified anywhere in the retrieved paper
excerpts. ``variant``, ``pretrained``, and ``image_size`` are therefore
**required** fields on :class:`~src.models.config.BackboneConfig` with no
default -- this adapter never chooses one on the paper's behalf.
"""

from __future__ import annotations

from pathlib import Path

import timm
from torch import Tensor

from src.models.backbones.base import RetinalBackbone
from src.models.config import BackboneConfig
from src.utils.logger import get_logger

__all__ = ["SwinBackboneAdapter"]

logger = get_logger(__name__)


def _pretrained_weights_cached(variant: str) -> bool:
    """Best-effort check for a previously-downloaded timm/HF-hub checkpoint."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:  # pragma: no cover - defensive
        HF_HUB_CACHE = str(Path.home() / ".cache" / "huggingface" / "hub")
    repo_id = f"timm/{variant}"
    repo_dir = f"models--{repo_id.replace('/', '--')}"
    cache_path = Path(HF_HUB_CACHE) / repo_dir
    if not cache_path.exists():
        return False
    # Look for any downloaded weight file.
    return any(cache_path.rglob(ext) for ext in ("*.bin", "*.safetensors", "*.pth"))


class SwinBackboneAdapter(RetinalBackbone):
    """Wraps a ``timm`` Swin Transformer as a :class:`RetinalBackbone`.

    Args:
        config: Backbone configuration; ``config.variant`` is passed to
            ``timm.create_model`` verbatim (PG-01).
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        if config.pretrained:
            if _pretrained_weights_cached(config.variant):
                logger.info("Found cached pretrained Swin weights for %s", config.variant)
            else:
                logger.info("Downloading pretrained Swin weights for %s...", config.variant)
            self._model = timm.create_model(
                config.variant,
                pretrained=True,
                features_only=True,
                img_size=config.image_size,
            )
            logger.info("Pretrained weights loaded for %s", config.variant)
        else:
            self._model = timm.create_model(
                config.variant,
                pretrained=False,
                features_only=True,
                img_size=config.image_size,
            )
        logger.info(
            "backbone: timm variant=%s pretrained=%s image_size=%s "
            "(variant/pretrained/image_size are PG-01/PG-02, not paper-confirmed)",
            config.variant,
            config.pretrained,
            config.image_size,
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """Return one ``[B, C_i, H_i, W_i]`` feature map per stage.

        ``timm``'s hierarchical Swin ``features_only=True`` output is
        channels-*last* (``[B, H, W, C]``), verified against the installed
        ``timm`` version rather than assumed; this permutes each stage to
        the channels-first (``[B, C, H, W]``) contract declared by
        :class:`~src.models.backbones.base.RetinalBackbone` so every
        downstream module can rely on a single, convolution-friendly
        layout regardless of backbone implementation.
        """
        return [feature.permute(0, 3, 1, 2).contiguous() for feature in self._model(x)]

    @property
    def out_channels(self) -> list[int]:
        """Per-stage channel counts, as reported by ``timm``'s feature_info."""
        return list(self._model.feature_info.channels())

    @property
    def out_strides(self) -> list[int]:
        """Per-stage downsampling factors, as reported by ``timm``'s feature_info."""
        return list(self._model.feature_info.reduction())
