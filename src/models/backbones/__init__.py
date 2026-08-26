"""Backbone registry and factory.

Only one implementation is registered in this milestone --
:class:`~src.models.backbones.swin.SwinBackboneAdapter`, sourced from
``timm`` (already a pinned dependency, so no new dependency is
introduced). Its concrete variant/pretrained/resolution values remain open
Paper Gaps (PG-01, PG-02) -- see ``docs/milestone_04_paper_gaps.md``.
"""

from __future__ import annotations

from src.models.backbones.base import RetinalBackbone
from src.models.backbones.swin import SwinBackboneAdapter
from src.models.config import BackboneConfig
from src.models.registry import Registry

__all__ = ["RetinalBackbone", "SwinBackboneAdapter", "BACKBONE_REGISTRY", "build_backbone"]

BACKBONE_REGISTRY: Registry[RetinalBackbone] = Registry("backbone")
BACKBONE_REGISTRY.register("timm_swin")(SwinBackboneAdapter)


def build_backbone(
    config: BackboneConfig, registry: Registry[RetinalBackbone] | None = None
) -> RetinalBackbone:
    """Instantiate the backbone named by ``config.name``.

    Args:
        config: Backbone configuration.
        registry: Registry to resolve ``config.name`` against; defaults to
            the module-level :data:`BACKBONE_REGISTRY`. Tests may pass a
            local registry to avoid mutating global state.

    Returns:
        The instantiated backbone.
    """
    active_registry = registry or BACKBONE_REGISTRY
    return active_registry.build(config.name, config)
