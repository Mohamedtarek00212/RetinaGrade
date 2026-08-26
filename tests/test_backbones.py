"""Tests for `src.models.backbones`: the RetinalBackbone interface and adapter."""

from __future__ import annotations

import torch

from src.models.backbones import BACKBONE_REGISTRY, build_backbone
from src.models.backbones.base import RetinalBackbone
from src.models.backbones.swin import SwinBackboneAdapter
from src.models.config import BackboneConfig


def test_timm_swin_is_registered() -> None:
    assert "timm_swin" in BACKBONE_REGISTRY.available()
    assert BACKBONE_REGISTRY.get("timm_swin") is SwinBackboneAdapter


def test_swin_backbone_adapter_forward_shapes() -> None:
    config = BackboneConfig(
        name="timm_swin",
        variant="swin_tiny_patch4_window7_224",
        pretrained=False,
        image_size=224,
    )
    backbone = build_backbone(config)
    assert isinstance(backbone, RetinalBackbone)

    x = torch.randn(2, 3, config.image_size, config.image_size)
    features = backbone(x)

    assert len(features) == 4
    assert [f.shape[1] for f in features] == backbone.out_channels
    # Channels-first contract: verified NOT channels-last (a real timm quirk).
    for feature, channels in zip(features, backbone.out_channels):
        assert feature.shape[1] == channels
        assert feature.ndim == 4


def test_out_strides_are_increasing() -> None:
    config = BackboneConfig(
        name="timm_swin", variant="swin_tiny_patch4_window7_224", pretrained=False, image_size=224
    )
    backbone = build_backbone(config)
    strides = backbone.out_strides
    assert strides == sorted(strides)
