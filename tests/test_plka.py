"""Tests for `src.models.attention.plka`: PLKA and the PLKAFusion interface."""

from __future__ import annotations

import pytest
import torch

from src.models.attention.plka import PLKA, PLKA_DILATION_RATES, PLKAFusion
from src.models.factories import activation_factory, normalization_factory
from tests.model_doubles import FakePLKAFusion


def test_dilation_rates_are_paper_fixed() -> None:
    assert PLKA_DILATION_RATES == (1, 2, 3)


def test_plka_fusion_is_abstract() -> None:
    with pytest.raises(TypeError):
        PLKAFusion(channels=8)  # type: ignore[abstract]


def test_plka_forward_shape_is_preserved() -> None:
    channels = 8
    fusion = FakePLKAFusion(channels=channels)
    plka = PLKA(
        channels=channels,
        kernel_size=3,
        activation_factory=activation_factory("relu"),
        normalization_factory=normalization_factory("batch_norm_2d"),
        fusion=fusion,
    )
    x = torch.randn(2, channels, 7, 7)
    output = plka(x)
    assert output.shape == x.shape


def test_plka_has_exactly_three_branches() -> None:
    fusion = FakePLKAFusion(channels=8)
    plka = PLKA(
        channels=8,
        kernel_size=3,
        activation_factory=activation_factory("identity"),
        normalization_factory=normalization_factory("identity"),
        fusion=fusion,
    )
    assert len(plka.branches) == len(PLKA_DILATION_RATES) == 3
