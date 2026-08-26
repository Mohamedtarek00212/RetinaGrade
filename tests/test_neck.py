"""Tests for `src.models.neck.shared_feature_neck`: NeckPooling and SharedFeatureNeck."""

from __future__ import annotations

import pytest
import torch

from src.models.factories import activation_factory
from src.models.neck.shared_feature_neck import NeckPooling, SharedFeatureNeck
from tests.model_doubles import FakeNeckPooling


def test_neck_pooling_is_abstract() -> None:
    with pytest.raises(TypeError):
        NeckPooling()  # type: ignore[abstract]


def test_shared_feature_neck_forward_shape() -> None:
    neck = SharedFeatureNeck(
        pooling=FakeNeckPooling(),
        in_channels=16,
        hidden_dim=32,
        dropout=0.0,
        activation_factory=activation_factory("identity"),
    )
    feature_map = torch.randn(4, 16, 7, 7)
    embedding = neck(feature_map)
    assert embedding.shape == (4, 32)


def test_dropout_zero_is_a_no_op_in_eval_mode() -> None:
    neck = SharedFeatureNeck(
        pooling=FakeNeckPooling(),
        in_channels=16,
        hidden_dim=32,
        dropout=0.0,
        activation_factory=activation_factory("identity"),
    ).eval()
    feature_map = torch.randn(1, 16, 5, 5)
    first = neck(feature_map)
    second = neck(feature_map)
    assert torch.equal(first, second)
