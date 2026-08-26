"""Tests for `src.models.heads`: ClassificationHead and OrdinalHead."""

from __future__ import annotations

import pytest
import torch

from src.models.dual_head import DualHead
from src.models.heads.classification_head import ClassificationHead
from src.models.heads.ordinal_head import OrdinalHead
from tests.model_doubles import FakeOrdinalHead


def test_classification_head_output_shape() -> None:
    head = ClassificationHead(hidden_dim=32, num_classes=5)
    embedding = torch.randn(4, 32)
    logits = head(embedding)
    assert logits.shape == (4, 5)


def test_ordinal_head_is_abstract() -> None:
    with pytest.raises(TypeError):
        OrdinalHead(num_classes=5)  # type: ignore[abstract]


def test_ordinal_head_thresholds_is_k_minus_one() -> None:
    head = FakeOrdinalHead(hidden_dim=32, num_classes=5)
    assert head.num_thresholds == 4
    embedding = torch.randn(4, 32)
    logits = head(embedding)
    assert logits.shape == (4, 4)


def test_dual_head_shares_the_identical_embedding() -> None:
    classification_head = ClassificationHead(hidden_dim=32, num_classes=5)
    ordinal_head = FakeOrdinalHead(hidden_dim=32, num_classes=5)
    dual_head = DualHead(classification_head, ordinal_head)

    embedding = torch.randn(4, 32)
    outputs = dual_head(embedding)

    assert outputs["classification_logits"].shape == (4, 5)
    assert outputs["ordinal_logits"].shape == (4, 4)
