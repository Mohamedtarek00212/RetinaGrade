"""Tests for `src.losses`."""

from __future__ import annotations

import pytest
import torch

from src.losses import build_total_loss
from src.losses.carm_loss import CARMLoss
from src.losses.classification_loss import ClassificationLoss
from src.losses.ordinal_loss import OrdinalLoss, ordinal_targets
from src.losses.total_loss import TotalLoss
from src.training.config import LossConfig


def test_ordinal_targets_thresholds() -> None:
    labels = torch.tensor([0, 2, 4])
    targets = ordinal_targets(labels, num_thresholds=4)
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert torch.equal(targets, expected)


def test_classification_loss_requires_valid_label_smoothing() -> None:
    with pytest.raises(ValueError):
        ClassificationLoss(label_smoothing=1.0)


def test_classification_loss_forward_shape_and_value() -> None:
    loss_fn = ClassificationLoss(label_smoothing=0.0)
    logits = torch.zeros(4, 5)
    labels = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(logits, labels)
    assert loss.shape == ()
    # Uniform logits -> uniform softmax -> loss == log(K).
    assert loss.item() == pytest.approx(torch.log(torch.tensor(5.0)).item(), abs=1e-5)


def test_ordinal_loss_forward_shape() -> None:
    loss_fn = OrdinalLoss(num_classes=5)
    logits = torch.randn(4, 4)
    labels = torch.tensor([0, 1, 2, 4])
    loss = loss_fn(logits, labels)
    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_carm_loss_matches_plain_ordinal_loss_by_default() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 5, (8,))

    ordinal = OrdinalLoss(num_classes=5)
    carm = CARMLoss(num_classes=5)
    assert carm(logits, labels).item() == pytest.approx(ordinal(logits, labels).item())


def test_carm_loss_describe_cites_paper_gap() -> None:
    carm = CARMLoss(num_classes=5)
    description = carm.describe()
    assert "PG-17" in description["paper_gap"]


def test_total_loss_combines_with_lambda() -> None:
    classification_loss = ClassificationLoss(label_smoothing=0.0)
    ordinal_loss = OrdinalLoss(num_classes=5)
    total_loss = TotalLoss(classification_loss, ordinal_loss, lambda_cls=0.5)

    outputs = {
        "classification_logits": torch.randn(4, 5),
        "ordinal_logits": torch.randn(4, 4),
    }
    labels = torch.tensor([0, 1, 2, 3])
    result = total_loss(outputs, labels)

    expected_total = 0.5 * result["classification"] + 0.5 * result["ordinal"]
    assert result["total"].item() == pytest.approx(expected_total.item())


def test_total_loss_rejects_lambda_out_of_range() -> None:
    classification_loss = ClassificationLoss(label_smoothing=0.0)
    ordinal_loss = OrdinalLoss(num_classes=5)
    with pytest.raises(ValueError):
        TotalLoss(classification_loss, ordinal_loss, lambda_cls=1.5)


def test_build_total_loss_from_config() -> None:
    config = LossConfig(label_smoothing=0.1, lambda_cls=0.3, carm_pos_weight_enabled=False)
    total_loss = build_total_loss(config, num_classes=5)

    assert isinstance(total_loss, TotalLoss)
    assert total_loss.lambda_cls == pytest.approx(0.3)
    assert isinstance(total_loss.ordinal_loss, CARMLoss)

    outputs = {
        "classification_logits": torch.randn(2, 5),
        "ordinal_logits": torch.randn(2, 4),
    }
    labels = torch.tensor([0, 4])
    result = total_loss(outputs, labels)
    assert torch.isfinite(result["total"])


def test_build_total_loss_with_class_weights() -> None:
    config = LossConfig(label_smoothing=0.0)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    total_loss = build_total_loss(config, num_classes=5, class_weights=weights)
    assert total_loss.classification_loss.class_weights is not None
    assert torch.equal(total_loss.classification_loss.class_weights, weights)
