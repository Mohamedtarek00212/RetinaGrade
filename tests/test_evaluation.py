"""Tests for `src.evaluation`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.calibration import expected_calibration_error, reliability_diagram
from src.evaluation.confusion_matrix import build_confusion_matrix, save_confusion_matrix
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import (
    accuracy,
    compute_all_metrics,
    compute_per_class_f1,
    compute_per_class_precision,
    compute_per_class_recall,
    macro_f1,
    mean_absolute_error,
    quadratic_weighted_kappa,
    referable_auc,
    referable_false_negative_rate,
    within_one_accuracy,
)


def test_quadratic_weighted_kappa_perfect_agreement() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    assert quadratic_weighted_kappa(labels, labels, num_classes=5) == 1.0


def test_accuracy_and_mae() -> None:
    labels = np.array([0, 1, 2, 3])
    predictions = np.array([0, 1, 1, 3])
    assert accuracy(labels, predictions) == 0.75
    assert mean_absolute_error(labels, predictions) == 0.25


def test_macro_f1_all_correct() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    assert macro_f1(labels, labels, num_classes=5) == 1.0


def test_within_one_accuracy() -> None:
    labels = np.array([0, 2, 4])
    predictions = np.array([1, 2, 2])
    assert within_one_accuracy(labels, predictions) == pytest.approx(2.0 / 3.0)


def test_referable_auc_and_fnr() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    probs = np.eye(5)  # perfect one-hot confidence
    predictions = probs.argmax(axis=1)

    auc = referable_auc(labels, probs)
    assert auc == 1.0

    fnr = referable_false_negative_rate(labels, predictions)
    assert fnr == 0.0


def test_referable_auc_returns_none_for_single_class() -> None:
    labels = np.array([0, 0, 1])  # all non-referable
    probs = np.random.rand(3, 5)
    assert referable_auc(labels, probs) is None


def test_compute_all_metrics_keys() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    predictions = np.array([0, 1, 2, 3, 4])
    probs = np.eye(5)
    class_names = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
    metrics = compute_all_metrics(labels, predictions, probs, num_classes=5, class_names=class_names)
    assert set(metrics) == {
        "qwk",
        "accuracy",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "macro_f1",
        "mae",
        "within_one_accuracy",
        "referable_auc",
        "referable_fnr",
    }
    assert isinstance(metrics["per_class_precision"], dict)
    assert isinstance(metrics["per_class_recall"], dict)
    assert isinstance(metrics["per_class_f1"], dict)
    assert set(metrics["per_class_recall"]) == set(class_names.values())
    assert metrics["qwk"] == 1.0


def test_per_class_metrics_perfect_predictions() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    predictions = np.array([0, 1, 2, 3, 4])
    class_names = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
    precision = compute_per_class_precision(labels, predictions, num_classes=5, class_names=class_names)
    recall = compute_per_class_recall(labels, predictions, num_classes=5, class_names=class_names)
    f1 = compute_per_class_f1(labels, predictions, num_classes=5, class_names=class_names)
    for name in class_names.values():
        assert precision[name] == 1.0
        assert recall[name] == 1.0
        assert f1[name] == 1.0


def test_build_confusion_matrix_shape() -> None:
    labels = np.array([0, 1, 1, 2])
    predictions = np.array([0, 1, 2, 2])
    matrix = build_confusion_matrix(labels, predictions, num_classes=3)
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 4


def test_save_confusion_matrix_writes_files(tmp_path: Path) -> None:
    matrix = build_confusion_matrix(np.array([0, 1]), np.array([0, 1]), num_classes=2)
    written = save_confusion_matrix(matrix, tmp_path, save_plot=True)
    assert written["csv"].exists()
    assert written["json"].exists()
    assert written["plot"].exists()


def test_expected_calibration_error_perfect_calibration() -> None:
    labels = np.array([0, 0, 0, 0])
    probs = np.tile(np.array([[1.0, 0.0]]), (4, 1))
    ece = expected_calibration_error(labels, probs, num_bins=10)
    assert ece == pytest.approx(0.0)


def test_reliability_diagram_shapes() -> None:
    labels = np.array([0, 1, 0, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    diagram = reliability_diagram(labels, probs, num_bins=5)
    assert diagram["bin_centers"].shape == (5,)
    assert diagram["bin_count"].sum() == 4


class _DictDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "label": self.labels[index]}


class _DummyModel(nn.Module):
    """Outputs logits that always predict the correct grade for tests."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        # Uses the first pixel (which we encode with the label) to fabricate
        # confident, correct logits -- purely a test double.
        labels = x[:, 0, 0, 0].long()
        classification_logits = torch.zeros(batch_size, self.num_classes)
        classification_logits.scatter_(1, labels.unsqueeze(1), 10.0)
        ordinal_logits = torch.zeros(batch_size, self.num_classes - 1)
        return {"classification_logits": classification_logits, "ordinal_logits": ordinal_logits}


def test_evaluator_evaluate_end_to_end() -> None:
    num_classes = 5
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    images = torch.zeros(len(labels), 3, 4, 4)
    images[:, 0, 0, 0] = labels.float()
    dataset = _DictDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = _DummyModel(num_classes)
    evaluator = Evaluator(model, device=torch.device("cpu"), num_classes=num_classes)
    result = evaluator.evaluate(loader)

    assert np.array_equal(result.predictions, labels.numpy())
    assert result.metrics["qwk"] == 1.0
    assert result.metrics["accuracy"] == 1.0
    assert result.loss is None
