"""Grading metrics.

The paper's own headline metric is Quadratic Weighted Kappa (QWK), reported
as ``0.9370`` on APTOS 2019 (abstract). This module reuses
:func:`sklearn.metrics.cohen_kappa_score` (``scikit-learn>=1.5`` is already
a pinned dependency) rather than reimplementing the QWK formula, plus a
handful of standard classification metrics needed to interpret model
behavior beyond the single headline number: accuracy, macro-F1,
mean-absolute-error (ordinal distance), within-one-grade accuracy, the
clinically-standard "referable DR" (grade >= 2) binary AUC/false-negative
rate, and per-class precision/recall/F1. All per-class metrics are computed
by collapsing the 5-way prediction, not by any dataset-level relabeling
(:mod:`src.data.datasets` continues to emit the full 5-class grade
unmodified).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "quadratic_weighted_kappa",
    "accuracy",
    "compute_per_class_precision",
    "compute_per_class_recall",
    "compute_per_class_f1",
    "macro_f1",
    "mean_absolute_error",
    "within_one_accuracy",
    "referable_auc",
    "referable_false_negative_rate",
    "compute_all_metrics",
]

#: Grade at/above which diabetic retinopathy is considered "referable"
#: (clinically standard threshold: moderate NPDR or worse, grade >= 2).
REFERABLE_THRESHOLD = 2


def quadratic_weighted_kappa(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    """Quadratic Weighted Kappa -- the paper's own headline metric.

    Args:
        labels: ``[N]`` integer ground-truth grades.
        predictions: ``[N]`` integer predicted grades.
        num_classes: ``K``.

    Returns:
        QWK in ``[-1, 1]``.
    """
    return float(cohen_kappa_score(labels, predictions, weights="quadratic", labels=list(range(num_classes))))


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Fraction of exactly correct grade predictions."""
    return float(np.mean(labels == predictions))


def macro_f1(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    """Macro-averaged F1 across all ``K`` grades (unweighted by class support)."""
    return float(f1_score(labels, predictions, labels=list(range(num_classes)), average="macro", zero_division=0))


def _build_class_name_map(
    num_classes: int,
    class_names: dict[int, str] | list[str] | None,
) -> dict[int, str]:
    """Build a class-index -> display-name map.

    Missing names fall back to ``class_{i}``.
    """
    if class_names is None:
        return {i: f"class_{i}" for i in range(num_classes)}
    if isinstance(class_names, list):
        return {i: name for i, name in enumerate(class_names)}
    return {int(k): v for k, v in class_names.items()}


def _compute_per_class_metric(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    class_names: dict[int, str] | list[str] | None,
    metric_fn: callable,
) -> dict[str, float]:
    """Compute a per-class metric via a sklearn ``average=None`` scorer."""
    name_map = _build_class_name_map(num_classes, class_names)
    labels_arr = np.asarray(labels)
    predictions_arr = np.asarray(predictions)
    scores = metric_fn(
        labels_arr,
        predictions_arr,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    return {name_map.get(i, f"class_{i}"): float(scores[i]) for i in range(num_classes)}


def compute_per_class_precision(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    class_names: dict[int, str] | list[str] | None = None,
) -> dict[str, float]:
    """Per-class Precision (positive predictive value) for each grade."""
    return _compute_per_class_metric(
        labels, predictions, num_classes, class_names, precision_score
    )


def compute_per_class_recall(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    class_names: dict[int, str] | list[str] | None = None,
) -> dict[str, float]:
    """Per-class Recall (sensitivity / true positive rate) for each grade."""
    return _compute_per_class_metric(
        labels, predictions, num_classes, class_names, recall_score
    )


def compute_per_class_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    class_names: dict[int, str] | list[str] | None = None,
) -> dict[str, float]:
    """Per-class F1-score (harmonic mean of precision and recall) for each grade."""
    return _compute_per_class_metric(
        labels, predictions, num_classes, class_names, f1_score
    )


def mean_absolute_error(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Mean absolute ordinal distance between predicted and true grade."""
    return float(np.mean(np.abs(labels.astype(np.float64) - predictions.astype(np.float64))))


def within_one_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Fraction of predictions within one grade of the ground truth."""
    return float(np.mean(np.abs(labels - predictions) <= 1))


def _referable_binary(grades: np.ndarray) -> np.ndarray:
    """Collapse 5-way grades to a binary "referable DR" (grade >= 2) label."""
    return (grades >= REFERABLE_THRESHOLD).astype(np.int64)


def referable_auc(labels: np.ndarray, classification_probs: np.ndarray) -> float | None:
    """AUC of the collapsed binary "referable DR" (grade >= 2) task.

    Args:
        labels: ``[N]`` integer ground-truth grades.
        classification_probs: ``[N, K]`` softmax probabilities from the
            Classification Head.

    Returns:
        The AUC, or ``None`` if only one class is present in ``labels``
        (``roc_auc_score`` is undefined in that degenerate case, which can
        happen on tiny validation splits).
    """
    binary_labels = _referable_binary(labels)
    if len(np.unique(binary_labels)) < 2:
        return None
    referable_prob = classification_probs[:, REFERABLE_THRESHOLD:].sum(axis=1)
    return float(roc_auc_score(binary_labels, referable_prob))


def referable_false_negative_rate(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    """False-negative rate of the collapsed binary "referable DR" task.

    Args:
        labels: ``[N]`` integer ground-truth grades.
        predictions: ``[N]`` integer predicted grades.

    Returns:
        ``FN / (FN + TP)``, or ``None`` if no referable-positive ground-truth
        sample exists in ``labels``.
    """
    binary_labels = _referable_binary(labels)
    binary_predictions = _referable_binary(predictions)
    positives = binary_labels == 1
    if positives.sum() == 0:
        return None
    false_negatives = np.sum(positives & (binary_predictions == 0))
    return float(false_negatives / positives.sum())


def compute_all_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    classification_probs: np.ndarray,
    num_classes: int,
    class_names: dict[int, str] | list[str] | None = None,
) -> dict[str, float | None | dict[str, float]]:
    """Compute every metric in this module in one call.

    Args:
        labels: ``[N]`` integer ground-truth grades.
        predictions: ``[N]`` integer predicted grades.
        classification_probs: ``[N, K]`` softmax probabilities.
        num_classes: ``K``.
        class_names: Optional mapping from class index to display name.

    Returns:
        A dict keyed by metric name. ``per_class_precision``, ``per_class_recall``
        and ``per_class_f1`` are nested dicts of class-name -> score.
    """
    return {
        "qwk": quadratic_weighted_kappa(labels, predictions, num_classes),
        "accuracy": accuracy(labels, predictions),
        "per_class_precision": compute_per_class_precision(labels, predictions, num_classes, class_names),
        "per_class_recall": compute_per_class_recall(labels, predictions, num_classes, class_names),
        "per_class_f1": compute_per_class_f1(labels, predictions, num_classes, class_names),
        "macro_f1": macro_f1(labels, predictions, num_classes),
        "mae": mean_absolute_error(labels, predictions),
        "within_one_accuracy": within_one_accuracy(labels, predictions),
        "referable_auc": referable_auc(labels, classification_probs),
        "referable_fnr": referable_false_negative_rate(labels, predictions),
    }
