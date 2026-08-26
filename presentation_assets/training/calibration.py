"""Confidence calibration: Expected Calibration Error and reliability diagrams.

Not a paper-reported metric (the retrieved excerpts report only QWK,
accuracy, and F1); included because clinical deployment claims motivated by
the paper's high QWK (0.9370) also need the model's confidence to be
trustworthy, which QWK alone cannot show. Uses the standard, textbook
Expected Calibration Error (ECE) formulation -- not a Dual-SwinOrd-specific
mechanism.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.utils.helpers import ensure_dir

__all__ = ["expected_calibration_error", "reliability_diagram", "save_reliability_diagram"]


def expected_calibration_error(
    labels: np.ndarray, classification_probs: np.ndarray, num_bins: int = 10
) -> float:
    """Standard Expected Calibration Error over the top-1 predicted class.

    Args:
        labels: ``[N]`` integer ground-truth grades.
        classification_probs: ``[N, K]`` softmax probabilities.
        num_bins: Number of equal-width confidence bins in ``[0, 1]``.

    Returns:
        ECE in ``[0, 1]``; lower is better-calibrated.
    """
    confidences = classification_probs.max(axis=1)
    predictions = classification_probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = correct[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def reliability_diagram(
    labels: np.ndarray, classification_probs: np.ndarray, num_bins: int = 10
) -> dict[str, np.ndarray]:
    """Per-bin accuracy/confidence/count, for plotting or reporting.

    Args:
        labels: ``[N]`` integer ground-truth grades.
        classification_probs: ``[N, K]`` softmax probabilities.
        num_bins: Number of equal-width confidence bins in ``[0, 1]``.

    Returns:
        ``{"bin_centers", "bin_accuracy", "bin_confidence", "bin_count"}``,
        each a length-``num_bins`` array (``nan`` for empty bins).
    """
    confidences = classification_probs.max(axis=1)
    predictions = classification_probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracy = np.full(num_bins, np.nan)
    bin_confidence = np.full(num_bins, np.nan)
    bin_count = np.zeros(num_bins, dtype=np.int64)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        in_bin = (confidences > lo) & (confidences <= hi)
        bin_count[i] = in_bin.sum()
        if bin_count[i] > 0:
            bin_accuracy[i] = correct[in_bin].mean()
            bin_confidence[i] = confidences[in_bin].mean()

    return {
        "bin_centers": bin_centers,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_count": bin_count,
    }


def save_reliability_diagram(
    labels: np.ndarray,
    classification_probs: np.ndarray,
    output_dir: str | Path,
    basename: str = "reliability_diagram",
    num_bins: int = 10,
) -> Path:
    """Render and save a reliability diagram PNG.

    Args:
        labels: ``[N]`` integer ground-truth grades.
        classification_probs: ``[N, K]`` softmax probabilities.
        output_dir: Destination directory.
        basename: File stem.
        num_bins: Number of equal-width confidence bins.

    Returns:
        The written PNG path.
    """
    diagram = reliability_diagram(labels, classification_probs, num_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    valid = ~np.isnan(diagram["bin_accuracy"])
    ax.bar(
        diagram["bin_centers"][valid],
        diagram["bin_accuracy"][valid],
        width=1.0 / num_bins,
        edgecolor="black",
        alpha=0.7,
        label="model",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()

    path = ensure_dir(output_dir) / f"{basename}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
