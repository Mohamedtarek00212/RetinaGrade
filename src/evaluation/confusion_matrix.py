"""Confusion matrix generation and persistence.

Reuses :func:`sklearn.metrics.confusion_matrix` for the counting logic and
:func:`src.utils.helpers.write_json`/``write_csv`` for atomic persistence,
plus Matplotlib/Seaborn (both already pinned dependencies, used the same way
by :mod:`src.data.statistics`) for the plot, rather than reimplementing any
of the three.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix

from src.utils.helpers import ensure_dir, write_csv, write_json

__all__ = ["build_confusion_matrix", "save_confusion_matrix"]


def build_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    """Return the ``[K, K]`` confusion matrix (rows=true, columns=predicted).

    Args:
        labels: ``[N]`` integer ground-truth grades.
        predictions: ``[N]`` integer predicted grades.
        num_classes: ``K``.

    Returns:
        A ``[K, K]`` integer NumPy array.
    """
    return _sk_confusion_matrix(labels, predictions, labels=list(range(num_classes)))


def save_confusion_matrix(
    matrix: np.ndarray,
    output_dir: str | Path,
    basename: str = "confusion_matrix",
    save_plot: bool = True,
) -> dict[str, Path]:
    """Persist a confusion matrix as CSV, JSON, and (optionally) a PNG heatmap.

    Args:
        matrix: ``[K, K]`` confusion matrix, as returned by
            :func:`build_confusion_matrix`.
        output_dir: Destination directory.
        basename: File stem shared by all written artifacts.
        save_plot: Whether to also render a PNG heatmap.

    Returns:
        Mapping of artifact kind (``"csv"``, ``"json"``, and optionally
        ``"plot"``) to the written path.
    """
    directory = ensure_dir(output_dir)
    num_classes = matrix.shape[0]

    rows = [
        {"true_grade": i, **{f"pred_{j}": int(matrix[i, j]) for j in range(num_classes)}}
        for i in range(num_classes)
    ]
    csv_path = write_csv(directory / f"{basename}.csv", rows)
    json_path = write_json(directory / f"{basename}.json", {"matrix": matrix.tolist()})

    written = {"csv": csv_path, "json": json_path}
    if save_plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted grade")
        ax.set_ylabel("True grade")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        plot_path = directory / f"{basename}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        written["plot"] = plot_path

    return written
