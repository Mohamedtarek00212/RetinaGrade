"""Validation/test evaluation loop.

Runs the model in ``eval()`` mode over a full DataLoader, collecting
predictions, labels, and probabilities, then delegates all metric math to
:mod:`src.evaluation.metrics`, :mod:`src.evaluation.confusion_matrix`, and
:mod:`src.evaluation.calibration`.

Paper-explicit inference rule (Figure 1 caption, quoted, out of scope for
:meth:`src.models.dual_swinord.DualSwinOrd.forward` per that module's own
docstring): "The final prediction is derived via arg max over the
classification output, refined by the ordinal constraints." The refinement
half of that rule is not defined anywhere in the retrieved excerpts (no
equation or algorithm for how ordinal logits "refine" the argmax is given),
so this evaluator implements only the unambiguous half -- plain ``argmax``
over the classification logits -- and does **not** invent a refinement
procedure. The raw ordinal probabilities are still returned in
:attr:`EvaluationResult.ordinal_probs` so a future, explicitly-cited
refinement rule can be added without re-running inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics
from src.utils.logger import get_logger

__all__ = ["EvaluationResult", "Evaluator"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Raw predictions plus computed metrics for one evaluated split.

    Attributes:
        labels: ``[N]`` integer ground-truth grades.
        predictions: ``[N]`` integer predicted grades (argmax of
            ``classification_probs``).
        classification_probs: ``[N, K]`` softmax probabilities.
        ordinal_probs: ``[N, K-1]`` sigmoid "> k" probabilities.
        metrics: Output of :func:`src.evaluation.metrics.compute_all_metrics`.
        loss: Mean total loss over the split, if a loss function was supplied.
    """

    labels: np.ndarray
    predictions: np.ndarray
    classification_probs: np.ndarray
    ordinal_probs: np.ndarray
    metrics: dict[str, float | None]
    loss: float | None = None


class Evaluator:
    """Runs one full-split evaluation pass.

    Args:
        model: The model to evaluate; not switched back to ``train()`` mode
            by this class -- callers own that transition.
        device: Device to move batches to before the forward pass.
        num_classes: ``K``, sourced from ``DataConfig.classes.num_classes``.
        class_names: Optional mapping from class index to display name, used
            for per-class metric dicts.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        class_names: dict[int, str] | list[str] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, loss_fn: nn.Module | None = None) -> EvaluationResult:
        """Run inference over an entire DataLoader and compute metrics.

        Args:
            dataloader: Yields batches with ``"image"`` and ``"label"`` keys,
                matching :mod:`src.data.datasets`' sample contract.
            loss_fn: Optional loss module (for example
                :class:`~src.losses.total_loss.TotalLoss`) used to also
                report the mean split loss. ``None`` skips loss computation.

        Returns:
            The populated :class:`EvaluationResult`.
        """
        self.model.eval()

        all_labels: list[np.ndarray] = []
        all_classification_probs: list[np.ndarray] = []
        all_ordinal_probs: list[np.ndarray] = []
        loss_sum = 0.0
        loss_count = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(images)
            if loss_fn is not None:
                loss_dict = loss_fn(outputs, labels)
                loss_sum += float(loss_dict["total"].item()) * labels.size(0)
                loss_count += labels.size(0)

            classification_probs = torch.softmax(outputs["classification_logits"], dim=1)
            ordinal_probs = torch.sigmoid(outputs["ordinal_logits"])

            all_labels.append(labels.detach().cpu().numpy())
            all_classification_probs.append(classification_probs.detach().cpu().numpy())
            all_ordinal_probs.append(ordinal_probs.detach().cpu().numpy())

        labels_arr = np.concatenate(all_labels)
        classification_probs_arr = np.concatenate(all_classification_probs)
        ordinal_probs_arr = np.concatenate(all_ordinal_probs)
        predictions_arr = classification_probs_arr.argmax(axis=1)

        metrics = compute_all_metrics(
            labels_arr,
            predictions_arr,
            classification_probs_arr,
            self.num_classes,
            class_names=self.class_names,
        )
        mean_loss = (loss_sum / loss_count) if loss_count > 0 else None

        return EvaluationResult(
            labels=labels_arr,
            predictions=predictions_arr,
            classification_probs=classification_probs_arr,
            ordinal_probs=ordinal_probs_arr,
            metrics=metrics,
            loss=mean_loss,
        )
