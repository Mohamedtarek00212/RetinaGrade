"""Training loop and engine.

Wires the model, dataloaders, optimizer, scheduler, and Eq. 9 total loss
into one ``fit()`` call, reusing every piece of already-built infrastructure
rather than re-implementing it:

* :func:`src.training.optim.build_optimizer` / :func:`src.training.scheduler.build_scheduler`
* :func:`src.losses.build_total_loss`
* :class:`src.training.amp.AMPContext`
* :class:`src.training.checkpoint.CheckpointManager`
* :class:`src.training.csv_logger.CSVEpochLogger` / :class:`src.training.tensorboard_logger.TensorBoardLogger`
* :class:`src.training.callbacks.EarlyStopping`
* :class:`src.evaluation.evaluator.Evaluator`
* :func:`src.utils.seed.set_seed`
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.evaluator import Evaluator
from src.losses.total_loss import TotalLoss
from src.training.amp import AMPContext
from src.training.callbacks import EarlyStopping
from src.training.checkpoint import CheckpointManager
from src.training.config import TrainingConfig
from src.training.csv_logger import CSVEpochLogger
from src.training.optim import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.tensorboard_logger import TensorBoardLogger
from src.utils.logger import get_logger, log_duration
from src.utils.seed import set_seed

__all__ = ["FitResult", "Trainer"]

logger = get_logger(__name__)


def _flatten_metrics(metrics: dict[str, dict[str, float] | float]) -> dict[str, float]:
    """Flatten a nested metrics dict for CSV/TensorBoard logging."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


@dataclass
class FitResult:
    """Outcome of one :meth:`Trainer.fit` call.

    Attributes:
        history: One dict of epoch metrics per completed epoch, in order.
            Values may be floats or nested per-class metric dicts.
        best_checkpoint_path: Path to the best checkpoint, if any epoch
            improved the monitored metric.
        stopped_early: Whether early stopping (if enabled) ended the run
            before ``config.epochs`` completed.
    """

    history: list[dict[str, Any]] = field(default_factory=list)
    best_checkpoint_path: Path | None = None
    stopped_early: bool = False


class Trainer:
    """Runs the full Dual-SwinOrd training loop.

    Args:
        model: The assembled :class:`~src.models.dual_swinord.DualSwinOrd`
            (or any module with the same ``forward()`` output contract).
        config: Validated training configuration.
        num_classes: ``K``, sourced from ``DataConfig.classes.num_classes``.
        device: Device to train on.
        class_weights: Optional ``[K]`` tensor, precomputed by
            :func:`src.data.statistics.compute_class_weights`, forwarded to
            the Classification Loss. ``None`` disables weighting.
        project_root: Root used to resolve every relative path in ``config``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        num_classes: int,
        device: torch.device,
        class_weights: torch.Tensor | None = None,
        class_names: dict[int, str] | list[str] | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.num_classes = num_classes
        self.device = device
        self.project_root = project_root or config.project_root

        set_seed(config.reproducibility.seed, deterministic=config.reproducibility.deterministic)

        self.model.to(device)
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler = build_scheduler(self.optimizer, config.scheduler, config.epochs)

        from src.losses import build_total_loss

        self.loss_fn: TotalLoss = build_total_loss(config.loss, num_classes, class_weights)
        self.loss_fn.to(device)

        self.amp = AMPContext(config.amp, device_type=device.type)
        self.checkpoint_manager = CheckpointManager(config.checkpoint, self.project_root)
        self.csv_logger = CSVEpochLogger(config.logging.log_dir, config.logging.csv_filename, self.project_root)
        self.tensorboard_logger = TensorBoardLogger(config.logging, self.project_root)
        self.early_stopping = EarlyStopping(config.early_stopping)
        self.evaluator = Evaluator(self.model, device, num_classes, class_names=class_names)

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Run one epoch of optimization; returns mean training losses."""
        self.model.train()
        total_loss_sum = 0.0
        classification_loss_sum = 0.0
        ordinal_loss_sum = 0.0
        num_samples = 0

        self.optimizer.zero_grad(set_to_none=True)
        num_batches = len(train_loader)
        epoch_start = time.perf_counter()
        pbar = tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{self.config.epochs} Train",
            unit="batch",
            dynamic_ncols=True,
        )
        for step, batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            batch_size = labels.size(0)

            with self.amp.autocast():
                outputs = self.model(images)
                losses = self.loss_fn(outputs, labels)
                scaled_loss = losses["total"] / self.config.gradient.accumulation_steps

            self.amp.backward(scaled_loss)

            is_accumulation_boundary = (step + 1) % self.config.gradient.accumulation_steps == 0
            is_last_batch = (step + 1) == num_batches
            if is_accumulation_boundary or is_last_batch:
                if self.config.gradient.clip_norm is not None:
                    self.amp.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.gradient.clip_norm)
                self.amp.step(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)

            total_loss_sum += float(losses["total"].item()) * batch_size
            classification_loss_sum += float(losses["classification"].item()) * batch_size
            ordinal_loss_sum += float(losses["ordinal"].item()) * batch_size
            num_samples += batch_size

            current_lr = self.optimizer.param_groups[0]["lr"]
            avg_loss = total_loss_sum / max(num_samples, 1)
            elapsed = time.perf_counter() - epoch_start
            img_per_sec = num_samples / elapsed if elapsed > 0 else 0.0
            pbar.set_postfix(
                {
                    "loss": f"{losses['total'].item():.4f}",
                    "avg": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "img/s": f"{img_per_sec:.1f}",
                }
            )

        self.scheduler.step()

        return {
            "train_loss": total_loss_sum / num_samples,
            "train_classification_loss": classification_loss_sum / num_samples,
            "train_ordinal_loss": ordinal_loss_sum / num_samples,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> FitResult:
        """Run the full training loop for ``config.epochs`` epochs.

        Args:
            train_loader: Training-split DataLoader.
            val_loader: Validation-split DataLoader, evaluated once per epoch.

        Returns:
            A :class:`FitResult` summarizing the run.
        """
        result = FitResult()
        total_train_start = time.perf_counter()

        for epoch in range(self.config.epochs):
            with log_duration(logger, f"epoch {epoch + 1}/{self.config.epochs}"):
                epoch_start = time.perf_counter()
                train_metrics = self._train_one_epoch(train_loader, epoch)
                train_time = time.perf_counter() - epoch_start

                val_start = time.perf_counter()
                val_pbar = tqdm(
                    val_loader,
                    total=len(val_loader),
                    desc=f"Epoch {epoch + 1}/{self.config.epochs} Val",
                    unit="batch",
                    dynamic_ncols=True,
                )
                eval_result = self.evaluator.evaluate(val_pbar, loss_fn=self.loss_fn)
                val_time = time.perf_counter() - val_start

                epoch_metrics = {
                    **train_metrics,
                    "val_loss": eval_result.loss,
                    "val_qwk": eval_result.metrics["qwk"],
                    "val_accuracy": eval_result.metrics["accuracy"],
                    "val_macro_f1": eval_result.metrics["macro_f1"],
                    "val_mae": eval_result.metrics["mae"],
                    "val_within_one_accuracy": eval_result.metrics["within_one_accuracy"],
                    "per_class_precision": eval_result.metrics["per_class_precision"],
                    "per_class_recall": eval_result.metrics["per_class_recall"],
                    "per_class_f1": eval_result.metrics["per_class_f1"],
                }

                logger.info(
                    "epoch %d/%d: train_loss=%.4f train_classification_loss=%.4f train_ordinal_loss=%.4f "
                    "val_loss=%.4f val_qwk=%.4f val_accuracy=%.4f val_macro_f1=%.4f val_mae=%.4f "
                    "val_within_one_accuracy=%.4f lr=%.6f",
                    epoch + 1,
                    self.config.epochs,
                    train_metrics["train_loss"],
                    train_metrics["train_classification_loss"],
                    train_metrics["train_ordinal_loss"],
                    eval_result.loss if eval_result.loss is not None else float("nan"),
                    eval_result.metrics["qwk"],
                    eval_result.metrics["accuracy"],
                    eval_result.metrics["macro_f1"],
                    eval_result.metrics["mae"],
                    eval_result.metrics["within_one_accuracy"],
                    train_metrics["lr"],
                )

                flat_epoch_metrics = _flatten_metrics(epoch_metrics)
                self.csv_logger.log(epoch, flat_epoch_metrics)
                self.tensorboard_logger.log_scalars(epoch, flat_epoch_metrics)

                print("[Checkpoint] Saving checkpoint...")
                written = self.checkpoint_manager.save(
                    epoch, self.model, self.optimizer, self.scheduler, epoch_metrics
                )
                print(f"[Checkpoint] Saved to: {written['epoch']}")
                is_best = "best" in written
                if is_best:
                    result.best_checkpoint_path = written["best"]
                    print(f"[Checkpoint] New best model saved to: {written['best']}")

                result.history.append(epoch_metrics)

                print("\n" + "=" * 50)
                print(f"Epoch {epoch + 1}/{self.config.epochs} completed")
                print(f"  Train time: {train_time:.2f}s")
                print(f"  Val time: {val_time:.2f}s")
                print(f"  Train loss: {epoch_metrics['train_loss']:.4f}")
                print(f"  Train classification loss: {epoch_metrics['train_classification_loss']:.4f}")
                print(f"  Train ordinal loss: {epoch_metrics['train_ordinal_loss']:.4f}")
                print(f"  Val loss: {epoch_metrics['val_loss']:.4f}")
                print(f"  QWK: {epoch_metrics['val_qwk']:.4f}")
                print(f"  Accuracy: {epoch_metrics['val_accuracy']:.4f}")
                print(f"  Macro F1: {epoch_metrics['val_macro_f1']:.4f}")
                print(f"  MAE: {epoch_metrics['val_mae']:.4f}")
                print(f"  Within-one accuracy: {epoch_metrics['val_within_one_accuracy']:.4f}")
                print(f"  LR: {epoch_metrics['lr']:.2e}")
                print(f"  Best model: {'YES' if is_best else 'NO'}")
                print("  Per-class Metrics:")
                per_class_precision = epoch_metrics["per_class_precision"]
                per_class_recall = epoch_metrics["per_class_recall"]
                per_class_f1 = epoch_metrics["per_class_f1"]
                for class_name in per_class_recall:
                    print(f"    {class_name}")
                    print(f"        Precision: {per_class_precision[class_name] * 100:.2f}%")
                    print(f"        Recall:    {per_class_recall[class_name] * 100:.2f}%")
                    print(f"        F1:        {per_class_f1[class_name] * 100:.2f}%")
                print("=" * 50)

                should_stop = self.early_stopping.step(epoch_metrics)
                if self.early_stopping.config.enabled:
                    print(
                        f"[EarlyStopping] counter: {self.early_stopping.num_bad_epochs} / "
                        f"{self.early_stopping.config.patience}"
                    )
                if should_stop:
                    result.stopped_early = True
                    print(f"[EarlyStopping] triggered after {epoch + 1} epochs")
                    break

        total_time = time.perf_counter() - total_train_start
        print(f"\n[Training] Total training time: {total_time:.2f}s ({total_time / 60:.2f} min)")

        self.tensorboard_logger.close()
        return result
