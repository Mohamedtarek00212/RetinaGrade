"""End-to-end training entry point for Dual-SwinOrd on APTOS 2019.

Run with:

    python scripts/train.py

or, from the repository root:

    python -m scripts.train
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataloader import build_dataloader
from src.data.datasets import BaseRetinalDataset, build_datasets
from src.evaluation.calibration import expected_calibration_error, save_reliability_diagram
from src.evaluation.confusion_matrix import build_confusion_matrix, save_confusion_matrix
from src.models import build_model
from src.models.config import load_model_config
from src.models.semantic_prior.text_adapter import HashingTextAdapter
from src.training.config import load_training_config
from src.training.manifest import write_run_manifest
from src.training.trainer import Trainer
from src.utils.config import load_data_config
from src.utils.helpers import ensure_dir, write_json
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

#: Clinical prompts used to derive semantic priors. The paper does not report
#: the exact prompt text (PG-04), so these are implementation assumptions
#: (not explicitly specified in the paper). Replace once the paper's prompts are known.
DEFAULT_TEXT_PROMPTS: list[str] = [
    "No diabetic retinopathy: healthy retina with no visible lesions.",
    "Mild diabetic retinopathy: presence of microaneurysms only.",
    "Moderate diabetic retinopathy: microaneurysms, dot-blot hemorrhages, and hard exudates.",
    "Severe diabetic retinopathy: extensive hemorrhages, venous beading, and intraretinal microvascular abnormalities.",
    "Proliferative diabetic retinopathy: neovascularization, preretinal hemorrhage, or fibrovascular proliferation.",
]


def _print_section(title: str) -> None:
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


def _print_done(message: str) -> None:
    print(f"[OK] {message}")


def _count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dual-SwinOrd on APTOS 2019")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--output-dir", default="outputs/final_report")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs (default: use config).")
    parser.add_argument(
        "--no-clean-epoch-checkpoints",
        dest="clean_epoch_checkpoints",
        action="store_false",
        default=True,
        help="Keep per-epoch checkpoint files after training (default: removed).",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_dataloaders_for_run(
    data_config,
) -> tuple[dict[str, BaseRetinalDataset], BaseRetinalDataset, DataLoader, BaseRetinalDataset, DataLoader]:
    """Build train and validation datasets and DataLoaders from the data configuration."""
    datasets = build_datasets(data_config)
    if "train" not in datasets:
        raise RuntimeError("Training dataset could not be built; check data/splits/train.csv and data/raw/train")
    if "val" not in datasets:
        raise RuntimeError("Validation dataset could not be built; check data/splits/valid.csv and data/raw/val")

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    train_loader = build_dataloader(train_dataset, data_config, "train")
    val_loader = build_dataloader(val_dataset, data_config, "val")
    return datasets, train_dataset, train_loader, val_dataset, val_loader


def compute_class_weights_if_requested(
    train_dataset: BaseRetinalDataset, num_classes: int, training_config
):
    """Return per-class weights tensor if configured, otherwise None."""
    strategy = training_config.loss.class_weight_strategy
    if strategy is None:
        return None

    from src.data.statistics import compute_class_weights

    counts = train_dataset.class_counts()
    weights = compute_class_weights(
        [counts.get(i, 0) for i in range(num_classes)],
        strategy=strategy,
    )
    return torch.tensor(weights, dtype=torch.float32)


def generate_final_report(
    output_dir: Path,
    trainer: Trainer,
    result,
    val_loader: DataLoader,
    data_config,
    model_config,
    training_config,
) -> Path:
    """Persist final evaluation artefacts and a metrics report."""
    output_dir = ensure_dir(output_dir)

    # 1. Best metrics table.
    best_epoch_idx = max(
        range(len(result.history)),
        key=lambda i: result.history[i].get(trainer.config.checkpoint.monitor_metric, float("-inf")),
    )
    best_metrics = dict(result.history[best_epoch_idx])
    metrics_path = output_dir / "best_metrics.json"
    write_json(
        metrics_path,
        {
            "best_epoch": best_epoch_idx + 1,
            "monitor_metric": trainer.config.checkpoint.monitor_metric,
            "metrics": best_metrics,
        },
    )

    # 2. Training curves as JSON for external plotting.
    curves_path = output_dir / "learning_curves.json"
    write_json(curves_path, result.history)

    # 3. Run manifest and configuration snapshots.
    manifest_path = output_dir / "run_manifest.json"
    write_run_manifest(
        manifest_path,
        data_config,
        model_config,
        training_config,
        extra={"best_epoch": best_epoch_idx + 1, "best_metrics": best_metrics},
    )
    shutil.copy(training_config.project_root / "configs" / "data.yaml", output_dir / "data_config_snapshot.yaml")
    shutil.copy(training_config.project_root / "configs" / "model.yaml", output_dir / "model_config_snapshot.yaml")
    shutil.copy(training_config.project_root / "configs" / "training.yaml", output_dir / "training_config_snapshot.yaml")

    # 4. Re-run evaluator on validation split with the best checkpoint for
    #    confusion-matrix and calibration plots.
    best_checkpoint = torch.load(
        trainer.checkpoint_manager.best_path,
        map_location=trainer.device,
        weights_only=False,
    )
    trainer.model.load_state_dict(best_checkpoint["model_state_dict"])

    eval_result = trainer.evaluator.evaluate(val_loader)

    cm = build_confusion_matrix(
        eval_result.labels,
        eval_result.predictions,
        num_classes=data_config.classes.num_classes,
    )
    save_confusion_matrix(cm, output_dir, basename="confusion_matrix", save_plot=True)

    save_reliability_diagram(
        eval_result.labels,
        eval_result.classification_probs,
        output_dir,
        basename="reliability_diagram",
        num_bins=10,
    )
    ece = expected_calibration_error(
        eval_result.labels,
        eval_result.classification_probs,
        num_bins=10,
    )

    report = {
        "best_checkpoint": str(trainer.checkpoint_manager.best_path),
        "last_checkpoint": str(trainer.checkpoint_manager.last_path),
        "best_epoch": best_epoch_idx + 1,
        "best_metrics": best_metrics,
        "validation_ece": ece,
        "confusion_matrix": cm.tolist(),
        "artefacts": {
            "best_metrics": str(metrics_path),
            "learning_curves": str(curves_path),
            "run_manifest": str(manifest_path),
            "confusion_matrix_csv": str(output_dir / "confusion_matrix.csv"),
            "confusion_matrix_plot": str(output_dir / "confusion_matrix.png"),
            "reliability_diagram": str(output_dir / "reliability_diagram.png"),
        },
    }
    report_path = output_dir / "final_report.json"
    write_json(report_path, report)
    logger.info("final report written to %s", report_path)
    return report_path


def clean_epoch_checkpoints(checkpoint_dir: Path) -> None:
    """Remove per-epoch files, leaving best.pt and last.pt."""
    for path in checkpoint_dir.glob("epoch_*.pt"):
        path.unlink(missing_ok=True)
    logger.info("removed per-epoch checkpoints from %s", checkpoint_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    configure_logging(level=logging.INFO, log_dir="logs", filename="train.log", tqdm_compatible=True)

    _print_section("RetinaGrade Training")

    # -----------------------------------------------------------------------
    # 1. Configuration
    # -----------------------------------------------------------------------
    print("\n[1/9] Loading configuration...")
    t0 = time.time()
    data_config = load_data_config(args.data_config)
    model_config = load_model_config(args.model_config)
    overrides = {"epochs": args.epochs} if args.epochs is not None else None
    training_config = load_training_config(args.training_config, overrides=overrides)
    _print_done(f"Configuration loaded ({time.time() - t0:.2f}s)")

    # -----------------------------------------------------------------------
    # 2. Device
    # -----------------------------------------------------------------------
    print("\n[2/9] Resolving device...")
    device = resolve_device(args.device)
    _print_done(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    # -----------------------------------------------------------------------
    # 3. Datasets
    # -----------------------------------------------------------------------
    print("\n[3/9] Building datasets...")
    t0 = time.time()
    datasets, train_dataset, train_loader, val_dataset, val_loader = build_dataloaders_for_run(data_config)
    print(f"  Train images: {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")
    test_dataset = datasets.get("test")
    if test_dataset is not None:
        print(f"  Test images: {len(test_dataset)}")
    else:
        print("  Test images: not used")
    _print_done(f"Datasets built ({time.time() - t0:.2f}s)")

    # -----------------------------------------------------------------------
    # 4. Normalization statistics
    # -----------------------------------------------------------------------
    print("\n[4/9] Loading normalization statistics...")
    stats_cache = data_config.resolve_path(data_config.statistics.normalization.cache_path)
    if stats_cache.is_file():
        _print_done(f"Using cached normalization statistics: {stats_cache}")
    else:
        _print_done("Computed normalization statistics (no cache found)")

    # -----------------------------------------------------------------------
    # 5. DataLoader summary
    # -----------------------------------------------------------------------
    print("\n[5/9] DataLoader summary...")
    print(f"  Batch size: {data_config.dataloader.batch_size}")
    print(f"  Workers: {data_config.dataloader.num_workers}")
    print(f"  Pin memory: {data_config.dataloader.pin_memory}")
    print(f"  Persistent workers: {data_config.dataloader.persistent_workers}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    _print_done("DataLoaders ready")

    # -----------------------------------------------------------------------
    # 6. Model
    # -----------------------------------------------------------------------
    print("\n[6/9] Building model...")
    t0 = time.time()
    text_adapter = HashingTextAdapter(embedding_dim=model_config.spm.text_embedding_dim)
    model = build_model(
        model_config,
        num_classes=data_config.classes.num_classes,
        text_adapter=text_adapter,
        text_prompts=DEFAULT_TEXT_PROMPTS,
    )
    total_params, trainable_params, frozen_params = _count_parameters(model)
    print(f"  Model: {model_config.model_name}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {device}")
    _print_done(f"Model built ({time.time() - t0:.2f}s)")

    # -----------------------------------------------------------------------
    # 7. Class weights + optimizer
    # -----------------------------------------------------------------------
    class_weights = compute_class_weights_if_requested(
        train_dataset, data_config.classes.num_classes, training_config
    )

    print("\n[7/9] Creating optimizer...")
    opt_display_name = {
        "adamw": "AdamW",
        "adam": "Adam",
        "sgd": "SGD",
    }.get(training_config.optimizer.name.lower(), training_config.optimizer.name.title())
    print(f"✓ {opt_display_name} initialized")
    print(f"✓ Learning rate: {training_config.optimizer.lr}")
    if training_config.scheduler.name:
        print(f"✓ Scheduler: {training_config.scheduler.name}")

    trainer = Trainer(
        model=model,
        config=training_config,
        num_classes=data_config.classes.num_classes,
        device=device,
        class_weights=class_weights,
        class_names=data_config.classes.names,
    )

    # -----------------------------------------------------------------------
    # 8. Trainer
    # -----------------------------------------------------------------------
    print("\n[8/9] Initializing trainer...")
    amp_status = "enabled" if trainer.amp.enabled else "disabled"
    _print_done("Trainer initialized")
    print(f"  AMP: {amp_status}")
    print(f"  Epochs: {training_config.epochs}")

    # -----------------------------------------------------------------------
    # 9. Training
    # -----------------------------------------------------------------------
    print("\n[9/9] Starting training...")
    result = trainer.fit(train_loader, val_loader)

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    print("\nGenerating final report...")
    t0 = time.time()
    report_path = generate_final_report(
        Path(args.output_dir),
        trainer,
        result,
        val_loader,
        data_config,
        model_config,
        training_config,
    )
    _print_done(f"Final report generated ({time.time() - t0:.2f}s)")

    if args.clean_epoch_checkpoints:
        print("\n[Cleanup] Removing per-epoch checkpoints...")
        clean_epoch_checkpoints(trainer.checkpoint_manager.directory)
        _print_done("Cleanup complete")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    best_epoch = 0
    best_value = float("-inf")
    monitor = training_config.checkpoint.monitor_metric
    for idx, row in enumerate(result.history):
        value = row.get(monitor, float("-inf"))
        if value > best_value:
            best_value = value
            best_epoch = idx + 1

    tb_path = None
    if trainer.tensorboard_logger.enabled:
        tb_path = trainer.tensorboard_logger._writer.log_dir

    _print_section("Training Summary")
    print(f"  Total epochs: {len(result.history)}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best {monitor}: {best_value:.4f}")
    print(f"  Final report path: {report_path}")
    print(f"  Best checkpoint path: {trainer.checkpoint_manager.best_path}")
    print(f"  Last checkpoint path: {trainer.checkpoint_manager.last_path}")
    print(f"  TensorBoard log path: {tb_path or 'disabled'}")
    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
