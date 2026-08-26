"""Evaluation entry point: run best.pt over one split and save artefacts.

Inference only. This script wires together infrastructure that already
exists elsewhere in the project -- it adds no new model, metric, or loss
code:

* configuration loading (identical to ``scripts/train.py``),
* :func:`src.data.datasets.build_datasets` for the split datasets,
* :func:`src.data.dataloader.build_dataloader` for the requested split,
* :func:`src.models.build_model` for the exact same model construction,
* :class:`src.evaluation.evaluator.Evaluator` for the ``torch.no_grad``
  inference pass and metrics,
* :func:`src.evaluation.confusion_matrix.build_confusion_matrix` /
  ``save_confusion_matrix``, :func:`src.evaluation.calibration.
  save_reliability_diagram` / ``expected_calibration_error`` for artefacts,
* :func:`src.utils.helpers.write_json` for the report.

No optimizer, scheduler, loss backward, or parameter update is created, so
``best.pt`` is read but never rewritten.

Run with::

    python scripts/evaluate.py                 # test split (default)
    python scripts/evaluate.py --split val     # smoke test against known val metrics
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataloader import build_dataloader
from src.data.datasets import BaseRetinalDataset, build_datasets
from src.evaluation.calibration import expected_calibration_error, save_reliability_diagram
from src.evaluation.confusion_matrix import build_confusion_matrix, save_confusion_matrix
from src.evaluation.evaluator import Evaluator
from src.models import build_model
from src.models.config import load_model_config
from src.models.semantic_prior.text_adapter import HashingTextAdapter
from src.training.config import load_training_config
from src.utils.config import load_data_config
from src.utils.helpers import ensure_dir, write_json
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

#: Identical clinical prompts to ``scripts/train.py`` (kept byte-for-byte in
#: sync). The semantic prior must be derived exactly as it was at training
#: time, or the loaded weights would see a different conditioning signal.
#: ``scripts`` is not an importable package under the editable install, so the
#: literal is repeated here rather than imported from ``scripts.train``.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Dual-SwinOrd on one split (inference only)")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/training/best.pt",
        help="Checkpoint to evaluate (read-only; never overwritten).",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="outputs/test_evaluation")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_split_dataloader(
    data_config, split: str
) -> tuple[BaseRetinalDataset, DataLoader]:
    """Build only the requested split's dataset and DataLoader."""
    datasets = build_datasets(data_config)
    if split not in datasets:
        raise RuntimeError(
            f"'{split}' dataset could not be built; check its CSV and image "
            f"directory in configs/data.yaml (available: {sorted(datasets)})"
        )
    dataset = datasets[split]
    loader = build_dataloader(dataset, data_config, split)
    return dataset, loader


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    configure_logging(level=logging.INFO, log_dir="logs", filename="evaluate.log", tqdm_compatible=True)

    _print_section("RetinaGrade Evaluation (inference only)")

    # -----------------------------------------------------------------------
    # 1. Configuration (identical to scripts/train.py)
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading configuration...")
    t0 = time.time()
    data_config = load_data_config(args.data_config)
    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)
    _print_done(f"Configuration loaded ({time.time() - t0:.2f}s)")

    # -----------------------------------------------------------------------
    # 2. Device
    # -----------------------------------------------------------------------
    print("\n[2/6] Resolving device...")
    device = resolve_device(args.device)
    _print_done(f"Device: {device}")

    # -----------------------------------------------------------------------
    # 3. Dataset + DataLoader for the requested split only
    # -----------------------------------------------------------------------
    print(f"\n[3/6] Building '{args.split}' split...")
    t0 = time.time()
    dataset, loader = build_split_dataloader(data_config, args.split)
    print(f"  {args.split} images: {len(dataset)}")
    print(f"  {args.split} batches: {len(loader)}")
    _print_done(f"DataLoader ready ({time.time() - t0:.2f}s)")

    # -----------------------------------------------------------------------
    # 4. Model (constructed exactly as scripts/train.py does)
    # -----------------------------------------------------------------------
    print("\n[4/6] Building model...")
    t0 = time.time()
    text_adapter = HashingTextAdapter(embedding_dim=model_config.spm.text_embedding_dim)
    model = build_model(
        model_config,
        num_classes=data_config.classes.num_classes,
        text_adapter=text_adapter,
        text_prompts=DEFAULT_TEXT_PROMPTS,
    )
    model.to(device)
    _print_done(f"Model built ({time.time() - t0:.2f}s)")

    # -----------------------------------------------------------------------
    # 5. Load checkpoint (read-only) and switch to eval mode
    # -----------------------------------------------------------------------
    print("\n[5/6] Loading checkpoint...")
    checkpoint_path = data_config.resolve_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _print_done(f"Loaded {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")

    # -----------------------------------------------------------------------
    # 6. Inference + artefacts (reusing the existing Evaluator and savers)
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Evaluating on '{args.split}'...")
    t0 = time.time()
    evaluator = Evaluator(
        model,
        device,
        num_classes=data_config.classes.num_classes,
        class_names=data_config.classes.names,
    )
    # Evaluator.evaluate is wrapped in @torch.no_grad(); no gradients are built.
    eval_result = evaluator.evaluate(loader)

    output_dir = ensure_dir(Path(args.output_dir))

    metrics_path = write_json(
        output_dir / "metrics.json",
        {
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "num_samples": int(len(dataset)),
            "metrics": eval_result.metrics,
        },
    )

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
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_samples": int(len(dataset)),
        "metrics": eval_result.metrics,
        "ece": ece,
        "confusion_matrix": cm.tolist(),
        "artefacts": {
            "metrics": str(metrics_path),
            "confusion_matrix_csv": str(output_dir / "confusion_matrix.csv"),
            "confusion_matrix_json": str(output_dir / "confusion_matrix.json"),
            "confusion_matrix_plot": str(output_dir / "confusion_matrix.png"),
            "reliability_diagram": str(output_dir / "reliability_diagram.png"),
        },
    }
    report_path = write_json(output_dir / "evaluation_report.json", report)
    _print_done(f"Evaluation complete ({time.time() - t0:.2f}s)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _print_section(f"Evaluation Summary ({args.split})")
    m = eval_result.metrics
    print(f"  Samples:          {len(dataset)}")
    print(f"  QWK:              {m.get('qwk'):.4f}")
    print(f"  Accuracy:         {m.get('accuracy'):.4f}")
    print(f"  Macro F1:         {m.get('macro_f1'):.4f}")
    print(f"  MAE:              {m.get('mae'):.4f}")
    print(f"  Within-one acc:   {m.get('within_one_accuracy'):.4f}")
    print(f"  ECE:              {ece:.4f}")
    print(f"  Report:           {report_path}")
    print("\nEvaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
