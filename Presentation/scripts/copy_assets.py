"""Copy real, verified project artefacts into the Presentation/ tree.

Read-only with respect to the source repository: every source file is opened
with shutil.copy2 (copy, never move, never modify). Originals are untouched.
Only files that actually exist are copied; missing sources are reported.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(r"d:\RetinaGrade")
PRES = ROOT / "Presentation"

# (source relative to ROOT, destination relative to PRES)
COPY_MAP = [
    # --- Evaluation figures (TEST split, primary) ---
    ("outputs/test_evaluation/confusion_matrix.png", "Figures/Evaluation/test_confusion_matrix_original.png"),
    ("outputs/test_evaluation/reliability_diagram.png", "Figures/Evaluation/test_reliability_diagram.png"),
    # --- Evaluation figures (VAL split) ---
    ("outputs/val_smoke_test/confusion_matrix.png", "Figures/Evaluation/val_confusion_matrix.png"),
    ("outputs/val_smoke_test/reliability_diagram.png", "Figures/Evaluation/val_reliability_diagram.png"),
    ("outputs/final_report/confusion_matrix.png", "Figures/Evaluation/finalreport_val_confusion_matrix.png"),
    ("outputs/final_report/reliability_diagram.png", "Figures/Evaluation/finalreport_val_reliability_diagram.png"),
    # --- Dataset figure ---
    ("outputs/data_prep/class_distribution.png", "Figures/Dataset/class_distribution_original.png"),
    # --- Preprocessing composite ---
    ("outputs/data_prep/preview/preview_b5954c864e28.png", "Figures/Preprocessing/preprocessing_preview_composite.png"),
    # --- Metrics (JSON/CSV) ---
    ("outputs/test_evaluation/evaluation_report.json", "Metrics/test_evaluation_report.json"),
    ("outputs/test_evaluation/metrics.json", "Metrics/test_metrics.json"),
    ("outputs/test_evaluation/confusion_matrix.csv", "Metrics/test_confusion_matrix.csv"),
    ("outputs/test_evaluation/confusion_matrix.json", "Metrics/test_confusion_matrix.json"),
    ("outputs/val_smoke_test/metrics.json", "Metrics/val_metrics.json"),
    ("outputs/final_report/final_report.json", "Metrics/final_report.json"),
    ("outputs/final_report/best_metrics.json", "Metrics/best_metrics.json"),
    ("outputs/logs/training/epoch_log.csv", "Metrics/epoch_log.csv"),
    ("outputs/final_report/run_manifest.json", "Metrics/run_manifest.json"),
    ("data/processed/normalization_stats.json", "Metrics/normalization_stats.json"),
    # --- Reports ---
    ("outputs/data_prep/dataset_statistics.json", "Reports/dataset_statistics.json"),
    ("outputs/data_prep/cleaning_report.json", "Reports/cleaning_report.json"),
    ("outputs/data_prep/split_verification_report.json", "Reports/split_verification_report.json"),
    ("outputs/data_prep/audit_report.json", "Reports/audit_report.json"),
    ("outputs/data_prep/class_distribution.csv", "Reports/class_distribution_by_split.csv"),
    ("data/processed/class_distribution.csv", "Reports/class_distribution_by_grade.csv"),
    # --- Configs ---
    ("configs/data.yaml", "Configs/data.yaml"),
    ("configs/model.yaml", "Configs/model.yaml"),
    ("configs/training.yaml", "Configs/training.yaml"),
    # --- Split label tables ---
    ("data/splits/train.csv", "Tables/split_train.csv"),
    ("data/splits/valid.csv", "Tables/split_valid.csv"),
    ("data/splits/test.csv", "Tables/split_test.csv"),
]


def main() -> None:
    copied, missing = [], []
    for src_rel, dst_rel in COPY_MAP:
        src = ROOT / src_rel
        dst = PRES / dst_rel
        if not src.is_file():
            missing.append(src_rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append((src_rel, dst_rel))

    print(f"COPIED {len(copied)} files:")
    for s, d in copied:
        print(f"  {s}  ->  Presentation/{d}")
    print(f"\nMISSING {len(missing)} sources (not copied):")
    for m in missing:
        print(f"  {m}")


if __name__ == "__main__":
    main()
