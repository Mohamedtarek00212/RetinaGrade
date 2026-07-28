# Dual-SwinOrd

A faithful research reproduction of **"Dual-SwinOrd: A Dual-Head Swin Transformer with Semantic Prior Injection for Ordinal Diabetic Retinopathy Grading"**.

This repository is a modular, production-quality PyTorch implementation for ordinal diabetic retinopathy grading on the APTOS 2019 dataset. The goal is maximum reproducibility: every algorithmic choice, preprocessing step, loss, and architectural block is taken directly from the paper and is not modified with unpublished improvements.

## Repository layout

```text
RetinaGrade/
├── configs/                 # Experiment and model configurations
├── data/
│   ├── raw/                 # Raw APTOS images (not committed)
│   ├── processed/           # Preprocessed/cached data (not committed)
│   └── splits/              # Train/validation/test CSV splits
├── notebooks/
│   └── 01_EDA_APTOS.ipynb   # Exploratory data analysis
├── src/
│   ├── data/                # dataset.py, preprocessing.py, dataloader.py
│   ├── models/              # swin_backbone.py, spm.py, plka.py, dual_head.py, dual_swinord.py
│   ├── losses/              # classification_loss.py, ordinal_loss.py, total_loss.py
│   ├── training/            # trainer.py, callbacks.py
│   ├── evaluation/          # metrics.py, evaluator.py, confusion_matrix.py
│   ├── visualization/       # gradcam.py, shap_analysis.py
│   └── utils/               # seed.py, logger.py, helpers.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── predict.py
├── outputs/                 # Experiment outputs and reports (not committed)
├── checkpoints/             # Saved model weights (not committed)
├── logs/                    # Training logs (not committed)
├── docs/                    # Documentation and literature
├── tests/                   # Unit tests
├── README.md
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## Milestones

- [x] Repository scaffold and project organization
- [ ] Data ingestion, preprocessing, and train/val/test split creation
- [ ] Swin Transformer backbone with SPM and PLKA
- [ ] Dual-head (classification + ordinal) architecture
- [ ] Classification, ordinal, and combined training losses
- [ ] Training loop, validation, and hyperparameter search
- [ ] Final locked-test evaluation and calibration
- [ ] Explainability (Grad-CAM/SHAP) and report generation

## Environment setup

1. Create and activate a Python 3.11 virtual environment.

   **Windows PowerShell:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **Linux/macOS:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the project in editable mode with development dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```

3. Verify the skeleton runs without errors:

   ```bash
   python scripts/train.py
   python scripts/evaluate.py
   pytest
   ```

   Each script currently raises `NotImplementedError`; the point is to confirm that imports resolve and the package is installed.

## Data setup

Place the APTOS 2019 data under `data/raw/`:

```text
data/raw/
├── train_images/
│   └── <id_code>.png
├── test_images/
│   └── <id_code>.png
└── val_images/
    └── <id_code>.png
```

Split CSVs belong in `data/splits/` (for example `train.csv`, `valid.csv`, `test.csv`).

Raw images and CSVs are **not** committed to Git (see `.gitignore`).

## Reproducibility and safety notes

- Only algorithms, preprocessing, augmentations, and architectural details explicitly described in the Dual-SwinOrd paper are allowed.
- All other changes are limited to software engineering: organization, readability, maintainability, logging, configuration, and documentation.
- This code is for research and education only and is not a medical device. It must not be used for diagnosis, triage, or treatment.
