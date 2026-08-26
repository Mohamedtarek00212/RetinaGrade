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
│   └── EDA_APTOS_Research.ipynb   # Publication-quality EDA with inline visualizations
├── src/
│   ├── data/                # dataset.py, preprocessing.py, dataloader.py
│   ├── models/              # config.py, registry.py, backbones/, semantic_prior/, attention/, neck/, heads/, dual_head.py, dual_swinord.py
│   ├── losses/              # base.py, classification_loss.py, ordinal_loss.py, carm_loss.py, total_loss.py
│   ├── training/            # config.py, optim.py, scheduler.py, amp.py, checkpoint.py, manifest.py, csv_logger.py, tensorboard_logger.py, callbacks.py, trainer.py
│   ├── evaluation/          # metrics.py, calibration.py, evaluator.py, confusion_matrix.py
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
- [x] Exploratory Data Analysis (`notebooks/EDA_APTOS_Research.ipynb`)
- [x] Data ingestion, preprocessing, and train/val/test split creation
- [x] Swin Transformer backbone with SPM and PLKA
- [x] Dual-head (classification + ordinal) architecture
- [x] Classification, ordinal, and combined training losses (Eq. 7-9; see `docs/milestone_04_paper_gaps.md`)
- [x] Training loop and per-epoch validation (`src/training/trainer.py`); hyperparameter search is not implemented -- the paper reports one fixed hyperparameter set (`configs/training.yaml`), not a search
- [ ] Final locked-test evaluation and calibration
- [ ] Explainability (Grad-CAM/SHAP) and report generation

## Environment setup

### Conda (recommended)

```powershell
conda env create -f environment.yml
conda activate dual-swinord
```

### venv

**Windows PowerShell:**
```powershell
python -m venv .venv-gpu
.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

**Linux/macOS:**
```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Verify installation

Run the test suite to confirm the package and dependencies are installed correctly:

```bash
pytest
```

> Note: `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`, and `scripts/explain.py` are still skeletons; running them directly will raise `NotImplementedError` until their implementations are added in later milestones.

## Data setup

Place the APTOS 2019 images under `data/raw/`:

```text
data/raw/
├── train/
│   └── <id_code>.png
├── val/
│   └── <id_code>.png
└── test/
    └── <id_code>.png
```

Split CSVs belong in `data/splits/` as `train.csv`, `valid.csv`, and `test.csv`.
The expected columns follow the candidates in `configs/data.yaml` (for example `id_code` and `diagnosis`).

Raw images and CSVs are **not** committed to Git (see `.gitignore`).

## Git and large-file notes

- `notebooks/EDA_APTOS_Research.ipynb` is ~22 MB because it stores inline visualizations. GitHub allows files up to 100 MB, but pushes and clones of this repository will be noticeably slower because of that notebook.
- `docs/literature/*.pdf` is intentionally ignored by `.gitignore`; keep the paper locally or reference its DOI/URL in `docs/milestone_02.md`.
- Data and generated artifacts (`data/raw/`, `data/processed/`, `outputs/`, `checkpoints/`, `logs/`) are not committed; only `.gitkeep` files are pushed so the directory structure is preserved when the project is cloned.

## Reproducibility and safety notes

- Only algorithms, preprocessing, augmentations, and architectural details explicitly described in the Dual-SwinOrd paper are allowed.
- All other changes are limited to software engineering: organization, readability, maintainability, logging, configuration, and documentation.
- This code is for research and education only and is not a medical device. It must not be used for diagnosis, triage, or treatment.
