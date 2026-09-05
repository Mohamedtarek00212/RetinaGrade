---
title: RetinaGrade
colorFrom: yellow
colorTo: green
sdk: static
app_build_command: cd frontend && npm ci && npm run build
app_file: frontend/dist/index.html
fullWidth: true
pinned: false
license: mit
short_description: Research demo for ordinal diabetic retinopathy grading
---

# Dual-SwinOrd

A faithful research reproduction of **"Dual-SwinOrd: A Dual-Head Swin Transformer with Semantic Prior Injection for Ordinal Diabetic Retinopathy Grading"**.

**Live demo:** https://huggingface.co/spaces/mohamed00212/RetinaGrade

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
│   ├── data/                # datasets, preprocessing/, augmentation, audit, splits
│   ├── models/              # config.py, registry.py, backbones/, semantic_prior/, attention/, neck/, heads/, dual_head.py, dual_swinord.py
│   ├── losses/              # base.py, classification_loss.py, ordinal_loss.py, carm_loss.py, total_loss.py
│   ├── training/            # config.py, optim.py, scheduler.py, amp.py, checkpoint.py, manifest.py, csv_logger.py, tensorboard_logger.py, callbacks.py, trainer.py
│   ├── evaluation/          # metrics.py, calibration.py, evaluator.py, confusion_matrix.py
│   ├── visualization/       # gradcam.py, shap_analysis.py
│   └── utils/               # seed.py, logger.py, helpers.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── deployment/              # Single-image inference utilities
├── frontend/                # React + TypeScript deployment interface
├── Presentation/            # Curated final presentation deliverables
├── presentation_assets/     # Source archive used to build the presentation
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
- [x] Final locked-test evaluation and calibration
- [x] Browser class-activation explanation and research guidance
- [x] Downloadable clinician and patient PDF reports

## Final test results

The validation-selected `best.pt` checkpoint was evaluated once on the locked
test split (342 images):

| Metric | Test result |
|---|---:|
| Accuracy | 86.84% |
| Quadratic Weighted Kappa (QWK) | 0.9074 |
| Macro F1 | 67.83% |
| Mean Absolute Error (MAE) | 0.1784 |
| Within-one-grade accuracy | 96.20% |
| Referable DR AUC | 98.18% |
| Referable DR false-negative rate | 3.42% |

The checkpoint was selected at epoch 24 using validation QWK (`0.9177`). The
test split was used only for final evaluation and did not influence checkpoint
selection.

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

`scripts/train.py`, `scripts/prepare_data.py`, `scripts/evaluate.py`, and
`scripts/predict.py` are implemented. The browser application also returns a
class-specific activation map with confidence context and conservative research
guidance. The map shows spatial contribution to the selected class; it is not a
lesion detector or a clinical explanation.

After clinician review, the browser can generate two local PDF documents: a
detailed clinician review report and a plain-language patient information
summary. Patient data remains in memory inside the browser and is not uploaded.

The FastAPI backend and React research interface are documented in
`deployment/README.md` and `frontend/README.md`.

Free browser-based ONNX deployment, plus optional CPU and NVIDIA GPU containers,
is documented in `docs/deployment.md`.

### Single-image prediction

After downloading `best.pt` as described below, export an inference-only copy:

```bash
python scripts/export_inference_checkpoint.py
python scripts/predict.py path/to/fundus.png \
  --checkpoint outputs/checkpoints/deployment/model_inference.pt
```

The command prints the predicted grade, class name, confidence, class
probabilities, and ordinal-threshold probabilities as JSON.

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

## Model checkpoint

The deployment checkpoint is `outputs/checkpoints/training/best.pt`. It is not
stored in Git because it is approximately 523 MB. Download it from the
[v0.1.0 model release](https://github.com/Mohamedtarek00212/RetinaGrade/releases/download/v0.1.0/best.pt)
and place it at the same path after cloning the repository.

Checkpoint integrity:

```text
SHA256: 5B979123FDA8179F6DFD59AD45EA0E7CE3D8B6B6DF47CFBA39776528779DF389
```

`best.pt` contains the model, optimizer, and scheduler states so training can be
resumed. `scripts/export_inference_checkpoint.py` removes the training-only state
and writes `outputs/checkpoints/deployment/model_inference.pt` for deployment.

## Git and large-file notes

- `notebooks/EDA_APTOS_Research.ipynb` is ~22 MB because it stores inline visualizations. GitHub allows files up to 100 MB, but pushes and clones of this repository will be noticeably slower because of that notebook.
- `docs/literature/*.pdf` is intentionally ignored by `.gitignore`; keep the paper locally or reference its DOI/URL in `docs/milestone_02.md`.
- Data and generated artifacts (`data/raw/`, `data/processed/`, `outputs/`, `checkpoints/`, `logs/`) are not committed; only `.gitkeep` files are pushed so the directory structure is preserved when the project is cloned.

## Reproducibility and safety notes

- Only algorithms, preprocessing, augmentations, and architectural details explicitly described in the Dual-SwinOrd paper are allowed.
- All other changes are limited to software engineering: organization, readability, maintainability, logging, configuration, and documentation.
- This code is for research and education only and is not a medical device. It must not be used for diagnosis, triage, or treatment.
