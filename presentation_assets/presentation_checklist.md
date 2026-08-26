# RetinaGrade / DualSwinOrd — Presentation Checklist

> All items verified against the copied asset package. SHA256 of every copied file is in `SHA256SUMS.txt`.
> **No repository file was modified.** This checklist is read-only discovery output.

---

## ✓ COPIED ASSETS (62 files, 0 failures)

### metrics/ (9 files)
- [x] `best_metrics.json` — QWK 0.9177, acc 0.8517, macro-F1 0.6959, MAE 0.186, within-one 0.9709, per-class P/R/F1
- [x] `final_report.json` — ECE 0.052, confusion matrix array, all best-epoch metrics
- [x] `confusion_matrix.csv` — raw 5×5 counts (rows=true, cols=pred)
- [x] `confusion_matrix.json` — same as CSV in JSON form
- [x] `learning_curves.json` — per-epoch metrics epoch 0 only (full history in epoch_log.csv)
- [x] `run_manifest.json` — Torch 2.5.1+cu124, Python 3.10, seed 42, best epoch 24
- [x] `best_state.json` — epoch 24, monitor val_qwk
- [x] `epoch_log.csv` — **50 epochs** of all metrics (train/val loss, QWK, F1, MAE, per-class P/R/F1, LR)
- [x] `normalization_stats.json` — mean [0.386, 0.204, 0.061], std [0.295, 0.161, 0.078], computed from 2,929 train images

### charts/ (3 files)
- [x] `confusion_matrix.png` — rendered normalised confusion matrix (21 KB — may need hi-res re-render)
- [x] `reliability_diagram.png` — calibration curve, ECE 0.052 (21 KB)
- [x] `class_distribution.png` — class imbalance bar chart (32 KB)

### images/ (1 file)
- [x] `preprocessing_preview.png` — multi-panel preprocessing composite (2.3 MB)

### reports/ (5 files)
- [x] `audit_report.json` — 3,662 images, 0 corrupt, 17 resolutions, quality stats
- [x] `cleaning_report.json` — 47 exact dupes excluded, 626 flagged, policy details
- [x] `split_verification_report.json` — 0 ID/MD5 overlap, 25 near-dup clusters (WARN), χ²=10.85 p=0.21
- [x] `dataset_statistics.json` — counts, imbalance metrics, class weights, normalization stats
- [x] `class_distribution.csv` — per-split counts (train/val/test/overall)

### eda/ (6 files)
- [x] `eda_class_distribution.csv` — early EDA class counts
- [x] `fundus_border_sample.csv` — border/padding ratio sample
- [x] `image_geometry_sample.csv` — resolution/aspect-ratio sample
- [x] `eda_run.log` — EDA pipeline log
- [x] `Report1_EDA_Analysis.docx` — written EDA findings
- [x] `EDA_APTOS_Research.ipynb` — 22 MB notebook with **all inline EDA plots**

### configs/ (6 files)
- [x] `data.yaml` — full preprocessing + augmentation policy with evidence tags
- [x] `model.yaml` — backbone, SPM, PLKA, neck, heads config with gap annotations
- [x] `training.yaml` — AdamW, cosine, λ=0.5, 50 epochs, AMP, paper-confirmed fields marked
- [x] `data_config_snapshot.yaml` — locked config hash for the training run
- [x] `model_config_snapshot.yaml` — locked model config for the training run
- [x] `training_config_snapshot.yaml` — locked training config for the training run

### architecture/ (19 files)
- [x] `dual_swinord.py` — top-level model forward pass
- [x] `dual_head.py` — dual-head wrapper
- [x] `spm.py` — Semantic Prior Modulation (FiLM sigmoid gate)
- [x] `plka.py` — PLKA parallel dilations 1/2/3 + SE fusion
- [x] `shared_feature_neck.py` — GAP → 512-d shared neck
- [x] `classification_head.py` — K=5 softmax head
- [x] `ordinal_head.py` — K−1 threshold BCE head
- [x] `swin_backbone.py` — timm Swin-Tiny wrapper
- [x] `model_config.py` — dataclass configs for all blocks
- [x] `ordinal_loss.py` — DPE/CARM per-threshold BCE
- [x] `classification_loss.py` — CE + label smoothing 0.1
- [x] `total_loss.py` — L_total = 0.5·L_cls + 0.5·L_ord
- [x] `carm_loss.py` — CARM with optional pos_weight (off by default)

### training/ (10 files)
- [x] `trainer.py` — training loop
- [x] `optim.py` — AdamW with no-decay patterns
- [x] `scheduler.py` — cosine annealing
- [x] `amp.py` — mixed precision
- [x] `checkpoint.py` — val_qwk monitor, best/last save
- [x] `preprocessing_pipeline.py` — canonical order: border→crop→resize→CLAHE→illum→aug→norm
- [x] `augmentation_policies.py` — enabled transforms + forbidden list
- [x] `cleaning.py` — MD5 + dual-hash dedup logic
- [x] `splits.py` — group-aware split logic
- [x] `metrics.py` — QWK, macro-F1, MAE, within-one, per-class P/R/F1
- [x] `calibration.py` — ECE computation

### deployment/ (3 files — all stubs)
- [x] `predict.py` — raises NotImplementedError (planned inference entry point)
- [x] `explain.py` — raises NotImplementedError (planned Grad-CAM CLI)
- [x] `gradcam.py` — 4-line docstring stub (no implementation)

### documents/ (4 files)
- [x] `README.md` — milestone checklist, repo layout, reproducibility notes
- [x] `milestone_02.md` — literature survey notes
- [x] `milestone_04_paper_gaps.md` — PG-01 to PG-19 open gap register
- [x] `Report2_Data_Preparation_Recommendations.docx` — data prep rationale

### papers/ (1 file)
- [x] `DualSwinOrd_paper.pdf` — 26.3 MB full paper PDF

---

## ✗ MISSING ASSETS

These do **not** exist anywhere in the repository and must be created before the presentation.

| # | Asset | Slide | Severity |
|---|---|---|---|
| M1 | Grade 0–4 fundus progression strip with lesion annotations | 3 | 🔴 Blocking |
| M2 | Grad-CAM / attention overlay images | 13 | 🔴 Blocking |
| M3 | Ablation study results (baseline / +SPM / +PLKA / +ordinal / full) | 11 | 🔴 Blocking |
| M4 | Test-set evaluation metrics (locked test split) | 12 | 🔴 Blocking |
| M5 | EDA plots exported from notebook as PNG | 4 | 🟠 High |
| M6 | Architecture diagram (redrawn from paper Figure 1) | 8 | 🟠 High |
| M7 | Preprocessing before/after image pairs | 6 | 🟠 High |
| M8 | Learning curve chart (val_qwk + loss over 50 epochs) | 11 | 🟡 Medium |
| M9 | Per-class radar chart (F1 across 5 grades) | 12 | 🟡 Medium |
| M10 | Hi-res confusion matrix heatmap (existing PNG is 21 KB) | 12 | 🟡 Medium |
| M11 | All 14 flow/architecture diagrams (D1–D14) | various | 🟡 Medium |
| M12 | ROC / PR curves per class | appendix | 🟡 Medium |
| M13 | TensorBoard screenshots (18 event files present, not rendered) | appendix | 🟢 Low |
| M14 | Augmentation policy visual (sample image grid) | 6 | 🟢 Low |

---

## ✓ METRICS AVAILABLE

All from `metrics/best_metrics.json` and `metrics/final_report.json`.

| Metric | Value | Available |
|---|---|---|
| QWK (val) | **0.9177** | ✓ |
| Accuracy (val) | 0.8517 | ✓ |
| Macro F1 (val) | 0.6959 | ✓ |
| MAE (val) | 0.186 | ✓ |
| Within-one accuracy (val) | **0.9709** | ✓ |
| ECE (val) | 0.052 | ✓ |
| Per-class Precision (all 5) | 0→0.994, 1→0.720, 2→0.780, 3→0.389, 4→0.655 | ✓ |
| Per-class Recall (all 5) | 0→0.994, 1→0.529, 2→0.848, 3→0.333, 4→0.760 | ✓ |
| Per-class F1 (all 5) | 0→0.994, 1→0.610, 2→0.813, 3→0.359, 4→0.704 | ✓ |
| Confusion matrix (5×5) | See CSV | ✓ |
| Train loss trajectory (50 ep) | 0.584→0.234 | ✓ |
| Val QWK trajectory (50 ep) | 0.815→0.918 (peak ep.24) | ✓ |
| Ordinal loss trajectory | 0.251→0.021 | ✓ |
| Best epoch | 24 / 50 | ✓ |
| Class imbalance ratio | 9.64:1 | ✓ |
| Effective num classes | 3.61 | ✓ |
| Exact dupes excluded | 47 / 3,662 | ✓ |
| Near-dup clusters (warn) | 25 clusters, 126 images | ✓ |
| Split sizes | 2929 / 344 / 342 | ✓ |
| Test-set metrics | **NONE** | ✗ |
| Ablation deltas | **NONE** | ✗ |
| ROC / AUC per class | **NONE** | ✗ |

---

## ✓ FIGURES AVAILABLE

| Figure | File | Usable |
|---|---|---|
| Confusion matrix (rendered) | `charts/confusion_matrix.png` | ✓ (may need hi-res re-render) |
| Reliability diagram | `charts/reliability_diagram.png` | ✓ |
| Class distribution bar | `charts/class_distribution.png` | ✓ |
| Preprocessing composite | `images/preprocessing_preview.png` | ✓ |
| EDA inline plots | `eda/EDA_APTOS_Research.ipynb` | ✓ (must export from notebook) |
| Paper Figure 1 | `papers/DualSwinOrd_paper.pdf` | ✓ (reference only — redraw for slides) |

---

## ✓ ARCHITECTURE AVAILABLE

| Component | File | Status |
|---|---|---|
| Full model wiring | `architecture/dual_swinord.py` | ✓ Implemented |
| Swin-Tiny backbone | `architecture/swin_backbone.py` | ✓ Implemented (timm wrapper) |
| SPM (FiLM sigmoid gate) | `architecture/spm.py` | ✓ Implemented |
| PLKA (dilations 1/2/3) | `architecture/plka.py` | ✓ Implemented |
| Shared neck (GAP→512d) | `architecture/shared_feature_neck.py` | ✓ Implemented |
| Classification head (K=5) | `architecture/classification_head.py` | ✓ Implemented |
| Ordinal head (K−1 BCE) | `architecture/ordinal_head.py` | ✓ Implemented |
| Total loss (λ=0.5) | `architecture/total_loss.py` | ✓ Implemented |
| Ordinal loss (DPE/CARM) | `architecture/ordinal_loss.py` | ✓ Implemented |
| Model weights (best.pt) | `outputs/checkpoints/training/best.pt` | ✓ 524 MB, epoch 24 |
| Architecture diagram | — | ✗ Must be drawn |
| Grad-CAM | `deployment/gradcam.py` | ✗ Stub only |

---

## ✓ REPORTS AVAILABLE

| Report | File | Key facts |
|---|---|---|
| Audit report | `reports/audit_report.json` | 3,662 images, 0 corrupt, 17 resolutions, 8.6 GB |
| Cleaning report | `reports/cleaning_report.json` | 47 excluded, 626 flagged, policy: never delete quality outliers |
| Split verification | `reports/split_verification_report.json` | 0 ID/MD5 leakage, 25 near-dup clusters WARN |
| Dataset statistics | `reports/dataset_statistics.json` | Gini 0.659, effective classes 3.61, class weights |
| EDA analysis | `eda/Report1_EDA_Analysis.docx` | Written EDA findings |
| Data prep recommendations | `documents/Report2_Data_Preparation_Recommendations.docx` | Cleaning rationale |
| Paper gaps register | `documents/milestone_04_paper_gaps.md` | PG-01 to PG-19 open gaps |
| Final training report | `metrics/final_report.json` | All best-epoch metrics + ECE |

---

## ✓ RECOMMENDED BACKUP SLIDES (Appendix)

| # | Topic | Source asset | Answers |
|---|---|---|---|
| A1 | Full 50-epoch training curves | `metrics/epoch_log.csv` | "Show overfitting / convergence" |
| A2 | Per-class P/R/F1 table | `metrics/best_metrics.json` | "Walk me through per-class performance" |
| A3 | Confusion matrix raw counts | `metrics/confusion_matrix.csv` | "Where do errors concentrate?" |
| A4 | Reliability diagram | `charts/reliability_diagram.png` | "Is the model calibrated? ECE?" |
| A5 | Split verification detail | `reports/split_verification_report.json` | "How did you prevent leakage?" |
| A6 | Near-duplicate WARN explanation | `reports/split_verification_report.json` | "Your split check returned WARN" |
| A7 | Cleaning report summary | `reports/cleaning_report.json` | "How many images removed and why?" |
| A8 | Class weights table | `reports/dataset_statistics.json` | "How did you handle imbalance?" |
| A9 | Model config + gap annotations | `configs/model.yaml` + `documents/milestone_04_paper_gaps.md` | "What is paper-confirmed vs. your choices?" |
| A10 | Training config | `configs/training.yaml` | "Optimizer, LR, schedule — paper-confirmed?" |
| A11 | Run manifest | `metrics/run_manifest.json` | "Hardware, environment, reproducibility?" |
| A12 | Ordinal loss source | `architecture/ordinal_loss.py` | "Is ordinal head more than regression?" |
| A13 | CARM gap (PG-17) | `documents/milestone_04_paper_gaps.md` | "What is CARM? Does your impl match?" |
| A14 | Normalization stats | `metrics/normalization_stats.json` | "What normalization? Why not ImageNet?" |

---

## ✓ RECOMMENDED ANIMATIONS

| Slide | Animation | Purpose |
|---|---|---|
| 8 | Architecture assembles block-by-block on click | Makes DualSwinOrd feel like a designed system, not a diagram |
| 3 | Grade 0→4 fundus cross-fade | Lesion accumulation becomes visceral |
| 5 | Pipeline flow builds left-to-right | Shows data integrity as a process, not a claim |
| 11 | Learning curve draws epoch-by-epoch | Shows training stability dynamically |
| 15 | Roadmap ribbon lights up section by section | Closes the journey narrative |

---

## ✓ RECOMMENDED VISUAL REPLACEMENTS FOR TEXT

| Text / JSON field | Replace with | Source |
|---|---|---|
| `best_metrics.json` headline numbers | Stat tiles: QWK 0.92 / Within-one 0.97 / ECE 0.052 | best_metrics.json |
| `confusion_matrix.csv` | Normalised heatmap, annotated, hi-res | confusion_matrix.csv |
| Per-class F1 five numbers | Radar chart (5 axes) | best_metrics.json |
| Per-class P/R/F1 table | Grouped bar chart (3 bars × 5 classes) | best_metrics.json |
| `epoch_log.csv` 50 rows | Dual-axis line chart (QWK + loss, best-epoch marker) | epoch_log.csv |
| `cleaning_report.json` flag counts | Horizontal bar chart (flag type vs count) | cleaning_report.json |
| `class_distribution.csv` counts | Horizontal bar chart, colour-coded by severity | class_distribution.csv |
| `configs/data.yaml` augmentation section | Two-column policy table (transform / evidence / p / why) | data.yaml |
| `configs/training.yaml` | Hyperparameter table (param / value / paper-confirmed?) | training.yaml |
| `configs/model.yaml` | Architecture block table (component / config / gap?) | model.yaml |
| `split_verification_report.json` checks | Traffic-light table (check / status / count) | split_verification_report.json |
| `dataset_statistics.json` imbalance | Imbalance gauge + majority:minority callout | dataset_statistics.json |
| Ordinal loss equations | Ladder diagram (K−1 binary sub-tasks) | ordinal_loss.py |
| Preprocessing pipeline order | Horizontal flow diagram with before/after thumbnails | preprocessing_pipeline.py |
