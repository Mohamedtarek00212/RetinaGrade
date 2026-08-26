# RetinaGrade / DualSwinOrd — Presentation Asset Inventory

> Read-only asset discovery package. **No repository file was modified, moved, renamed, or deleted.** Every file below was *copied* into `presentation_assets/` and verified byte-for-byte with SHA256 (see `SHA256SUMS.txt` and the copy log). All numbers quoted in this document are read directly from the copied artifacts — nothing is invented.

====================================

## PROJECT OVERVIEW

====================================

**RetinaGrade** is a faithful research reproduction of the paper *"Dual-SwinOrd: A Dual-Head Swin Transformer with Semantic Prior Injection for Ordinal Diabetic Retinopathy Grading."* It performs **5-class ordinal** diabetic-retinopathy grading (0 No DR → 4 Proliferative DR) on **APTOS 2019** fundus images.

**Model — DualSwinOrd:** Input → Swin-Tiny backbone (ImageNet-pretrained, `swin_tiny_patch4_window7_224`, 512²) → Semantic Prior Modulation (FiLM-style sigmoid gate, injected at stage 3) → PLKA attention (parallel dilations 1/2/3, SE-style fusion) → shared feature neck (GAP → 512-d) → **dual heads**: a K=5 softmax classification head and a K−1 threshold ordinal head. Loss `L_total = λ·L_cls + (1−λ)·L_ord`, `λ=0.5`; classification = cross-entropy + 0.1 label smoothing; ordinal = per-threshold BCE ("Deep Progressive Enhancement" / CARM).

**Headline result (VALIDATION set, best epoch 24 of 50):** QWK **0.9177**, accuracy **0.8517**, macro-F1 **0.6959**, MAE **0.186**, within-one-accuracy **0.9709**, ECE **0.052**. Optimizer AdamW (lr 1e-4, wd 1e-4), cosine annealing, AMP, batch 16, seed 42, Torch 2.5.1+cu124 / CUDA.

**Dataset reality:** 3,662 raw PNGs audited (0 corrupt, RGB, 8.6 GB) → 47 exact cross-split duplicates excluded → **3,615 images** used (train 2,929 / val 344 / test 342). Strong imbalance (majority:minority ≈ 9.6:1; Gini 0.659; effective classes 3.61). Class homogeneity across splits passes (χ²=10.85, p=0.21).

**Three findings that shape the whole deck (be honest about these):**
1. **All reported metrics are VALIDATION metrics.** The "Final locked-test evaluation" milestone is *unchecked* in the README — there are no test-set results, ROC, or PR curves in the repo.
2. **Explainability does not exist yet.** `src/visualization/gradcam.py` is a docstring stub; `scripts/explain.py`, `predict.py`, `evaluate.py` all raise `NotImplementedError`. **There are no Grad-CAM/SHAP images to copy.**
3. **No ablation study exists.** Only one full-model run is present; there are no baseline / +SPM / +PLKA / +ordinal runs. An "ablation waterfall" would have to be *generated* — it is not backed by current artifacts.

A fourth caveat for Q&A: split verification returns **`warn`** — exact-MD5 leakage is 0, but **25 perceptual near-duplicate clusters (126 images) still span splits** (flagged, never auto-removed by policy), a real val-metric-inflation risk.

====================================

## SLIDE-BY-SLIDE ASSETS

====================================

---

### Slide 1 — Clinical Problem

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Hero fundus image (grade 4) | `images/preprocessing_preview.png` | Preprocessed fundus sample — use highest-severity crop as hero | Required | image |
| APTOS paper PDF | `papers/DualSwinOrd_paper.pdf` | Source for DR prevalence statistics quoted on slide | Required | reference |
| EDA report (docx) | `eda/Report1_EDA_Analysis.docx` | Contains dataset context and clinical framing | Optional | document |

**Hero visual:** A single high-severity fundus image (grade 4, neovascularization visible) occupying 60–70% of the slide. **Must be sourced from `data/raw/train/` — pick a visually striking grade-4 image manually.** The preview PNG in the package is a preprocessing-pipeline composite, not a single clinical image.

---

### Slide 2 — Research Background

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Paper PDF | `papers/DualSwinOrd_paper.pdf` | Prior-art comparison table source | Required | reference |
| Milestone 02 doc | `documents/milestone_02.md` | Literature survey notes | Optional | document |
| Paper gaps register | `documents/milestone_04_paper_gaps.md` | Shows what the paper explicitly states vs. gaps | Optional | reference |

**Hero visual:** A hand-drawn comparison table (Gulshan 2016, IDx-DR, APTOS top solutions) — no existing file; must be created. See DIAGRAMS TO DRAW.

---

### Slide 3 — Task Definition & Why Ordinal

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Class distribution PNG | `charts/class_distribution.png` | Shows 5-class imbalance — use as supporting visual | Required | chart |
| Class distribution CSV | `reports/class_distribution.csv` | Raw counts: 0→1803, 1→362, 2→977, 3→187, 4→286 | Required | csv |
| Dataset statistics JSON | `reports/dataset_statistics.json` | Imbalance metrics: majority:minority 9.64:1, Gini 0.659 | Required | json |
| Grade 0–4 fundus strip | `data/raw/train/` (manual selection) | One representative image per grade | Required | image |

**Hero visual:** Grade 0→4 fundus progression strip with annotated lesion callouts. **Must be assembled manually from `data/raw/train/`.** No pre-built strip exists.

> Convert `class_distribution.csv` into a **Horizontal Bar Chart** (sorted by count, colour-coded by severity). The existing `class_distribution.png` is usable but may need re-styling.

---

### Slide 4 — Dataset & EDA

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Audit report JSON | `reports/audit_report.json` | 3,662 images, 17 distinct resolutions, quality stats | Required | json |
| Image geometry CSV | `eda/image_geometry_sample.csv` | Resolution/aspect-ratio scatter source | Required | csv |
| EDA notebook | `eda/EDA_APTOS_Research.ipynb` | 22 MB notebook with inline visualisations — extract plots | Required | notebook |
| EDA report docx | `eda/Report1_EDA_Analysis.docx` | Written EDA findings | Optional | document |
| Fundus border sample CSV | `eda/fundus_border_sample.csv` | Border/padding ratio data | Optional | csv |

**Hero visual:** Extract the class-distribution + resolution scatter from the EDA notebook. The `eda_plots/` directory is **empty** — all plots are embedded in the notebook cells and must be exported.

> Convert `image_geometry_sample.csv` into a **Scatter Plot** (width vs height, coloured by grade).
> Convert `audit_report.json` quality fields into a **Box-plot or Violin** (brightness, contrast, sharpness distributions).

---

### Slide 5 — Data Integrity: Cleaning, Duplicates, Leakage

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Cleaning report JSON | `reports/cleaning_report.json` | 3,662→3,615; 47 exact dupes excluded; 626 flagged | Required | json |
| Split verification JSON | `reports/split_verification_report.json` | 0 ID overlap, 0 MD5 overlap, 25 near-dup clusters (WARN) | Required | json |
| Audit report JSON | `reports/audit_report.json` | MD5: 3,662 computed, 3,534 unique | Required | json |
| Data prep recommendations | `documents/Report2_Data_Preparation_Recommendations.docx` | Written rationale for cleaning decisions | Optional | document |

**Hero visual:** A pipeline flow diagram showing: Raw 3,662 → MD5 audit → 47 exact dupes removed → 3,615 clean → group-aware split → 0 ID/MD5 cross-split leakage. **Must be drawn** — no existing figure.

**⚠ Honest note for presenter:** Split verification status is `warn` — 25 near-duplicate clusters (126 images) span splits. This is flagged, not fixed. Presenter must be ready to explain the policy decision.

> Convert `cleaning_report.json` flag counts into a **Stacked Bar** (per-split: exact dupes, near-dupes, quality flags).

---

### Slide 6 — Preprocessing & Augmentation

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Preprocessing preview PNG | `images/preprocessing_preview.png` | Multi-panel before/after preprocessing composite | Required | image |
| Data config YAML | `configs/data.yaml` | Full preprocessing + augmentation policy with evidence tags | Required | config |
| Data config snapshot YAML | `configs/data_config_snapshot.yaml` | Locked config used for the training run | Required | config |
| Preprocessing pipeline source | `training/preprocessing_pipeline.py` | Code showing canonical order: border→crop→resize→CLAHE→illum | Optional | source |
| Augmentation policies source | `training/augmentation_policies.py` | Shows forbidden list (MixUp, CutMix, CutOut, etc.) | Optional | source |

**Hero visual:** `images/preprocessing_preview.png` — the existing composite is the best available asset. Supplement with a before/after pair extracted from the EDA notebook.

> Convert `configs/data.yaml` augmentation section into a **Two-column Policy Table** (transform | evidence | p | why enabled/disabled).

---

### Slide 7 — Why Swin, Not a CNN

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Model config YAML | `configs/model.yaml` | Backbone: `swin_tiny_patch4_window7_224`, pretrained, 512² | Required | config |
| Paper PDF | `papers/DualSwinOrd_paper.pdf` | Architecture justification from paper | Required | reference |
| Paper gaps register | `documents/milestone_04_paper_gaps.md` | Shows what is paper-confirmed vs. implementation assumption | Required | reference |
| Swin backbone source | `architecture/swin_backbone.py` | timm wrapper — shows variant selection | Optional | source |

**Hero visual:** A scored decision matrix (ResNet / EfficientNet / ViT / Swin) across axes: receptive field, data efficiency, lesion-scale sensitivity, pretrain availability, inference cost. **Must be drawn** — no existing figure.

---

### Slide 8 — DualSwinOrd Architecture

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Paper PDF | `papers/DualSwinOrd_paper.pdf` | Figure 1 — the canonical architecture diagram | Required | figure |
| Model config YAML | `configs/model.yaml` | All block configs with implementation-assumption annotations | Required | config |
| Model config snapshot | `configs/model_config_snapshot.yaml` | Locked config for the run | Required | config |
| `dual_swinord.py` | `architecture/dual_swinord.py` | Top-level forward pass wiring | Required | source |
| `spm.py` | `architecture/spm.py` | SPM FiLM-style gate | Required | source |
| `plka.py` | `architecture/plka.py` | PLKA parallel dilations 1/2/3 | Required | source |
| `shared_feature_neck.py` | `architecture/shared_feature_neck.py` | GAP → 512-d neck | Required | source |
| `classification_head.py` | `architecture/classification_head.py` | K=5 softmax head | Required | source |
| `ordinal_head.py` | `architecture/ordinal_head.py` | K−1 threshold BCE head | Required | source |
| `model_config.py` | `architecture/model_config.py` | Dataclass definitions for all blocks | Optional | source |

**Hero visual:** Reproduce / redraw Figure 1 from the paper PDF as an animated block-by-block build. This is the single most important diagram in the deck.

---

### Slide 9 — Ordinal Learning & Loss

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| `ordinal_loss.py` | `architecture/ordinal_loss.py` | DPE / CARM per-threshold BCE implementation | Required | source |
| `classification_loss.py` | `architecture/classification_loss.py` | Cross-entropy + label smoothing 0.1 | Required | source |
| `total_loss.py` | `architecture/total_loss.py` | λ=0.5 convex combination | Required | source |
| `carm_loss.py` | `architecture/carm_loss.py` | CARM with optional pos_weight (off by default) | Required | source |
| Training config YAML | `configs/training.yaml` | `lambda_cls: 0.5`, paper-confirmed | Required | config |
| Paper gaps register | `documents/milestone_04_paper_gaps.md` | PG-17: CARM discrepancy between abstract and Eq. 8 | Required | reference |
| Best metrics JSON | `metrics/best_metrics.json` | `train_ordinal_loss: 0.080` at best epoch | Optional | json |

**Hero visual:** Equation diagram: L_total = 0.5·L_cls + 0.5·L_ord, with K−1 threshold sub-tasks shown as a ladder. **Must be drawn.**

---

### Slide 10 — Training Strategy

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Training config YAML | `configs/training.yaml` | AdamW lr=1e-4, wd=1e-4, cosine, 50 epochs, AMP, seed 42 | Required | config |
| Training config snapshot | `configs/training_config_snapshot.yaml` | Locked config for the run | Required | config |
| Run manifest JSON | `metrics/run_manifest.json` | Environment: Torch 2.5.1+cu124, Python 3.10, seed 42 | Required | json |
| `trainer.py` | `training/trainer.py` | Training loop | Optional | source |
| `optim.py` | `training/optim.py` | AdamW with no-decay patterns | Optional | source |
| `scheduler.py` | `training/scheduler.py` | Cosine annealing | Optional | source |
| `amp.py` | `training/amp.py` | Mixed precision | Optional | source |

**Hero visual:** A hyperparameter table (parameter | value | paper-confirmed?) — directly derivable from `configs/training.yaml`. All paper-confirmed fields are annotated in the YAML.

---

### Slide 11 — Experiments & Ablation

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Epoch log CSV | `metrics/epoch_log.csv` | 50 epochs of all metrics — full training history | Required | csv |
| Learning curves JSON | `metrics/learning_curves.json` | Per-epoch train/val loss + QWK | Required | json |
| Best metrics JSON | `metrics/best_metrics.json` | Best epoch 24 headline numbers | Required | json |

**⚠ CRITICAL MISSING ASSET:** There is **no ablation study** in the repository. Only one model configuration was trained. A "waterfall" chart showing Baseline → +SPM → +PLKA → +Ordinal → Full **cannot be built from existing data.** This must either be (a) honestly omitted, (b) replaced with a learning-curve progression showing training stability, or (c) run as new experiments before the presentation.

> Convert `epoch_log.csv` into a **Dual-axis Line Chart**: val_qwk (left axis) + val_loss (right axis) over 50 epochs, with best-epoch marker at epoch 24.
> Convert `learning_curves.json` into a **Multi-line Chart**: train_loss, val_loss, train_ordinal_loss over epochs.

---

### Slide 12 — Results

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Confusion matrix PNG | `charts/confusion_matrix.png` | Rendered normalised confusion matrix | Required | image |
| Confusion matrix CSV | `metrics/confusion_matrix.csv` | Raw counts for re-rendering | Required | csv |
| Best metrics JSON | `metrics/best_metrics.json` | QWK 0.9177, acc 0.8517, macro-F1 0.6959, MAE 0.186, within-one 0.9709 | Required | json |
| Final report JSON | `metrics/final_report.json` | ECE 0.052, per-class precision/recall/F1 | Required | json |
| Reliability diagram PNG | `charts/reliability_diagram.png` | Calibration curve (ECE 0.052) | Required | image |

**Hero visual:** `charts/confusion_matrix.png` — already rendered. Verify it is normalised (row-normalised). The raw CSV confirms strong diagonal concentration.

> Convert `best_metrics.json` per-class F1 into a **Radar Chart** (5 axes = 5 classes).
> Convert `best_metrics.json` per-class precision/recall/F1 into a **Grouped Bar Chart**.
> The confusion matrix CSV should be re-rendered as a **Heatmap** with annotation if the existing PNG is too small (21 KB — likely low resolution).

**⚠ Honest note:** All metrics are **validation-set** results. The test-set evaluation milestone is unchecked. State this explicitly on the slide.

---

### Slide 13 — Error Analysis & Explainability

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| Confusion matrix CSV | `metrics/confusion_matrix.csv` | Off-diagonal cells identify error patterns | Required | csv |
| Final report JSON | `metrics/final_report.json` | Per-class recall — Severe DR recall 0.333 is the weakest | Required | json |
| Best metrics JSON | `metrics/best_metrics.json` | Per-class precision/recall/F1 | Required | json |
| Grade-1 and grade-3 raw images | `data/raw/train/` (manual) | Misclassified-class examples | Optional | image |

**⚠ CRITICAL MISSING ASSET:** `src/visualization/gradcam.py` is a **4-line docstring stub**. `scripts/explain.py` raises `NotImplementedError`. **There are zero Grad-CAM or attention-map images in the repository.** This slide cannot show explainability visuals from existing artifacts. Options: (a) show the confusion matrix error analysis only and honestly state Grad-CAM is future work, (b) generate Grad-CAM images before the presentation using the saved `best.pt` checkpoint.

> Convert confusion matrix off-diagonal cells into an **Error Taxonomy Table** (true grade | predicted grade | count | likely cause).

---

### Slide 14 — Deployment & Limitations

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| `predict.py` | `deployment/predict.py` | Stub — shows planned inference entry point | Optional | source |
| `gradcam.py` | `deployment/gradcam.py` | Stub — shows planned explainability hook | Optional | source |
| Best checkpoint | `outputs/checkpoints/training/best.pt` | 524 MB — model weights exist and are loadable | Required | model |
| Best state JSON | `metrics/best_state.json` | Epoch 24, monitor metric val_qwk | Required | json |
| Run manifest JSON | `metrics/run_manifest.json` | Environment fingerprint for reproducibility | Required | json |
| Paper gaps register | `documents/milestone_04_paper_gaps.md` | Honest list of unresolved architectural gaps | Required | reference |

**Hero visual:** An inference pipeline flow diagram: fundus image → quality gate → preprocessing → DualSwinOrd → ordinal decode → referral decision. **Must be drawn.** The `predict.py` stub confirms the pipeline is designed but not yet implemented.

**Limitations to state explicitly on this slide:**
- Validation metrics only (no locked test set)
- No Grad-CAM / explainability yet
- 25 near-duplicate clusters span splits (potential val inflation)
- Single training run (no ablation, no seed variance)
- Single-source dataset (APTOS only, no external validation)
- Several architectural gaps remain unresolved (PG-01 through PG-19)

---

### Slide 15 — Future Work & Conclusion

| Asset | Path in package | Description | Priority | Visual type |
|---|---|---|---|---|
| README.md | `documents/README.md` | Milestone checklist — shows what is done vs. pending | Required | document |
| Paper gaps register | `documents/milestone_04_paper_gaps.md` | Open gaps PG-01 to PG-19 drive future work | Required | reference |
| Learning curves JSON | `metrics/learning_curves.json` | Shows model has not plateaued — room for improvement | Optional | json |

**Hero visual:** A research roadmap ribbon: Done (backbone, dual heads, losses, training) → In progress (test eval, Grad-CAM) → Future (external validation, uncertainty, SSL pretraining, multimodal OCT, ablation study). **Must be drawn.**

---

====================================

## METRICS TO VISUALIZE

====================================

All values are from `metrics/best_metrics.json` and `metrics/final_report.json` unless noted.

| Metric | Value | Best Visualization | Source file |
|---|---|---|---|
| **QWK** | 0.9177 | Gauge / single large number with context bar | best_metrics.json |
| **Accuracy** | 0.8517 | Single number (secondary to QWK) | best_metrics.json |
| **Macro F1** | 0.6959 | Single number | best_metrics.json |
| **MAE** | 0.186 | Single number | best_metrics.json |
| **Within-one accuracy** | 0.9709 | Single number — very strong, highlight | best_metrics.json |
| **ECE** | 0.052 | Gauge (lower = better calibration) | final_report.json |
| **Per-class Precision** | 0→0.994, 1→0.720, 2→0.780, 3→0.389, 4→0.655 | Grouped Bar Chart | best_metrics.json |
| **Per-class Recall** | 0→0.994, 1→0.529, 2→0.848, 3→0.333, 4→0.760 | Grouped Bar Chart | best_metrics.json |
| **Per-class F1** | 0→0.994, 1→0.610, 2→0.813, 3→0.359, 4→0.704 | Radar Chart (5 axes) | best_metrics.json |
| **Confusion Matrix** | See CSV | Normalised Heatmap with annotation | confusion_matrix.csv |
| **Val QWK over epochs** | 0.815→0.918 (best ep.24) | Line Chart with best-epoch marker | epoch_log.csv |
| **Train vs Val Loss** | Converging, no divergence | Dual-line Chart | epoch_log.csv |
| **Ordinal loss trajectory** | 0.251→0.080 | Line Chart (shows ordinal head learning) | epoch_log.csv |
| **LR schedule** | 1e-4 → 0 cosine | Line Chart (secondary) | epoch_log.csv |
| **Class distribution** | 1803/362/977/187/286 | Horizontal Bar Chart | class_distribution.csv |
| **Majority:minority ratio** | 9.64:1 | Callout number | dataset_statistics.json |
| **Effective num classes** | 3.61 / 5 | Gauge | dataset_statistics.json |
| **Cleaning: excluded** | 47 exact dupes / 3,662 total | Donut / Stacked Bar | cleaning_report.json |
| **Quality flags** | blurry 37, dark 47, bright 8, noisy 37, low-contrast 21 | Horizontal Bar | cleaning_report.json |
| **Split sizes** | train 2929 / val 344 / test 342 | Stacked Bar | class_distribution.csv |
| **Reliability diagram** | Already rendered | Use existing PNG | reliability_diagram.png |

---

====================================

## DIAGRAMS TO DRAW

====================================

None of these exist as files in the repository. All must be created for the presentation.

| # | Diagram | Slide | Description |
|---|---|---|---|
| D1 | **Grade 0→4 Fundus Progression Strip** | 3 | 5 fundus images side-by-side with lesion callouts (microaneurysms, hemorrhages, exudates, neovascularization). Source images from `data/raw/train/` — manual selection required. |
| D2 | **Data Pipeline Flow** | 5 | Raw 3,662 → MD5 audit → 47 excluded → 3,615 → group-aware split → train/val/test with counts. Show 0 ID/MD5 leakage, 25 near-dup clusters flagged. |
| D3 | **Preprocessing Pipeline** | 6 | Canonical order: border removal → circular crop → resize (INTER_AREA) → [CLAHE off] → normalize → augment (train only). Before/after image pair at each step. |
| D4 | **Backbone Decision Matrix** | 7 | Table: ResNet / EfficientNet / ViT / Swin scored on receptive field, data efficiency, lesion sensitivity, pretrain, FLOPs. |
| D5 | **DualSwinOrd Architecture** | 8 | Full block diagram: Input → Swin-Tiny (4 stages) → SPM (stage 3 injection, sigmoid gate) → PLKA (dilations 1/2/3, SE fusion) → Shared Neck (GAP→512d) → {Cls Head K=5, Ordinal Head K−1}. Tensor shapes at each transition. |
| D6 | **SPM Detail** | 8 | FiLM-style: text embedding → linear → sigmoid gate → element-wise modulation of stage-3 backbone features. |
| D7 | **PLKA Detail** | 8 | Three parallel conv branches (dilation 1, 2, 3) → SE-style channel attention fusion → output features. |
| D8 | **Ordinal Loss Ladder** | 9 | K−1=4 binary sub-tasks: P(grade>0), P(grade>1), P(grade>2), P(grade>3). Show how argmax of classification + ordinal constraints combine at inference. |
| D9 | **Loss Equation Visual** | 9 | L_total = 0.5·L_cls + 0.5·L_ord. Show L_cls = CE+LS(0.1), L_ord = DPE/CARM BCE. |
| D10 | **Training Pipeline** | 10 | DataLoader → AMP forward → dual-loss backward → AdamW step → cosine LR → EMA checkpoint → early-stop monitor (val_qwk). |
| D11 | **Inference Pipeline** | 14 | Fundus image → quality gate (ungradable → reject) → preprocessing → DualSwinOrd → ordinal decode → grade + confidence → referral decision (≥2 = refer). |
| D12 | **Research Roadmap** | 15 | Timeline ribbon: Completed / In Progress / Future Work. |
| D13 | **Prior Art Comparison Table** | 2 | Gulshan 2016 / IDx-DR / APTOS top solutions vs. DualSwinOrd — method, dataset, metric, limitation. |
| D14 | **Two-lane Screening Flow** | 1 | Current: patient → ophthalmologist (bottleneck) → months. AI-assisted: fundus camera → model → instant triage → specialist only for positives. |

---

====================================

## MISSING ASSETS

====================================

These visuals **do not exist** in the repository and must be created before the presentation.

| # | Missing asset | Slide | Severity | How to obtain |
|---|---|---|---|---|
| M1 | **Grade 0–4 fundus progression strip** | 3 | 🔴 Blocking | Manually select 5 representative images from `data/raw/train/`, annotate lesions |
| M2 | **Grad-CAM / attention overlays** | 13 | 🔴 Blocking | Run `src/visualization/gradcam.py` (currently a stub) against `best.pt`; requires implementation |
| M3 | **Ablation study results** | 11 | 🔴 Blocking | Requires 4 additional training runs (baseline, +SPM, +PLKA, +ordinal); no existing data |
| M4 | **Test-set evaluation metrics** | 12 | 🔴 Blocking | Run `scripts/evaluate.py` (currently a stub) on the held-out test split using `best.pt` |
| M5 | **EDA plots (exported)** | 4 | 🟠 High | Export all inline plots from `notebooks/EDA_APTOS_Research.ipynb` as PNG |
| M6 | **Architecture diagram (redrawn)** | 8 | 🟠 High | Redraw Figure 1 from paper PDF as editable vector/slide diagram |
| M7 | **Preprocessing before/after pairs** | 6 | 🟠 High | Run preprocessing pipeline on 2–3 sample images and save intermediate outputs |
| M8 | **Learning curve chart** | 11 | 🟡 Medium | Generate from `epoch_log.csv` — data exists, chart does not |
| M9 | **Per-class radar chart** | 12 | 🟡 Medium | Generate from `best_metrics.json` per-class F1 |
| M10 | **Confusion matrix heatmap (hi-res)** | 12 | 🟡 Medium | Re-render from `confusion_matrix.csv` — existing PNG is only 21 KB |
| M11 | **All D1–D14 diagrams** | various | 🟡 Medium | See DIAGRAMS TO DRAW section |
| M12 | **ROC / PR curves** | appendix | 🟡 Medium | Requires running evaluation script with probability outputs |
| M13 | **TensorBoard screenshots** | appendix | 🟢 Low | Launch TensorBoard against `outputs/logs/training/tensorboard/` (18 event files present) |
| M14 | **Augmentation policy visual** | 6 | 🟢 Low | Apply augmentation transforms to a sample image and save grid |

---

====================================

## FILES RECOMMENDED FOR APPENDIX

====================================

These are backup slides for Q&A — one per likely professor question.

| Appendix slide | Asset(s) | Answers question |
|---|---|---|
| A1 — Full training curves (all metrics) | `metrics/epoch_log.csv` | "Show me train/val curves — is there overfitting?" |
| A2 — Per-class precision/recall/F1 table | `metrics/best_metrics.json` | "Walk me through per-class performance" |
| A3 — Confusion matrix (raw counts) | `metrics/confusion_matrix.csv` | "Where exactly do errors concentrate?" |
| A4 — Reliability diagram | `charts/reliability_diagram.png` | "Is the model well-calibrated? What is ECE?" |
| A5 — Split verification report | `reports/split_verification_report.json` | "How did you prevent leakage? What about near-duplicates?" |
| A6 — Cleaning report summary | `reports/cleaning_report.json` | "How many images were removed and why?" |
| A7 — Class weights table | `reports/dataset_statistics.json` | "How did you handle imbalance?" |
| A8 — Model config with gap annotations | `configs/model.yaml` + `documents/milestone_04_paper_gaps.md` | "What is paper-confirmed vs. your engineering choices?" |
| A9 — Training config | `configs/training.yaml` | "What optimizer, LR, schedule? Are these paper-confirmed?" |
| A10 — Run manifest | `metrics/run_manifest.json` | "What hardware/environment? Is this reproducible?" |
| A11 — Ordinal loss source | `architecture/ordinal_loss.py` | "Is the ordinal head more than regression-with-rounding?" |
| A12 — CARM gap (PG-17) | `documents/milestone_04_paper_gaps.md` | "What is CARM and does your implementation match the paper?" |
| A13 — Near-duplicate warn detail | `reports/split_verification_report.json` | "Your split verification returned WARN — explain" |
| A14 — Normalization stats | `metrics/normalization_stats.json` | "What normalization statistics did you use and why?" |

