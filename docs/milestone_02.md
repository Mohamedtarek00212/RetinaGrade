# Milestone 02 — Exploratory Data Analysis (EDA)

## Goal

Create a professional, publication-quality EDA notebook to understand the APTOS 2019 dataset structure, integrity, image statistics, and class distribution before any model implementation.

## Files Created

- `notebooks/EDA_APTOS_Research.ipynb` — Publication-quality EDA notebook (16 sections: integrity, class distribution, resolution/quality/color-space analysis, visual and lesion-level exploration, statistical hypothesis testing, outlier/near-duplicate/leakage audits, clinical interpretation, and research insights). Each section explains its scientific rationale, presents visualizations, interprets them, and states its implications for later milestones (preprocessing, augmentation, loss function, sampling, evaluation, explainability).

## Execution Instructions

1. Activate the project virtual environment:
   ```powershell
   .venv-gpu\Scripts\Activate.ps1
   ```

2. Launch Jupyter and run the notebook:
   ```powershell
   jupyter notebook notebooks/EDA_APTOS_Research.ipynb
   ```

3. Run all cells from top to bottom (Cell → Run All). No manual edits are required.

**Prerequisites:**
- Raw images must be present in `data/raw/{train,val,test}/`
- Split CSVs must be present in `data/splits/` (`train.csv`, `valid.csv`, `test.csv`)
- Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `Pillow`, `tqdm`, `scipy`

## Validation Checklist

- [ ] Notebook runs start-to-finish without errors
- [ ] Section 2: Dataset structure and split sizes printed
- [ ] Section 3: Integrity check reports missing/corrupted/zero-byte images
- [ ] Section 4: Width, height, aspect ratio histograms and statistics displayed
- [ ] Section 5: Class distribution tables, bar charts, and pie chart rendered
- [ ] Section 6: Representative sample images from each DR grade displayed
- [ ] Section 7: Summary statistics printed with key observations

## Next Milestone

**Milestone 03 — Data Pipeline**: Dataset class, data loading, label processing, DataLoader, and paper-specified preprocessing.
