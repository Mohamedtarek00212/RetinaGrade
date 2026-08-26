"""Generate prediction-example figures from best.pt over the TEST split.

Inference only. This reuses the exact project inference stack (same as
scripts/evaluate.py): configuration loading, build_datasets, build_dataloader,
build_model, and a torch.no_grad forward pass. No source code, config, or
checkpoint is modified -- best.pt is read with map_location and never rewritten.
All output is written strictly under Presentation/Figures/Predictions/.

For each selected TEST image the panel shows:
  * the original fundus image (loaded from the manifest 'path'),
  * ground-truth grade,
  * predicted grade,
  * confidence score (max softmax probability),
  * a Correct / Incorrect indicator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\RetinaGrade")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "Presentation" / "Figures" / "Predictions"

# Byte-for-byte identical to scripts/evaluate.py and scripts/train.py.
DEFAULT_TEXT_PROMPTS = [
    "No diabetic retinopathy: healthy retina with no visible lesions.",
    "Mild diabetic retinopathy: presence of microaneurysms only.",
    "Moderate diabetic retinopathy: microaneurysms, dot-blot hemorrhages, and hard exudates.",
    "Severe diabetic retinopathy: extensive hemorrhages, venous beading, and intraretinal microvascular abnormalities.",
    "Proliferative diabetic retinopathy: neovascularization, preretinal hemorrhage, or fibrovascular proliferation.",
]

CHECKPOINT = "outputs/checkpoints/training/best.pt"

def load_raw_rgb(path: str) -> np.ndarray:
    """Load the original fundus image as RGB uint8 using the project reader."""
    from src.utils.helpers import read_image_rgb

    img = read_image_rgb(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def run_inference():
    """Return (records, class_names). records: per-sample dicts in dataset order."""
    from src.data.dataloader import build_dataloader
    from src.data.datasets import build_datasets
    from src.models import build_model
    from src.models.config import load_model_config
    from src.models.semantic_prior.text_adapter import HashingTextAdapter
    from src.utils.config import load_data_config

    data_config = load_data_config("configs/data.yaml")
    model_config = load_model_config("configs/model.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = build_datasets(data_config)
    dataset = datasets["test"]
    loader = build_dataloader(dataset, data_config, "test")

    text_adapter = HashingTextAdapter(embedding_dim=model_config.spm.text_embedding_dim)
    model = build_model(
        model_config,
        num_classes=data_config.classes.num_classes,
        text_adapter=text_adapter,
        text_prompts=DEFAULT_TEXT_PROMPTS,
    )
    model.to(device)

    ckpt_path = data_config.resolve_path(CHECKPOINT)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = data_config.classes.names
    manifest = dataset.manifest  # id_code, path, label, ...
    path_by_id = dict(zip(manifest["id_code"].astype(str), manifest["path"].astype(str)))

    records = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].numpy()
            id_codes = batch["id_code"]
            logits = model(images)["classification_logits"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for i in range(len(id_codes)):
                idc = str(id_codes[i])
                records.append(
                    {
                        "id_code": idc,
                        "path": path_by_id.get(idc),
                        "label": int(labels[i]),
                        "pred": int(preds[i]),
                        "conf": float(probs[i, preds[i]]),
                        "probs": probs[i].tolist(),
                    }
                )
    return records, class_names, str(ckpt_path), checkpoint.get("epoch")

def select_examples(records):
    """Pick 8-15 representative examples: correct hits spanning all 5 grades,
    plus the clinically notable confusions, chosen deterministically (no RNG).

    Selection order within each bucket is by descending confidence so panels
    are reproducible run-to-run.
    """
    by_conf = sorted(records, key=lambda r: r["conf"], reverse=True)
    chosen: list[dict] = []
    seen_ids: set[str] = set()

    def take(pred, filt, n=1):
        added = 0
        for r in by_conf:
            if added >= n:
                break
            if r["id_code"] in seen_ids:
                continue
            if not filt(r):
                continue
            chosen.append(r)
            seen_ids.add(r["id_code"])
            added += 1

    # 1) One confident CORRECT example per grade (0..4).
    for g in range(5):
        take(g, lambda r, g=g: r["label"] == g and r["pred"] == g, n=1)

    # 2) A second confident correct for the two majority grades (0 and 2).
    take(0, lambda r: r["label"] == 0 and r["pred"] == 0, n=1)
    take(2, lambda r: r["label"] == 2 and r["pred"] == 2, n=1)

    # 3) Representative INCORRECT cases (the confusion matrix's frequent
    #    off-diagonals): Mild<->Moderate, Severe->Moderate, and any other error.
    take(None, lambda r: r["label"] == 1 and r["pred"] == 2, n=1)   # Mild -> Moderate
    take(None, lambda r: r["label"] == 2 and r["pred"] == 1, n=1)   # Moderate -> Mild
    take(None, lambda r: r["label"] == 3 and r["pred"] == 2, n=1)   # Severe -> Moderate
    take(None, lambda r: r["label"] == 4 and r["pred"] == 2, n=1)   # PDR  -> Moderate
    take(None, lambda r: r["label"] != r["pred"], n=1)              # any remaining error

    return chosen


def render(chosen, class_names, ckpt_path, epoch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    short = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

    # --- Combined grid ----------------------------------------------------
    n = len(chosen)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for k, r in enumerate(chosen):
        ax = axes[k]
        img = load_raw_rgb(r["path"])
        ax.imshow(img)
        ax.axis("off")
        correct = r["label"] == r["pred"]
        colour = "#1a9850" if correct else "#d73027"
        tag = "CORRECT" if correct else "INCORRECT"
        title = (
            f"{r['id_code']}\n"
            f"GT: {short[r['label']]}  |  Pred: {short[r['pred']]}\n"
            f"Conf: {r['conf']*100:.1f}%   [{tag}]"
        )
        ax.set_title(title, fontsize=10, color=colour, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(colour)
            spine.set_linewidth(4)
        ax.patch.set_edgecolor(colour)

    fig.suptitle(
        "RetinaGrade TEST-split prediction examples (best.pt, epoch "
        f"{epoch}) -- green=correct, red=incorrect",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = OUT_DIR / "prediction_examples_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Individual panels ------------------------------------------------
    for k, r in enumerate(chosen, 1):
        f, ax = plt.subplots(figsize=(5, 5.6))
        img = load_raw_rgb(r["path"])
        ax.imshow(img)
        ax.axis("off")
        correct = r["label"] == r["pred"]
        colour = "#1a9850" if correct else "#d73027"
        tag = "CORRECT" if correct else "INCORRECT"
        probs = r["probs"]
        prob_line = "  ".join(f"{short[j]}:{probs[j]*100:.0f}%" for j in range(5))
        ax.set_title(
            f"{r['id_code']}   [{tag}]\n"
            f"Ground truth: {class_names[r['label']]}\n"
            f"Predicted: {class_names[r['pred']]}   (conf {r['conf']*100:.1f}%)\n"
            f"{prob_line}",
            fontsize=9,
            color=colour,
            fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(colour)
            spine.set_linewidth(4)
        status = "correct" if correct else "incorrect"
        panel_path = OUT_DIR / f"prediction_{k:02d}_gt{r['label']}_pred{r['pred']}_{status}.png"
        f.tight_layout()
        f.savefig(panel_path, dpi=140, bbox_inches="tight")
        plt.close(f)

    return grid_path


def main() -> int:
    records, class_names, ckpt_path, epoch = run_inference()
    n_correct = sum(1 for r in records if r["label"] == r["pred"])
    print(f"Inference over {len(records)} TEST images | correct={n_correct} "
          f"({100*n_correct/len(records):.1f}%) | epoch={epoch}")
    chosen = select_examples(records)
    grid_path = render(chosen, class_names, ckpt_path, epoch)
    print(f"Selected {len(chosen)} prediction examples:")
    for r in chosen:
        ok = "CORRECT" if r["label"] == r["pred"] else "INCORRECT"
        print(f"  {r['id_code']}  GT={r['label']} pred={r['pred']} "
              f"conf={r['conf']*100:.1f}% [{ok}]")
    print(f"Grid: {grid_path}")
    print(f"Panels written under: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


