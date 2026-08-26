"""Generate derived figures for the RetinaGrade presentation.

Every figure is built ONLY from verified project artefacts already copied into
Presentation/ (CSV/JSON) or from the real raw fundus images under data/raw/.
No numbers are invented. Nothing outside Presentation/ is written or modified.

Outputs land in the appropriate Presentation/Figures/<Category>/ subfolder.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(r"d:\RetinaGrade")
PRES = ROOT / "Presentation"
FIG = PRES / "Figures"
MET = PRES / "Metrics"
REP = PRES / "Reports"

CLASS_SHORT = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
CLASS_FULL = [
    "No DR (Normal)",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR (PDR)",
]
PALETTE = ["#2c7fb8", "#41b6c4", "#7fcdbb", "#fecc5c", "#f03b20"]


def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig, path: Path, dpi=150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(PRES)}")


def fig_class_distribution():
    """Grouped bar: per-split effective (post-cleaning) class counts + overall."""
    stats = _load_json(REP / "dataset_statistics.json")
    per = stats["counts"]["per_split"]
    overall = stats["counts"]["overall"]
    splits = ["train", "val", "test"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    x = np.arange(5)
    w = 0.25
    for i, sp in enumerate(splits):
        vals = [per[sp][str(g)] for g in range(5)]
        ax1.bar(x + (i - 1) * w, vals, w, label=sp, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_SHORT)
    ax1.set_ylabel("Image count")
    ax1.set_title("Per-split class distribution (effective, post-cleaning)")
    ax1.legend(title="Split")
    for i, sp in enumerate(splits):
        for g in range(5):
            v = per[sp][str(g)]
            ax1.text(g + (i - 1) * w, v + 5, str(v), ha="center", va="bottom", fontsize=7)

    ov = [overall[str(g)] for g in range(5)]
    bars = ax2.bar(x, ov, color=PALETTE, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASS_SHORT)
    ax2.set_ylabel("Image count")
    total = stats["counts"]["total_images"]
    ax2.set_title(f"Overall class distribution (n = {total})")
    for b, v in zip(bars, ov):
        pct = 100 * v / total
        ax2.text(b.get_x() + b.get_width() / 2, v + 5, f"{v}\n{pct:.1f}%",
                 ha="center", va="bottom", fontsize=8)

    imb = stats["imbalance"]
    ax2.text(0.97, 0.95,
             f"Majority:minority = {imb['majority_minority_ratio']:.2f}:1\n"
             f"Gini = {imb['gini']:.3f}\n"
             f"Norm. imbalance = {imb['normalized_imbalance_index']:.3f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="#fff4c2", edgecolor="grey"))
    _save(fig, FIG / "Dataset" / "class_distribution_derived.png")


def fig_class_weights():
    """Reference class-weight strategies (reported, not applied in final run)."""
    stats = _load_json(REP / "dataset_statistics.json")
    rw = stats["reference_class_weights"]
    strategies = ["inverse", "inverse_sqrt", "effective_number", "balanced"]
    x = np.arange(5)
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, s in enumerate(strategies):
        ax.bar(x + (i - 1.5) * w, rw[s], w, label=s, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_SHORT)
    ax.set_ylabel("Reference weight")
    ax.set_title("Reference class-weight strategies (reported; final run uses none)")
    ax.legend()
    _save(fig, FIG / "Dataset" / "reference_class_weights.png")


def _read_col(path: Path, col: str) -> list[float]:
    out: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append(float(row[col]))
            except (ValueError, KeyError, TypeError):
                continue
    return out


def fig_eda_quality():
    """EDA: image geometry & border/brightness distributions from sample CSVs."""
    geo = ROOT / "data" / "processed" / "image_geometry_sample.csv"
    bor = ROOT / "data" / "processed" / "fundus_border_sample.csv"

    widths = _read_col(geo, "width")
    heights = _read_col(geo, "height")
    aspects = _read_col(geo, "aspect_ratio")
    tissue = _read_col(bor, "tissue_pct")
    brightness = _read_col(bor, "mean_brightness")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].hist(widths, bins=30, color="#2c7fb8", edgecolor="black", linewidth=0.3)
    axes[0, 0].set_title(f"Image width distribution (n={len(widths)} sampled)")
    axes[0, 0].set_xlabel("width (px)")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(aspects, bins=30, color="#41b6c4", edgecolor="black", linewidth=0.3)
    axes[0, 1].set_title(f"Aspect-ratio distribution (n={len(aspects)} sampled)")
    axes[0, 1].set_xlabel("width / height")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].scatter(widths, heights, s=6, alpha=0.4, color="#7a0177")
    axes[1, 0].set_title("Resolution scatter (width vs height, sampled)")
    axes[1, 0].set_xlabel("width (px)")
    axes[1, 0].set_ylabel("height (px)")

    if brightness:
        axes[1, 1].hist(brightness, bins=30, color="#fdae61", edgecolor="black", linewidth=0.3)
        axes[1, 1].set_title(f"Mean brightness distribution (n={len(brightness)} sampled)")
        axes[1, 1].set_xlabel("mean brightness (0-255)")
        axes[1, 1].set_ylabel("count")
    elif tissue:
        axes[1, 1].hist(tissue, bins=30, color="#fdae61", edgecolor="black", linewidth=0.3)
        axes[1, 1].set_title(f"Retinal tissue coverage % (n={len(tissue)} sampled)")
        axes[1, 1].set_xlabel("tissue %")
        axes[1, 1].set_ylabel("count")
    else:
        axes[1, 1].axis("off")

    fig.suptitle("EDA: image geometry, resolution & intensity (sampled)",
                 fontsize=14, fontweight="bold")
    _save(fig, FIG / "EDA" / "eda_geometry_quality.png")

def fig_cleaning():
    """Cleaning: before/after per split + flag counts."""
    rep = _load_json(REP / "cleaning_report.json")
    before = rep["class_distribution_before"]
    after = rep["class_distribution_after"]
    splits = ["train", "val", "test"]

    def tot(d, sp):
        return sum(int(v) for v in d[sp].values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))

    x = np.arange(3)
    w = 0.38
    b_tot = [tot(before, sp) for sp in splits]
    a_tot = [tot(after, sp) for sp in splits]
    ax1.bar(x - w / 2, b_tot, w, label="before cleaning", color="#bdbdbd", edgecolor="black")
    ax1.bar(x + w / 2, a_tot, w, label="after cleaning", color="#2c7fb8", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.set_ylabel("Image count")
    ax1.set_title("Split sizes before vs after cleaning")
    ax1.legend()
    for i in range(3):
        ax1.text(i - w / 2, b_tot[i] + 8, str(b_tot[i]), ha="center", fontsize=8)
        ax1.text(i + w / 2, a_tot[i] + 8, str(a_tot[i]), ha="center", fontsize=8)
        d = b_tot[i] - a_tot[i]
        ax1.text(i, max(b_tot[i], a_tot[i]) + 40, f"-{d}", ha="center",
                 fontsize=9, color="#d73027", fontweight="bold")

    flags = rep["flag_counts"]
    names = list(flags.keys())
    vals = [flags[n] for n in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    ax2.barh(names, vals, color="#fd8d3c", edgecolor="black", linewidth=0.4)
    ax2.set_xlabel("count")
    ax2.set_title("Quality/duplicate FLAGS (investigation only — never deleted)")
    for i, v in enumerate(vals):
        ax2.text(v + 2, i, str(v), va="center", fontsize=8)

    excl = rep["totals"]["excluded"]
    ax2.text(0.97, 0.05,
             f"Total excluded (exact MD5 dup only): {excl}\n"
             f"Files deleted from disk: {rep['policy']['files_deleted']}",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="#e5f5e0", edgecolor="grey"))
    _save(fig, FIG / "Cleaning" / "cleaning_before_after_flags.png")


def _load_epoch_log():
    rows = []
    with open(MET / "epoch_log.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def fig_training_curves():
    rows = _load_epoch_log()
    ep = [int(r["epoch"]) for r in rows]
    tl = [float(r["train_loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    qwk = [float(r["val_qwk"]) for r in rows]
    best_i = int(np.argmax(qwk))

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(ep, tl, "-o", ms=3, color="#2c7fb8", label="train loss")
    ax1.plot(ep, vl, "-s", ms=3, color="#f03b20", label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ep, qwk, "-^", ms=3, color="#238b45", label="val QWK")
    ax2.set_ylabel("validation QWK", color="#238b45")
    ax2.tick_params(axis="y", labelcolor="#238b45")
    ax2.axvline(best_i, ls="--", color="grey", lw=1)
    ax2.annotate(f"best epoch {best_i}\nval QWK = {qwk[best_i]:.4f}",
                 xy=(best_i, qwk[best_i]), xytext=(best_i + 3, qwk[best_i] - 0.06),
                 arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="#ffffcc", edgecolor="grey"))

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right")
    ax1.set_title("Training / validation loss and validation QWK over 50 epochs")
    _save(fig, FIG / "Training" / "training_curves.png")


def fig_training_perclass_f1():
    rows = _load_epoch_log()
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    for g in range(5):
        key = f"per_class_f1.{CLASS_FULL[g]}"
        vals = [float(r[key]) for r in rows]
        ax.plot(ep, vals, "-", lw=1.6, color=PALETTE[g], label=CLASS_SHORT[g])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation per-class F1")
    ax.set_title("Validation per-class F1 across training")
    ax.grid(alpha=0.3)
    ax.legend(title="grade")
    _save(fig, FIG / "Training" / "training_perclass_f1.png")

def fig_evaluation_confusion_matrix():
    """Hi-res annotated TEST confusion matrix (counts + row-normalised)."""
    rep = _load_json(MET / "test_evaluation_report.json")
    cm = np.array(rep["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, data, fmt, title in [
        (ax1, cm, "d", "TEST confusion matrix (counts)"),
        (ax2, cm_norm, ".2f", "TEST confusion matrix (row-normalised recall)"),
    ]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(CLASS_SHORT, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_SHORT)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        thresh = data.max() / 2
        for i in range(5):
            for j in range(5):
                v = cm[i, j] if fmt == "d" else cm_norm[i, j]
                txt = (f"{v:.2f}" if fmt == ".2f" else str(v))
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=10)
    _save(fig, FIG / "Evaluation" / "test_confusion_matrix_hires.png", dpi=160)


def fig_evaluation_perclass_bar():
    """Grouped bar: per-class precision, recall, F1 on TEST split."""
    rep = _load_json(MET / "test_evaluation_report.json")
    m = rep["metrics"]
    precision = [m["per_class_precision"][c] for c in CLASS_FULL]
    recall = [m["per_class_recall"][c] for c in CLASS_FULL]
    f1 = [m["per_class_f1"][c] for c in CLASS_FULL]

    x = np.arange(5)
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w, precision, w, label="Precision", color="#2c7fb8", edgecolor="black")
    ax.bar(x,     recall,    w, label="Recall",    color="#7fcdbb", edgecolor="black")
    ax.bar(x + w, f1,        w, label="F1",        color="#fd8d3c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_SHORT)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1 — TEST split (n = 342)")
    ax.legend()
    ax.axhline(m["macro_f1"], ls="--", color="grey", lw=1.2)
    ax.text(4.45, m["macro_f1"] + 0.01, f"macro F1 = {m['macro_f1']:.3f}", fontsize=9, color="grey")
    for i in range(5):
        for offset, vals in [(-w, precision), (0, recall), (w, f1)]:
            ax.text(i + offset, vals[i] + 0.01, f"{vals[i]:.2f}", ha="center", fontsize=7.5)
    _save(fig, FIG / "Evaluation" / "test_perclass_prf1_bar.png")


def fig_evaluation_f1_radar():
    """Radar/spider chart of per-class F1 for TEST and VAL."""
    rep_t = _load_json(MET / "test_evaluation_report.json")
    rep_v = _load_json(MET / "val_metrics.json")
    f1_t = [rep_t["metrics"]["per_class_f1"][c] for c in CLASS_FULL]
    f1_v = [rep_v["metrics"]["per_class_f1"][c] for c in CLASS_FULL]

    N = 5
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for vals, label, color in [
        (f1_t, "TEST", "#e34a33"),
        (f1_v, "VAL",  "#2c7fb8"),
    ]:
        vals_c = vals + vals[:1]
        ax.plot(angles, vals_c, "-o", lw=2, color=color, label=label)
        ax.fill(angles, vals_c, alpha=0.12, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CLASS_SHORT, size=11)
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 radar — TEST vs VAL", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    _save(fig, FIG / "Evaluation" / "test_val_f1_radar.png")


def fig_evaluation_test_val_bars():
    """Bar comparison of key aggregate metrics: TEST vs VAL."""
    rep_t = _load_json(MET / "test_evaluation_report.json")
    rep_v = _load_json(MET / "val_metrics.json")
    keys = ["qwk", "accuracy", "macro_f1", "within_one_accuracy"]
    labels = ["QWK", "Accuracy", "Macro F1", "Within-one\nAccuracy"]
    t_vals = [rep_t["metrics"][k] for k in keys]
    v_vals = [rep_v["metrics"][k] for k in keys]

    x = np.arange(4)
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, t_vals, w, label="TEST", color="#e34a33", edgecolor="black")
    ax.bar(x + w / 2, v_vals, w, label="VAL",  color="#2c7fb8", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Key metrics — TEST vs VAL (best checkpoint, epoch 23/24)")
    ax.legend()
    for i, (tv, vv) in enumerate(zip(t_vals, v_vals)):
        ax.text(i - w / 2, tv + 0.01, f"{tv:.3f}", ha="center", fontsize=8, color="#c0392b")
        ax.text(i + w / 2, vv + 0.01, f"{vv:.3f}", ha="center", fontsize=8, color="#2980b9")

    extra_keys = [("referable_auc", "Ref. AUC"), ("mae", "MAE")]
    note = " | ".join(
        f"TEST {lb}: {rep_t['metrics'][k]:.4f}   VAL: {rep_v['metrics'][k]:.4f}"
        for k, lb in extra_keys
    )
    val_ece = _load_json(MET / "final_report.json")["validation_ece"]
    note += f"\nECE  TEST: {rep_t['ece']:.4f}   VAL: {val_ece:.4f}"
    ax.text(0.5, -0.14, note, transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    _save(fig, FIG / "Evaluation" / "test_val_metrics_comparison.png")


def fig_error_analysis():
    """Error-analysis confusion heatmap highlighting key off-diagonal cells."""
    rep = _load_json(MET / "test_evaluation_report.json")
    cm = np.array(rep["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 7))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_SHORT, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_SHORT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Error analysis: row-normalised TEST confusion (shading = error severity)")
    plt.colorbar(im, ax=ax, label="fraction")

    notable = [(1, 2, "Mild→Moderate\n(14/28)"),
               (3, 2, "Severe→Moderate\n(9/13)"),
               (4, 2, "PDR→Moderate\n(7/27)"),
               (2, 1, "Moderate→Mild\n(2/77)")]

    for i, j, label in notable:
        ax.add_patch(FancyBboxPatch((j - 0.45, i - 0.45), 0.9, 0.9,
                                    boxstyle="round,pad=0.05",
                                    linewidth=2.5, edgecolor="navy", fill=False))

    thresh = cm_norm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > thresh else "black")
    _save(fig, FIG / "ErrorAnalysis" / "error_analysis_confusion.png", dpi=160)

def _box(ax, xy, w, h, text, fc, fontsize=9, ec="black"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def _arrow(ax, xy1, xy2, color="#333"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=color, shrinkA=2, shrinkB=2))


def fig_architecture():
    """Model architecture block diagram (DualSwinOrd)."""
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    _box(ax, (0.2, 3.0), 1.7, 1.0, "Input fundus\n512x512x3", "#deebf7")
    _box(ax, (2.2, 3.0), 1.9, 1.0, "Deterministic\npreprocessing\n(crop+resize+norm)", "#deebf7")
    _box(ax, (4.4, 2.7), 2.0, 1.6, "Swin-Tiny\nbackbone\n(4 stages,\nImageNet-pretrained)", "#c7e9c0")
    _box(ax, (6.8, 4.4), 1.9, 1.1, "SPM\n(semantic prior,\nFiLM gate @ stage 3)", "#fdd0a2")
    _box(ax, (6.8, 1.2), 1.9, 1.1, "PLKA\n(parallel dilated\nattn @ stage 3)", "#fdd0a2")
    _box(ax, (9.0, 2.7), 1.8, 1.6, "Shared neck\nGAP -> 512-d\n(dropout 0.0)", "#c7e9c0")
    _box(ax, (11.1, 4.2), 2.0, 1.2, "Classification head\nK=5 logits -> softmax", "#dadaeb")
    _box(ax, (11.1, 1.4), 2.0, 1.2, "Ordinal head\nK-1 thresholds -> BCE", "#dadaeb")
    _box(ax, (13.4, 3.0), 1.4, 1.0, "argmax\n-> grade 0..4", "#fee0d2")

    _arrow(ax, (1.9, 3.5), (2.2, 3.5))
    _arrow(ax, (4.1, 3.5), (4.4, 3.5))
    _arrow(ax, (6.4, 3.9), (6.8, 4.7))
    _arrow(ax, (6.4, 3.1), (6.8, 1.9))
    _arrow(ax, (8.7, 4.7), (9.0, 3.9))
    _arrow(ax, (8.7, 1.9), (9.0, 3.1))
    _arrow(ax, (10.8, 3.9), (11.1, 4.6))
    _arrow(ax, (10.8, 3.1), (11.1, 2.0))
    _arrow(ax, (13.1, 4.6), (13.7, 4.0))
    _arrow(ax, (13.1, 2.0), (13.7, 3.5), color="#999")

    ax.text(13.9, 2.4, "ordinal probs\ncollected,\nnot used for\nprediction",
            fontsize=7, color="#777", ha="center")
    ax.text(7.5, 6.5, "DualSwinOrd architecture", fontsize=15, fontweight="bold", ha="center")
    ax.text(7.5, 0.25,
            "Text prompts (5 clinical descriptions) -> frozen HashingTextAdapter -> 512-d embedding -> SPM conditioning",
            fontsize=8.5, color="#555", ha="center", style="italic")
    _save(fig, FIG / "Architecture" / "model_architecture.png", dpi=160)


def fig_workflow():
    """End-to-end pipeline workflow diagram."""
    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    steps = [
        ("Raw fundus\nimage", "#deebf7"),
        ("Cleaning\n(MD5 dedup,\nintegrity)", "#deebf7"),
        ("Preprocessing\n(crop, resize,\nnormalize)", "#c7e9c0"),
        ("Augmentation\n(train only)", "#c7e9c0"),
        ("Training\n(50 epochs,\nAdamW+cosine)", "#fdd0a2"),
        ("Validation\n(QWK monitor)", "#fdd0a2"),
        ("Checkpoint\nselection\n(best val QWK)", "#dadaeb"),
        ("Testing\n(held-out)", "#dadaeb"),
        ("Prediction\n(argmax)", "#fee0d2"),
        ("Final grade\n0..4", "#fee0d2"),
    ]
    n = len(steps)
    bw = 1.28
    gap = (15 - n * bw) / (n + 1)
    y = 1.7
    xs = []
    for i, (txt, fc) in enumerate(steps):
        x = gap + i * (bw + gap)
        xs.append(x)
        _box(ax, (x, y), bw, 1.1, txt, fc, fontsize=7.5)
    for i in range(n - 1):
        _arrow(ax, (xs[i] + bw, y + 0.55), (xs[i + 1], y + 0.55))

    ax.text(7.5, 3.7, "RetinaGrade end-to-end workflow", fontsize=14, fontweight="bold", ha="center")
    ax.text(7.5, 0.6, "Deterministic preprocessing is identical across train/val/test; "
                      "augmentation applies to the train split only.",
            fontsize=8.5, color="#555", ha="center", style="italic")
    _save(fig, FIG / "Workflow" / "end_to_end_workflow.png", dpi=160)

def _ensure_root_on_path():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _wide_aspect_candidates(min_ar=1.32, max_ar=1.60, want=2):
    """Deterministically pick existing raw images with visible black borders."""
    geo = ROOT / "data" / "processed" / "image_geometry_sample.csv"
    rows = []
    with open(geo, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ar = float(r["aspect_ratio"])
            except (ValueError, KeyError):
                continue
            p = Path(r["filepath"])
            if min_ar <= ar <= max_ar and p.is_file():
                rows.append((ar, str(p), r.get("diagnosis", "?"), r.get("split", "?")))
    rows.sort(key=lambda t: (-t[0], t[1]))  # widest first, then path (deterministic)
    return rows[:want]


def fig_preprocessing_beforeafter():
    """Deterministic geometry pipeline shown stage-by-stage on real images.

    Reuses the project transforms exactly: BlackBorderRemoval -> CircularCrop
    -> FundusResize, with the parameters from configs/data.yaml.
    """
    _ensure_root_on_path()
    from src.data.preprocessing import BlackBorderRemoval, CircularCrop, FundusResize
    from src.utils.helpers import read_image_rgb

    border = BlackBorderRemoval(threshold=10, blur_kernel=5, min_area_ratio=0.10, padding=0)
    circ = CircularCrop(margin_ratio=0.0, fill_value=0)
    resize = FundusResize(size=512, interpolation_down="area",
                          interpolation_up="linear", keep_aspect_ratio=False)

    cands = _wide_aspect_candidates(want=2)
    if not cands:
        print("  [skip] no wide-aspect raw image found for preprocessing demo")
        return
    nrows = len(cands)
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 4.3 * nrows))
    axes = np.atleast_2d(axes)
    for ri, (ar, path, diag, split) in enumerate(cands):
        img0 = read_image_rgb(path)
        img1 = border.transform(img0)
        img2 = circ.transform(img1)
        img3 = resize.transform(img2)
        stages = [
            (img0, f"0. Original\n{img0.shape[1]}x{img0.shape[0]} (AR {ar:.2f})"),
            (img1, f"1. Black-border removal\n{img1.shape[1]}x{img1.shape[0]}"),
            (img2, f"2. Circular crop\n{img2.shape[1]}x{img2.shape[0]}"),
            (img3, f"3. Resize to 512x512\n{img3.shape[1]}x{img3.shape[0]}"),
        ]
        for ci, (im, title) in enumerate(stages):
            ax = axes[ri, ci]
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.suptitle("Deterministic preprocessing pipeline (applied identically to train / val / test)",
                 fontsize=14, fontweight="bold")
    _save(fig, FIG / "Preprocessing" / "preprocessing_stages_beforeafter.png", dpi=150)


def fig_augmentation_grid():
    """Show each ENABLED train-time augmentation applied to one real image.

    The augmentation policy is built by the project (build_policy) from
    configs/data.yaml, so the transforms and their parameters are exactly the
    ones training uses. Each transform is shown in isolation; probability is
    forced to 1.0 only so the (otherwise stochastic) effect is visible, and a
    fixed seed makes the panel reproducible. No project file is modified.
    """
    import random

    _ensure_root_on_path()
    import albumentations as A

    from src.data.augmentation.policies import build_policy
    from src.data.preprocessing import BlackBorderRemoval, CircularCrop, FundusResize
    from src.utils.config import load_data_config
    from src.utils.helpers import read_image_rgb

    cfg = load_data_config("configs/data.yaml")
    tagged = build_policy(cfg)  # only enabled transforms, in canonical order

    # Preprocess one representative raw image so augmentation sees a 512x512
    # fundus, exactly as during training.
    cands = _wide_aspect_candidates(want=1)
    if not cands:
        print("  [skip] no raw image found for augmentation demo")
        return
    _, path, diag, split = cands[0]
    border = BlackBorderRemoval(threshold=10, blur_kernel=5, min_area_ratio=0.10)
    circ = CircularCrop(margin_ratio=0.0, fill_value=0)
    resize = FundusResize(size=512, interpolation_down="area", interpolation_up="linear")
    base = resize.transform(circ.transform(border.transform(read_image_rgb(path))))

    panels = [(base, "Preprocessed input\n(no augmentation)")]
    for t in tagged:
        random.seed(42)
        np.random.seed(42)
        tr = t.transform
        orig_p = getattr(tr, "p", 1.0)
        try:
            tr.p = 1.0  # force the effect to appear for illustration only
            out = A.Compose([tr])(image=base)["image"]
        finally:
            tr.p = orig_p
        label = t.key.replace("_", " ")
        p_txt = f"p={orig_p:g}"
        panels.append((out, f"{label}\n({p_txt} in training)"))

    n = len(panels)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, (im, title) in enumerate(panels):
        axes[k].imshow(im)
        axes[k].set_title(title, fontsize=10)
        axes[k].axis("off")
    fig.suptitle("Train-split augmentations (each shown in isolation; forbidden: "
                 "MixUp / CutMix / CutOut / RandomErasing)",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG / "Augmentation" / "augmentation_examples_grid.png", dpi=150)


ALL_FIGS = [
    fig_class_distribution,
    fig_class_weights,
    fig_eda_quality,
    fig_cleaning,
    fig_training_curves,
    fig_training_perclass_f1,
    fig_evaluation_confusion_matrix,
    fig_evaluation_perclass_bar,
    fig_evaluation_f1_radar,
    fig_evaluation_test_val_bars,
    fig_error_analysis,
    fig_architecture,
    fig_workflow,
    fig_preprocessing_beforeafter,
    fig_augmentation_grid,
]


def main():
    ok, failed = 0, 0
    for fn in ALL_FIGS:
        print(f"[{fn.__name__}]")
        try:
            fn()
            ok += 1
        except Exception as exc:  # keep going; one figure must not abort the rest
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\nFigures complete: {ok} ok, {failed} failed.")
    return 0


if __name__ == "__main__":
    main()
