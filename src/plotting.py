import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "font.size": 9,
})

def _load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _find_model_dirs(outputs_dir: Path):
    # samo direktorijumi koji imaju makar val_metrics.json
    dirs = []
    for p in outputs_dir.iterdir():
        if p.is_dir() and (p / "val_metrics.json").exists():
            dirs.append(p)
    return dirs


def _plot_group_generic(outputs_dir: Path, split: str, kind: str):
    """
    kind in {"roc","pr"}
    Čita odgovarajuće JSON-ove i prikazuje grupni graf.
    """
    label_x, label_y = ("FPR", "TPR") if kind == "roc" else ("Recall", "Precision")
    diag = kind == "roc"

    plt.figure()
    any_curve = False
    for mdir in _find_model_dirs(outputs_dir):
        curve_path = mdir / f"{split}_{kind}.json"
        j = _load_json(curve_path)
        if not j:
            continue

        if kind == "roc":
            xs = np.array(j.get("fpr", []), dtype=float)
            ys = np.array(j.get("tpr", []), dtype=float)
            area = j.get("auc", None)
            metric_name = "AUC"
        else:
            xs = np.array(j.get("recall", []), dtype=float)
            ys = np.array(j.get("precision", []), dtype=float)
            area = j.get("ap", None)
            metric_name = "AP"

        if xs.size and ys.size:
            lbl = mdir.name if area is None else f"{mdir.name} ({metric_name}={area:.3f})"
            plt.plot(xs, ys, lw=1.8, label=lbl)
            any_curve = True

    if not any_curve:
        plt.close()
        return None

    if diag:
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)

    title = "ROC curves" if kind == "roc" else "Precision–Recall curves"
    plt.title(f"{title} — {split}")
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=True)
    out_path = outputs_dir / f"{title.replace(' ', '_')}_{split}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def _plot_confusion_matrix(cm, classes=("0", "1"), title="Confusion matrix", out_path=None):
    cm = np.array(cm, dtype=float)
    # normalizacija po stvarnoj klasi (po redovima)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots()
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # upis i procenta i apsolutnog broja
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{int(cm[i, j])}\n({norm[i, j]*100:.1f}%)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all_plots(outputs_dir="outputs"):
    """
    Generates:
      - outputs/ROC_curves_val.png, outputs/ROC_curves_test.png
      - outputs/Precision–Recall_curves_val.png, outputs/Precision–Recall_curves_test.png
      - outputs/<Model>/(val|test)_confusion_matrix.png (ako postoje confusion matrice u metrics json-ovima)
    """
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grupni ROC i PR
    _plot_group_generic(out_dir, "val", "roc")
    _plot_group_generic(out_dir, "test", "roc")
    _plot_group_generic(out_dir, "val", "pr")
    _plot_group_generic(out_dir, "test", "pr")

    # Per-model confusion matrices (ako postoje)
    for mdir in _find_model_dirs(out_dir):
        for split in ("val", "test"):
            metrics_path = mdir / f"{split}_metrics.json"
            m = _load_json(metrics_path)
            if not m:
                continue
            cm = m.get("confusion_matrix") or m.get("confusion") or m.get("cm")
            if cm:
                out_png = mdir / f"{split}_confusion_matrix.png"
                _plot_confusion_matrix(
                    cm,
                    title=f"{mdir.name} — {split} confusion matrix",
                    out_path=out_png,
                )
