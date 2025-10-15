import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _find_model_dirs(outputs_dir: Path):
    return [p for p in outputs_dir.iterdir() if p.is_dir()]


def _plot_roc_group(outputs_dir: Path, split: str = "val"):
    plt.figure()
    any_curve = False
    for mdir in _find_model_dirs(outputs_dir):
        roc_path = mdir / f"{split}_roc.json"
        roc = _load_json(roc_path)
        if not roc:
            continue
        fpr = np.array(roc.get("fpr", []), dtype=float)
        tpr = np.array(roc.get("tpr", []), dtype=float)
        auc = roc.get("auc", None)
        label = mdir.name if auc is None else f"{mdir.name} (AUC={auc:.3f})"
        if fpr.size and tpr.size:
            plt.plot(fpr, tpr, label=label)
            any_curve = True
    if not any_curve:
        return None
    plt.plot([0,1],[0,1], linestyle="--")
    plt.title(f"ROC curves — {split}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    out_path = outputs_dir / f"ROC_{split}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _plot_confusion_matrix(cm, classes=("0","1"), title="Confusion matrix", out_path=None):

    cm = np.array(cm, dtype=float)
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_plots(outputs_dir="outputs"):
    """
    Generates:
      - outputs/ROC_val.png
      - outputs/ROC_test.png
      - outputs/<Model>/(val|test)_confusion_matrix.png (if confusion matrices exist in metrics jsons)
    Safe to call even if JSONs are missing – it will just skip.
    """
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC comparison charts
    _plot_roc_group(out_dir, "val")
    _plot_roc_group(out_dir, "test")

    # Per-model confusion matrices if available in metrics JSONs
    for mdir in _find_model_dirs(out_dir):
        for split in ("val", "test"):
            metrics_path = mdir / f"{split}_metrics.json"
            m = _load_json(metrics_path)
            if not m:
                continue
            cm = m.get("confusion_matrix") or m.get("confusion") or m.get("cm")
            if cm:
                out_png = mdir / f"{split}_confusion_matrix.png"
                _plot_confusion_matrix(cm, title=f"{mdir.name} — {split} confusion matrix",
                                       out_path=out_png)
