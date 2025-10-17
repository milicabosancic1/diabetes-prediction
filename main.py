import os
import sys
import json

import numpy as np

from src.plotting import generate_all_plots
from src.train_eval import train_and_evaluate_all

BASE_DIR = os.path.dirname(__file__)


def resolve_csv_path() -> str:
    data_csv = os.path.join(BASE_DIR, "data", "diabetes.csv")
    if os.path.isfile(data_csv):
        return data_csv

    # fallback na sample ako postoji
    sample_csv = os.path.join(BASE_DIR, "data", "diabetes_sample.csv")
    if os.path.isfile(sample_csv):
        print("⚠️  Nema data/diabetes.csv – koristim data/diabetes_sample.csv")
        return sample_csv

    raise FileNotFoundError(
        "Nisam našao ni data/diabetes.csv ni data/diabetes_sample.csv. "
        "Dodaj CSV u folder data/ pa pokreni ponovo."
    )


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _print_box(title: str):
    line = "═" * (len(title) + 2)
    print(f"\n╔{line}╗")
    print(f"║ {title} ║")
    print(f"╚{line}╝")


def _fmt(x, digits=3):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _pretty_print_summary(out_dir: str):
    info = _read_json(os.path.join(out_dir, "dataset_info.json")) or {}
    summary = _read_json(os.path.join(out_dir, "summary.json")) or []

    # Dataset info
    _print_box("Dataset info")
    print(f" train: {info.get('n_train', '?')}  "
          f"val: {info.get('n_val', '?')}  "
          f"test: {info.get('n_test', '?')}")
    cb_tr = info.get("class_balance_train", {})
    cb_va = info.get("class_balance_val", {})
    cb_te = info.get("class_balance_test", {})
    def pct(d, k): return f"{d.get(k, 0)*100:.1f}%"
    print(f" balance train  -> 0:{pct(cb_tr, 0)}  1:{pct(cb_tr, 1)}")
    print(f" balance val    -> 0:{pct(cb_va, 0)}  1:{pct(cb_va, 1)}")
    print(f" balance test   -> 0:{pct(cb_te, 0)}  1:{pct(cb_te, 1)}")

    # Models table
    if not summary:
        return
    _print_box("Rezime modela (val→odabir thr, test→izveštaj)")

    headers = ["Model", "Thr", "Val F1", "Val AUC", "Val AP", "Test F1", "Test AUC", "Test AP"]
    widths = [max(len(h), 6) for h in headers]
    rows = []
    for s in summary:
        row = [
            s["model"],
            _fmt(s["threshold"], 4),
            _fmt(s["val_f1"]),
            _fmt(s["val_auc"]),
            _fmt(s["val_ap"]),
            _fmt(s["test_f1"]),
            _fmt(s["test_auc"]),
            _fmt(s["test_ap"]),
        ]
        rows.append(row)
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(ch="-"):
        print(" " + "+".join(ch * (w+2) for w in widths))

    # print header
    line("=")
    print(" " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    line("=")

    # print rows
    for r in rows:
        print(" " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
    line("=")

    best = None
    for s in summary:
        if best is None:
            best = s
        else:
            if (s["test_f1"] or 0) > (best["test_f1"] or 0) or (
                    np.isclose(s["test_f1"], best["test_f1"]) and (s["test_auc"] or 0) > (best["test_auc"] or 0)
            ):
                best = s

    if best:
        print("\n🏆 Najbolji model:",
              f"{best['model']}  (test F1={_fmt(best['test_f1'])}, "
              f"AUC={_fmt(best['test_auc'])}, AP={_fmt(best['test_ap'])})")

    # helpful pointers
    print("\n📁 Grafike su u 'outputs/':")
    print("   • ROC_curves_val.png, ROC_curves_test.png")
    print("   • Precision–Recall_curves_val.png, Precision–Recall_curves_test.png")
    print("   • <Model>/(val|test)_confusion_matrix.png")
    print("🧾 Sažetak: 'outputs/summary.json' i 'outputs/summary.csv'")


def main() -> int:
    try:
        csv_path = resolve_csv_path()
        out_dir = os.path.join(BASE_DIR, "outputs")
        train_and_evaluate_all(csv_path=csv_path, out_dir=out_dir, seed=42)
        generate_all_plots(outputs_dir=out_dir)
        _pretty_print_summary(out_dir)
        print("\n✅ Gotovo.")
        return 0
    except Exception as e:
        print(f"❌ Greška: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
