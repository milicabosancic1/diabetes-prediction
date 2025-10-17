from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List
import importlib
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, accuracy_score, confusion_matrix
)
from .metrics import (
    scores_from_model, best_threshold_by_f1,
    confusion_at_threshold, prec_recall_f1
)
from .utils import prepare_dataset, class_balance


@dataclass
class EvalResult:
    # Jednostavan paket rezultata da lako serijalizujem i dalje obradjujem
    name: str
    val: Dict[str, float]
    test: Dict[str, float]
    threshold: float
    roc_val: Dict[str, List[float]]
    roc_test: Dict[str, List[float]]
    pr_val: Dict[str, List[float]]
    pr_test: Dict[str, List[float]]


# evaluacija jednog modela, pomoćna funkcija da skratim eval all
def _evaluate_single(name: str, model: Any, data, out_dir: str) -> EvalResult:
    Xtr, ytr = data.X_train, data.y_train
    Xva, yva = data.X_val,   data.y_val
    Xte, yte = data.X_test,  data.y_test

    # Trening
    if hasattr(model, "fit"):
        model.fit(Xtr, ytr)
    else:
        raise AttributeError(f"Model '{name}' nema fit/train metod")

    # Skorovi na validaciji i izbor praga po F1
    s_val = scores_from_model(model, Xva)
    thr, best_val = best_threshold_by_f1(yva, s_val)  # prosledim rezultate modela i valid. rez da odredim najbolji prag klasif.

    # Fiksiram prag na testu
    s_test = scores_from_model(model, Xte)
    y_pred_test = (s_test >= thr).astype(int)

    # Test metrike
    p, r, f1, _ = precision_recall_fscore_support(yte, y_pred_test, average="binary", zero_division=0)
    acc = accuracy_score(yte, y_pred_test)
    tn, fp, fn, tp = confusion_matrix(yte, y_pred_test, labels=[0, 1]).ravel()

    # ROC/PR krive i povrsine
    fpr_v, tpr_v, _ = roc_curve(yva, s_val)  #false positive, true positive
    fpr_t, tpr_t, _ = roc_curve(yte, s_test)
    auc_v = auc(fpr_v, tpr_v)  # povrsina ispod krive
    auc_t = auc(fpr_t, tpr_t)

    prec_v, rec_v, _ = precision_recall_curve(yva, s_val)
    prec_t, rec_t, _ = precision_recall_curve(yte, s_test)
    ap_v = average_precision_score(yva, s_val)
    ap_t = average_precision_score(yte, s_test)

    # pakujem rezultate – val koristi najbolji prag, test koristi taj isti prag
    val_metrics = {**best_val, "roc_auc": float(auc_v), "pr_ap": float(ap_v)}
    test_metrics = {
        "precision": float(p), "recall": float(r), "f1": float(f1), "accuracy": float(acc),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "roc_auc": float(auc_t), "pr_ap": float(ap_t), "threshold_used": float(thr),
    }

    # cuvanje svega u fajlove da kasnije lakse generisem grafikone/izvestaje
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    with open(os.path.join(out_dir, name, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(out_dir, name, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    with open(os.path.join(out_dir, name, "val_roc.json"), "w", encoding="utf-8") as f:
        json.dump({"fpr": fpr_v.tolist(), "tpr": tpr_v.tolist(), "auc": float(auc_v)}, f, indent=2)
    with open(os.path.join(out_dir, name, "test_roc.json"), "w", encoding="utf-8") as f:
        json.dump({"fpr": fpr_t.tolist(), "tpr": tpr_t.tolist(), "auc": float(auc_t)}, f, indent=2)

    with open(os.path.join(out_dir, name, "val_pr.json"), "w", encoding="utf-8") as f:
        json.dump({"precision": prec_v.tolist(), "recall": rec_v.tolist(), "ap": float(ap_v)}, f, indent=2)
    with open(os.path.join(out_dir, name, "test_pr.json"), "w", encoding="utf-8") as f:
        json.dump({"precision": prec_t.tolist(), "recall": rec_t.tolist(), "ap": float(ap_t)}, f, indent=2)


    return EvalResult(
        name=name,
        val=val_metrics,
        test=test_metrics,
        threshold=thr,
        roc_val={"fpr": fpr_v.tolist(), "tpr": tpr_v.tolist()},
        roc_test={"fpr": fpr_t.tolist(), "tpr": tpr_t.tolist()},
        pr_val={"precision": prec_v.tolist(), "recall": rec_v.tolist()},
        pr_test={"precision": prec_t.tolist(), "recall": rec_t.tolist()},
    )


# glavni ulaz: 4 modela
def train_and_evaluate_all(csv_path: str = "data/diabetes.csv",
                           out_dir: str = "outputs",
                           seed: int = 42) -> List[EvalResult]:
    # Glavni pipeline: priprema podataka → treniranje → evaluacija → izvoz rezultata
    os.makedirs(out_dir, exist_ok=True)
    data = prepare_dataset(csv_path, seed=seed)

    # Info o skupu za log/izvestaj
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_train": int(len(data.y_train)),
            "n_val": int(len(data.y_val)),
            "n_test": int(len(data.y_test)),
            "class_balance_train": class_balance(data.y_train),
            "class_balance_val": class_balance(data.y_val),
            "class_balance_test": class_balance(data.y_test),
        }, f, indent=2)

    results: List[EvalResult] = []

    def try_make(module_name: str, class_names: List[str], kwargs: Dict[str, Any]) -> Any:
        """
        Pokusaj da importujes modul i konstruses klasu po imenu.
        """
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError:
            return None  # preskočim ako model nije implementiran
        for cname in class_names:
            if hasattr(mod, cname):
                cls = getattr(mod, cname)
                try:
                    return cls(**kwargs)  # probam sa zadatim argumentima
                except TypeError:
                    return cls()          # fallback bez argumenata
        return None

    # 1) Naive Bayes
    nb = try_make("src.models.naive_bayes", ["GaussianNB"], {})
    if nb is not None:
        results.append(_evaluate_single("NaiveBayes", nb, data, out_dir))

    # 2) KNN
    knn = try_make("src.models.knn", ["KNN"], {"n_neighbors": 5})
    if knn is not None:
        results.append(_evaluate_single("KNN", knn, data, out_dir))

    # 3) Logistic Regression (custom implementacija – lr, l2, epohe)
    lr = try_make("src.models.logistic_regression", ["LogisticRegression"], {"lr": 0.1, "l2": 0.0, "max_epochs": 200})
    if lr is not None:
        results.append(_evaluate_single("LogReg", lr, data, out_dir))

    # 4) MLP
    mlp = try_make("src.models.mlp", ["MLP"], {"hidden_sizes": [32, 16], "lr": 0.01, "l2": 0.0, "max_epochs": 300})
    if mlp is not None:
        results.append(_evaluate_single("MLP", mlp, data, out_dir))

    # sazetak po modelima
    summary = [{
        "model": r.name,
        "threshold": r.threshold,
        "val_f1": r.val.get("f1"),
        "val_auc": r.val.get("roc_auc"),
        "val_ap": r.val.get("pr_ap"),
        "test_f1": r.test.get("f1"),
        "test_auc": r.test.get("roc_auc"),
        "test_ap": r.test.get("pr_ap"),
    } for r in results]

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Izvoz i u summary CSV
    lines = ["model,threshold,val_f1,val_auc,val_ap,test_f1,test_auc,test_ap"]
    for s in summary:
        lines.append(",".join([
            str(s["model"]),
            f"{s['threshold']:.6f}" if s["threshold"] is not None else "",
            f"{s['val_f1']:.6f}" if s["val_f1"] is not None else "",
            f"{s['val_auc']:.6f}" if s["val_auc"] is not None else "",
            f"{s['val_ap']:.6f}" if s["val_ap"] is not None else "",
            f"{s['test_f1']:.6f}" if s["test_f1"] is not None else "",
            f"{s['test_auc']:.6f}" if s["test_auc"] is not None else "",
            f"{s['test_ap']:.6f}" if s["test_ap"] is not None else "",
        ]))
    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return results


if __name__ == "__main__":
    train_and_evaluate_all()
