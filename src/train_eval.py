from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import importlib
from .utils import prepare_dataset, class_balance

def _confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Tuple[int,int,int,int]:
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn

def _prec_recall_f1(tp:int, tn:int, fp:int, fn:int) -> Tuple[float,float,float,float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0.0
    return precision, recall, f1, acc

def _roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, List[float]]:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    P = float((y_true == 1).sum())
    N = float((y_true == 0).sum())
    tps = 0.0
    fps = 0.0
    tpr = [0.0]
    fpr = [0.0]
    last_score = np.inf
    for i in range(len(y_true)):
        if y_score[i] != last_score:
            tpr.append(tps / P if P > 0 else 0.0)
            fpr.append(fps / N if N > 0 else 0.0)
            last_score = y_score[i]
        if y_true[i] == 1:
            tps += 1.0
        else:
            fps += 1.0
    tpr.append(1.0)
    fpr.append(1.0)
    tpr = tpr[1:]
    fpr = fpr[1:]
    return {"tpr": tpr, "fpr": fpr}

def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, List[float]]:
    # sort po skorovima opadajuće
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    tp = 0.0
    fp = 0.0
    P = float((y_true == 1).sum())
    precision = []
    recall = []
    last_score = np.inf
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1.0
        else:
            fp += 1.0
        # izračunaj na svakoj tački (ili samo kad se promeni skor); biramo svaku tačku (gušće)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / P if P > 0 else 0.0
        precision.append(prec)
        recall.append(rec)
        last_score = y_score[i]
    # dodaj (0,1) stilizovano? — nije potrebno; koristićemo direktno integral
    return {"precision": precision, "recall": recall}

def _auc(xs: List[float], ys: List[float]) -> float:
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i-1]) * (ys[i] + ys[i-1]) * 0.5
    return float(area)

def _average_precision(prec: List[float], rec: List[float]) -> float:
    # standardna aproksimacija površine ispod PR krive (trapezoid nad recall-om)
    # prvo sort po recall-u rastuće
    r = np.array(rec, dtype=float)
    p = np.array(prec, dtype=float)
    order = np.argsort(r)
    r = r[order]
    p = p[order]
    ap = 0.0
    for i in range(1, len(r)):
        ap += (r[i] - r[i-1]) * (p[i] + p[i-1]) * 0.5
    return float(ap)

def _best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    uniq = np.unique(y_score)
    cands = np.unique(np.concatenate([uniq, np.array([0.5])]))
    best = {"f1": -1.0}
    best_thr = 0.5
    for thr in cands:
        tp, tn, fp, fn = _confusion_at_threshold(y_true, y_score, thr)
        precision, recall, f1, acc = _prec_recall_f1(tp, tn, fp, fn)
        if f1 > best["f1"] or (np.isclose(f1, best["f1"]) and recall > best.get("recall", 0)):
            best = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": acc,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }
            best_thr = float(thr)
    return best_thr, best

def _scores_from_model(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return np.asarray(proba).ravel()
    for name in ("decision_function", "predict_scores", "forward", "predict"):
        if hasattr(model, name):
            out = getattr(model, name)(X)
            return out.ravel() if isinstance(out, np.ndarray) else np.asarray(out).ravel()
    raise AttributeError("Model nema metod za izračunavanje skorova (predict_proba/decision_function/forward/predict)")

def _maybe_instantiate(mod, class_candidates: List[str], kwargs: Dict[str, Any]) -> Any:
    for cname in class_candidates:
        if hasattr(mod, cname):
            cls = getattr(mod, cname)
            try:
                return cls(**kwargs)
            except TypeError:
                return cls()
    raise AttributeError(f"Ni jedna od klasa {class_candidates} nije pronađena u {mod.__name__}")

@dataclass
class EvalResult:
    name: str
    val: Dict[str, Any]
    test: Dict[str, Any]
    threshold: float
    roc_val: Dict[str, List[float]]
    roc_test: Dict[str, List[float]]
    pr_val: Dict[str, List[float]]
    pr_test: Dict[str, List[float]]

# ============================
# Core pipeline
# ============================

def _evaluate_single(name: str, model: Any, data, out_dir: str) -> EvalResult:
    Xtr, ytr = data.X_train, data.y_train
    Xva, yva = data.X_val,   data.y_val
    Xte, yte = data.X_test,  data.y_test

    # Fit / Train
    if hasattr(model, "fit"):
        model.fit(Xtr, ytr)
    elif hasattr(model, "train"):
        model.train(Xtr, ytr)
    else:
        raise AttributeError(f"Model '{name}' nema fit/train metod")

    # Scores
    s_val = _scores_from_model(model, Xva)
    thr, best_val = _best_threshold_by_f1(yva, s_val)

    s_test = _scores_from_model(model, Xte)
    tp, tn, fp, fn = _confusion_at_threshold(yte, s_test, thr)
    precision, recall, f1, acc = _prec_recall_f1(tp, tn, fp, fn)

    # ROC & AUC
    roc_v = _roc_curve(yva, s_val)
    roc_t = _roc_curve(yte, s_test)
    auc_v = _auc(roc_v["fpr"], roc_v["tpr"])
    auc_t = _auc(roc_t["fpr"], roc_t["tpr"])

    # PR & AP
    pr_v = _precision_recall_curve(yva, s_val)
    pr_t = _precision_recall_curve(yte, s_test)
    ap_v = _average_precision(pr_v["precision"], pr_v["recall"])
    ap_t = _average_precision(pr_t["precision"], pr_t["recall"])

    # Pack
    val_metrics = dict(best_val)
    val_metrics.update({"roc_auc": auc_v, "pr_ap": ap_v})
    test_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "roc_auc": auc_t,
        "pr_ap": ap_t,
        "threshold_used": thr,
    }

    # Save JSON
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    with open(os.path.join(out_dir, name, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(out_dir, name, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(out_dir, name, "val_roc.json"), "w", encoding="utf-8") as f:
        json.dump({**roc_v, "auc": auc_v}, f)
    with open(os.path.join(out_dir, name, "test_roc.json"), "w", encoding="utf-8") as f:
        json.dump({**roc_t, "auc": auc_t}, f)
    with open(os.path.join(out_dir, name, "val_pr.json"), "w", encoding="utf-8") as f:
        json.dump({**pr_v, "ap": ap_v}, f)
    with open(os.path.join(out_dir, name, "test_pr.json"), "w", encoding="utf-8") as f:
        json.dump({**pr_t, "ap": ap_t}, f)
    with open(os.path.join(out_dir, name, "threshold.txt"), "w", encoding="utf-8") as f:
        f.write(str(thr))

    return EvalResult(
        name=name, val=val_metrics, test=test_metrics, threshold=thr,
        roc_val=roc_v, roc_test=roc_t, pr_val=pr_v, pr_test=pr_t
    )

def train_and_evaluate_all(csv_path: str = "data/diabetes.csv", out_dir: str = "outputs", seed: int = 42) -> List[EvalResult]:
    os.makedirs(out_dir, exist_ok=True)
    data = prepare_dataset(csv_path, seed=seed)

    # Log basic info
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

    def safe_import(module_name: str):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            return None

    # Naive Bayes
    nb_mod = safe_import("src.models.naive_bayes")
    if nb_mod is not None:
        model = _maybe_instantiate(nb_mod, ["GaussianNB"], kwargs={})
        results.append(_evaluate_single("NaiveBayes", model, data, out_dir))

    # KNN
    knn_mod = safe_import("src.models.knn")
    if knn_mod is not None:
        model = _maybe_instantiate(knn_mod, ["KNN"], kwargs={"n_neighbors": 5})
        results.append(_evaluate_single("KNN", model, data, out_dir))

    # Logistic Regression
    lr_mod = safe_import("src.models.logistic_regression")
    if lr_mod is not None:
        model = _maybe_instantiate(lr_mod, ["LogisticRegression"], kwargs={"lr": 0.1, "l2": 0.0, "max_epochs": 200})
        results.append(_evaluate_single("LogReg", model, data, out_dir))

    # MLP
    mlp_mod = safe_import("src.models.mlp")
    if mlp_mod is not None:
        model = _maybe_instantiate(mlp_mod, ["MLP"], kwargs={"hidden_sizes": [32, 16], "lr": 0.01, "l2": 0.0, "max_epochs": 300})
        results.append(_evaluate_single("MLP", model, data, out_dir))

    # Summary files (JSON + CSV)
    summary = []
    for r in results:
        summary.append({
            "model": r.name,
            "threshold": r.threshold,
            "val_f1": r.val.get("f1"),
            "val_auc": r.val.get("roc_auc"),
            "val_ap": r.val.get("pr_ap"),
            "test_f1": r.test.get("f1"),
            "test_auc": r.test.get("roc_auc"),
            "test_ap": r.test.get("pr_ap"),
        })
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV friendly export
    csv_lines = ["model,threshold,val_f1,val_auc,val_ap,test_f1,test_auc,test_ap"]
    for s in summary:
        csv_lines.append(",".join([
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
        f.write("\n".join(csv_lines))

    return results

if __name__ == "__main__":
    train_and_evaluate_all()
