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
    # Sort by score descending
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

    # remove the first duplicate (0,0)
    tpr = tpr[1:]
    fpr = fpr[1:]
    return {"tpr": tpr, "fpr": fpr}


def _auc(fpr: List[float], tpr: List[float]) -> float:
    # Trapezoidal rule (assumes fpr increasing)
    area = 0.0
    for i in range(1, len(fpr)):
        area += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) * 0.5
    return float(area)


def _best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    # Evaluate on all unique scores + {0.5} fall-back
    uniq = np.unique(y_score)
    cands = np.unique(np.concatenate([uniq, np.array([0.5])]))
    best = {"f1": -1.0}
    best_thr = 0.5
    for thr in cands:
        tp, tn, fp, fn = _confusion_at_threshold(y_true, y_score, thr)
        precision, recall, f1, acc = _prec_recall_f1(tp, tn, fp, fn)
        if f1 > best["f1"] or (np.isclose(f1, best["f1"]) and recall > best.get("recall",0)):
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
    # Try common APIs: predict_proba -> decision_function -> predict_scores -> forward -> predict (as scores)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        # if binary stored as single proba of positive class
        return proba.ravel()
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


# ============================
# Core pipeline
# ============================

def _evaluate_single(name: str, model: Any, data, out_dir: str) -> EvalResult:
    Xtr, ytr = data.X_train, data.y_train
    Xva, yva = data.X_val,   data.y_val
    Xte, yte = data.X_test,  data.y_test

    # Fit
    if hasattr(model, "fit"):
        model.fit(Xtr, ytr)
    else:
        # Minimal API for scratch: train(X, y)
        if hasattr(model, "train"):
            model.train(Xtr, ytr)
        else:
            raise AttributeError(f"Model '{name}' nema fit/train metod")

    # Scores
    s_val = _scores_from_model(model, Xva)
    thr, best_val = _best_threshold_by_f1(yva, s_val)

    s_test = _scores_from_model(model, Xte)
    tp, tn, fp, fn = _confusion_at_threshold(yte, s_test, thr)
    precision, recall, f1, acc = _prec_recall_f1(tp, tn, fp, fn)

    roc_v = _roc_curve(yva, s_val)
    roc_t = _roc_curve(yte, s_test)

    auc_v = _auc(roc_v["fpr"], roc_v["tpr"])
    auc_t = _auc(roc_t["fpr"], roc_t["tpr"])

    # Pack
    val_metrics = dict(best_val)
    val_metrics.update({"roc_auc": auc_v})

    test_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "roc_auc": auc_t,
    }

    # Save JSON
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    with open(os.path.join(out_dir, name, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(out_dir, name, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(out_dir, name, "val_roc.json"), "w", encoding="utf-8") as f:
        json.dump(roc_v, f)
    with open(os.path.join(out_dir, name, "test_roc.json"), "w", encoding="utf-8") as f:
        json.dump(roc_t, f)
    with open(os.path.join(out_dir, name, "threshold.txt"), "w", encoding="utf-8") as f:
        f.write(str(thr))

    return EvalResult(name=name, val=val_metrics, test=test_metrics, threshold=thr, roc_val=roc_v, roc_test=roc_t)


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
        model = _maybe_instantiate(mlp_mod, ["MLP"], kwargs={"hidden_sizes": [32,16], "lr": 0.01, "l2": 0.0, "max_epochs": 300})
        results.append(_evaluate_single("MLP", model, data, out_dir))

    # Summary file
    summary = []
    for r in results:
        summary.append({
            "model": r.name,
            "threshold": r.threshold,
            "val_f1": r.val.get("f1", None),
            "val_auc": r.val.get("roc_auc", None),
            "test_f1": r.test.get("f1", None),
            "test_auc": r.test.get("roc_auc", None),
        })
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == "__main__":
    train_and_evaluate_all()
