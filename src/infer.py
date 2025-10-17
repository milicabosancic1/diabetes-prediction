from __future__ import annotations
import os
import json
import importlib
from typing import Dict, Any, Optional
import numpy as np

from .utils import prepare_dataset, FEATURES


# pomocne metrike
def _confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def _prec_recall_f1(tp: int, tn: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0.0
    return precision, recall, f1, acc


def _best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray):
    uniq = np.unique(y_score)
    cands = np.unique(np.concatenate([uniq, np.array([0.5])]))
    best = {"f1": -1.0}
    best_thr = 0.5
    for thr in cands:
        tp, tn, fp, fn = _confusion_at_threshold(y_true, y_score, thr)
        precision, recall, f1, acc = _prec_recall_f1(tp, tn, fp, fn)
        if f1 > best["f1"] or (np.isclose(f1, best["f1"]) and recall > best.get("recall", 0)):
            best = {
                "precision": precision, "recall": recall, "f1": f1, "accuracy": acc,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }
            best_thr = float(thr)
    return best_thr, best


def _scores_from_model(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return np.asarray(proba).ravel()
    for name in ("decision_function", "predict_scores", "forward", "predict"):
        if hasattr(model, name):
            out = getattr(model, name)(X)
            return out.ravel() if isinstance(out, np.ndarray) else np.asarray(out).ravel()
    raise AttributeError("Model nema metod za skor (predict_proba/decision_function/forward/predict)")


# izbor modela na osnovu outputs/summary.json
def _detect_best_family(outputs_dir: str) -> str:
    path = os.path.join(outputs_dir, "summary.json")
    if not os.path.isfile(path):
        return "KNN"
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if not summary:
        return "KNN"
    best = None
    for s in summary:
        if best is None:
            best = s
        else:
            if (s.get("test_f1", 0) > best.get("test_f1", 0)) or (
                np.isclose(s.get("test_f1", 0), best.get("test_f1", 0)) and s.get("test_auc", 0) > best.get("test_auc", 0)
            ):
                best = s
    name = best["model"]
    if name.startswith("KNN"):
        return "KNN"
    if name.startswith("LogReg"):
        return "LogReg"
    if name.startswith("MLP"):
        return "MLP"
    if name.startswith("NaiveBayes"):
        return "NaiveBayes"
    return "LogReg"


def _instantiate_model(family: str):
    if family == "NaiveBayes":
        mod = importlib.import_module("src.models.naive_bayes")
        return getattr(mod, "GaussianNB")()
    if family == "KNN":
        mod = importlib.import_module("src.models.knn")
        return getattr(mod, "KNN")(n_neighbors=7)
    if family == "MLP":
        mod = importlib.import_module("src.models.mlp")
        return getattr(mod, "MLP")(hidden_sizes=[32, 16], lr=0.02, l2=0.0005, max_epochs=400)
    mod = importlib.import_module("src.models.logistic_regression")
    return getattr(mod, "LogisticRegression")(lr=0.1, l2=0.01, max_epochs=400)


def predict_single(
    user_input: Dict[str, float],
    csv_path: str = "data/diabetes.csv",
    outputs_dir: str = "outputs",
    family: Optional[str] = None
) -> Dict[str, Any]:

    # validacija ključeva
    missing = [k for k in FEATURES if k not in user_input]
    if missing:
        raise ValueError(f"Nedostaju polja: {missing}")

    fam = family or _detect_best_family(outputs_dir)

    # pripremi dataset + stats
    data = prepare_dataset(csv_path, seed=42)

    # istreniraj model
    model = _instantiate_model(fam)
    if hasattr(model, "fit"):
        model.fit(data.X_train, data.y_train)
    elif hasattr(model, "train"):
        model.train(data.X_train, data.y_train)

    # prag po F1 na validaciji
    s_val = _scores_from_model(model, data.X_val)
    thr, _ = _best_threshold_by_f1(data.y_val, s_val)

    # pripremi x0 kroz isti PREPROCESSING kao i dataset
    x0 = np.array([[float(user_input[k]) for k in FEATURES]], dtype=np.float64)

    # zero->NaN (kao u utils.ZERO_IS_MISSING)
    for cname in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        j = FEATURES.index(cname)
        if x0[0, j] == 0.0:
            x0[0, j] = np.nan

    # median input sa train-medijanama
    med = data.standardize_stats["median"]
    inds = np.where(np.isnan(x0))
    if inds[0].size > 0:
        x0[inds] = np.take(med, inds[1])

    # standardizacija istim mean/std sa train-a
    mean = data.standardize_stats["mean"]
    std = data.standardize_stats["std"]
    x0 = (x0 - mean) / std

    # skor i odluka
    score = float(_scores_from_model(model, x0)[0])
    label = int(score >= thr)
    return {"family": fam, "threshold": float(thr), "score": score, "label": label}
