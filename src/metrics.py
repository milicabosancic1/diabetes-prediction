from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, accuracy_score, confusion_matrix
)


def scores_from_model(model, X) -> np.ndarray:
    """
    Vraća "score" za pozit. klasu (float u [0,1]).
    Preferira predict_proba[:,1]; fallback su decision_function/predict/...
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)  #saljem x_val
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] == 2:  # ako je binarna klasif. uzimamo drugu kolonu
            return proba[:, 1]
        return proba.ravel()  # ako nije 2-dim spljosti se

    for name in ("decision_function", "predict_scores", "forward", "predict"):
        if hasattr(model, name):
            out = getattr(model, name)(X)
            return np.asarray(out).ravel()

    raise AttributeError("Model ne izbacuje skor/probabilnosti kompatibilne sa evaluacijom.")


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def prec_recall_f1(tp: int, tn: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # TP / (TP + FP)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0       # TP / (TP + FN)
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0.0
    return precision, recall, f1, acc


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Kandidati: jedinstveni skorovi + "razumni" pragovi
    uniq = np.unique(y_score)  # svi jedins. scorovi kao pot. prag
    candidates = np.unique(np.concatenate([uniq, [0.25, 0.5, 0.75]]))

    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "acc": 0.0, "thr": 0.5}
    for thr in candidates:
        tp, tn, fp, fn = confusion_at_threshold(y_true, y_score, thr)
        p, r, f1, acc = prec_recall_f1(tp, tn, fp, fn)
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": p, "recall": r, "acc": acc, "thr": float(thr)}
    return best["thr"], best
