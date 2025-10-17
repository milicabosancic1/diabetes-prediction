from __future__ import annotations
import os
import json
import importlib
from typing import Dict, Any, Optional
import numpy as np
from .utils import prepare_dataset, FEATURES
from .metrics import (
    scores_from_model, best_threshold_by_f1,
    confusion_at_threshold, prec_recall_f1
)


# izbor modela na osnovu outputs/summary.json
def _detect_best_family(outputs_dir: str) -> str:
    path = os.path.join(outputs_dir, "summary.json")
    if not os.path.isfile(path):
        return "KNN"  # pametan default ako nema rezultata
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if not summary:
        return "KNN"
    best = None
    for s in summary:
        if best is None:
            best = s
        else:
            # Primarno poređenje po test F1; ako je izjednačeno, koristi test AUC
            if (s.get("test_f1", 0) > best.get("test_f1", 0)) or (
                np.isclose(s.get("test_f1", 0), best.get("test_f1", 0)) and s.get("test_auc", 0) > best.get("test_auc", 0)
            ):
                best = s
    name = best["model"]
    # Mapiranje na familiju (nazivi foldera ≠ nazivi klasa)
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
    # Konstrukcija modela po familiji – vrednosti hiperparametara su razumni startovi
    if family == "NaiveBayes":
        mod = importlib.import_module("src.models.naive_bayes")
        return getattr(mod, "GaussianNB")()
    if family == "KNN":
        mod = importlib.import_module("src.models.knn")
        return getattr(mod, "KNN")(n_neighbors=7)
    if family == "MLP":
        mod = importlib.import_module("src.models.mlp")
        return getattr(mod, "MLP")(hidden_sizes=[32, 16], lr=0.02, l2=0.0005, max_epochs=400)
    # Default: Logistička regresija
    mod = importlib.import_module("src.models.logistic_regression")
    return getattr(mod, "LogisticRegression")(lr=0.1, l2=0.01, max_epochs=400)


def predict_single(
    user_input: Dict[str, float],
    csv_path: str = "data/diabetes.csv",
    outputs_dir: str = "outputs",
    family: Optional[str] = None
) -> Dict[str, Any]:

    # provera da li su popunjena sva ocekivana polja
    missing = [k for k in FEATURES if k not in user_input]
    if missing:
        raise ValueError(f"Nedostaju polja: {missing}")

    fam = family or _detect_best_family(outputs_dir)

    # pripremi ceo datase – koristi se i za izvlacenje train statistike
    data = prepare_dataset(csv_path, seed=42)

    # istreniraj izabrani model
    model = _instantiate_model(fam)
    if hasattr(model, "fit"):
        model.fit(data.X_train, data.y_train)
    elif hasattr(model, "train"):
        model.train(data.X_train, data.y_train)

    # odaberi prag po F1 na validaciji – taj prag koristimo i za pojedinacnu prognozu
    s_val = scores_from_model(model, data.X_val)
    thr, _ = best_threshold_by_f1(data.y_val, s_val)

    # pripremi x0 – mora kroz isti preprocessing kao i tren/test
    x0 = np.array([[float(user_input[k]) for k in FEATURES]], dtype=np.float64)

    # Zero -> NaN za kolone gde je 0 placeholder za missing
    for cname in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        j = FEATURES.index(cname)
        if x0[0, j] == 0.0:
            x0[0, j] = np.nan

    # imputacija medijanom iz train-a
    med = data.standardize_stats["median"]
    inds = np.where(np.isnan(x0))
    if inds[0].size > 0:
        x0[inds] = np.take(med, inds[1])

    # Standardizacija istim mean/std iz train-a
    mean = data.standardize_stats["mean"]
    std = data.standardize_stats["std"]
    x0 = (x0 - mean) / std

    # Konacan skor i binarna odluka po pragu
    score = float(scores_from_model(model, x0)[0])
    label = int(score >= thr)
    return {"family": fam, "threshold": float(thr), "score": score, "label": label}
