# src/utils.py
from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42) -> None:
    """Setuje seed za NumPy (dovoljno jer radimo scratch modele)."""
    np.random.seed(seed)

# -----------------------------
# Učitavanje CSV-a (bez pandas)
# -----------------------------

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    standardize_stats: Dict[str, np.ndarray]

def _load_csv_numeric(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Učitava Pima CSV kao čiste numerike (bez pandas), vraća X, y, feature_names."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    # Validacija kolona
    missing_cols = [c for c in FEATURES + [TARGET] if c not in fieldnames]
    if missing_cols:
        raise ValueError(f"CSV nema očekivane kolone: {missing_cols}")

    X, y = [], []
    for r in rows:
        fv = [float(r[c]) for c in FEATURES]
        X.append(fv)
        y.append(int(float(r[TARGET])))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    # (opciono) warning ako broj instanci ne odgovara tipičnom Pima broju
    if X.shape[0] not in (768, 767):
        print(f"[WARN] Očekivano ~768 instanci, pronađeno {X.shape[0]}")

    return X, y, FEATURES.copy()

# -----------------------------
# Pretprocesiranje (bez sklearn)
# -----------------------------

ZERO_IS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

def _replace_zeros_with_nan(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    X = X.copy()
    for col_name in ZERO_IS_MISSING:
        j = feature_names.index(col_name)
        zeros = (X[:, j] == 0.0)
        X[zeros, j] = np.nan
    return X

def _median_impute(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Median imputacija po koloni (računa median na nansuprimovanim vrednostima)."""
    X = X.copy()
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    return X, medians

def _standardize_fit(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std = np.where(std == 0.0, 1.0, std)  # izbegni deljenje nulom
    return mean, std

def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std

# -----------------------------
# Stratifikovan split 70/15/15 (bez sklearn)
# -----------------------------

@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

def _stratified_indices(y: np.ndarray, train_p=0.70, val_p=0.15, test_p=0.15, seed=42) -> SplitIndices:
    assert abs(train_p + val_p + test_p - 1.0) < 1e-9, "Procenti moraju dati 1.0"
    set_seed(seed)

    idx = np.arange(len(y))
    class0 = idx[y == 0]
    class1 = idx[y == 1]
    np.random.shuffle(class0)
    np.random.shuffle(class1)

    def take_splits(cls_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(cls_idx)
        n_train = int(round(n * train_p))
        n_val = int(round(n * val_p))
        n_test = n - n_train - n_val
        train_i = cls_idx[:n_train]
        val_i = cls_idx[n_train:n_train + n_val]
        test_i = cls_idx[n_train + n_val:]
        return train_i, val_i, test_i

    t0, v0, s0 = take_splits(class0)
    t1, v1, s1 = take_splits(class1)

    train = np.concatenate([t0, t1])
    val = np.concatenate([v0, v1])
    test = np.concatenate([s0, s1])
    for arr in (train, val, test):
        np.random.shuffle(arr)
    return SplitIndices(train=train, val=val, test=test)

# -----------------------------
# Glavna funkcija za pripremu podataka
# -----------------------------

def prepare_dataset(
    csv_path: str,
    seed: int = 42,
    train_p: float = 0.70,
    val_p: float = 0.15,
    test_p: float = 0.15,
) -> Dataset:
    """
    - Učitava CSV (Pima),
    - nule u definisanim kolonama tretira kao NaN,
    - stratifikovan split 70/15/15,
    - imputacija medianom naučenom na train-u,
    - standardizacija po train statistici,
    - vraća Dataset + mean/std/median.
    """
    set_seed(seed)

    X_raw, y, feature_names = _load_csv_numeric(csv_path)
    X_nan = _replace_zeros_with_nan(X_raw, feature_names)

    split = _stratified_indices(y, train_p, val_p, test_p, seed)

    X_train_raw, y_train = X_nan[split.train], y[split.train]
    X_val_raw,   y_val   = X_nan[split.val],   y[split.val]
    X_test_raw,  y_test  = X_nan[split.test],  y[split.test]

    # Fit median na train-u, apply na ostale
    X_train_imp, med = _median_impute(X_train_raw)

    def apply_imputer(Xpart: np.ndarray, med: np.ndarray) -> np.ndarray:
        Xpart = Xpart.copy()
        inds = np.where(np.isnan(Xpart))
        Xpart[inds] = np.take(med, inds[1])
        return Xpart

    X_val_imp  = apply_imputer(X_val_raw,  med)
    X_test_imp = apply_imputer(X_test_raw, med)

    mean, std = _standardize_fit(X_train_imp)
    X_train = _standardize_apply(X_train_imp, mean, std)
    X_val   = _standardize_apply(X_val_imp,   mean, std)
    X_test  = _standardize_apply(X_test_imp,  mean, std)

    stats = {"mean": mean, "std": std, "median": med}

    return Dataset(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        feature_names=feature_names,
        standardize_stats=stats,
    )

# -----------------------------
# Pomoćne funkcije
# -----------------------------

def class_balance(y: np.ndarray) -> Dict[int, float]:
    counts = {0: int((y == 0).sum()), 1: int((y == 1).sum())}
    total = len(y)
    return {k: counts[k] / total for k in counts}
