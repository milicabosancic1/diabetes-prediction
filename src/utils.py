# src/utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42) -> None:
    """Setuje globalni random seed za NumPy i scikit-learn."""
    np.random.seed(seed)


# -----------------------------
# Kolone i ciljni atribut
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


# -----------------------------
# Dataset struktura
# -----------------------------
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


# -----------------------------
# Učitavanje i pretprocesiranje
# -----------------------------
def _load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV fajl nema očekivane kolone: {missing}")

    # Konverzija u numeričke vrednosti (greške → NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    if len(df) not in (767, 768):
        print(f"[WARN] Neočekivan broj instanci ({len(df)}), očekivano oko 768.")

    return df


def _replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Zamenjuje nule NaN vrednostima u kolonama gde 0 znači 'nedostaje'."""
    zero_is_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df = df.copy()
    for col in zero_is_missing:
        df.loc[df[col] == 0, col] = np.nan
    return df

def prepare_dataset(
    csv_path: str,
    seed: int = 42,
    train_p: float = 0.70,
    val_p: float = 0.15,
    test_p: float = 0.15,
) -> Dataset:
    set_seed(seed)

    # ucitavanje i ciscenje
    df = _load_csv(csv_path)
    df = _replace_zeros_with_nan(df)

    X = df[FEATURES].values
    y = df[TARGET].values.astype(int)

    # split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_p), random_state=seed, stratify=y
    )
    rel_val = val_p / (val_p + test_p)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - rel_val), random_state=seed, stratify=y_temp
    )

    # median imputacija (fit samo na train)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # standardizacija po train statistici
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_imp)
    X_val_std = scaler.transform(X_val_imp)
    X_test_std = scaler.transform(X_test_imp)

    stats = {
        "mean": scaler.mean_,
        "std": scaler.scale_,
        "median": imputer.statistics_,
    }

    return Dataset(
        X_train=X_train_std,
        y_train=y_train,
        X_val=X_val_std,
        y_val=y_val,
        X_test=X_test_std,
        y_test=y_test,
        feature_names=FEATURES.copy(),
        standardize_stats=stats,
    )


# -----------------------------
# Pomoćne funkcije

def class_balance(y: np.ndarray) -> Dict[int, float]:
    """Vraća procentualnu zastupljenost klasa u skupu."""
    counts = {0: int((y == 0).sum()), 1: int((y == 1).sum())}
    total = len(y)
    return {k: counts[k] / total for k in counts}