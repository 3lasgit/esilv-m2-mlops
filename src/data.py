# ============================================================
# src/data.py
# Module de chargement et preprocessing — Heart Disease UCI
# ============================================================

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.2
URL_DATASET = (
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
CAT_TO_ENCODE = ["cp", "restecg", "slope", "thal"]
OUTLIER_COLS = ["trestbps", "chol", "oldpeak"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_raw() -> pd.DataFrame:
    """Charge le dataset brut depuis l'URL ou OpenML en fallback."""
    try:
        df = pd.read_csv(URL_DATASET)
        print("✅ Dataset chargé depuis URL")
    except Exception:
        from sklearn.datasets import fetch_openml

        data = fetch_openml("heart-statlog", version=1, as_frame=True)
        df = data.frame
        df.rename(columns={"class": "target"}, inplace=True)
        df["target"] = (df["target"] == "present").astype(int)
        print("✅ Dataset chargé depuis sklearn OpenML (fallback)")

    if "target" not in df.columns:
        df.columns = COLUMNS

    return df


def _cap_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Écrêtage des outliers par méthode IQR."""
    df = df.copy()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n > 0:
            print(f"  {col}: {n} outlier(s) écrêté(s) [{lower:.1f}, {upper:.1f}]")
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering : 3 variables d'interaction cliniques."""
    df = df.copy()
    # Réserve cardiaque normalisée par l'âge
    df["hr_age_ratio"] = df["thalach"] / df["age"]
    # Score de risque cardiovasculaire composé
    df["cardio_risk_score"] = (
        (df["trestbps"] / 120) * (df["chol"] / 200) * (df["age"] / 60)
    )
    # Stress cardiaque : dépression ST × angine d'effort
    df["exang_oldpeak"] = df["exang"] * df["oldpeak"]
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding des variables catégorielles multi-classes."""
    df = df.copy()
    for col in CAT_TO_ENCODE:
        df[col] = df[col].astype(int)
    return pd.get_dummies(df, columns=CAT_TO_ENCODE, drop_first=False)


# ------------------------------------------------------------------
# API publique
# ------------------------------------------------------------------

def load_and_preprocess():
    """
    Pipeline complet de chargement et preprocessing.

    Returns
    -------
    X_train_all : pd.DataFrame  — features train (non scalées, toutes colonnes)
    X_test_all  : pd.DataFrame  — features test  (non scalées, toutes colonnes)
    X_train_scaled : np.ndarray — features train scalées (StandardScaler)
    X_test_scaled  : np.ndarray — features test  scalées
    y_train : pd.Series
    y_test  : pd.Series
    feature_names : list[str]   — noms de colonnes de X_train_all
    scaler  : StandardScaler    — scaler fitté sur le train set
    """
    np.random.seed(RANDOM_STATE)

    # 1. Chargement
    df = _load_raw()

    # 2. Encodage de la cible en binaire (si non déjà fait)
    if df["target"].nunique() > 2:
        df["target"] = (df["target"] > 0).astype(int)

    # 3. Outliers
    print("=== Gestion des outliers (IQR) ===")
    df = _cap_outliers_iqr(df, OUTLIER_COLS)

    # 4. Feature engineering
    df = _add_features(df)

    # 5. One-hot encoding
    df = _encode(df)

    # 6. Split X / y
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 7. Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    X_test_scaled = scaler.transform(X_test_all)

    print(f"\n✅ Données prêtes — Train: {X_train_all.shape[0]} | Test: {X_test_all.shape[0]}")
    print(f"   Features : {X_train_all.shape[1]}")

    return (
        X_train_all,
        X_test_all,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        list(X_train_all.columns),
        scaler,
    )