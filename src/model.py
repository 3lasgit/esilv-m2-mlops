# ============================================================
# src/model.py
# Définition des modèles ML, tuning et architecture ANN
# ============================================================

import numpy as np
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CV_SPLITS = 5


# ------------------------------------------------------------------
# Modèles baseline
# ------------------------------------------------------------------

def get_models(best_k: int = 9) -> dict:
    """
    Retourne un dictionnaire de modèles sklearn prêts à l'emploi.

    LR, KNN, SVM utilisent un Pipeline avec StandardScaler intégré
    (ils reçoivent donc les features brutes non scalées).
    DT et RF travaillent directement sur les features brutes.

    Parameters
    ----------
    best_k : int — valeur de k pour KNN (déterminée lors de la sélection)

    Returns
    -------
    dict[str, estimator]
    """
    models = {
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=best_k)),
        ]),
        "DT": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }
    return models


def select_best_k(X_train_scaled: np.ndarray, y_train, cv=None) -> int:
    """
    Sélectionne le k optimal pour KNN par cross-validation (scoring F1).

    Parameters
    ----------
    X_train_scaled : features scalées
    y_train        : labels train
    cv             : StratifiedKFold (optionnel, créé si None)

    Returns
    -------
    best_k : int
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    from sklearn.model_selection import cross_val_score

    k_scores = []
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring="f1").mean()
        k_scores.append((k, score))

    best_k = max(k_scores, key=lambda x: x[1])[0]
    print(f"  KNN — meilleur k = {best_k}")
    return best_k


# ------------------------------------------------------------------
# Hyperparameter Tuning
# ------------------------------------------------------------------

def tune_rf(
    X_train, y_train, cv=None
) -> tuple[RandomForestClassifier, dict]:
    """
    GridSearchCV sur Random Forest.

    Returns
    -------
    best_estimator : RandomForestClassifier
    best_params    : dict
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"  RF Tuned — meilleurs params : {grid.best_params_}")
    print(f"  RF Tuned — CV AUC : {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def tune_svm(
    X_train_scaled: np.ndarray, y_train, cv=None
) -> tuple[SVC, dict]:
    """
    RandomizedSearchCV sur SVM (50 itérations).

    Returns
    -------
    best_estimator : SVC
    best_params    : dict
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    param_dist = {
        "C": loguniform(0.01, 100),
        "gamma": loguniform(1e-4, 1),
        "kernel": ["rbf", "poly"],
        "degree": [2, 3, 4],
    }

    rand = RandomizedSearchCV(
        SVC(probability=True, random_state=RANDOM_STATE),
        param_dist,
        n_iter=50,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    rand.fit(X_train_scaled, y_train)
    print(f"  SVM Tuned — meilleurs params : {rand.best_params_}")
    print(f"  SVM Tuned — CV AUC : {rand.best_score_:.4f}")
    return rand.best_estimator_, rand.best_params_


# ------------------------------------------------------------------
# ANN (Keras)
# ------------------------------------------------------------------

def build_ann(input_dim: int, dropout_rate: float = 0.3, l2_reg: float = 0.001):
    """
    Construit et compile un ANN binaire (Dense 128→64→32→1).

    Architecture
    ------------
    Input → Dense(128) + BN + ReLU + Dropout
          → Dense(64)  + BN + ReLU + Dropout
          → Dense(32)  + ReLU + Dropout/2
          → Dense(1, sigmoid)

    Parameters
    ----------
    input_dim    : nombre de features en entrée
    dropout_rate : taux de dropout (défaut 0.3)
    l2_reg       : coefficient de régularisation L2 (défaut 0.001)

    Returns
    -------
    model : keras.Sequential (compilé)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    tf.random.set_seed(RANDOM_STATE)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Bloc 1
        layers.Dense(128, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(dropout_rate),

        # Bloc 2
        layers.Dense(64, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(dropout_rate),

        # Bloc 3
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout_rate / 2),

        # Sortie
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model