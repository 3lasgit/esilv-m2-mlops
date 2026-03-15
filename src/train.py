# ============================================================
# src/train.py
# Script d'entraînement principal — Heart Disease UCI
# ============================================================
# Usage :
#   uv run python src/train.py
#
# Sorties :
#   models/rf_tuned.joblib
#   models/svm_tuned.joblib
#   models/ann.keras
# ============================================================

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data import load_and_preprocess
from model import build_ann, get_models, select_best_k, tune_rf, tune_svm

RANDOM_STATE = 42
CV_SPLITS = 5
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def evaluate_model(name: str, model, X_tr, X_te, y_tr, y_te, cv) -> dict:
    """
    Cross-validation + évaluation finale sur le test set.

    Parameters
    ----------
    name  : label du modèle
    model : estimateur sklearn (Pipeline ou estimateur direct)
    X_tr, X_te : features train / test (array ou DataFrame)
    y_tr, y_te : labels train / test

    Returns
    -------
    dict contenant toutes les métriques + l'objet modèle fitté
    """
    cv_acc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1")
    cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "CV_Acc": cv_acc.mean(),
        "CV_Acc_std": cv_acc.std(),
        "CV_F1": cv_f1.mean(),
        "CV_AUC": cv_auc.mean(),
        "Test_Acc": accuracy_score(y_te, y_pred),
        "Test_F1": f1_score(y_te, y_pred),
        "Test_AUC": roc_auc_score(y_te, y_prob) if y_prob is not None else None,
        "model_obj": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    print(
        f"[{name:<22}] "
        f"Test Acc: {metrics['Test_Acc']:.3f} | "
        f"F1: {metrics['Test_F1']:.3f} | "
        f"AUC: {metrics['Test_AUC']:.3f} | "
        f"CV_Acc: {cv_acc.mean():.3f}±{cv_acc.std():.3f}"
    )
    return metrics


def print_summary(results: dict, ann_metrics: dict) -> None:
    """Affiche le tableau récapitulatif final trié par AUC."""
    all_results = {**results, "ANN (Deep Learning)": ann_metrics}

    rows = [
        {
            "Modèle": v.get("Model", k),
            "Accuracy": round(v["Test_Acc"], 4),
            "F1-Score": round(v["Test_F1"], 4),
            "AUC-ROC": round(v["Test_AUC"], 4),
        }
        for k, v in all_results.items()
        if v.get("Test_AUC") is not None
    ]

    df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
    sep = "=" * 60
    print(f"\n{sep}")
    print("     TABLEAU RÉCAPITULATIF FINAL — TEST SET")
    print(sep)
    print(df.to_string(index=False))
    print(sep)


# ------------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------------


def main() -> None:
    np.random.seed(RANDOM_STATE)

    # ----------------------------------------------------------
    # 1. Données
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ÉTAPE 1 — Chargement & preprocessing")
    print("=" * 60)
    (
        X_train_all,
        X_test_all,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_names,
        scaler,
    ) = load_and_preprocess()

    cv = _make_cv()

    # ----------------------------------------------------------
    # 2. Sélection du k optimal pour KNN
    # ----------------------------------------------------------
    print("\n=== Sélection du k optimal (KNN) ===")
    best_k = select_best_k(X_train_scaled, y_train, cv)

    # ----------------------------------------------------------
    # 3. Modèles baseline
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ÉTAPE 2 — Entraînement des modèles baseline")
    print("=" * 60)

    models = get_models(best_k=best_k)
    results = {}

    # LR et SVM (Pipeline avec scaler interne) → features brutes
    for key in ("LR", "KNN", "SVM"):
        results[key] = evaluate_model(
            models[key].steps[-1][1].__class__.__name__ if key != "KNN" else "KNN",
            models[key],
            X_train_all,
            X_test_all,
            y_train,
            y_test,
            cv,
        )
        results[key]["Model"] = {
            "LR": "Logistic Regression",
            "KNN": "KNN",
            "SVM": "SVM (RBF)",
        }[key]

    # DT et RF → features brutes (pas besoin de scaler)
    for key in ("DT", "RF"):
        results[key] = evaluate_model(
            {"DT": "Decision Tree", "RF": "Random Forest"}[key],
            models[key],
            X_train_all.values,
            X_test_all.values,
            y_train,
            y_test,
            cv,
        )

    # ----------------------------------------------------------
    # 4. Hyperparameter tuning
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ÉTAPE 3 — Hyperparameter Tuning")
    print("=" * 60)

    print("\n--- GridSearchCV : Random Forest ---")
    rf_tuned, _ = tune_rf(X_train_all.values, y_train, cv)
    y_pred_rf_t = rf_tuned.predict(X_test_all.values)
    y_prob_rf_t = rf_tuned.predict_proba(X_test_all.values)[:, 1]
    results["RF_tuned"] = {
        "Model": "RF (Tuned)",
        "Test_Acc": accuracy_score(y_test, y_pred_rf_t),
        "Test_F1": f1_score(y_test, y_pred_rf_t),
        "Test_AUC": roc_auc_score(y_test, y_prob_rf_t),
        "y_pred": y_pred_rf_t,
        "y_prob": y_prob_rf_t,
        "model_obj": rf_tuned,
    }

    print("\n--- RandomizedSearchCV : SVM ---")
    svm_tuned, _ = tune_svm(X_train_scaled, y_train, cv)
    y_pred_svm_t = svm_tuned.predict(X_test_scaled)
    y_prob_svm_t = svm_tuned.predict_proba(X_test_scaled)[:, 1]
    results["SVM_tuned"] = {
        "Model": "SVM (Tuned)",
        "Test_Acc": accuracy_score(y_test, y_pred_svm_t),
        "Test_F1": f1_score(y_test, y_pred_svm_t),
        "Test_AUC": roc_auc_score(y_test, y_prob_svm_t),
        "y_pred": y_pred_svm_t,
        "y_prob": y_prob_svm_t,
        "model_obj": svm_tuned,
    }

    # ----------------------------------------------------------
    # 5. ANN
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ÉTAPE 4 — Entraînement ANN")
    print("=" * 60)

    ann = build_ann(input_dim=X_train_scaled.shape[1])
    ann.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=20,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    history = ann.fit(
        X_train_scaled,
        y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"\n✅ ANN — entraînement terminé (époque {len(history.history['loss'])})")

    y_prob_ann = ann.predict(X_test_scaled, verbose=0).ravel()
    y_pred_ann = (y_prob_ann >= 0.5).astype(int)

    ann_metrics = {
        "Model": "ANN (Deep Learning)",
        "Test_Acc": accuracy_score(y_test, y_pred_ann),
        "Test_F1": f1_score(y_test, y_pred_ann),
        "Test_AUC": roc_auc_score(y_test, y_prob_ann),
    }
    print(
        f"[{'ANN (Deep Learning)':<22}] "
        f"Test Acc: {ann_metrics['Test_Acc']:.3f} | "
        f"F1: {ann_metrics['Test_F1']:.3f} | "
        f"AUC: {ann_metrics['Test_AUC']:.3f}"
    )

    # ----------------------------------------------------------
    # 6. Résumé final
    # ----------------------------------------------------------
    print_summary(results, ann_metrics)

    # ----------------------------------------------------------
    # 7. Sauvegarde des modèles
    # ----------------------------------------------------------
    print("\n=== Sauvegarde des modèles ===")
    joblib.dump(rf_tuned, MODELS_DIR / "rf_tuned.joblib")
    joblib.dump(svm_tuned, MODELS_DIR / "svm_tuned.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    ann.save(MODELS_DIR / "ann.keras")
    print(f"✅ Modèles sauvegardés dans {MODELS_DIR}/")


if __name__ == "__main__":
    main()
