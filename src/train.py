# ============================================================
# src/train.py
# Script d'entraînement principal — Heart Disease UCI
# ============================================================

import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

from data import load_and_preprocess
from model import build_ann, get_models, select_best_k, tune_rf, tune_svm

RANDOM_STATE = 42
CV_SPLITS = 5
EXPERIMENT_NAME = "heart-disease-prediction"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def evaluate_model(name: str, model, X_tr, X_te, y_tr, y_te, cv) -> dict:
    cv_acc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    cv_f1  = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1")
    cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model":      name,
        "CV_Acc":     cv_acc.mean(),
        "CV_Acc_std": cv_acc.std(),
        "CV_F1":      cv_f1.mean(),
        "CV_AUC":     cv_auc.mean(),
        "Test_Acc":   accuracy_score(y_te, y_pred),
        "Test_F1":    f1_score(y_te, y_pred),
        "Test_AUC":   roc_auc_score(y_te, y_prob) if y_prob is not None else None,
        "model_obj":  model,
        "y_pred":     y_pred,
        "y_prob":     y_prob,
    }

    print(
        f"[{name:<22}] "
        f"Test Acc: {metrics['Test_Acc']:.3f} | "
        f"F1: {metrics['Test_F1']:.3f} | "
        f"AUC: {metrics['Test_AUC']:.3f} | "
        f"CV_Acc: {cv_acc.mean():.3f}±{cv_acc.std():.3f}"
    )
    return metrics


def log_sklearn_run(run_name: str, model_name: str, model, params: dict, metrics: dict):
    """Log un modèle sklearn dans MLflow (params + metrics + artifact)."""
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("model_type", model_name)
        mlflow.log_params(params)
        mlflow.log_metrics({
            "cv_accuracy":     metrics["CV_Acc"],
            "cv_accuracy_std": metrics["CV_Acc_std"],
            "cv_f1":           metrics["CV_F1"],
            "cv_auc":          metrics["CV_AUC"],
            "test_accuracy":   metrics["Test_Acc"],
            "test_f1":         metrics["Test_F1"],
            "test_auc":        metrics["Test_AUC"],
        })
        mlflow.sklearn.log_model(model, artifact_path="model")


def print_summary(results: dict, ann_metrics: dict) -> None:
    all_results = {**results, "ANN (Deep Learning)": ann_metrics}
    rows = [
        {
            "Modèle":   v.get("Model", k),
            "Accuracy": round(v["Test_Acc"], 4),
            "F1-Score": round(v["Test_F1"], 4),
            "AUC-ROC":  round(v["Test_AUC"], 4),
        }
        for k, v in all_results.items()
        if v.get("Test_AUC") is not None
    ]
    df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
    sep = "=" * 60
    print(f"\n{sep}\n     TABLEAU RÉCAPITULATIF FINAL — TEST SET\n{sep}")
    print(df.to_string(index=False))
    print(sep)


# ------------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------------

def main() -> None:
    np.random.seed(RANDOM_STATE)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="full_training"):
        mlflow.set_tag("framework", "sklearn+keras")
        mlflow.set_tag("dataset", "heart-disease-uci-cleveland")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_splits", CV_SPLITS)
        mlflow.log_param("test_size", 0.2)

        # ----------------------------------------------------------
        # 1. Données
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("  ÉTAPE 1 — Chargement & preprocessing")
        print("=" * 60)
        (
            X_train_all, X_test_all,
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            feature_names, scaler,
        ) = load_and_preprocess()

        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_train", X_train_all.shape[0])
        mlflow.log_param("n_test", X_test_all.shape[0])

        cv = _make_cv()

        # ----------------------------------------------------------
        # 2. Sélection du k optimal pour KNN
        # ----------------------------------------------------------
        print("\n=== Sélection du k optimal (KNN) ===")
        best_k = select_best_k(X_train_scaled, y_train, cv)
        mlflow.log_param("knn_best_k", best_k)

        # ----------------------------------------------------------
        # 3. Modèles baseline
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("  ÉTAPE 2 — Entraînement des modèles baseline")
        print("=" * 60)

        models = get_models(best_k=best_k)
        results = {}

        baseline_configs = [
            ("LR",  "Logistic Regression", models["LR"],  X_train_all, X_test_all,  {"max_iter": 1000}),
            ("KNN", "KNN",                 models["KNN"], X_train_all, X_test_all,  {"n_neighbors": best_k}),
            ("SVM", "SVM (RBF)",           models["SVM"], X_train_all, X_test_all,  {"kernel": "rbf"}),
            ("DT",  "Decision Tree",       models["DT"],  X_train_all.values, X_test_all.values, {"max_depth": 5}),
            ("RF",  "Random Forest",       models["RF"],  X_train_all.values, X_test_all.values, {"n_estimators": 200}),
        ]

        for key, name, model, X_tr, X_te, params in baseline_configs:
            results[key] = evaluate_model(name, model, X_tr, X_te, y_train, y_test, cv)
            log_sklearn_run(f"baseline_{key}", name, model, params, results[key])

        # ----------------------------------------------------------
        # 4. Hyperparameter tuning
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("  ÉTAPE 3 — Hyperparameter Tuning")
        print("=" * 60)

        print("\n--- GridSearchCV : Random Forest ---")
        rf_tuned, rf_best_params = tune_rf(X_train_all.values, y_train, cv)
        y_pred_rf_t = rf_tuned.predict(X_test_all.values)
        y_prob_rf_t = rf_tuned.predict_proba(X_test_all.values)[:, 1]
        results["RF_tuned"] = {
            "Model":     "RF (Tuned)",
            "CV_Acc":    0.0, "CV_Acc_std": 0.0, "CV_F1": 0.0, "CV_AUC": 0.0,
            "Test_Acc":  accuracy_score(y_test, y_pred_rf_t),
            "Test_F1":   f1_score(y_test, y_pred_rf_t),
            "Test_AUC":  roc_auc_score(y_test, y_prob_rf_t),
            "y_pred":    y_pred_rf_t,
            "y_prob":    y_prob_rf_t,
            "model_obj": rf_tuned,
        }
        log_sklearn_run("tuned_RF", "RF (Tuned)", rf_tuned, rf_best_params, results["RF_tuned"])

        print("\n--- RandomizedSearchCV : SVM ---")
        svm_tuned, svm_best_params = tune_svm(X_train_scaled, y_train, cv)
        y_pred_svm_t = svm_tuned.predict(X_test_scaled)
        y_prob_svm_t = svm_tuned.predict_proba(X_test_scaled)[:, 1]
        results["SVM_tuned"] = {
            "Model":     "SVM (Tuned)",
            "CV_Acc":    0.0, "CV_Acc_std": 0.0, "CV_F1": 0.0, "CV_AUC": 0.0,
            "Test_Acc":  accuracy_score(y_test, y_pred_svm_t),
            "Test_F1":   f1_score(y_test, y_pred_svm_t),
            "Test_AUC":  roc_auc_score(y_test, y_prob_svm_t),
            "y_pred":    y_pred_svm_t,
            "y_prob":    y_prob_svm_t,
            "model_obj": svm_tuned,
        }
        log_sklearn_run("tuned_SVM", "SVM (Tuned)", svm_tuned, svm_best_params, results["SVM_tuned"])

        # ----------------------------------------------------------
        # 5. ANN
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("  ÉTAPE 4 — Entraînement ANN")
        print("=" * 60)

        ann_params = {"input_dim": X_train_scaled.shape[1], "dropout_rate": 0.3, "l2_reg": 0.001,
                      "epochs": 200, "batch_size": 32, "learning_rate": 0.001}

        ann = build_ann(input_dim=ann_params["input_dim"])
        ann.summary()

        callbacks = [
            EarlyStopping(monitor="val_auc", patience=20, restore_best_weights=True, mode="max", verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
        ]
        history = ann.fit(
            X_train_scaled, y_train,
            epochs=ann_params["epochs"],
            batch_size=ann_params["batch_size"],
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1,
        )
        actual_epochs = len(history.history["loss"])
        print(f"\n✅ ANN — entraînement terminé (époque {actual_epochs})")

        y_prob_ann = ann.predict(X_test_scaled, verbose=0).ravel()
        y_pred_ann = (y_prob_ann >= 0.5).astype(int)

        ann_metrics = {
            "Model":    "ANN (Deep Learning)",
            "Test_Acc": accuracy_score(y_test, y_pred_ann),
            "Test_F1":  f1_score(y_test, y_pred_ann),
            "Test_AUC": roc_auc_score(y_test, y_prob_ann),
        }

        with mlflow.start_run(run_name="ann", nested=True):
            mlflow.set_tag("model_type", "ANN")
            mlflow.log_params({**ann_params, "actual_epochs": actual_epochs})
            mlflow.log_metrics({
                "test_accuracy": ann_metrics["Test_Acc"],
                "test_f1":       ann_metrics["Test_F1"],
                "test_auc":      ann_metrics["Test_AUC"],
            })
            mlflow.tensorflow.log_model(ann, artifact_path="model")

        # Log best overall metrics sur le run parent
        best_auc = max(
            results["RF_tuned"]["Test_AUC"],
            results["SVM_tuned"]["Test_AUC"],
            ann_metrics["Test_AUC"],
        )
        mlflow.log_metric("best_test_auc", best_auc)

        # ----------------------------------------------------------
        # 6. Résumé + sauvegarde
        # ----------------------------------------------------------
        print_summary(results, ann_metrics)

        joblib.dump(rf_tuned,  MODELS_DIR / "rf_tuned.joblib")
        joblib.dump(svm_tuned, MODELS_DIR / "svm_tuned.joblib")
        joblib.dump(scaler,    MODELS_DIR / "scaler.joblib")
        ann.save(MODELS_DIR / "ann.keras")

        mlflow.log_artifact(str(MODELS_DIR / "scaler.joblib"), artifact_path="preprocessing")
        print(f"\n✅ Modèles sauvegardés dans {MODELS_DIR}/")
        print(f"✅ Run MLflow loggé — expérience : '{EXPERIMENT_NAME}'")


if __name__ == "__main__":
    main()