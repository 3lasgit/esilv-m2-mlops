"""Unit tests for src/model.py — model definitions and tuning."""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from model import RANDOM_STATE, get_models, select_best_k

# ── get_models ───────────────────────────────────────────────


class TestGetModels:
    def test_returns_dict(self):
        models = get_models()
        assert isinstance(models, dict)

    def test_contains_all_keys(self):
        models = get_models()
        expected = {"LR", "KNN", "DT", "RF", "SVM"}
        assert set(models.keys()) == expected

    def test_lr_is_pipeline(self):
        models = get_models()
        assert isinstance(models["LR"], Pipeline)

    def test_knn_is_pipeline(self):
        models = get_models()
        assert isinstance(models["KNN"], Pipeline)

    def test_svm_is_pipeline(self):
        models = get_models()
        assert isinstance(models["SVM"], Pipeline)

    def test_custom_k_applied(self):
        models = get_models(best_k=5)
        knn_step = models["KNN"].named_steps["clf"]
        assert knn_step.n_neighbors == 5

    def test_default_k_is_nine(self):
        models = get_models()
        knn_step = models["KNN"].named_steps["clf"]
        assert knn_step.n_neighbors == 9

    def test_all_models_have_fit_predict(self):
        models = get_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit"
            assert hasattr(model, "predict"), f"{name} missing predict"

    def test_dt_has_max_depth(self):
        models = get_models()
        assert models["DT"].max_depth == 5

    def test_rf_n_estimators(self):
        models = get_models()
        assert models["RF"].n_estimators == 200


# ── select_best_k ────────────────────────────────────────────


class TestSelectBestK:
    def test_returns_int(self, xy_scaled):
        _, _, X_train_s, _, y_train, _ = xy_scaled
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        best_k = select_best_k(X_train_s, y_train, cv=cv)
        assert isinstance(best_k, (int, np.integer))

    def test_k_in_valid_range(self, xy_scaled):
        _, _, X_train_s, _, y_train, _ = xy_scaled
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        best_k = select_best_k(X_train_s, y_train, cv=cv)
        assert 1 <= best_k <= 20

    def test_deterministic(self, xy_scaled):
        _, _, X_train_s, _, y_train, _ = xy_scaled
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        k1 = select_best_k(X_train_s, y_train, cv=cv)
        cv2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        k2 = select_best_k(X_train_s, y_train, cv=cv2)
        assert k1 == k2


# ── Model training smoke tests ──────────────────────────────


class TestModelTraining:
    def test_lr_fits_and_predicts(self, xy_scaled):
        X_train, X_test, _, _, y_train, _ = xy_scaled
        models = get_models()
        models["LR"].fit(X_train, y_train)
        preds = models["LR"].predict(X_test)
        assert len(preds) == len(X_test)

    def test_dt_fits_and_predicts(self, xy_scaled):
        X_train, X_test, _, _, y_train, _ = xy_scaled
        models = get_models()
        models["DT"].fit(X_train, y_train)
        preds = models["DT"].predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_rf_fits_and_predicts(self, xy_scaled):
        X_train, X_test, _, _, y_train, _ = xy_scaled
        models = get_models()
        models["RF"].fit(X_train, y_train)
        proba = models["RF"].predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_svm_fits_and_predicts(self, xy_scaled):
        X_train, X_test, _, _, y_train, _ = xy_scaled
        models = get_models()
        models["SVM"].fit(X_train, y_train)
        preds = models["SVM"].predict(X_test)
        assert len(preds) == len(X_test)
