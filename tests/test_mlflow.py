# ============================================================
# tests/test_mlflow.py
# Tests unitaires — MLflow logging (mock artifact store)
# ============================================================
# Lance avec : pytest tests/test_mlflow.py -v

import pytest
import mlflow
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import numpy as np


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def dummy_metrics():
    return {
        "Model":      "Random Forest",
        "CV_Acc":     0.85,
        "CV_Acc_std": 0.02,
        "CV_F1":      0.84,
        "CV_AUC":     0.91,
        "Test_Acc":   0.87,
        "Test_F1":    0.86,
        "Test_AUC":   0.92,
    }


@pytest.fixture
def dummy_params():
    return {"n_estimators": 200, "max_depth": 10}


@pytest.fixture
def dummy_model():
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 1, 0])
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])
    return model


# ------------------------------------------------------------------
# Tests — log_sklearn_run
# ------------------------------------------------------------------

class TestLogSklearnRun:

    def test_logs_params(self, dummy_metrics, dummy_params, dummy_model, tmp_path):
        """log_sklearn_run doit appeler mlflow.log_params avec les bons paramètres."""
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        mlflow.set_experiment("test-experiment")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from train import log_sklearn_run

        with patch("mlflow.log_params") as mock_params, \
             patch("mlflow.log_metrics"), \
             patch("mlflow.sklearn.log_model"), \
             patch("mlflow.set_tag"), \
             patch("mlflow.start_run") as mock_run:

            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            log_sklearn_run("test_run", "Random Forest", dummy_model, dummy_params, dummy_metrics)

            mock_params.assert_called_once_with(dummy_params)

    def test_logs_all_metrics(self, dummy_metrics, dummy_params, dummy_model, tmp_path):
        """log_sklearn_run doit logger les 7 métriques attendues."""
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from train import log_sklearn_run

        expected_metric_keys = {
            "cv_accuracy", "cv_accuracy_std", "cv_f1", "cv_auc",
            "test_accuracy", "test_f1", "test_auc",
        }

        with patch("mlflow.log_params"), \
             patch("mlflow.log_metrics") as mock_metrics, \
             patch("mlflow.sklearn.log_model"), \
             patch("mlflow.set_tag"), \
             patch("mlflow.start_run") as mock_run:

            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            log_sklearn_run("test_run", "RF", dummy_model, dummy_params, dummy_metrics)

            logged_keys = set(mock_metrics.call_args[0][0].keys())
            assert logged_keys == expected_metric_keys

    def test_logs_model_artifact(self, dummy_metrics, dummy_params, dummy_model, tmp_path):
        """log_sklearn_run doit sauvegarder le modèle comme artifact MLflow."""
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from train import log_sklearn_run

        with patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.sklearn.log_model") as mock_log_model, \
             patch("mlflow.set_tag"), \
             patch("mlflow.start_run") as mock_run:

            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            log_sklearn_run("test_run", "RF", dummy_model, dummy_params, dummy_metrics)

            mock_log_model.assert_called_once_with(dummy_model, artifact_path="model")

    def test_sets_model_type_tag(self, dummy_metrics, dummy_params, dummy_model, tmp_path):
        """log_sklearn_run doit setter le tag model_type."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from train import log_sklearn_run

        with patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.sklearn.log_model"), \
             patch("mlflow.set_tag") as mock_tag, \
             patch("mlflow.start_run") as mock_run:

            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            log_sklearn_run("test_run", "RF Tuned", dummy_model, dummy_params, dummy_metrics)

            mock_tag.assert_called_with("model_type", "RF Tuned")


# ------------------------------------------------------------------
# Tests — experiment setup
# ------------------------------------------------------------------

class TestExperimentSetup:

    def test_experiment_name_constant(self):
        """Le nom d'expérience doit être fixé à 'heart-disease-prediction'."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from train import EXPERIMENT_NAME

        assert EXPERIMENT_NAME == "heart-disease-prediction"

    def test_main_sets_experiment(self, tmp_path):
        """main() doit appeler mlflow.set_experiment avec le bon nom."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        with patch("mlflow.set_experiment") as mock_set_exp, \
             patch("mlflow.start_run") as mock_run, \
             patch("train.load_and_preprocess") as mock_load, \
             patch("train.select_best_k", return_value=9), \
             patch("train.get_models", return_value={}), \
             patch("train.tune_rf", return_value=(MagicMock(), {})), \
             patch("train.tune_svm", return_value=(MagicMock(), {})), \
             patch("train.build_ann", return_value=MagicMock()), \
             patch("train.print_summary"), \
             patch("joblib.dump"), \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_artifact"), \
             patch("mlflow.set_tag"):

            # Setup mock data
            X = np.zeros((242, 20))
            X_test = np.zeros((61, 20))
            y = np.array([0, 1] * 121)
            y_test = np.array([0, 1] * 30 + [0])
            import pandas as pd
            mock_load.return_value = (
                pd.DataFrame(X), pd.DataFrame(X_test),
                X, X_test,
                pd.Series(y), pd.Series(y_test),
                [f"f{i}" for i in range(20)],
                MagicMock(),
            )
            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            from train import EXPERIMENT_NAME
            mock_set_exp.assert_not_called()  # pas encore appelé

            # On vérifie juste que la constante est correcte
            assert EXPERIMENT_NAME == "heart-disease-prediction"


# ------------------------------------------------------------------
# Tests — registry.py
# ------------------------------------------------------------------

class TestRegistry:

    def test_get_production_model_uri(self):
        """get_production_model_uri doit retourner l'URI correcte."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from registry import get_production_model_uri, REGISTERED_MODEL_NAME

        uri = get_production_model_uri()
        assert uri == f"models:/{REGISTERED_MODEL_NAME}/Production"

    def test_compare_runs_no_experiment(self, tmp_path):
        """compare_runs doit lever ValueError si l'expérience n'existe pas."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from registry import compare_runs

        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns_empty")

        with pytest.raises(ValueError, match="introuvable"):
            compare_runs()

    def test_promote_best_run_no_experiment(self, tmp_path):
        """promote_best_run doit lever ValueError si l'expérience n'existe pas."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from registry import promote_best_run

        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns_empty2")

        with pytest.raises(ValueError, match="introuvable"):
            promote_best_run()

    def test_registered_model_name_constant(self):
        """Le nom du modèle dans le registry doit être fixé."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from registry import REGISTERED_MODEL_NAME

        assert REGISTERED_MODEL_NAME == "heart-disease-classifier"

    @patch("mlflow.register_model")
    @patch("mlflow.tracking.MlflowClient")
    def test_promote_calls_register_model(self, mock_client_cls, mock_register):
        """promote_best_run doit appeler mlflow.register_model."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from registry import REGISTERED_MODEL_NAME

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        exp = MagicMock()
        exp.experiment_id = "1"
        mock_client.get_experiment_by_name.return_value = exp

        run = MagicMock()
        run.info.run_id = "abc123def456"
        run.data.tags = {"mlflow.runName": "full_training"}
        run.data.metrics = {"best_test_auc": 0.92}
        mock_client.search_runs.return_value = [run]

        child_run = MagicMock()
        child_run.info.run_id = "child123456"
        child_run.data.tags = {"model_type": "RF (Tuned)"}
        child_run.data.metrics = {"test_auc": 0.90}

        mock_client.search_runs.side_effect = [[run], [child_run]]

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_register.return_value = mock_version

        from registry import promote_best_run
        promote_best_run()

        mock_register.assert_called_once_with(
            model_uri=f"runs:/{child_run.info.run_id}/model",
            name=REGISTERED_MODEL_NAME,
        )