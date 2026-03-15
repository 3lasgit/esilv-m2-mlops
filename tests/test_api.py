# ============================================================
# tests/test_api.py
# Tests de l'API FastAPI — endpoint /predict et /health
# ============================================================
# Lance avec : pytest tests/test_api.py -v

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Payload valide réutilisable dans tous les tests
VALID_PAYLOAD = {
    "age": 54.0, "sex": 1.0, "trestbps": 130.0, "chol": 256.0,
    "fbs": 0.0, "thalach": 147.0, "exang": 0.0, "oldpeak": 1.4, "ca": 0.0,
    "cp_0": 1.0, "cp_1": 0.0, "cp_2": 0.0, "cp_3": 0.0,
    "restecg_0": 1.0, "restecg_1": 0.0, "restecg_2": 0.0,
    "slope_0": 0.0, "slope_1": 1.0, "slope_2": 0.0,
    "thal_0": 0.0, "thal_1": 0.0, "thal_2": 1.0,
    "hr_age_ratio": 2.72, "cardio_risk_score": 1.30, "exang_oldpeak": 0.0,
}


# ------------------------------------------------------------------
# Fixture : client avec modèle mocké chargé
# ------------------------------------------------------------------

@pytest.fixture
def client_with_model():
    """Crée un TestClient avec un modèle et un scaler mockés."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 26))

    import api.app as app_module
    app_module._model = mock_model
    app_module._scaler = mock_scaler
    app_module._model_version = "Staging"
    app_module._model_source = "mlflow:http://localhost:5000"

    with TestClient(app_module.app) as c:
        yield c

    app_module._model = None
    app_module._scaler = None


@pytest.fixture
def client_no_model():
    """Crée un TestClient sans modèle chargé (simule un démarrage raté)."""
    import api.app as app_module
    app_module._model = None
    app_module._scaler = None

    with TestClient(app_module.app) as c:
        yield c


# ------------------------------------------------------------------
# Tests — /health
# ------------------------------------------------------------------

class TestHealth:

    def test_health_ok_when_model_loaded(self, client_with_model):
        """GET /health doit retourner status=ok quand le modèle est chargé."""
        resp = client_with_model.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_health_degraded_when_no_model(self, client_no_model):
        """GET /health doit retourner status=degraded si le modèle est absent."""
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False

    def test_health_returns_model_version(self, client_with_model):
        """GET /health doit exposer la version et la source du modèle."""
        resp = client_with_model.get("/health")
        body = resp.json()
        assert "model_version" in body
        assert "model_source" in body


# ------------------------------------------------------------------
# Tests — /predict (nominal)
# ------------------------------------------------------------------

class TestPredictNominal:

    def test_predict_returns_200(self, client_with_model):
        """POST /predict doit retourner 200 avec un payload valide."""
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client_with_model):
        """La réponse doit contenir tous les champs attendus."""
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        body = resp.json()
        assert "prediction"    in body
        assert "probability"   in body
        assert "label"         in body
        assert "model_name"    in body
        assert "model_version" in body
        assert "latency_ms"    in body

    def test_predict_prediction_is_binary(self, client_with_model):
        """Le champ prediction doit être 0 ou 1."""
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["prediction"] in (0, 1)

    def test_predict_probability_in_range(self, client_with_model):
        """La probabilité doit être dans [0.0, 1.0]."""
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        proba = resp.json()["probability"]
        assert 0.0 <= proba <= 1.0

    def test_predict_label_positive(self, client_with_model):
        """Avec prediction=1, le label doit mentionner 'détectée'."""
        resp = client_with_model.post("/predict", json=VALID_PAYLOAD)
        body = resp.json()
        if body["prediction"] == 1:
            assert "détectée" in body["label"].lower()
        else:
            assert "aucune" in body["label"].lower()

    def test_predict_calls_scaler_transform(self, client_with_model):
        """Le scaler doit être appelé exactement une fois par requête."""
        import api.app as app_module
        app_module._scaler.transform.reset_mock()
        client_with_model.post("/predict", json=VALID_PAYLOAD)
        app_module._scaler.transform.assert_called_once()

    def test_predict_calls_model_predict(self, client_with_model):
        """Le modèle doit être appelé exactement une fois par requête."""
        import api.app as app_module
        app_module._model.predict.reset_mock()
        client_with_model.post("/predict", json=VALID_PAYLOAD)
        app_module._model.predict.assert_called_once()


# ------------------------------------------------------------------
# Tests — /predict (cas d'erreur)
# ------------------------------------------------------------------

class TestPredictErrors:

    def test_predict_503_when_no_model(self, client_no_model):
        """POST /predict doit retourner 503 si le modèle n'est pas chargé."""
        resp = client_no_model.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 503

    def test_predict_422_missing_required_field(self, client_with_model):
        """POST /predict doit retourner 422 si un champ obligatoire est absent."""
        payload = VALID_PAYLOAD.copy()
        del payload["age"]
        resp = client_with_model.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_422_age_out_of_range(self, client_with_model):
        """POST /predict doit retourner 422 si age est hors bornes (ge=1, le=120)."""
        payload = {**VALID_PAYLOAD, "age": 200}
        resp = client_with_model.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_422_invalid_binary_field(self, client_with_model):
        """POST /predict doit retourner 422 si sex n'est pas 0 ou 1."""
        payload = {**VALID_PAYLOAD, "sex": 5.0}
        resp = client_with_model.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_422_empty_body(self, client_with_model):
        """POST /predict doit retourner 422 si le body est vide."""
        resp = client_with_model.post("/predict", json={})
        assert resp.status_code == 422