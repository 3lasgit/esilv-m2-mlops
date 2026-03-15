# ============================================================
# tests/test_monitoring.py
# Tests des endpoints de monitoring et du logging structuré
# ============================================================

import json
import logging
import time
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

VALID_PAYLOAD = {
    "age": 54.0,
    "sex": 1.0,
    "trestbps": 130.0,
    "chol": 256.0,
    "fbs": 0.0,
    "thalach": 147.0,
    "exang": 0.0,
    "oldpeak": 1.4,
    "ca": 0.0,
    "cp_0": 1.0,
    "cp_1": 0.0,
    "cp_2": 0.0,
    "cp_3": 0.0,
    "restecg_0": 1.0,
    "restecg_1": 0.0,
    "restecg_2": 0.0,
    "slope_0": 0.0,
    "slope_1": 1.0,
    "slope_2": 0.0,
    "thal_0": 0.0,
    "thal_1": 0.0,
    "thal_2": 1.0,
    "hr_age_ratio": 2.72,
    "cardio_risk_score": 1.30,
    "exang_oldpeak": 0.0,
}


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def client_with_model():
    """Cree un TestClient avec un modele et un scaler mockes."""
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
    app_module._start_time = time.time()
    app_module._metrics = defaultdict(int)
    app_module._latencies = []

    with TestClient(app_module.app) as c:
        yield c

    app_module._model = None
    app_module._scaler = None


@pytest.fixture
def client_no_model():
    """Cree un TestClient sans modele charge."""
    import api.app as app_module

    app_module._model = None
    app_module._scaler = None
    app_module._start_time = time.time()
    app_module._metrics = defaultdict(int)
    app_module._latencies = []

    with TestClient(app_module.app) as c:
        yield c


# ------------------------------------------------------------------
# Tests — /health
# ------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_uptime(self, client_with_model):
        resp = client_with_model.get("/health")
        body = resp.json()
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0.0

    def test_health_status_ok(self, client_with_model):
        resp = client_with_model.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_status_degraded(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.json()["status"] == "degraded"


# ------------------------------------------------------------------
# Tests — /metrics
# ------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client_with_model):
        resp = client_with_model.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_initial_state(self, client_with_model):
        resp = client_with_model.get("/metrics")
        body = resp.json()
        assert body["total_predictions"] == 0
        assert body["predictions_positive"] == 0
        assert body["predictions_negative"] == 0
        assert body["errors_count"] == 0

    def test_metrics_increment_after_predict(self, client_with_model):
        client_with_model.post("/predict", json=VALID_PAYLOAD)
        client_with_model.post("/predict", json=VALID_PAYLOAD)

        resp = client_with_model.get("/metrics")
        body = resp.json()
        assert body["total_predictions"] == 2
        assert body["predictions_positive"] == 2

    def test_metrics_latency_populated(self, client_with_model):
        client_with_model.post("/predict", json=VALID_PAYLOAD)
        resp = client_with_model.get("/metrics")
        body = resp.json()
        assert body["avg_latency_ms"] > 0.0
        assert body["p95_latency_ms"] >= 0.0

    def test_metrics_has_uptime(self, client_with_model):
        resp = client_with_model.get("/metrics")
        assert resp.json()["uptime_seconds"] >= 0.0

    def test_metrics_errors_on_503(self, client_no_model):
        client_no_model.post("/predict", json=VALID_PAYLOAD)
        resp = client_no_model.get("/metrics")
        body = resp.json()
        assert body["errors_count"] >= 1


# ------------------------------------------------------------------
# Tests — Structured Logging
# ------------------------------------------------------------------


class TestStructuredLogging:
    def test_json_formatter_output(self):
        from logging_config import JSONFormatter

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "test message"
        assert "timestamp" in parsed

    def test_json_formatter_includes_extras(self):
        from logging_config import JSONFormatter

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="prediction",
            args=(),
            exc_info=None,
        )
        record.latency_ms = 12.5
        record.prediction = 1
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["latency_ms"] == 12.5
        assert parsed["prediction"] == 1

    def test_setup_logging_dev_mode(self):
        from logging_config import setup_logging

        setup_logging(dev=True)
        root = logging.getLogger()
        assert len(root.handlers) > 0

    def test_setup_logging_prod_mode(self):
        from logging_config import setup_logging

        setup_logging(dev=False)
        root = logging.getLogger()
        assert len(root.handlers) > 0
