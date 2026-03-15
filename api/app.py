# ============================================================
# api/app.py
# API REST — Heart Disease Prediction
# ============================================================
# Usage :
#   uvicorn api.app:app --reload --port 8000
#   POST http://localhost:8000/predict
#
# Variables d'environnement :
#   MLFLOW_TRACKING_URI   (défaut : http://localhost:5000)
#   MODEL_STAGE           (défaut : Production, fallback : Staging)
#   USE_LOCAL_MODEL       (défaut : false) — bypass MLflow, charge depuis models/
#   APP_ENV               (défaut : production) — "dev" pour logs lisibles
# ============================================================

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Logging structuré — import conditionnel pour ne pas casser si absent
try:
    from logging_config import setup_logging

    setup_logging()
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("api")

_model = None
_scaler = None
_model_version = "local"
_model_source = "local"
_start_time: float = 0.0

# Compteurs de monitoring (in-memory, reset au redémarrage)
_metrics: dict = defaultdict(int)
_latencies: list[float] = []


# ------------------------------------------------------------------
# Schémas Pydantic
# ------------------------------------------------------------------


class PredictRequest(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: float = Field(..., ge=0, le=1)
    trestbps: float = Field(..., ge=60, le=250)
    chol: float = Field(..., ge=100, le=600)
    fbs: float = Field(..., ge=0, le=1)
    thalach: float = Field(..., ge=60, le=220)
    exang: float = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0, le=10.0)
    ca: float = Field(..., ge=0, le=3)
    cp_0: float = Field(0.0, ge=0, le=1)
    cp_1: float = Field(0.0, ge=0, le=1)
    cp_2: float = Field(0.0, ge=0, le=1)
    cp_3: float = Field(0.0, ge=0, le=1)
    restecg_0: float = Field(0.0, ge=0, le=1)
    restecg_1: float = Field(0.0, ge=0, le=1)
    restecg_2: float = Field(0.0, ge=0, le=1)
    slope_0: float = Field(0.0, ge=0, le=1)
    slope_1: float = Field(0.0, ge=0, le=1)
    slope_2: float = Field(0.0, ge=0, le=1)
    thal_0: float = Field(0.0, ge=0, le=1)
    thal_1: float = Field(0.0, ge=0, le=1)
    thal_2: float = Field(0.0, ge=0, le=1)
    hr_age_ratio: float = Field(...)
    cardio_risk_score: float = Field(...)
    exang_oldpeak: float = Field(...)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 54,
                "sex": 1,
                "trestbps": 130,
                "chol": 256,
                "fbs": 0,
                "thalach": 147,
                "exang": 0,
                "oldpeak": 1.4,
                "ca": 0,
                "cp_0": 1,
                "cp_1": 0,
                "cp_2": 0,
                "cp_3": 0,
                "restecg_0": 1,
                "restecg_1": 0,
                "restecg_2": 0,
                "slope_0": 0,
                "slope_1": 1,
                "slope_2": 0,
                "thal_0": 0,
                "thal_1": 0,
                "thal_2": 1,
                "hr_age_ratio": 2.72,
                "cardio_risk_score": 1.30,
                "exang_oldpeak": 0.0,
            }
        }
    }


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    model_name: str
    model_version: str
    model_source: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    model_source: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_predictions: int
    predictions_positive: int
    predictions_negative: int
    errors_count: int
    avg_latency_ms: float
    p95_latency_ms: float
    uptime_seconds: float


# ------------------------------------------------------------------
# Helpers de chargement
# ------------------------------------------------------------------


def _load_from_mlflow() -> tuple:
    """
    Charge le modèle depuis le MLflow Model Registry.
    Essaie d'abord le stage Production, puis Staging en fallback.
    """
    import mlflow.sklearn

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    registered_name = "heart-disease-classifier"
    stage = os.getenv("MODEL_STAGE", "Production")
    stages_to_try = [stage, "Staging"] if stage == "Production" else [stage]

    for s in stages_to_try:
        model_uri = f"models:/{registered_name}/{s}"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(
                "Modele charge depuis MLflow",
                extra={"model_uri": model_uri, "stage": s},
            )
            return model, s, f"mlflow:{tracking_uri}"
        except Exception as e:
            logger.warning(
                "Impossible de charger depuis MLflow",
                extra={"stage": s, "error": str(e)},
            )

    raise RuntimeError(
        f"Aucun modele disponible dans le registry MLflow ({tracking_uri}). "
        "Lance python src/registry.py --promote pour enregistrer un modele."
    )


def _load_from_local() -> tuple:
    """Fallback : charge le modèle depuis le système de fichiers local."""
    import joblib
    from pathlib import Path

    model = joblib.load(Path("models") / "rf_tuned.joblib")
    logger.info(
        "Modele charge depuis le filesystem local",
        extra={"path": "models/rf_tuned.joblib"},
    )
    return model, "local", "filesystem"


def _load_scaler():
    import joblib
    from pathlib import Path

    return joblib.load(Path("models") / "scaler.joblib")


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler, _model_version, _model_source, _start_time

    _start_time = time.time()
    use_local = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

    try:
        if use_local:
            _model, _model_version, _model_source = _load_from_local()
        else:
            try:
                _model, _model_version, _model_source = _load_from_mlflow()
            except RuntimeError as e:
                logger.warning("Fallback vers le modele local", extra={"reason": str(e)})
                _model, _model_version, _model_source = _load_from_local()

        _scaler = _load_scaler()
        logger.info(
            "API prete",
            extra={"model_version": _model_version, "model_source": _model_source},
        )

    except Exception as e:
        logger.error("Impossible de charger le modele", extra={"error": str(e)})

    yield
    _model = _scaler = None


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predit la presence d'une maladie cardiaque a partir de donnees cliniques.",
    version="1.0.0",
    lifespan=lifespan,
)

FEATURE_ORDER = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "oldpeak",
    "ca",
    "cp_0",
    "cp_1",
    "cp_2",
    "cp_3",
    "restecg_0",
    "restecg_1",
    "restecg_2",
    "slope_0",
    "slope_1",
    "slope_2",
    "thal_0",
    "thal_1",
    "thal_2",
    "hr_age_ratio",
    "cardio_risk_score",
    "exang_oldpeak",
]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Verifie l'etat de l'API et du modele."""
    uptime = time.time() - _start_time if _start_time else 0.0
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_version=_model_version,
        model_source=_model_source,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
def metrics():
    """Expose les metriques applicatives (compteurs, latences)."""
    uptime = time.time() - _start_time if _start_time else 0.0
    sorted_lat = sorted(_latencies) if _latencies else [0.0]
    p95_idx = int(len(sorted_lat) * 0.95)

    return MetricsResponse(
        total_predictions=_metrics["total"],
        predictions_positive=_metrics["positive"],
        predictions_negative=_metrics["negative"],
        errors_count=_metrics["errors"],
        avg_latency_ms=round(sum(sorted_lat) / max(len(sorted_lat), 1), 2),
        p95_latency_ms=round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 2),
        uptime_seconds=round(uptime, 1),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Predit la presence d'une maladie cardiaque."""
    if _model is None or _scaler is None:
        _metrics["errors"] += 1
        raise HTTPException(status_code=503, detail="Modele non disponible.")

    t0 = time.perf_counter()

    data = request.model_dump()
    features = np.array([[data[f] for f in FEATURE_ORDER]])
    features_scaled = _scaler.transform(features)

    pred = int(_model.predict(features_scaled)[0])
    proba = float(_model.predict_proba(features_scaled)[0][1])

    latency_ms = (time.perf_counter() - t0) * 1000

    # Mise à jour des compteurs
    _metrics["total"] += 1
    _metrics["positive" if pred == 1 else "negative"] += 1
    _latencies.append(latency_ms)

    logger.info(
        "Prediction effectuee",
        extra={
            "prediction": pred,
            "probability": round(proba, 4),
            "latency_ms": round(latency_ms, 2),
            "age": data["age"],
        },
    )

    return PredictResponse(
        prediction=pred,
        probability=round(proba, 4),
        label="Maladie cardiaque detectee" if pred == 1 else "Aucune maladie detectee",
        model_name="heart-disease-classifier",
        model_version=_model_version,
        model_source=_model_source,
        latency_ms=round(latency_ms, 2),
    )
