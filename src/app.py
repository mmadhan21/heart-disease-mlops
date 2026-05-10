"""
=============================================================
Heart Disease MLOps Pipeline
File    : src/app.py
Purpose : FastAPI model-serving application.
          Endpoints
          ─────────
          GET  /          → health check (root)
          GET  /health    → detailed health
          POST /predict   → prediction + confidence
          GET  /metrics   → Prometheus metrics scrape
Run     : uvicorn src.app:app --host 0.0.0.0 --port 8000
          OR via Docker
=============================================================
"""

import os
import time
import logging
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from fastapi              import FastAPI, HTTPException
from fastapi.responses    import Response, JSONResponse
from pydantic             import BaseModel, Field
from prometheus_client    import (
    Counter, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY,
)

# ─────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_FILE   = "api_requests.log"

logging.basicConfig(
    level    = logging.INFO,
    format   = LOG_FORMAT,
    handlers = [
        logging.StreamHandler(),            # console
        logging.FileHandler(LOG_FILE),      # file
    ]
)
logger = logging.getLogger("heart_api")


# ─────────────────────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total number of /predict requests received",
)

REQUEST_LATENCY = Histogram(
    "heart_api_request_latency_seconds",
    "Latency of /predict requests in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTION_BY_CLASS = Counter(
    "heart_api_predictions_total",
    "Prediction counts broken down by predicted class",
    ["predicted_class"],
)

ERROR_COUNT = Counter(
    "heart_api_errors_total",
    "Total number of prediction errors",
)


# ─────────────────────────────────────────────────────────────
# APP INITIALISATION
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Heart Disease Prediction API",
    description = (
        "MLOps Assignment — BITS Pilani WILP | MTech AI & ML\n\n"
        "Predicts the risk of heart disease from patient health data."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Global model handle — loaded once at startup
model      = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")


# ─────────────────────────────────────────────────────────────
# STARTUP EVENT
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_model() -> None:
    """Load the serialised model pipeline from disk at startup."""
    global model
    logger.info(f"Loading model from: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(
            f"Model file not found: {MODEL_PATH}. "
            "Run  python src/train.py  first."
        )
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    except Exception as exc:
        logger.error(f"Unexpected error loading model: {exc}")
        raise


# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────

class PatientData(BaseModel):
    """
    Input schema — 13 clinical features for one patient.
    All values are numeric (float) to match the training schema.
    """
    age      : float = Field(..., ge=1,  le=120, example=63,
                             description="Age in years")
    sex      : float = Field(..., ge=0,  le=1,   example=1,
                             description="1 = Male, 0 = Female")
    cp       : float = Field(..., ge=0,  le=3,   example=3,
                             description="Chest pain type (0‑3)")
    trestbps : float = Field(..., ge=80, le=250,  example=145,
                             description="Resting blood pressure (mm Hg)")
    chol     : float = Field(..., ge=100, le=600, example=233,
                             description="Serum cholesterol (mg/dl)")
    fbs      : float = Field(..., ge=0,  le=1,   example=1,
                             description="Fasting blood sugar > 120 mg/dl")
    restecg  : float = Field(..., ge=0,  le=2,   example=0,
                             description="Resting ECG results (0‑2)")
    thalach  : float = Field(..., ge=60, le=250,  example=150,
                             description="Maximum heart rate achieved")
    exang    : float = Field(..., ge=0,  le=1,   example=0,
                             description="Exercise-induced angina")
    oldpeak  : float = Field(..., ge=0,  le=10,  example=2.3,
                             description="ST depression induced by exercise")
    slope    : float = Field(..., ge=0,  le=2,   example=0,
                             description="Slope of peak exercise ST segment")
    ca       : float = Field(..., ge=0,  le=4,   example=0,
                             description="Number of major vessels coloured (0‑3)")
    thal     : float = Field(..., ge=0,  le=3,   example=1,
                             description="Thalassemia: 1=normal,2=fixed,3=reversible")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 3,
                "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150,
                "exang": 0, "oldpeak": 2.3,
                "slope": 0, "ca": 0, "thal": 1,
            }
        }


class PredictionResponse(BaseModel):
    prediction  : int
    diagnosis   : str
    confidence  : float
    risk_level  : str
    latency_ms  : float


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def _risk_level(probability: float) -> str:
    """Bucket a disease probability into a human-readable risk tier."""
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    return "High"


def _patient_to_dataframe(patient: PatientData) -> pd.DataFrame:
    """Convert a PatientData Pydantic object to a single-row DataFrame."""
    return pd.DataFrame([{
        "age"      : patient.age,
        "sex"      : patient.sex,
        "cp"       : patient.cp,
        "trestbps" : patient.trestbps,
        "chol"     : patient.chol,
        "fbs"      : patient.fbs,
        "restecg"  : patient.restecg,
        "thalach"  : patient.thalach,
        "exang"    : patient.exang,
        "oldpeak"  : patient.oldpeak,
        "slope"    : patient.slope,
        "ca"       : patient.ca,
        "thal"     : patient.thal,
    }])


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Root health check")
def root() -> Dict[str, Any]:
    """Returns a simple alive message."""
    logger.info("GET /  — health check")
    return {
        "status"  : "healthy",
        "service" : "Heart Disease Prediction API",
        "version" : "1.0.0",
    }


@app.get("/health", tags=["Health"], summary="Detailed health check")
def health() -> Dict[str, Any]:
    """Returns model load status and environment info."""
    return {
        "status"       : "ok",
        "model_loaded" : model is not None,
        "model_path"   : MODEL_PATH,
    }


@app.post(
    "/predict",
    response_model = PredictionResponse,
    tags           = ["Prediction"],
    summary        = "Predict heart disease risk",
)
def predict(patient: PatientData) -> PredictionResponse:
    """
    Accepts 13 clinical features and returns:
    - **prediction**  : 0 (No Disease) or 1 (Disease)
    - **diagnosis**   : human-readable label
    - **confidence**  : probability of disease (0–1)
    - **risk_level**  : Low / Medium / High
    - **latency_ms**  : inference time in milliseconds
    """
    REQUEST_COUNT.inc()
    start = time.perf_counter()

    if model is None:
        ERROR_COUNT.inc()
        raise HTTPException(
            status_code = 503,
            detail      = "Model is not loaded. Try again later."
        )

    try:
        features    = _patient_to_dataframe(patient)
        prediction  = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        risk        = _risk_level(probability)
        latency_ms  = round((time.perf_counter() - start) * 1000, 2)

        REQUEST_LATENCY.observe((time.perf_counter() - start))
        PREDICTION_BY_CLASS.labels(
            predicted_class=str(prediction)
        ).inc()

        logger.info(
            f"POST /predict | "
            f"prediction={prediction} | "
            f"confidence={probability:.4f} | "
            f"risk={risk} | "
            f"latency={latency_ms}ms"
        )

        return PredictionResponse(
            prediction = prediction,
            diagnosis  = (
                "Heart Disease Detected"
                if prediction == 1
                else "No Heart Disease Detected"
            ),
            confidence = round(probability, 4),
            risk_level = risk,
            latency_ms = latency_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        ERROR_COUNT.inc()
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/metrics",
    tags    = ["Monitoring"],
    summary = "Prometheus metrics scrape endpoint",
)
def metrics() -> Response:
    """Exposes Prometheus-compatible metrics for scraping."""
    return Response(
        content      = generate_latest(),
        media_type   = CONTENT_TYPE_LATEST,
    )