"""
=============================================================
Heart Disease MLOps Pipeline
File    : tests/test_model.py
Purpose : Unit tests for model inference and the
          FastAPI /predict endpoint.
          Requires models/best_model.pkl to exist.
          Run src/train.py before running these tests.
Run     : pytest tests/ -v
=============================================================
"""

import os
import sys
import pytest
import numpy  as np
import pandas as pd

# ── Suppress Git warning ──────────────────────────────────
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# ── Make src importable ───────────────────────────────────
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

# ── Resolve absolute model path ───────────────────────────
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

# ── Set environment variable BEFORE importing app ─────────
os.environ["MODEL_PATH"] = MODEL_PATH

# Sample row matching training schema exactly
SAMPLE_INPUT = {
    "age"      : 52.0,
    "sex"      : 1.0,
    "cp"       : 0.0,
    "trestbps" : 125.0,
    "chol"     : 212.0,
    "fbs"      : 0.0,
    "restecg"  : 1.0,
    "thalach"  : 168.0,
    "exang"    : 0.0,
    "oldpeak"  : 1.0,
    "slope"    : 2.0,
    "ca"       : 2.0,
    "thal"     : 3.0,
}


# ─────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model():
    """
    Load the trained pipeline once for the entire module.
    Skips all tests gracefully if model not trained yet.
    """
    import joblib
    if not os.path.exists(MODEL_PATH):
        pytest.skip(
            f"Model not found at '{MODEL_PATH}'. "
            "Run  python src/train.py  first.",
            allow_module_level=True,
        )
    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_df():
    """Single-row DataFrame matching training feature schema."""
    return pd.DataFrame([SAMPLE_INPUT])


@pytest.fixture(scope="module")
def test_client():
    """
    Create a FastAPI TestClient with model path
    correctly set before app startup.
    """
    # Must set env var before importing app
    os.environ["MODEL_PATH"] = MODEL_PATH

    from fastapi.testclient import TestClient
    from app import app

    # Use context manager to trigger startup event
    with TestClient(app) as client:
        yield client


# ─────────────────────────────────────────────────────────
# TEST : Model file
# ─────────────────────────────────────────────────────────

class TestModelFile:
    def test_best_model_exists(self):
        """best_model.pkl must exist after training."""
        assert os.path.exists(MODEL_PATH), (
            f"'{MODEL_PATH}' not found. "
            "Run src/train.py first."
        )

    def test_model_is_loadable(self, trained_model):
        """Loaded object must not be None."""
        assert trained_model is not None


# ─────────────────────────────────────────────────────────
# TEST : Model prediction
# ─────────────────────────────────────────────────────────

class TestModelPrediction:
    def test_predict_returns_array(
        self, trained_model, sample_df
    ):
        """predict() must return an array."""
        result = trained_model.predict(sample_df)
        assert hasattr(result, "__len__")

    def test_predict_single_value(
        self, trained_model, sample_df
    ):
        """Single-row input → single prediction."""
        result = trained_model.predict(sample_df)
        assert len(result) == 1

    def test_prediction_is_binary(
        self, trained_model, sample_df
    ):
        """Prediction must be 0 or 1."""
        result = trained_model.predict(sample_df)
        assert result[0] in [0, 1], (
            f"Expected 0 or 1, got {result[0]}"
        )

    def test_predict_proba_shape(
        self, trained_model, sample_df
    ):
        """predict_proba must return shape (1, 2)."""
        proba = trained_model.predict_proba(sample_df)
        assert proba.shape == (1, 2)

    def test_probabilities_sum_to_one(
        self, trained_model, sample_df
    ):
        """Class probabilities must sum to 1.0."""
        proba = trained_model.predict_proba(sample_df)
        total = proba[0].sum()
        assert abs(total - 1.0) < 1e-6, (
            f"Probabilities sum to {total}, expected 1.0"
        )

    def test_probabilities_in_range(
        self, trained_model, sample_df
    ):
        """Both class probabilities must be in [0,1]."""
        proba = trained_model.predict_proba(sample_df)
        assert 0.0 <= proba[0][0] <= 1.0
        assert 0.0 <= proba[0][1] <= 1.0

    def test_batch_prediction(self, trained_model):
        """Model must handle a batch of 5 rows."""
        rows  = pd.DataFrame([SAMPLE_INPUT] * 5)
        preds = trained_model.predict(rows)
        assert len(preds) == 5

    def test_all_predictions_binary_in_batch(
        self, trained_model
    ):
        """Every prediction in a batch must be 0 or 1."""
        rows  = pd.DataFrame([SAMPLE_INPUT] * 5)
        preds = trained_model.predict(rows)
        assert all(p in [0, 1] for p in preds)


# ─────────────────────────────────────────────────────────
# TEST : FastAPI endpoint
# ─────────────────────────────────────────────────────────

class TestPredictEndpoint:
    """
    Tests use FastAPI TestClient — no running
    server required. Model path is set via
    environment variable before app loads.
    """

    def test_root_returns_200(self, test_client):
        """GET / should return HTTP 200."""
        response = test_client.get("/")
        assert response.status_code == 200

    def test_health_endpoint(self, test_client):
        """GET /health should report model_loaded=True."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True, (
            f"model_loaded is False. "
            f"MODEL_PATH={MODEL_PATH} "
            f"exists={os.path.exists(MODEL_PATH)}"
        )

    def test_predict_endpoint_returns_200(
        self, test_client
    ):
        """POST /predict with valid data → HTTP 200."""
        response = test_client.post(
            "/predict",
            json=SAMPLE_INPUT
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. "
            f"Response: {response.json()}"
        )

    def test_predict_response_schema(self, test_client):
        """Response must contain required keys."""
        response = test_client.post(
            "/predict",
            json=SAMPLE_INPUT
        )
        data = response.json()

        assert "prediction"  in data
        assert "diagnosis"   in data
        assert "confidence"  in data
        assert "risk_level"  in data
        assert "latency_ms"  in data

        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["risk_level"] in [
            "Low", "Medium", "High"
        ]

    def test_predict_invalid_input_returns_422(
        self, test_client
    ):
        """Missing required fields → HTTP 422."""
        response = test_client.post(
            "/predict",
            json={"age": 63}
        )
        assert response.status_code == 422

    def test_metrics_endpoint(self, test_client):
        """GET /metrics → Prometheus text format."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "heart_api" in response.text