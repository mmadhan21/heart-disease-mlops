"""
=============================================================
Heart Disease MLOps Pipeline
File    : tests/conftest.py
Purpose : Pytest configuration and shared session-level
          fixtures available to ALL test files automatically.
=============================================================
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# ── Ensure src/ is on the Python path for all tests ──────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────
# SESSION-SCOPED FIXTURES
# (Created once per pytest session — shared across test files)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def full_dataset() -> pd.DataFrame:
    """
    Load the actual cleaned CSV if it exists.
    Skips the fixture (and any test using it) if the file
    is not present — avoids false failures in CI before
    download_data.py has run.
    """
    csv_path = os.path.join("data", "heart_cleaned.csv")
    if not os.path.exists(csv_path):
        pytest.skip(
            "heart_cleaned.csv not found. "
            "Run  python data/download_data.py  first."
        )
    return pd.read_csv(csv_path)


@pytest.fixture(scope="session")
def minimal_df() -> pd.DataFrame:
    """
    A deterministic 10-row DataFrame that satisfies the
    heart dataset schema — used as a lightweight stand-in
    when the real CSV is not needed.
    """
    np.random.seed(0)
    n = 10
    return pd.DataFrame({
        "age"      : np.random.randint(30,  80,  n).astype(float),
        "sex"      : np.random.randint(0,   2,   n).astype(float),
        "cp"       : np.random.randint(0,   4,   n).astype(float),
        "trestbps" : np.random.randint(90,  200, n).astype(float),
        "chol"     : np.random.randint(150, 400, n).astype(float),
        "fbs"      : np.random.randint(0,   2,   n).astype(float),
        "restecg"  : np.random.randint(0,   3,   n).astype(float),
        "thalach"  : np.random.randint(80,  200, n).astype(float),
        "exang"    : np.random.randint(0,   2,   n).astype(float),
        "oldpeak"  : np.round(np.random.uniform(0, 6, n), 1),
        "slope"    : np.random.randint(0,   3,   n).astype(float),
        "ca"       : np.random.randint(0,   4,   n).astype(float),
        "thal"     : np.random.randint(0,   4,   n).astype(float),
        "target"   : np.random.randint(0,   2,   n),
    })


# ─────────────────────────────────────────────────────────────
# PYTEST CONFIGURATION HOOKS
# ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Register custom markers to suppress PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring external resources"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests marked @pytest.mark.integration
    unless --run-integration flag is passed.
    """
    if not config.getoption("--run-integration", default=False):
        skip_integration = pytest.mark.skip(
            reason="Use --run-integration to run integration tests"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom CLI options to pytest."""
    parser.addoption(
        "--run-integration",
        action  = "store_true",
        default = False,
        help    = "Run integration tests that require external services",
    )