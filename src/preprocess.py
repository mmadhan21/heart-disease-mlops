"""
=============================================================
Heart Disease MLOps Pipeline
File    : src/preprocess.py
Purpose : All data preprocessing logic.
          - load_data()
          - clean_data()
          - build_preprocessing_pipeline()
          - split_data()
          - save / load preprocessor helpers
          Imported by train.py, app.py, and test files.
=============================================================
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────
# SCHEMA CONSTANTS
# (Single source of truth — imported everywhere)
# ─────────────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET       = "target"


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError if the path does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] Dataset not found at: {filepath}\n"
            "        Run  python data/download_data.py  first."
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Data loaded from '{filepath}' — shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in any column.
    Strategy : replace NaN with the column median.
    This is safe for both numerical and ordinal-encoded
    categorical columns present in this dataset.

    Parameters
    ----------
    df : pd.DataFrame  (raw or partially processed)

    Returns
    -------
    pd.DataFrame  (no NaN values)
    """
    df = df.copy()
    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        print("[INFO] clean_data : no missing values detected.")
        return df

    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            fill_value = df[col].median()
            df[col]    = df[col].fillna(fill_value)
            print(
                f"[INFO] clean_data : '{col}' — "
                f"filled {n_missing} NaN(s) with median={fill_value:.4f}"
            )

    print(f"[INFO] clean_data : {total_missing} missing value(s) resolved.")
    return df


# ─────────────────────────────────────────────────────────────
# FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────────────────────

def get_features_and_target(df: pd.DataFrame):
    """
    Separate the DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame  — must contain ALL_FEATURES + TARGET columns

    Returns
    -------
    X : pd.DataFrame  shape (n, 13)
    y : pd.Series     shape (n,)   values in {0, 1}
    """
    missing_cols = [
        c for c in ALL_FEATURES + [TARGET] if c not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"[ERROR] DataFrame missing required columns: {missing_cols}"
        )

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


# ─────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
      - Applies StandardScaler to numerical features
      - Passes categorical features through unchanged
        (they are already integer-encoded in this dataset)

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",               # name
                StandardScaler(),    # transformer
                NUMERICAL_FEATURES   # columns to apply to
            ),
            (
                "cat",
                "passthrough",       # no transformation
                CATEGORICAL_FEATURES
            ),
        ],
        remainder="drop"             # drop any unexpected columns
    )
    return preprocessor


# ─────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float   = 0.20,
    random_state: int  = 42
):
    """
    Stratified train/test split to preserve class balance.

    Parameters
    ----------
    X            : feature DataFrame
    y            : target Series
    test_size    : fraction for test set (default 0.20)
    random_state : seed for reproducibility (default 42)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y
    )
    print(
        f"[INFO] split_data  : "
        f"train={len(X_train)}, test={len(X_test)} "
        f"(test_size={test_size}, stratified)"
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# PERSIST HELPERS
# ─────────────────────────────────────────────────────────────

def save_preprocessor(preprocessor, path: str) -> None:
    """Serialize a fitted preprocessor to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"[INFO] Preprocessor saved → {path}")


def load_preprocessor(path: str):
    """Load a serialized preprocessor from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Preprocessor not found: {path}")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# QUICK SELF‑TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data("data/heart_cleaned.csv")
    df = clean_data(df)
    X, y = get_features_and_target(df)
    preprocessor = build_preprocessing_pipeline()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Fit and transform
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    print(f"[INFO] X_train transformed shape : {X_train_t.shape}")
    print(f"[INFO] X_test  transformed shape : {X_test_t.shape}")
    print("[DONE] preprocess.py self-test passed.\n")