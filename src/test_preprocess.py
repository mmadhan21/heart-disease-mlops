"""
=============================================================
Heart Disease MLOps Pipeline
File    : tests/test_preprocess.py
Purpose : Unit tests for all functions in src/preprocess.py
Run     : pytest tests/ -v
=============================================================
"""

import os
import sys
import pytest
import numpy  as np
import pandas as pd

# Make src importable when running from project root
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

from preprocess import (
    load_data,
    clean_data,
    get_features_and_target,
    build_preprocessing_pipeline,
    split_data,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)


# ─────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def clean_sample_df() -> pd.DataFrame:
    """
    A small, fully clean DataFrame that mirrors the
    heart dataset schema (no missing values).
    """
    return pd.DataFrame({
        "age"      : [63, 37, 41, 56, 57, 44, 52, 48],
        "sex"      : [1,  1,  0,  1,  0,  1,  1,  0 ],
        "cp"       : [3,  2,  1,  1,  0,  2,  3,  1 ],
        "trestbps" : [145,130,130,120,120,130,172,110],
        "chol"     : [233,250,204,236,354,233,199,229],
        "fbs"      : [1,  0,  0,  0,  0,  0,  1,  0 ],
        "restecg"  : [0,  1,  0,  1,  1,  0,  1,  1 ],
        "thalach"  : [150,187,172,178,163,179,162,168],
        "exang"    : [0,  0,  0,  0,  1,  0,  0,  0 ],
        "oldpeak"  : [2.3,3.5,1.4,0.8,0.6,0.0,0.5,1.0],
        "slope"    : [0,  0,  2,  2,  2,  2,  2,  2 ],
        "ca"       : [0,  0,  0,  0,  0,  0,  0,  0 ],
        "thal"     : [1,  2,  2,  2,  2,  2,  3,  2 ],
        "target"   : [1,  1,  1,  1,  0,  0,  0,  1 ],
    })


@pytest.fixture
def dirty_sample_df(clean_sample_df) -> pd.DataFrame:
    """Same DataFrame but with intentional NaN values."""
    df          = clean_sample_df.copy()
    df.loc[0, "ca"]   = np.nan
    df.loc[2, "thal"] = np.nan
    df.loc[4, "age"]  = np.nan
    return df


# ─────────────────────────────────────────────────────────────
# TEST : load_data
# ─────────────────────────────────────────────────────────────

class TestLoadData:
    def test_raises_if_file_missing(self):
        """load_data must raise FileNotFoundError for a bad path."""
        with pytest.raises(FileNotFoundError):
            load_data("data/nonexistent_file.csv")

    def test_loads_cleaned_csv_if_present(self, tmp_path, clean_sample_df):
        """load_data should return a DataFrame when the file exists."""
        csv_path = tmp_path / "heart.csv"
        clean_sample_df.to_csv(csv_path, index=False)
        df = load_data(str(csv_path))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == clean_sample_df.shape


# ─────────────────────────────────────────────────────────────
# TEST : clean_data
# ─────────────────────────────────────────────────────────────

class TestCleanData:
    def test_no_nulls_remain_after_cleaning(self, dirty_sample_df):
        """After clean_data, zero NaN values must remain."""
        cleaned = clean_data(dirty_sample_df)
        assert cleaned.isnull().sum().sum() == 0

    def test_row_count_unchanged(self, dirty_sample_df):
        """clean_data must not drop any rows."""
        cleaned = clean_data(dirty_sample_df)
        assert len(cleaned) == len(dirty_sample_df)

    def test_column_count_unchanged(self, dirty_sample_df):
        """clean_data must not add or remove columns."""
        cleaned = clean_data(dirty_sample_df)
        assert list(cleaned.columns) == list(dirty_sample_df.columns)

    def test_already_clean_df_passes_through(self, clean_sample_df):
        """clean_data on an already-clean df should not alter values."""
        cleaned = clean_data(clean_sample_df)
        assert cleaned.isnull().sum().sum() == 0
        pd.testing.assert_frame_equal(cleaned, clean_sample_df)

    def test_fill_value_is_median(self, dirty_sample_df):
        """NaN in 'ca' should be replaced by the median of 'ca'."""
        expected_median = dirty_sample_df["ca"].median()
        cleaned         = clean_data(dirty_sample_df)
        # Row 0 was set to NaN — check it now holds the median
        assert cleaned.loc[0, "ca"] == expected_median


# ─────────────────────────────────────────────────────────────
# TEST : get_features_and_target
# ─────────────────────────────────────────────────────────────

class TestGetFeaturesAndTarget:
    def test_X_has_correct_number_of_columns(self, clean_sample_df):
        X, _ = get_features_and_target(clean_sample_df)
        assert X.shape[1] == len(ALL_FEATURES)

    def test_X_column_names_match_schema(self, clean_sample_df):
        X, _ = get_features_and_target(clean_sample_df)
        assert list(X.columns) == ALL_FEATURES

    def test_y_is_binary(self, clean_sample_df):
        _, y = get_features_and_target(clean_sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_X_and_y_same_length(self, clean_sample_df):
        X, y = get_features_and_target(clean_sample_df)
        assert len(X) == len(y)

    def test_raises_on_missing_column(self, clean_sample_df):
        """Should raise ValueError if a required column is absent."""
        df_missing = clean_sample_df.drop(columns=["age"])
        with pytest.raises(ValueError):
            get_features_and_target(df_missing)


# ─────────────────────────────────────────────────────────────
# TEST : split_data
# ─────────────────────────────────────────────────────────────

class TestSplitData:
    def test_total_size_preserved(self, clean_sample_df):
        X, y = get_features_and_target(clean_sample_df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)
        assert len(X_train) + len(X_test) == len(X)

    def test_no_index_overlap(self, clean_sample_df):
        X, y = get_features_and_target(clean_sample_df)
        X_train, X_test, _, _ = split_data(X, y, test_size=0.25)
        train_idx = set(X_train.index)
        test_idx  = set(X_test.index)
        assert train_idx.isdisjoint(test_idx)

    def test_returns_four_objects(self, clean_sample_df):
        X, y  = get_features_and_target(clean_sample_df)
        result = split_data(X, y)
        assert len(result) == 4

    def test_reproducibility(self, clean_sample_df):
        """Same random_state must produce identical splits."""
        X, y = get_features_and_target(clean_sample_df)
        X_train_a, X_test_a, _, _ = split_data(X, y, random_state=0)
        X_train_b, X_test_b, _, _ = split_data(X, y, random_state=0)
        pd.testing.assert_frame_equal(X_train_a, X_train_b)
        pd.testing.assert_frame_equal(X_test_a,  X_test_b)


# ─────────────────────────────────────────────────────────────
# TEST : build_preprocessing_pipeline
# ─────────────────────────────────────────────────────────────

class TestBuildPreprocessingPipeline:
    def test_pipeline_instantiates(self):
        """Pipeline object should be created without error."""
        pp = build_preprocessing_pipeline()
        assert pp is not None

    def test_fit_transform_shape(self, clean_sample_df):
        """
        Output row count must equal input row count.
        Column count = len(NUMERICAL) + len(CATEGORICAL).
        """
        X, _ = get_features_and_target(clean_sample_df)
        pp   = build_preprocessing_pipeline()
        out  = pp.fit_transform(X)
        assert out.shape[0] == X.shape[0]
        assert out.shape[1] == len(ALL_FEATURES)

    def test_numerical_columns_are_scaled(self, clean_sample_df):
        """
        Scaled numerical columns should have near-zero mean
        across a reasonably large sample.
        """
        X, _ = get_features_and_target(clean_sample_df)
        pp   = build_preprocessing_pipeline()
        out  = pp.fit_transform(X)
        # Numerical cols are indices 0..len(NUMERICAL_FEATURES)-1
        n    = len(NUMERICAL_FEATURES)
        col_means = np.abs(out[:, :n].mean(axis=0))
        # Mean should be close to 0 after StandardScaler
        assert np.all(col_means < 5), (
            "Numerical columns do not appear to be standardised."
        )