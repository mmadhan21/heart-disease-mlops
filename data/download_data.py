"""
=============================================================
Heart Disease MLOps Pipeline
File    : data/download_data.py
Purpose : Downloads the Heart Disease UCI dataset,
          adds column headers, binarizes the target,
          handles missing values, and saves a cleaned CSV.
Run     : python data/download_data.py
=============================================================
"""

import urllib.request
import os
import pandas as pd


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

RAW_PATH     = os.path.join("data", "heart_raw.csv")
CLEANED_PATH = os.path.join("data", "heart_cleaned.csv")

# Column names as defined in the UCI documentation
COLUMNS = [
    "age",      # Age in years
    "sex",      # 1 = male, 0 = female
    "cp",       # Chest pain type (0‑3)
    "trestbps", # Resting blood pressure (mm Hg)
    "chol",     # Serum cholesterol (mg/dl)
    "fbs",      # Fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",  # Resting ECG results (0‑2)
    "thalach",  # Maximum heart rate achieved
    "exang",    # Exercise induced angina (1 = yes)
    "oldpeak",  # ST depression induced by exercise
    "slope",    # Slope of peak exercise ST segment
    "ca",       # Number of major vessels (0‑3)
    "thal",     # Thalassemia type
    "target"    # Diagnosis (0 = no disease, 1‑4 = disease)
]


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def ensure_data_dir():
    """Create the data/ directory if it does not exist."""
    os.makedirs("data", exist_ok=True)
    print("[INFO] data/ directory ready.")


def download_raw(url: str, save_path: str) -> None:
    """
    Download the raw CSV file from UCI repository.
    The file has no header row and uses '?' for missing values.
    """
    print(f"[INFO] Downloading dataset from:\n       {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"[INFO] Raw file saved → {save_path}")


def load_raw(path: str) -> pd.DataFrame:
    """
    Read the raw file into a DataFrame.
    '?' is treated as NaN automatically via na_values.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values="?"
    )
    print(f"[INFO] Raw shape       : {df.shape}")
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    The original target has values 0‑4.
    Convert to binary:
        0  → 0  (no disease)
        1‑4 → 1  (disease present)
    """
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)
    print("[INFO] Target binarized: 0 = No Disease | 1 = Disease")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with the column median.
    Only 'ca' and 'thal' have missing values in this dataset.
    """
    missing_before = df.isnull().sum()
    cols_with_na   = missing_before[missing_before > 0].index.tolist()

    if not cols_with_na:
        print("[INFO] No missing values found.")
        return df

    for col in cols_with_na:
        median_val = df[col].median()
        df[col]    = df[col].fillna(median_val)
        print(
            f"[INFO] Filled '{col}' — "
            f"{missing_before[col]} NaN(s) replaced with median={median_val}"
        )

    return df


def save_cleaned(df: pd.DataFrame, path: str) -> None:
    """Save the cleaned DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"[INFO] Cleaned file saved → {path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the cleaned dataset."""
    print("\n" + "=" * 50)
    print("  DATASET SUMMARY")
    print("=" * 50)
    print(f"  Rows      : {df.shape[0]}")
    print(f"  Columns   : {df.shape[1]}")
    print(f"  Missing   : {df.isnull().sum().sum()}")
    print(f"  Target 0  : {(df['target'] == 0).sum()}  (No Disease)")
    print(f"  Target 1  : {(df['target'] == 1).sum()}  (Disease)")
    print("=" * 50)
    print(df.dtypes.to_string())
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ensure_data_dir()
    download_raw(URL, RAW_PATH)

    df = load_raw(RAW_PATH)
    df = binarize_target(df)
    df = handle_missing(df)

    save_cleaned(df, CLEANED_PATH)
    print_summary(df)

    print("[DONE] Dataset ready for EDA and training.\n")


if __name__ == "__main__":
    main()