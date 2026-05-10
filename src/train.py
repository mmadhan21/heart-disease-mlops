"""
=============================================================
Heart Disease MLOps Pipeline
File    : src/train.py
Purpose : Train Logistic Regression and Random Forest models,
          track all experiments with MLflow, evaluate both
          models, select the best one, and save artefacts.
Run     : python src/train.py
          mlflow ui          (to inspect runs in browser)
=============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for CI
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ── Add project root to path so src.preprocess is importable ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import (
    load_data,
    clean_data,
    get_features_and_target,
    build_preprocessing_pipeline,
    split_data,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

DATA_PATH    = os.path.join("data", "heart_cleaned.csv")
MODEL_DIR    = "models"
SCREENSHOTS  = "screenshots"
MLFLOW_URI   = "mlruns"
EXPERIMENT   = "Heart_Disease_Classification"

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(SCREENSHOTS, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


# ─────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_test,
    y_pred,
    model_name: str,
    save_dir: str = SCREENSHOTS
) -> str:
    """
    Plot a colour-coded confusion matrix and save to disk.
    Returns the saved file path (logged as MLflow artefact).
    """
    cm     = confusion_matrix(y_test, y_pred)
    labels = ["No Disease", "Disease"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im      = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1]);  ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {model_name}",
        fontsize=13, fontweight="bold"
    )

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    fname = os.path.join(
        save_dir, f"cm_{model_name.replace(' ', '_')}.png"
    )
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_roc_curve(
    y_test,
    y_prob,
    model_name: str,
    save_dir: str = SCREENSHOTS
) -> str:
    """
    Plot the ROC curve and save to disk.
    Returns the saved file path.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val      = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        fpr, tpr, color="darkorange", lw=2,
        label=f"ROC Curve  (AUC = {auc_val:.3f})"
    )
    ax.plot(
        [0, 1], [0, 1], color="navy", lw=1,
        linestyle="--", label="Random Classifier"
    )
    ax.fill_between(fpr, tpr, alpha=0.08, color="darkorange")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(
        f"ROC Curve — {model_name}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(
        save_dir, f"roc_{model_name.replace(' ', '_')}.png"
    )
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_feature_importance(
    pipeline,
    feature_names: list,
    model_name: str,
    save_dir: str = SCREENSHOTS
) -> str:
    """
    Plot feature importances for Random Forest.
    Falls back silently for models that don't support it.
    Returns the saved file path or empty string.
    """
    try:
        classifier   = pipeline.named_steps["classifier"]
        importances  = classifier.feature_importances_
        indices      = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_vals  = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            sorted_names[::-1], sorted_vals[::-1],
            color="steelblue", edgecolor="white"
        )
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(
            f"Feature Importances — {model_name}",
            fontsize=13, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(
            save_dir,
            f"feat_importance_{model_name.replace(' ', '_')}.png"
        )
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        return fname
    except AttributeError:
        return ""


# ─────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────

def evaluate_model(pipeline, X_test, y_test, model_name: str):
    """
    Compute all classification metrics.

    Returns
    -------
    metrics : dict  { metric_name: value }
    y_pred  : np.array
    y_prob  : np.array  (probability of class 1)
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"  : round(float(accuracy_score (y_test, y_pred)),            4),
        "precision" : round(float(precision_score(y_test, y_pred)),             4),
        "recall"    : round(float(recall_score   (y_test, y_pred)),             4),
        "f1_score"  : round(float(f1_score       (y_test, y_pred)),             4),
        "roc_auc"   : round(float(roc_auc_score  (y_test, y_prob)),             4),
    }

    print(f"\n")
    print(f"  {model_name}  —  Evaluation Results")
    print(f"\n")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print()
    print(classification_report(y_test, y_pred,
          target_names=["No Disease", "Disease"]))

    return metrics, y_pred, y_prob


# ─────────────────────────────────────────────────────────────
# CORE TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────

def train_single_model(
    model_name:   str,
    classifier,
    params:       dict,
    X_train:      pd.DataFrame,
    X_test:       pd.DataFrame,
    y_train:      pd.Series,
    y_test:       pd.Series,
    feature_names: list,
) -> tuple:
    """
    Build a full Pipeline (preprocessor + classifier),
    train it, evaluate it, and log everything to MLflow.

    Returns
    -------
    fitted_pipeline, metrics_dict
    """
    # Fresh preprocessor for every model
    preprocessor = build_preprocessing_pipeline()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])

    with mlflow.start_run(run_name=model_name):

        # ── Log hyper-parameters ─────────────────────────
        mlflow.log_params(params)
        mlflow.log_param("model_name",   model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples",  len(X_test))
        mlflow.log_param("n_features",    len(X_train.columns))
        mlflow.log_param("cv_folds",      5)

        # ── Train ────────────────────────────────────────
        print(f"\n[INFO] Training  →  {model_name} ...")
        pipeline.fit(X_train, y_train)

        # ── Cross-validation ─────────────────────────────
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=5, scoring="roc_auc", n_jobs=-1
        )
        mlflow.log_metric("cv_roc_auc_mean", round(float(cv_scores.mean()), 4))
        mlflow.log_metric("cv_roc_auc_std",  round(float(cv_scores.std()),  4))
        print(
            f"[INFO] 5-Fold CV ROC-AUC : "
            f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        # ── Evaluate ─────────────────────────────────────
        metrics, y_pred, y_prob = evaluate_model(
            pipeline, X_test, y_test, model_name
        )
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        # ── Plots & artefacts ─────────────────────────────
        cm_path   = plot_confusion_matrix(y_test, y_pred, model_name)
        roc_path  = plot_roc_curve(y_test, y_prob, model_name)
        feat_path = plot_feature_importance(pipeline, feature_names, model_name)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        if feat_path:
            mlflow.log_artifact(feat_path)

        # ── Save model as .pkl ────────────────────────────
        pkl_path = os.path.join(
            MODEL_DIR, f"{model_name.replace(' ', '_')}.pkl"
        )
        joblib.dump(pipeline, pkl_path)
        mlflow.log_artifact(pkl_path)

        # ── Register model in MLflow Model Registry ───────
        mlflow.sklearn.log_model(
            sk_model        = pipeline,
            artifact_path   = model_name.replace(" ", "_"),
            registered_model_name = model_name,
        )

        run_id = mlflow.active_run().info.run_id
        print(f"[INFO] MLflow run ID : {run_id}")

    return pipeline, metrics


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  HEART DISEASE — MODEL TRAINING PIPELINE")
    print("=" * 55)

    # ── Load & prepare data ──────────────────────────────
    df      = load_data(DATA_PATH)
    df      = clean_data(df)
    X, y    = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_names = list(X.columns)

    # ─────────────────────────────────────────────────────
    # MODEL 1 : Logistic Regression
    # ─────────────────────────────────────────────────────
    lr_params = {
        "C"            : 1.0,
        "max_iter"     : 1000,
        "solver"       : "lbfgs",
        "random_state" : 42,
    }
    lr_pipeline, lr_metrics = train_single_model(
        model_name    = "Logistic_Regression",
        classifier    = LogisticRegression(**lr_params),
        params        = lr_params,
        X_train       = X_train,
        X_test        = X_test,
        y_train       = y_train,
        y_test        = y_test,
        feature_names = feature_names,
    )

    # ─────────────────────────────────────────────────────
    # MODEL 2 : Random Forest
    # ─────────────────────────────────────────────────────
    rf_params = {
        "n_estimators"    : 100,
        "max_depth"       : 6,
        "min_samples_split": 4,
        "random_state"    : 42,
    }
    rf_pipeline, rf_metrics = train_single_model(
        model_name    = "Random_Forest",
        classifier    = RandomForestClassifier(**rf_params),
        params        = rf_params,
        X_train       = X_train,
        X_test        = X_test,
        y_train       = y_train,
        y_test        = y_test,
        feature_names = feature_names,
    )

    # ─────────────────────────────────────────────────────
    # MODEL SELECTION
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    print(f"  Logistic Regression  ROC-AUC : {lr_metrics['roc_auc']}")
    print(f"  Random Forest        ROC-AUC : {rf_metrics['roc_auc']}")

    if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"]:
        best_pipeline = rf_pipeline
        best_name     = "Random_Forest"
        best_metrics  = rf_metrics
    else:
        best_pipeline = lr_pipeline
        best_name     = "Logistic_Regression"
        best_metrics  = lr_metrics

    print(f"\n  ✅  Best model selected : {best_name}")
    print(f"      ROC-AUC             : {best_metrics['roc_auc']}")
    print(f"      Accuracy            : {best_metrics['accuracy']}")

    # Save best model
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_pipeline, best_path)
    print(f"\n[INFO] Best model saved → {best_path}")

    print("\n[DONE] Training complete.")
    print(f"[INFO] Run  'mlflow ui'  and open http://localhost:5000\n")


if __name__ == "__main__":
    main()