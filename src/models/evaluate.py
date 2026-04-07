"""
src/models/evaluate.py  —  Model Evaluation + Version Comparison
Loads the best saved model, runs inference, compares against previous
MLflow runs, and optionally writes predictions to CSV.

Usage:
    python src/models/evaluate.py
    python src/models/evaluate.py --input data/test.csv --output data/predictions.csv
    python src/models/evaluate.py --compare-runs   # print MLflow run comparison table
"""
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
GOLD_DIR   = ROOT / "data" / "gold"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

LABEL_ORDER = ["high", "medium", "low"]


# ── public helpers ─────────────────────────────────────────────────────────

def load_model(models_dir: Path = MODELS_DIR):
    """Load the best saved model from disk."""
    model_path = models_dir / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}\nRun train.py first."
        )
    model = joblib.load(model_path)
    log.info(f"Loaded model from {model_path}")
    return model


def evaluate_on_gold(model, gold_dir: Path = GOLD_DIR) -> dict:
    """
    Run evaluation on the Gold test split.
    Returns a metrics dict.
    """
    test_path = gold_dir / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Gold test set not found: {test_path}\nRun featurize.py first."
        )

    test_df = pd.read_parquet(test_path)
    X_test  = test_df["text"]
    y_test  = test_df["priority"]

    preds   = model.predict(X_test)
    metrics = {
        "f1_macro":  float(f1_score(y_test, preds, average="macro",  zero_division=0)),
        "f1_high":   float(f1_score(y_test, preds, labels=["high"],   average="macro", zero_division=0)),
        "f1_medium": float(f1_score(y_test, preds, labels=["medium"], average="macro", zero_division=0)),
        "f1_low":    float(f1_score(y_test, preds, labels=["low"],    average="macro", zero_division=0)),
        "accuracy":  float(accuracy_score(y_test, preds)),
        "n_test":    len(test_df),
    }

    log.info(f"\n{'='*50}")
    log.info(f"  F1 macro  : {metrics['f1_macro']:.4f}")
    log.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    log.info(f"  F1 high   : {metrics['f1_high']:.4f}")
    log.info(f"  F1 medium : {metrics['f1_medium']:.4f}")
    log.info(f"  F1 low    : {metrics['f1_low']:.4f}")
    log.info(f"\n{classification_report(y_test, preds, labels=LABEL_ORDER, zero_division=0)}")

    return metrics


def run_inference(
    model,
    input_path: Path,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run inference on a CSV file with 'title' and 'body' columns.
    Saves predictions CSV if output_path is given.
    Returns a dataframe with predictions.
    """
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df):,} rows from {input_path}")

    if "text" not in df.columns:
        if "title" not in df.columns:
            raise ValueError("Input CSV must have a 'title' column (and optionally 'body')")
        df["text"] = df["title"].fillna("") + " " + df.get("body", pd.Series("")).fillna("")

    df["predicted_priority"] = model.predict(df["text"])

    # Add confidence if model supports predict_proba
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(df["text"])
        df["confidence"] = probas.max(axis=1).round(4)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        log.info(f"Predictions saved → {output_path}")

    log.info(f"Prediction distribution:\n{df['predicted_priority'].value_counts().to_string()}")
    return df


def compare_mlflow_runs(
    experiment_name: str = "github-issue-priority",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Pull the top N MLflow runs for the experiment and print a comparison table.
    Useful for tracking model improvement across data versions.
    """
    try:
        mlflow.set_tracking_uri(str(ROOT / "mlruns"))
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            log.warning(f"No MLflow experiment named '{experiment_name}' found.")
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_macro DESC"],
            max_results=top_n,
        )

        if runs.empty:
            log.warning("No runs found.")
            return pd.DataFrame()

        cols = ["run_id", "tags.data_version", "tags.classifier",
                "metrics.f1_macro", "metrics.accuracy",
                "metrics.cv_f1_macro_mean", "metrics.cv_f1_macro_std"]
        cols = [c for c in cols if c in runs.columns]
        summary = runs[cols].copy()

        # Shorten run_id for readability
        summary["run_id"] = summary["run_id"].str[:8]

        log.info(f"\n{'='*70}")
        log.info("MLflow Run Comparison (sorted by F1 macro):")
        log.info(f"\n{summary.to_string(index=False)}")
        log.info(f"{'='*70}\n")

        return summary

    except Exception as e:
        log.warning(f"MLflow comparison failed: {e}")
        return pd.DataFrame()


def check_improvement(
    current_metrics: dict,
    threshold: float = 0.60,
    metrics_path: Path = ROOT / "metrics.json",
) -> bool:
    """
    Compare current run against the last saved metrics.json.
    Returns True if the model improved or meets threshold.
    Used as a CI/CD quality gate.
    """
    f1 = current_metrics.get("f1_macro", 0.0)

    # Load previous best if it exists
    if metrics_path.exists():
        prev = json.loads(metrics_path.read_text())
        prev_f1 = prev.get("f1_macro", 0.0)
        log.info(f"Previous best F1: {prev_f1:.4f} | Current: {f1:.4f}")
        if f1 >= prev_f1:
            log.info("✅ Model improved or matched — promoting.")
        else:
            log.warning(f"⚠️  Model regressed ({f1:.4f} < {prev_f1:.4f}).")
    else:
        log.info(f"No previous metrics found. Current F1: {f1:.4f}")

    passed = f1 >= threshold
    log.info(f"Quality gate ({threshold}): {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


# ── main ───────────────────────────────────────────────────────────────────

def main(
    input_path: Optional[Path],
    output_path: Optional[Path],
    compare_runs: bool,
) -> None:
    log.info("=== Model Evaluation ===")

    model = load_model()

    # 1. Evaluate on Gold test split
    metrics = evaluate_on_gold(model)

    # 2. Check improvement / quality gate
    check_improvement(metrics)

    # 3. Run inference on external CSV if provided
    if input_path:
        run_inference(model, input_path=input_path, output_path=output_path)

    # 4. Print MLflow run comparison table
    if compare_runs:
        compare_mlflow_runs()

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--input",        type=Path, default=None,
                        help="Path to input CSV (title, body columns)")
    parser.add_argument("--output",       type=Path, default=None,
                        help="Path to save predictions CSV")
    parser.add_argument("--compare-runs", action="store_true",
                        help="Print MLflow run comparison table")
    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        compare_runs=args.compare_runs,
    )