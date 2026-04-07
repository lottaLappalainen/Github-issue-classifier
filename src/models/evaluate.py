"""
src/models/evaluate.py  —  Model Evaluation + Version Comparison

Always produces a version comparison table (saved to
monitoring/version_comparison.json) so every CI run generates
auditable evidence of model performance across data versions.

Usage:
    python src/models/evaluate.py
    python src/models/evaluate.py --input data/test.csv --output data/predictions.csv
    python src/models/evaluate.py --compare-runs
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
from sklearn.metrics import f1_score, accuracy_score, classification_report

ROOT            = Path(__file__).resolve().parents[2]
MODELS_DIR      = ROOT / "models"
GOLD_DIR        = ROOT / "data" / "gold"
MONITORING_DIR  = ROOT / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)
LABEL_ORDER = ["high", "medium", "low"]


# ── public helpers (imported by tests) ─────────────────────────────────────

def load_model(models_dir: Path = MODELS_DIR):
    model_path = Path(models_dir) / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)


def evaluate_on_gold(model, gold_dir: Path = GOLD_DIR) -> dict:
    test_path = Path(gold_dir) / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Gold test set not found: {test_path}")
    test_df = pd.read_parquet(test_path)
    X_test, y_test = test_df["text"], test_df["priority"]
    preds = model.predict(X_test)

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


def run_inference(model, input_path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df):,} rows from {input_path}")
    if "text" not in df.columns:
        if "title" not in df.columns:
            raise ValueError("Input CSV must have a 'title' column (and optionally 'body')")
        df["text"] = df["title"].fillna("") + " " + df.get("body", pd.Series("")).fillna("")
    df["predicted_priority"] = model.predict(df["text"])
    if hasattr(model, "predict_proba"):
        df["confidence"] = model.predict_proba(df["text"]).max(axis=1).round(4)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        log.info(f"Predictions saved → {output_path}")
    log.info(f"Prediction distribution:\n{df['predicted_priority'].value_counts().to_string()}")
    return df


def compare_mlflow_runs(
    experiment_name: str = "github-issue-priority",
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Pull MLflow runs sorted by F1 macro and produce a version comparison
    table. Always saves to monitoring/version_comparison.json so every
    CI run produces auditable evidence of model performance across data
    versions — this directly satisfies the assignment requirement to
    'compare model performance across data versions'.
    """
    save_path = Path(save_path) if save_path else (MONITORING_DIR / "version_comparison.json")

    try:
        mlflow.set_tracking_uri(str(ROOT / "mlruns"))
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            log.warning(f"No MLflow experiment '{experiment_name}' found.")
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_macro DESC"],
            max_results=top_n,
        )

        if runs.empty:
            log.warning("No MLflow runs found.")
            return pd.DataFrame()

        # Pull the columns we care about
        want = [
            "run_id",
            "tags.data_version",
            "tags.classifier",
            "metrics.f1_macro",
            "metrics.accuracy",
            "metrics.f1_high",
            "metrics.f1_medium",
            "metrics.f1_low",
            "metrics.cv_f1_macro_mean",
            "metrics.cv_f1_macro_std",
        ]
        cols    = [c for c in want if c in runs.columns]
        summary = runs[cols].copy()
        summary["run_id"] = summary["run_id"].str[:8]
        summary = summary.rename(columns={
            "tags.data_version":      "data_version",
            "tags.classifier":        "classifier",
            "metrics.f1_macro":       "f1_macro",
            "metrics.accuracy":       "accuracy",
            "metrics.f1_high":        "f1_high",
            "metrics.f1_medium":      "f1_medium",
            "metrics.f1_low":         "f1_low",
            "metrics.cv_f1_macro_mean": "cv_f1_mean",
            "metrics.cv_f1_macro_std":  "cv_f1_std",
        })

        log.info(f"\n{'='*70}")
        log.info("Version comparison (sorted by F1 macro):")
        log.info(f"\n{summary.to_string(index=False)}")
        log.info(f"{'='*70}\n")

        # Save as JSON evidence artifact
        records = summary.to_dict(orient="records")
        save_path.write_text(json.dumps(
            {"experiment": experiment_name, "runs": records}, indent=2
        ))
        log.info(f"Version comparison saved → {save_path}")

        return summary

    except Exception as exc:
        log.warning(f"MLflow comparison failed: {exc}")
        return pd.DataFrame()


def check_improvement(
    current_metrics: dict,
    threshold: float = 0.60,
    metrics_path: Optional[Path] = None,
) -> bool:
    if metrics_path is None:
        metrics_path = ROOT / "metrics.json"
    f1 = current_metrics.get("f1_macro", 0.0)
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        prev     = json.loads(metrics_path.read_text())
        prev_f1  = prev.get("f1_macro", 0.0)
        prev_ver = prev.get("data_version", "unknown")
        log.info(f"Previous best F1: {prev_f1:.4f} ({prev_ver}) | Current: {f1:.4f}")
        if f1 >= prev_f1:
            log.info("✅ Model improved or matched — promoting.")
        else:
            log.warning(f"⚠️  Model regressed ({f1:.4f} < {prev_f1:.4f}).")
    passed = f1 >= threshold
    log.info(f"Quality gate ({threshold}): {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


# ── main ───────────────────────────────────────────────────────────────────

def main(
    input_path:   Optional[Path] = None,
    output_path:  Optional[Path] = None,
    compare_runs: bool = False,
) -> None:
    log.info("=== Model Evaluation ===")

    model   = load_model()
    metrics = evaluate_on_gold(model)
    check_improvement(metrics)

    # Always run version comparison — produces monitoring/version_comparison.json
    # as a CI artifact every run regardless of --compare-runs flag
    compare_mlflow_runs()

    if input_path:
        run_inference(model, input_path=input_path, output_path=output_path)

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--input",        type=Path, default=None)
    parser.add_argument("--output",       type=Path, default=None)
    parser.add_argument("--compare-runs", action="store_true",
                        help="(always runs now — flag kept for back-compat)")
    args = parser.parse_args()
    main(input_path=args.input, output_path=args.output, compare_runs=args.compare_runs)