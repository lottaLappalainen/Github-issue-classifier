"""
src/models/train.py  —  Model Training + MLflow Logging + Registry
Trains 3 classifiers, logs to MLflow, registers the best model in the
MLflow Model Registry, and promotes it to Production if F1 >= threshold.

MLflow Registry lifecycle:
  Staging    → best model from this run, pending validation
  Production → promoted automatically if F1 >= PRODUCTION_THRESHOLD
  Archived   → previous Production models

Usage:
    python src/models/train.py
    python src/models/train.py --gold-version v2
"""
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

ROOT       = Path(__file__).resolve().parents[2]
GOLD_DIR   = ROOT / "data" / "gold"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

LABEL_ORDER          = ["high", "medium", "low"]
REGISTRY_NAME        = "github-issue-priority"
PRODUCTION_THRESHOLD = 0.70


def build_pipeline(
    classifier: str = "logistic_regression",
    max_features: int = 10_000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    n_estimators: int = 100,
) -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=max_features, ngram_range=ngram_range, sublinear_tf=True,
        strip_accents="unicode", analyzer="word", token_pattern=r"\w{2,}", min_df=2,
    )
    if classifier == "logistic_regression":
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", solver="lbfgs")
    elif classifier == "random_forest":
        clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
    elif classifier == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def train_model(pipeline: Pipeline, X: pd.Series, y: pd.Series) -> Pipeline:
    pipeline.fit(X, y)
    return pipeline


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1_high":   float(f1_score(y_true, y_pred, labels=["high"],   average="macro", zero_division=0)),
        "f1_medium": float(f1_score(y_true, y_pred, labels=["medium"], average="macro", zero_division=0)),
        "f1_low":    float(f1_score(y_true, y_pred, labels=["low"],    average="macro", zero_division=0)),
    }


def register_best_model(run_id: str, f1_macro: float, data_version: str) -> Optional[str]:
    """
    Register the best run in the MLflow Model Registry and handle
    staging → production promotion with automatic archiving of the
    previous Production version.

    This implements the governance workflow from the ModelOps lecture:
    model lineage is preserved, every version is tagged with its data
    version and F1, and promotion requires passing the quality gate.
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    try:
        log.info(f"  Registering run {run_id[:8]} → '{REGISTRY_NAME}'")
        mv      = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
        version = mv.version

        # Tag for full traceability (data version ↔ model version)
        client.set_model_version_tag(REGISTRY_NAME, version, "data_version", data_version)
        client.set_model_version_tag(REGISTRY_NAME, version, "f1_macro",     str(round(f1_macro, 4)))

        if f1_macro >= PRODUCTION_THRESHOLD:
            # Archive previous Production versions
            for old in client.get_latest_versions(REGISTRY_NAME, stages=["Production"]):
                log.info(f"  Archiving previous Production v{old.version}")
                client.transition_model_version_stage(
                    name=REGISTRY_NAME, version=old.version,
                    stage="Archived", archive_existing_versions=False,
                )
            client.transition_model_version_stage(
                name=REGISTRY_NAME, version=version, stage="Production",
            )
            log.info(f"  ✅ v{version} → Production  (F1={f1_macro:.4f} ≥ {PRODUCTION_THRESHOLD})")
        else:
            client.transition_model_version_stage(
                name=REGISTRY_NAME, version=version, stage="Staging",
            )
            log.info(f"  ⚠️  v{version} → Staging  (F1={f1_macro:.4f} < {PRODUCTION_THRESHOLD})")

        return version
    except Exception as exc:
        log.warning(f"  Registry step failed (non-fatal): {exc}")
        return None


def run_training(gold_version: Optional[str] = None) -> None:
    log.info("=== Model Training ===")

    train_path = GOLD_DIR / "train.parquet"
    test_path  = GOLD_DIR / "test.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Gold train set not found: {train_path}\nRun featurize.py first.")

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)
    log.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    X_train, y_train = train_df["text"], train_df["priority"]
    X_test,  y_test  = test_df["text"],  test_df["priority"]

    mlflow.set_experiment("github-issue-priority")
    data_version = gold_version or "v1"

    configs = [
        {"classifier": "logistic_regression", "C": 0.5,  "max_features": 5_000},
        {"classifier": "logistic_regression", "C": 1.0,  "max_features": 10_000},
        {"classifier": "random_forest",       "n_estimators": 100, "max_features": 10_000},
    ]

    best_f1 = -1.0; best_model = None; best_run_id = None

    for cfg in configs:
        clf_name = cfg["classifier"]
        log.info(f"\nTraining: {clf_name} | config: {cfg}")

        with mlflow.start_run(run_name=f"{clf_name}__{data_version}") as run:
            mlflow.set_tag("data_version", data_version)
            mlflow.set_tag("classifier",   clf_name)
            mlflow.log_params(cfg)

            pipeline = build_pipeline(**cfg)
            trained  = train_model(pipeline, X_train, y_train)

            cv_scores = cross_val_score(
                build_pipeline(**cfg), X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1
            )
            mlflow.log_metric("cv_f1_macro_mean", float(cv_scores.mean()))
            mlflow.log_metric("cv_f1_macro_std",  float(cv_scores.std()))

            preds   = trained.predict(X_test)
            metrics = compute_metrics(y_test, preds)
            mlflow.log_metrics(metrics)

            report      = classification_report(y_test, preds, labels=LABEL_ORDER, zero_division=0)
            report_path = MODELS_DIR / f"report_{clf_name}.txt"
            report_path.write_text(report)
            mlflow.log_artifact(str(report_path))

            log.info(f"  F1 macro: {metrics['f1_macro']:.4f}  accuracy: {metrics['accuracy']:.4f}")
            log.info(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            mlflow.sklearn.log_model(trained, artifact_path="model")

            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]; best_model = trained; best_run_id = run.info.run_id

    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_path)
    log.info(f"\nBest model → {best_path}  (F1={best_f1:.4f}, run={best_run_id})")

    registry_version = register_best_model(
        run_id=best_run_id, f1_macro=best_f1, data_version=data_version
    )

    metrics_out = {
        "f1_macro":         best_f1,
        "best_run_id":      best_run_id,
        "data_version":     data_version,
        "registry_version": registry_version,
        "registry_name":    REGISTRY_NAME,
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    log.info("metrics.json written")
    if registry_version:
        log.info(f"Registry: '{REGISTRY_NAME}' v{registry_version} — mlflow ui → http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-version", default=None, help="e.g. v1, v2")
    args = parser.parse_args()
    run_training(gold_version=args.gold_version)