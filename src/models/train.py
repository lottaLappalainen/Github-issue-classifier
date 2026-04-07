"""
train.py — Model Training
Trains multiple classifiers, logs everything to MLflow,
and registers the best model.
"""

import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

GOLD_DIR    = Path("data/gold")
MODEL_DIR   = Path("models")
PARAMS_FILE = Path("params.yaml")

MLFLOW_EXPERIMENT = "github-issue-priority"


# ── Load Gold data ─────────────────────────────────────────────────────────────

def load_gold():
    X_train = joblib.load(GOLD_DIR / "X_train.joblib")
    X_test  = joblib.load(GOLD_DIR / "X_test.joblib")
    y_train = joblib.load(GOLD_DIR / "y_train.joblib")
    y_test  = joblib.load(GOLD_DIR / "y_test.joblib")
    le      = joblib.load(GOLD_DIR / "label_encoder.joblib")

    with open(GOLD_DIR / "meta.json") as f:
        meta = json.load(f)

    return X_train, X_test, y_train, y_test, le, meta


# ── Models to try ─────────────────────────────────────────────────────────────
# We train all three and MLflow tracks them.
# The best F1 (macro) wins and gets registered.

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    ),
}


# ── Train + log one model ──────────────────────────────────────────────────────

def train_and_log(name, model, X_train, X_test, y_train, y_test, le, meta):
    with mlflow.start_run(run_name=name):

        # Log params
        mlflow.log_param("model_type",  name)
        mlflow.log_param("n_train",     meta["n_train"])
        mlflow.log_param("n_test",      meta["n_test"])
        mlflow.log_param("n_features",  meta["n_features"])
        mlflow.log_params(model.get_params())

        # Train
        print(f"  Training {name}...")
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        f1        = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
        accuracy  = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("f1_macro",   f1)
        mlflow.log_metric("precision",  precision)
        mlflow.log_metric("recall",     recall)
        mlflow.log_metric("accuracy",   accuracy)

        # Per-class metrics
        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            output_dict=True,
        )
        for cls in le.classes_:
            mlflow.log_metric(f"f1_{cls}",        report[cls]["f1-score"])
            mlflow.log_metric(f"precision_{cls}", report[cls]["precision"])
            mlflow.log_metric(f"recall_{cls}",    report[cls]["recall"])

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Print summary
        print(f"    F1 (macro): {f1:.4f}  |  Accuracy: {accuracy:.4f}")
        print(f"    {classification_report(y_test, y_pred, target_names=le.classes_)}")

        return mlflow.active_run().info.run_id, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔄 Starting model training...")
    print()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, X_test, y_train, y_test, le, meta = load_gold()

    results = []
    for name, model in MODELS.items():
        run_id, f1 = train_and_log(
            name, model,
            X_train, X_test,
            y_train, y_test,
            le, meta,
        )
        results.append((name, run_id, f1))

    # Find the best model
    best_name, best_run_id, best_f1 = max(results, key=lambda x: x[2])

    print()
    print("📊 Results summary:")
    for name, _, f1 in sorted(results, key=lambda x: -x[2]):
        marker = " ← best" if name == best_name else ""
        print(f"   {name:30s}: F1={f1:.4f}{marker}")

    # Save best model locally for serving
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model = MODELS[best_name]

    joblib.dump(best_model,                             MODEL_DIR / "model.joblib")
    joblib.dump(joblib.load(GOLD_DIR / "vectorizer.joblib"), MODEL_DIR / "vectorizer.joblib")
    joblib.dump(joblib.load(GOLD_DIR / "label_encoder.joblib"), MODEL_DIR / "label_encoder.joblib")

    # Save best model metadata
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump({
            "best_model":  best_name,
            "best_run_id": best_run_id,
            "f1_macro":    best_f1,
        }, f, indent=2)

    print()
    print(f"✅ Training complete. Best model: {best_name} (F1={best_f1:.4f})")
    print(f"   Saved to {MODEL_DIR.resolve()}")
    print(f"   MLflow UI: run `mlflow ui` to explore experiments")


if __name__ == "__main__":
    main()