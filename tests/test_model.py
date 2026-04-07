"""
Tests for src/models/train.py and the saved model artifact.
Run with: python -m pytest tests/test_model.py -v
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import joblib
import tempfile
import os


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gold_df():
    """Minimal Gold-layer style dataframe for testing."""
    texts = [
        "App crashes on startup bug critical",
        "Add dark mode enhancement nice feature",
        "Update README documentation typo fix",
        "Login fails on mobile bug regression",
        "Improve performance enhancement speed",
        "Fix broken link docs good first issue",
        "Null pointer exception bug crash critical",
        "Add search feature enhancement request",
        "Spelling mistake in docs documentation",
    ]
    labels = ["high", "medium", "low", "high", "medium", "low",
              "high", "medium", "low"]
    return pd.DataFrame({"text": texts, "priority": labels})


@pytest.fixture
def trained_model(gold_df, tmp_path):
    """Train a quick model and return (pipeline, model_path)."""
    from src.models.train import build_pipeline, train_model
    pipeline = build_pipeline()
    X = gold_df["text"]
    y = gold_df["priority"]
    trained = train_model(pipeline, X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(trained, model_path)
    return trained, model_path


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

class TestBuildPipeline:

    def test_pipeline_has_tfidf(self):
        from src.models.train import build_pipeline
        from sklearn.pipeline import Pipeline
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)
        step_names = [name for name, _ in pipeline.steps]
        assert "tfidf" in step_names

    def test_pipeline_has_classifier(self):
        from src.models.train import build_pipeline
        pipeline = build_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert "clf" in step_names

    def test_pipeline_is_fittable(self, gold_df):
        from src.models.train import build_pipeline
        pipeline = build_pipeline()
        pipeline.fit(gold_df["text"], gold_df["priority"])  # should not raise


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTrainModel:

    def test_model_trains_without_error(self, gold_df):
        from src.models.train import build_pipeline, train_model
        pipeline = build_pipeline()
        train_model(pipeline, gold_df["text"], gold_df["priority"])

    def test_model_predicts_valid_labels(self, gold_df):
        from src.models.train import build_pipeline, train_model
        pipeline = build_pipeline()
        trained = train_model(pipeline, gold_df["text"], gold_df["priority"])
        preds = trained.predict(gold_df["text"])
        valid = {"high", "medium", "low"}
        assert set(preds).issubset(valid)

    def test_model_returns_probabilities(self, gold_df):
        from src.models.train import build_pipeline, train_model
        pipeline = build_pipeline()
        trained = train_model(pipeline, gold_df["text"], gold_df["priority"])
        proba = trained.predict_proba(gold_df["text"])
        assert proba.shape[0] == len(gold_df)
        assert proba.shape[1] == 3  # high / medium / low
        # probabilities must sum to ~1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_metrics_dict_has_required_keys(self, gold_df):
        from src.models.train import build_pipeline, train_model, compute_metrics
        pipeline = build_pipeline()
        trained = train_model(pipeline, gold_df["text"], gold_df["priority"])
        preds = trained.predict(gold_df["text"])
        metrics = compute_metrics(gold_df["priority"], preds)
        assert "f1_macro" in metrics
        assert "accuracy" in metrics

    def test_f1_is_between_0_and_1(self, gold_df):
        from src.models.train import build_pipeline, train_model, compute_metrics
        pipeline = build_pipeline()
        trained = train_model(pipeline, gold_df["text"], gold_df["priority"])
        preds = trained.predict(gold_df["text"])
        metrics = compute_metrics(gold_df["priority"], preds)
        assert 0.0 <= metrics["f1_macro"] <= 1.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestModelPersistence:

    def test_model_saved_as_joblib(self, trained_model):
        _, model_path = trained_model
        assert model_path.exists()

    def test_saved_model_loads_and_predicts(self, trained_model):
        trained, model_path = trained_model
        loaded = joblib.load(model_path)
        preds = loaded.predict(["app crashes on login bug"])
        assert preds[0] in {"high", "medium", "low"}

    def test_predictions_consistent_after_reload(self, trained_model, gold_df):
        trained, model_path = trained_model
        loaded = joblib.load(model_path)
        original_preds = trained.predict(gold_df["text"])
        loaded_preds = loaded.predict(gold_df["text"])
        assert list(original_preds) == list(loaded_preds)


# ---------------------------------------------------------------------------
# Production inference on test.csv  (Assignment requirement)
# ---------------------------------------------------------------------------

class TestProductionInference:
    """
    The assignment explicitly requires running test.csv through the model
    and providing the result. These tests validate that inference works
    on the expected input format.
    """

    def test_inference_on_csv_format(self, trained_model, tmp_path):
        """Simulate test.csv → predictions output."""
        trained, _ = trained_model

        # Create a mock test.csv
        test_data = pd.DataFrame({
            "title": ["App crashes on load", "Add export button", "Fix typo in docs"],
            "body": ["Crash on startup", "Would be useful", "Small fix"],
        })
        test_csv = tmp_path / "test.csv"
        test_data.to_csv(test_csv, index=False)

        # Load and run inference
        df = pd.read_csv(test_csv)
        df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
        preds = trained.predict(df["text"])

        assert len(preds) == len(df)
        assert all(p in {"high", "medium", "low"} for p in preds)

    def test_predictions_saved_to_csv(self, trained_model, tmp_path):
        """Output CSV must have title, body, predicted_priority columns."""
        trained, _ = trained_model

        test_data = pd.DataFrame({
            "title": ["Login bug", "Dark mode", "Fix docs"],
            "body":  ["Crash", "Feature", "Typo"],
        })
        test_data["text"] = test_data["title"] + " " + test_data["body"]
        test_data["predicted_priority"] = trained.predict(test_data["text"])

        out = tmp_path / "predictions.csv"
        test_data.to_csv(out, index=False)

        loaded = pd.read_csv(out)
        assert "predicted_priority" in loaded.columns
        assert len(loaded) == 3