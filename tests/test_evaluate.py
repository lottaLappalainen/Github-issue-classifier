import json
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def trained_model(tmp_path):
    from src.models.train import build_pipeline, train_model
    texts  = ["App crash bug"] * 4 + ["Add feature enhancement"] * 4 + ["Fix docs documentation"] * 4
    labels = ["high"] * 4 + ["medium"] * 4 + ["low"] * 4
    pipeline = build_pipeline()
    trained  = train_model(pipeline, pd.Series(texts), pd.Series(labels))
    path = tmp_path / "models" / "best_model.joblib"
    path.parent.mkdir()
    joblib.dump(trained, path)
    return trained, path


@pytest.fixture
def gold_test_dir(tmp_path, trained_model):
    """Write a small Gold test.parquet for evaluate_on_gold tests."""
    trained, _ = trained_model
    df = pd.DataFrame({
        "text":     ["app crash bug", "add feature enhancement", "fix docs typo"],
        "priority": ["high",          "medium",                  "low"],
    })
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    df.to_parquet(gold_dir / "test.parquet", index=False)
    return gold_dir


@pytest.fixture
def test_csv(tmp_path):
    df = pd.DataFrame({
        "title": ["App crashes", "Add feature", "Fix docs"],
        "body":  ["on startup",  "would help",  "typo fix"],
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:

    def test_loads_successfully(self, trained_model, tmp_path):
        from src.models.evaluate import load_model
        _, model_path = trained_model
        loaded = load_model(model_path.parent)
        assert loaded is not None

    def test_loaded_model_can_predict(self, trained_model, tmp_path):
        from src.models.evaluate import load_model
        _, model_path = trained_model
        loaded = load_model(model_path.parent)
        preds = loaded.predict(["app crash bug"])
        assert preds[0] in {"high", "medium", "low"}

    def test_missing_model_raises_file_not_found(self, tmp_path):
        from src.models.evaluate import load_model
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent_dir")

    def test_returns_sklearn_pipeline(self, trained_model):
        from src.models.evaluate import load_model
        from sklearn.pipeline import Pipeline
        _, model_path = trained_model
        assert isinstance(load_model(model_path.parent), Pipeline)


# ---------------------------------------------------------------------------
# evaluate_on_gold
# ---------------------------------------------------------------------------

class TestEvaluateOnGold:

    def test_returns_dict(self, trained_model, gold_test_dir):
        from src.models.evaluate import evaluate_on_gold
        trained, _ = trained_model
        metrics = evaluate_on_gold(trained, gold_dir=gold_test_dir)
        assert isinstance(metrics, dict)

    def test_has_f1_macro(self, trained_model, gold_test_dir):
        from src.models.evaluate import evaluate_on_gold
        trained, _ = trained_model
        assert "f1_macro" in evaluate_on_gold(trained, gold_dir=gold_test_dir)

    def test_has_accuracy(self, trained_model, gold_test_dir):
        from src.models.evaluate import evaluate_on_gold
        trained, _ = trained_model
        assert "accuracy" in evaluate_on_gold(trained, gold_dir=gold_test_dir)

    def test_has_n_test(self, trained_model, gold_test_dir):
        from src.models.evaluate import evaluate_on_gold
        trained, _ = trained_model
        metrics = evaluate_on_gold(trained, gold_dir=gold_test_dir)
        assert "n_test" in metrics
        assert metrics["n_test"] == 3

    def test_f1_in_valid_range(self, trained_model, gold_test_dir):
        from src.models.evaluate import evaluate_on_gold
        trained, _ = trained_model
        f1 = evaluate_on_gold(trained, gold_dir=gold_test_dir)["f1_macro"]
        assert 0.0 <= f1 <= 1.0

    def test_accuracy_in_valid_range(self, trained_model, gold_test_dir):
        from src.models.evaluate import evaluate_on_gold
        trained, _ = trained_model
        acc = evaluate_on_gold(trained, gold_dir=gold_test_dir)["accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_missing_gold_raises_file_not_found(self, trained_model, tmp_path):
        from src.models.evaluate import evaluate_on_gold
        trained, _ = trained_model
        with pytest.raises(FileNotFoundError):
            evaluate_on_gold(trained, gold_dir=tmp_path / "nonexistent")

    def test_per_class_f1_present(self, trained_model, gold_test_dir):
        from src.models.evaluate import evaluate_on_gold
        trained, _ = trained_model
        metrics = evaluate_on_gold(trained, gold_dir=gold_test_dir)
        assert "f1_high" in metrics
        assert "f1_medium" in metrics
        assert "f1_low" in metrics


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

class TestRunInference:

    def test_returns_dataframe(self, trained_model, test_csv):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        result = run_inference(trained, test_csv)
        assert isinstance(result, pd.DataFrame)

    def test_predicted_priority_column_added(self, trained_model, test_csv):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        result = run_inference(trained, test_csv)
        assert "predicted_priority" in result.columns

    def test_all_predictions_valid(self, trained_model, test_csv):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        result = run_inference(trained, test_csv)
        assert set(result["predicted_priority"].unique()).issubset({"high", "medium", "low"})

    def test_row_count_preserved(self, trained_model, test_csv):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        original = pd.read_csv(test_csv)
        result = run_inference(trained, test_csv)
        assert len(result) == len(original)

    def test_saves_csv_when_output_path_given(self, trained_model, test_csv, tmp_path):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        out = tmp_path / "predictions.csv"
        run_inference(trained, test_csv, output_path=out)
        assert out.exists()

    def test_no_file_saved_without_output_path(self, trained_model, test_csv, tmp_path):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        run_inference(trained, test_csv, output_path=None)
        assert not (tmp_path / "predictions.csv").exists()

    def test_confidence_column_added_when_proba_available(self, trained_model, test_csv):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        result = run_inference(trained, test_csv)
        if hasattr(trained, "predict_proba"):
            assert "confidence" in result.columns
            assert (result["confidence"] >= 0).all() and (result["confidence"] <= 1).all()

    def test_csv_with_text_column_accepted(self, trained_model, tmp_path):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        df = pd.DataFrame({"text": ["app crash bug", "add feature docs"]})
        path = tmp_path / "with_text.csv"
        df.to_csv(path, index=False)
        result = run_inference(trained, path)
        assert "predicted_priority" in result.columns

    def test_missing_title_column_raises(self, trained_model, tmp_path):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        df = pd.DataFrame({"body": ["some body only"]})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError):
            run_inference(trained, path)

    def test_null_body_handled_gracefully(self, trained_model, tmp_path):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        df = pd.DataFrame({"title": ["App crash"], "body": [None]})
        path = tmp_path / "null_body.csv"
        df.to_csv(path, index=False)
        result = run_inference(trained, path)
        assert len(result) == 1
        assert result.iloc[0]["predicted_priority"] in {"high", "medium", "low"}

    def test_saved_csv_has_predictions(self, trained_model, test_csv, tmp_path):
        from src.models.evaluate import run_inference
        trained, _ = trained_model
        out = tmp_path / "preds.csv"
        run_inference(trained, test_csv, output_path=out)
        loaded = pd.read_csv(out)
        assert "predicted_priority" in loaded.columns
        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# check_improvement
# ---------------------------------------------------------------------------

class TestCheckImprovement:

    def test_returns_true_above_threshold(self, tmp_path):
        from src.models.evaluate import check_improvement
        assert check_improvement({"f1_macro": 0.8}, threshold=0.6, metrics_path=tmp_path / "m.json") is True

    def test_returns_false_below_threshold(self, tmp_path):
        from src.models.evaluate import check_improvement
        assert check_improvement({"f1_macro": 0.4}, threshold=0.6, metrics_path=tmp_path / "m.json") is False

    def test_equal_to_threshold_passes(self, tmp_path):
        from src.models.evaluate import check_improvement
        assert check_improvement({"f1_macro": 0.6}, threshold=0.6, metrics_path=tmp_path / "m.json") is True

    def test_no_previous_metrics_still_works(self, tmp_path):
        from src.models.evaluate import check_improvement
        result = check_improvement({"f1_macro": 0.75}, threshold=0.6, metrics_path=tmp_path / "nonexistent.json")
        assert result is True

    def test_compares_against_previous_metrics(self, tmp_path):
        from src.models.evaluate import check_improvement
        prev = tmp_path / "metrics.json"
        prev.write_text(json.dumps({"f1_macro": 0.85}))
        # Current is worse but above threshold — should still return True (threshold check)
        result = check_improvement({"f1_macro": 0.70}, threshold=0.6, metrics_path=prev)
        assert result is True

    def test_zero_f1_fails_threshold(self, tmp_path):
        from src.models.evaluate import check_improvement
        assert check_improvement({"f1_macro": 0.0}, threshold=0.6, metrics_path=tmp_path / "m.json") is False