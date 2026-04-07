import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib


@pytest.fixture
def gold_df():
    texts  = [
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
    labels = ["high", "medium", "low", "high", "medium", "low", "high", "medium", "low"]
    return pd.DataFrame({"text": texts, "priority": labels})


@pytest.fixture
def trained_model(gold_df, tmp_path):
    from src.models.train import build_pipeline, train_model
    pipeline = build_pipeline()
    trained  = train_model(pipeline, gold_df["text"], gold_df["priority"])
    model_path = tmp_path / "model.joblib"
    joblib.dump(trained, model_path)
    return trained, model_path


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:

    def test_returns_sklearn_pipeline(self):
        from src.models.train import build_pipeline
        from sklearn.pipeline import Pipeline
        assert isinstance(build_pipeline(), Pipeline)

    def test_has_tfidf_step(self):
        from src.models.train import build_pipeline
        names = [n for n, _ in build_pipeline().steps]
        assert "tfidf" in names

    def test_has_clf_step(self):
        from src.models.train import build_pipeline
        names = [n for n, _ in build_pipeline().steps]
        assert "clf" in names

    def test_tfidf_is_first_step(self):
        from src.models.train import build_pipeline
        assert build_pipeline().steps[0][0] == "tfidf"

    def test_clf_is_last_step(self):
        from src.models.train import build_pipeline
        assert build_pipeline().steps[-1][0] == "clf"

    def test_default_is_logistic_regression(self):
        from src.models.train import build_pipeline
        from sklearn.linear_model import LogisticRegression
        assert isinstance(build_pipeline().named_steps["clf"], LogisticRegression)

    def test_random_forest_classifier(self):
        from src.models.train import build_pipeline
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(build_pipeline("random_forest").named_steps["clf"], RandomForestClassifier)

    def test_gradient_boosting_classifier(self):
        from src.models.train import build_pipeline
        from sklearn.ensemble import GradientBoostingClassifier
        assert isinstance(build_pipeline("gradient_boosting").named_steps["clf"], GradientBoostingClassifier)

    def test_unknown_classifier_raises(self):
        from src.models.train import build_pipeline
        with pytest.raises(ValueError):
            build_pipeline("nonexistent_model")

    def test_max_features_passed_to_tfidf(self):
        from src.models.train import build_pipeline
        p = build_pipeline(max_features=500)
        assert p.named_steps["tfidf"].max_features == 500

    def test_ngram_range_passed_to_tfidf(self):
        from src.models.train import build_pipeline
        p = build_pipeline(ngram_range=(1, 3))
        assert p.named_steps["tfidf"].ngram_range == (1, 3)

    def test_c_param_passed_to_logistic_regression(self):
        from src.models.train import build_pipeline
        p = build_pipeline(C=0.01)
        assert p.named_steps["clf"].C == 0.01

    def test_pipeline_is_fittable(self, gold_df):
        from src.models.train import build_pipeline
        build_pipeline().fit(gold_df["text"], gold_df["priority"])


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:

    def test_returns_fitted_pipeline(self, gold_df):
        from src.models.train import build_pipeline, train_model
        pipeline = build_pipeline()
        trained  = train_model(pipeline, gold_df["text"], gold_df["priority"])
        assert hasattr(trained, "predict")

    def test_trains_without_error(self, gold_df):
        from src.models.train import build_pipeline, train_model
        train_model(build_pipeline(), gold_df["text"], gold_df["priority"])

    def test_predicts_valid_labels(self, gold_df):
        from src.models.train import build_pipeline, train_model
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        preds = trained.predict(gold_df["text"])
        assert set(preds).issubset({"high", "medium", "low"})

    def test_prediction_length_matches_input(self, gold_df):
        from src.models.train import build_pipeline, train_model
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        assert len(trained.predict(gold_df["text"])) == len(gold_df)

    def test_predict_single_string(self, gold_df):
        from src.models.train import build_pipeline, train_model
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        preds = trained.predict(["app crashes on login"])
        assert preds[0] in {"high", "medium", "low"}

    def test_returns_probabilities(self, gold_df):
        from src.models.train import build_pipeline, train_model
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        proba = trained.predict_proba(gold_df["text"])
        assert proba.shape == (len(gold_df), 3)

    def test_probabilities_sum_to_one(self, gold_df):
        from src.models.train import build_pipeline, train_model
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        proba = trained.predict_proba(gold_df["text"])
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_all_three_classes_in_classes_(self, gold_df):
        from src.models.train import build_pipeline, train_model
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        assert set(trained.classes_) == {"high", "medium", "low"}

    def test_random_forest_also_trains(self, gold_df):
        from src.models.train import build_pipeline, train_model
        trained = train_model(build_pipeline("random_forest"), gold_df["text"], gold_df["priority"])
        preds = trained.predict(gold_df["text"])
        assert set(preds).issubset({"high", "medium", "low"})


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_returns_dict(self, gold_df):
        from src.models.train import build_pipeline, train_model, compute_metrics
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        preds   = trained.predict(gold_df["text"])
        assert isinstance(compute_metrics(gold_df["priority"], preds), dict)

    def test_has_f1_macro(self, gold_df):
        from src.models.train import build_pipeline, train_model, compute_metrics
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        preds   = trained.predict(gold_df["text"])
        assert "f1_macro" in compute_metrics(gold_df["priority"], preds)

    def test_has_accuracy(self, gold_df):
        from src.models.train import build_pipeline, train_model, compute_metrics
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        preds   = trained.predict(gold_df["text"])
        assert "accuracy" in compute_metrics(gold_df["priority"], preds)

    def test_has_per_class_f1(self, gold_df):
        from src.models.train import build_pipeline, train_model, compute_metrics
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        preds   = trained.predict(gold_df["text"])
        metrics = compute_metrics(gold_df["priority"], preds)
        assert "f1_high" in metrics and "f1_medium" in metrics and "f1_low" in metrics

    def test_f1_in_0_to_1(self, gold_df):
        from src.models.train import build_pipeline, train_model, compute_metrics
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        preds   = trained.predict(gold_df["text"])
        f1 = compute_metrics(gold_df["priority"], preds)["f1_macro"]
        assert 0.0 <= f1 <= 1.0

    def test_accuracy_in_0_to_1(self, gold_df):
        from src.models.train import build_pipeline, train_model, compute_metrics
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        preds   = trained.predict(gold_df["text"])
        acc = compute_metrics(gold_df["priority"], preds)["accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_perfect_predictions_give_f1_1(self):
        from src.models.train import compute_metrics
        y = pd.Series(["high", "medium", "low", "high"])
        metrics = compute_metrics(y, y.values)
        assert metrics["f1_macro"] == pytest.approx(1.0)

    def test_all_wrong_gives_f1_0_or_low(self):
        from src.models.train import compute_metrics
        y_true = pd.Series(["high", "high", "high"])
        y_pred = np.array(["low",  "low",  "low"])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["f1_macro"] < 0.5

    def test_values_are_floats(self, gold_df):
        from src.models.train import build_pipeline, train_model, compute_metrics
        trained = train_model(build_pipeline(), gold_df["text"], gold_df["priority"])
        preds   = trained.predict(gold_df["text"])
        metrics = compute_metrics(gold_df["priority"], preds)
        for v in metrics.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

class TestModelPersistence:

    def test_saved_file_exists(self, trained_model):
        _, model_path = trained_model
        assert model_path.exists()

    def test_loaded_model_predicts(self, trained_model):
        _, model_path = trained_model
        loaded = joblib.load(model_path)
        assert loaded.predict(["app crashes bug"])[0] in {"high", "medium", "low"}

    def test_predictions_identical_after_reload(self, trained_model, gold_df):
        trained, model_path = trained_model
        loaded = joblib.load(model_path)
        assert list(trained.predict(gold_df["text"])) == list(loaded.predict(gold_df["text"]))

    def test_probabilities_identical_after_reload(self, trained_model, gold_df):
        trained, model_path = trained_model
        loaded = joblib.load(model_path)
        np.testing.assert_array_almost_equal(
            trained.predict_proba(gold_df["text"]),
            loaded.predict_proba(gold_df["text"])
        )

    def test_model_file_is_nonzero_bytes(self, trained_model):
        _, model_path = trained_model
        assert model_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Production inference on test.csv (assignment requirement)
# ---------------------------------------------------------------------------

class TestProductionInference:

    def test_inference_on_csv_format(self, trained_model, tmp_path):
        trained, _ = trained_model
        test_data = pd.DataFrame({
            "title": ["App crashes on load", "Add export button", "Fix typo in docs"],
            "body":  ["Crash on startup",   "Would be useful",   "Small fix"],
        })
        test_csv = tmp_path / "test.csv"
        test_data.to_csv(test_csv, index=False)
        df = pd.read_csv(test_csv)
        df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
        preds = trained.predict(df["text"])
        assert len(preds) == 3
        assert all(p in {"high", "medium", "low"} for p in preds)

    def test_predictions_saved_to_csv(self, trained_model, tmp_path):
        trained, _ = trained_model
        df = pd.DataFrame({"title": ["Login bug", "Dark mode", "Fix docs"], "body": ["Crash", "Feature", "Typo"]})
        df["text"] = df["title"] + " " + df["body"]
        df["predicted_priority"] = trained.predict(df["text"])
        out = tmp_path / "predictions.csv"
        df.to_csv(out, index=False)
        loaded = pd.read_csv(out)
        assert "predicted_priority" in loaded.columns
        assert len(loaded) == 3

    def test_inference_with_missing_body(self, trained_model, tmp_path):
        trained, _ = trained_model
        df = pd.DataFrame({"title": ["App crash"], "body": [None]})
        df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
        preds = trained.predict(df["text"])
        assert preds[0] in {"high", "medium", "low"}

    def test_confidence_scores_available(self, trained_model):
        trained, _ = trained_model
        texts = pd.Series(["app crash bug", "add dark mode", "fix typo"])
        probas = trained.predict_proba(texts)
        confidence = probas.max(axis=1)
        assert all(0.0 <= c <= 1.0 for c in confidence)

    def test_output_csv_has_correct_columns(self, trained_model, tmp_path):
        trained, _ = trained_model
        df = pd.DataFrame({"title": ["Bug"], "body": ["crash"]})
        df["text"] = df["title"] + " " + df["body"]
        df["predicted_priority"] = trained.predict(df["text"])
        out = tmp_path / "pred.csv"
        df.to_csv(out, index=False)
        loaded = pd.read_csv(out)
        assert "title" in loaded.columns
        assert "predicted_priority" in loaded.columns