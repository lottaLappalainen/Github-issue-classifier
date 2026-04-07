import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch

from src.monitoring.monitor import (
    check_class_distribution,
    check_dataset_size,
    check_class_balance,
    check_new_data_version,
    check_prediction_confidence,
    run_monitoring,
    CHI2_P_THRESHOLD,
    SIZE_DRIFT_THRESHOLD,
    BALANCE_RATIO_MAX,
    CONFIDENCE_MIN,
)


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def stable_meta():
    return {
        "gold_version": "v1",
        "silver_hash": "abc123",
        "n_train": 6996,
        "n_test": 1749,
        "class_distribution": {"high": 2332, "medium": 2332, "low": 2332},
        "class_proportions": {"high": 0.333, "medium": 0.333, "low": 0.334},
        "test_size": 0.2,
        "columns": ["text", "priority"],
    }


@pytest.fixture
def drifted_meta():
    return {
        "gold_version": "v2",
        "silver_hash": "def456",
        "n_train": 9000,
        "n_test": 2250,
        "class_distribution": {"high": 7000, "medium": 1000, "low": 1000},
        "class_proportions": {"high": 0.777, "medium": 0.111, "low": 0.111},
        "test_size": 0.2,
        "columns": ["text", "priority"],
    }


def _make_pred_db(tmp_path, confidences):
    """Write a minimal prediction_log.db with given confidence values."""
    db = tmp_path / "prediction_log.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, title TEXT, body TEXT,
                predicted_priority TEXT, confidence REAL,
                prob_high REAL, prob_medium REAL, prob_low REAL,
                model_f1 REAL, data_version TEXT
            )
        """)
        for c in confidences:
            conn.execute(
                "INSERT INTO predictions (timestamp, predicted_priority, confidence) VALUES (?,?,?)",
                ("2024-01-01T00:00:00Z", "high", c)
            )
        conn.commit()
    return db


# ── check_class_distribution ──────────────────────────────────────────────

def test_no_distribution_drift(stable_meta):
    result = check_class_distribution(stable_meta, stable_meta)
    assert result["drift_detected"] is False


def test_distribution_drift_detected(stable_meta, drifted_meta):
    result = check_class_distribution(stable_meta, drifted_meta)
    assert result["drift_detected"] is True


def test_distribution_result_has_required_keys(stable_meta):
    result = check_class_distribution(stable_meta, stable_meta)
    for key in ("chi2_stat", "p_value", "threshold", "drift_detected", "baseline_dist", "current_dist"):
        assert key in result


def test_distribution_p_value_range(stable_meta):
    result = check_class_distribution(stable_meta, stable_meta)
    assert 0.0 <= result["p_value"] <= 1.0


def test_distribution_threshold_value(stable_meta):
    result = check_class_distribution(stable_meta, stable_meta)
    assert result["threshold"] == CHI2_P_THRESHOLD


def test_distribution_missing_class_in_current(stable_meta):
    current = dict(stable_meta)
    current["class_distribution"] = {"high": 5000, "medium": 2000}
    result = check_class_distribution(stable_meta, current)
    assert result["drift_detected"] is True


# ── check_dataset_size ────────────────────────────────────────────────────

def test_no_size_drift(stable_meta):
    assert check_dataset_size(stable_meta, stable_meta)["drift_detected"] is False


def test_size_drift_large_increase(stable_meta):
    current = dict(stable_meta, n_train=20_000)
    assert check_dataset_size(stable_meta, current)["drift_detected"] is True


def test_size_drift_large_decrease(stable_meta):
    current = dict(stable_meta, n_train=100)
    assert check_dataset_size(stable_meta, current)["drift_detected"] is True


def test_size_drift_small_change(stable_meta):
    current = dict(stable_meta, n_train=7100)
    assert check_dataset_size(stable_meta, current)["drift_detected"] is False


def test_size_drift_relative_change_computed(stable_meta):
    current = dict(stable_meta, n_train=stable_meta["n_train"] * 2)
    result = check_dataset_size(stable_meta, current)
    assert abs(result["relative_change"] - 1.0) < 0.01


def test_size_drift_threshold_key(stable_meta):
    assert check_dataset_size(stable_meta, stable_meta)["threshold"] == SIZE_DRIFT_THRESHOLD


# ── check_class_balance ───────────────────────────────────────────────────

def test_balanced_classes_no_drift(stable_meta):
    assert check_class_balance(stable_meta)["drift_detected"] is False


def test_imbalanced_classes_drift(drifted_meta):
    assert check_class_balance(drifted_meta)["drift_detected"] is True


def test_zero_count_class_fires(stable_meta):
    meta = dict(stable_meta)
    meta["class_distribution"] = {"high": 5000, "medium": 2000, "low": 0}
    assert check_class_balance(meta)["drift_detected"] is True


def test_balance_ratio_threshold(stable_meta):
    assert check_class_balance(stable_meta)["threshold"] == BALANCE_RATIO_MAX


# ── check_new_data_version ────────────────────────────────────────────────

def test_same_version_no_new_batch(stable_meta):
    assert check_new_data_version(stable_meta, stable_meta)["new_batch_detected"] is False


def test_different_version_new_batch(stable_meta, drifted_meta):
    assert check_new_data_version(stable_meta, drifted_meta)["new_batch_detected"] is True


def test_version_keys_present(stable_meta, drifted_meta):
    result = check_new_data_version(stable_meta, drifted_meta)
    assert "baseline_version" in result and "current_version" in result


# ── check_prediction_confidence ───────────────────────────────────────────

def test_confidence_no_db_returns_no_drift(tmp_path):
    result = check_prediction_confidence(db_path=tmp_path / "nonexistent.db")
    assert result["drift_detected"] is False
    assert result["n_predictions"] == 0


def test_confidence_empty_db_returns_no_drift(tmp_path):
    db = tmp_path / "empty.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, timestamp TEXT, predicted_priority TEXT, confidence REAL)")
    result = check_prediction_confidence(db_path=db)
    assert result["drift_detected"] is False
    assert result["n_predictions"] == 0


def test_confidence_high_mean_no_drift(tmp_path):
    db = _make_pred_db(tmp_path, [0.90, 0.92, 0.88, 0.95, 0.91])
    result = check_prediction_confidence(db_path=db, threshold=0.70)
    assert result["drift_detected"] is False
    assert result["mean_confidence"] > 0.70


def test_confidence_low_mean_drift_detected(tmp_path):
    db = _make_pred_db(tmp_path, [0.51, 0.55, 0.52, 0.49, 0.53])
    result = check_prediction_confidence(db_path=db, threshold=0.70)
    assert result["drift_detected"] is True


def test_confidence_clearly_above_threshold_no_drift(tmp_path):
    db = _make_pred_db(tmp_path, [0.80, 0.80, 0.80])
    result = check_prediction_confidence(db_path=db, threshold=0.70)
    assert result["drift_detected"] is False


def test_confidence_result_has_required_keys(tmp_path):
    db = _make_pred_db(tmp_path, [0.85, 0.90])
    result = check_prediction_confidence(db_path=db)
    for key in ("drift_detected", "mean_confidence", "threshold", "n_predictions"):
        assert key in result


def test_confidence_window_limits_rows(tmp_path):
    # 50 high-confidence rows, then 10 low-confidence (most recent)
    db = _make_pred_db(tmp_path, [0.95] * 50 + [0.40] * 10)
    result = check_prediction_confidence(db_path=db, window=10, threshold=0.70)
    assert result["drift_detected"] is True
    assert result["n_predictions"] == 10


def test_confidence_threshold_in_result(tmp_path):
    db = _make_pred_db(tmp_path, [0.85])
    result = check_prediction_confidence(db_path=db, threshold=0.65)
    assert result["threshold"] == 0.65


# ── run_monitoring (integration) ──────────────────────────────────────────

def test_run_monitoring_creates_baseline(tmp_path, stable_meta):
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"
    meta_file.write_text(json.dumps(stable_meta))
    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path), \
         patch("src.monitoring.monitor.PRED_LOG_DB",  tmp_path / "nope.db"):
        report = run_monitoring(baseline_path=baseline_file, meta_path=meta_file)
    assert baseline_file.exists()
    assert report["retrain_required"] is False
    assert report["status"] == "baseline_created"


def test_run_monitoring_no_drift(tmp_path, stable_meta):
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"
    meta_file.write_text(json.dumps(stable_meta))
    baseline_file.write_text(json.dumps(stable_meta))
    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path), \
         patch("src.monitoring.monitor.PRED_LOG_DB",  tmp_path / "nope.db"):
        report = run_monitoring(baseline_path=baseline_file, meta_path=meta_file)
    assert report["retrain_required"] is False
    assert report["status"] == "no_drift"


def test_run_monitoring_drift_detected(tmp_path, stable_meta, drifted_meta):
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"
    meta_file.write_text(json.dumps(drifted_meta))
    baseline_file.write_text(json.dumps(stable_meta))
    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path), \
         patch("src.monitoring.monitor.PRED_LOG_DB",  tmp_path / "nope.db"):
        report = run_monitoring(baseline_path=baseline_file, meta_path=meta_file)
    assert report["retrain_required"] is True
    assert report["status"] == "drift_detected"


def test_run_monitoring_writes_drift_report(tmp_path, stable_meta, drifted_meta):
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"
    meta_file.write_text(json.dumps(drifted_meta))
    baseline_file.write_text(json.dumps(stable_meta))
    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path), \
         patch("src.monitoring.monitor.PRED_LOG_DB",  tmp_path / "nope.db"):
        run_monitoring(baseline_path=baseline_file, meta_path=meta_file)
    assert drift_report.exists()
    content = json.loads(drift_report.read_text())
    assert "checks" in content and "retrain_required" in content


def test_run_monitoring_confidence_drift_triggers_retrain(tmp_path, stable_meta):
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"
    meta_file.write_text(json.dumps(stable_meta))
    baseline_file.write_text(json.dumps(stable_meta))
    low_conf_result = {"drift_detected": True, "mean_confidence": 0.40,
                       "threshold": 0.70, "n_predictions": 3}
    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path), \
         patch("src.monitoring.monitor.check_prediction_confidence", return_value=low_conf_result):
        report = run_monitoring(baseline_path=baseline_file, meta_path=meta_file)
    assert report["retrain_required"] is True
    assert report["checks"]["prediction_confidence"]["drift_detected"] is True


def test_run_monitoring_confidence_ok_no_extra_retrain(tmp_path, stable_meta):
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"
    meta_file.write_text(json.dumps(stable_meta))
    baseline_file.write_text(json.dumps(stable_meta))
    high_conf_result = {"drift_detected": False, "mean_confidence": 0.91,
                        "threshold": 0.70, "n_predictions": 3}
    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path), \
         patch("src.monitoring.monitor.check_prediction_confidence", return_value=high_conf_result):
        report = run_monitoring(baseline_path=baseline_file, meta_path=meta_file)
    assert report["checks"]["prediction_confidence"]["drift_detected"] is False


def test_run_monitoring_has_five_checks(tmp_path, stable_meta):
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"
    meta_file.write_text(json.dumps(stable_meta))
    baseline_file.write_text(json.dumps(stable_meta))
    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path), \
         patch("src.monitoring.monitor.PRED_LOG_DB",  tmp_path / "nope.db"):
        report = run_monitoring(baseline_path=baseline_file, meta_path=meta_file)
    assert len(report["checks"]) == 5