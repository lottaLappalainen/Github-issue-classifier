import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.monitoring.monitor import (
    check_class_distribution,
    check_dataset_size,
    check_class_balance,
    check_new_data_version,
    run_monitoring,
    CHI2_P_THRESHOLD,
    SIZE_DRIFT_THRESHOLD,
    BALANCE_RATIO_MAX,
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
    """Class distribution has shifted heavily toward 'high'."""
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


# ── check_class_distribution ──────────────────────────────────────────────

def test_no_distribution_drift(stable_meta):
    """Identical distributions should not trigger drift."""
    result = check_class_distribution(stable_meta, stable_meta)
    assert result["drift_detected"] is False


def test_distribution_drift_detected(stable_meta, drifted_meta):
    result = check_class_distribution(stable_meta, drifted_meta)
    assert result["drift_detected"] is True


def test_distribution_result_has_required_keys(stable_meta):
    result = check_class_distribution(stable_meta, stable_meta)
    for key in ("chi2_stat", "p_value", "threshold", "drift_detected",
                "baseline_dist", "current_dist"):
        assert key in result


def test_distribution_p_value_range(stable_meta):
    result = check_class_distribution(stable_meta, stable_meta)
    assert 0.0 <= result["p_value"] <= 1.0


def test_distribution_threshold_value(stable_meta):
    result = check_class_distribution(stable_meta, stable_meta)
    assert result["threshold"] == CHI2_P_THRESHOLD


def test_distribution_missing_class_in_current(stable_meta):
    """If a class disappears entirely, drift should fire."""
    current = dict(stable_meta)
    current["class_distribution"] = {"high": 5000, "medium": 2000}  # 'low' gone
    result = check_class_distribution(stable_meta, current)
    assert result["drift_detected"] is True


# ── check_dataset_size ────────────────────────────────────────────────────

def test_no_size_drift(stable_meta):
    result = check_dataset_size(stable_meta, stable_meta)
    assert result["drift_detected"] is False


def test_size_drift_large_increase(stable_meta):
    current = dict(stable_meta)
    current["n_train"] = 20_000   # >> 20 % increase
    result = check_dataset_size(stable_meta, current)
    assert result["drift_detected"] is True


def test_size_drift_large_decrease(stable_meta):
    current = dict(stable_meta)
    current["n_train"] = 100      # massive drop
    result = check_dataset_size(stable_meta, current)
    assert result["drift_detected"] is True


def test_size_drift_small_change(stable_meta):
    current = dict(stable_meta)
    current["n_train"] = 7100     # ~1.5 % change — below threshold
    result = check_dataset_size(stable_meta, current)
    assert result["drift_detected"] is False


def test_size_drift_relative_change_computed(stable_meta):
    current = dict(stable_meta)
    current["n_train"] = stable_meta["n_train"] * 2
    result = check_dataset_size(stable_meta, current)
    assert abs(result["relative_change"] - 1.0) < 0.01


def test_size_drift_threshold_key(stable_meta):
    result = check_dataset_size(stable_meta, stable_meta)
    assert result["threshold"] == SIZE_DRIFT_THRESHOLD


# ── check_class_balance ───────────────────────────────────────────────────

def test_balanced_classes_no_drift(stable_meta):
    result = check_class_balance(stable_meta)
    assert result["drift_detected"] is False


def test_imbalanced_classes_drift(drifted_meta):
    result = check_class_balance(drifted_meta)
    assert result["drift_detected"] is True


def test_zero_count_class_fires(stable_meta):
    meta = dict(stable_meta)
    meta["class_distribution"] = {"high": 5000, "medium": 2000, "low": 0}
    result = check_class_balance(meta)
    assert result["drift_detected"] is True


def test_balance_ratio_threshold(stable_meta):
    result = check_class_balance(stable_meta)
    assert result["threshold"] == BALANCE_RATIO_MAX


# ── check_new_data_version ────────────────────────────────────────────────

def test_same_version_no_new_batch(stable_meta):
    result = check_new_data_version(stable_meta, stable_meta)
    assert result["new_batch_detected"] is False


def test_different_version_new_batch(stable_meta, drifted_meta):
    result = check_new_data_version(stable_meta, drifted_meta)
    assert result["new_batch_detected"] is True


def test_version_keys_present(stable_meta, drifted_meta):
    result = check_new_data_version(stable_meta, drifted_meta)
    assert "baseline_version" in result
    assert "current_version" in result


# ── run_monitoring (integration) ──────────────────────────────────────────

def test_run_monitoring_creates_baseline(tmp_path, stable_meta):
    """First run with no baseline should create one and not trigger retrain."""
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"

    meta_file.write_text(json.dumps(stable_meta))

    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path):
        report = run_monitoring(
            baseline_path=baseline_file,
            meta_path=meta_file,
        )

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
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path):
        report = run_monitoring(
            baseline_path=baseline_file,
            meta_path=meta_file,
        )

    assert report["retrain_required"] is False
    assert report["status"] == "no_drift"


def test_run_monitoring_drift_detected(tmp_path, stable_meta, drifted_meta):
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"

    meta_file.write_text(json.dumps(drifted_meta))
    baseline_file.write_text(json.dumps(stable_meta))

    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path):
        report = run_monitoring(
            baseline_path=baseline_file,
            meta_path=meta_file,
        )

    assert report["retrain_required"] is True
    assert report["status"] == "drift_detected"


def test_run_monitoring_writes_drift_report(tmp_path, stable_meta, drifted_meta):
    meta_file     = tmp_path / "meta.json"
    baseline_file = tmp_path / "baseline_meta.json"
    drift_report  = tmp_path / "drift_report.json"

    meta_file.write_text(json.dumps(drifted_meta))
    baseline_file.write_text(json.dumps(stable_meta))

    with patch("src.monitoring.monitor.DRIFT_REPORT", drift_report), \
         patch("src.monitoring.monitor.GOLD_DIR",     tmp_path):
        run_monitoring(baseline_path=baseline_file, meta_path=meta_file)

    assert drift_report.exists()
    content = json.loads(drift_report.read_text())
    assert "checks" in content
    assert "retrain_required" in content