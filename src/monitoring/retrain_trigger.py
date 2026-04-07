"""
Tests for src/monitoring/retrain_trigger.py
Run with: python -m pytest tests/test_retrain_trigger.py -v
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call


@pytest.fixture
def no_retrain_report():
    return {"status": "no_drift", "retrain_required": False, "checks": {}}


@pytest.fixture
def retrain_report():
    return {
        "status": "drift_detected",
        "retrain_required": True,
        "checks": {
            "class_distribution": {"drift_detected": True},
            "dataset_size":       {"drift_detected": False},
        },
    }


@pytest.fixture
def drift_report_file_no_retrain(tmp_path, no_retrain_report):
    path = tmp_path / "drift_report.json"
    path.write_text(json.dumps(no_retrain_report))
    return path


@pytest.fixture
def drift_report_file_retrain(tmp_path, retrain_report):
    path = tmp_path / "drift_report.json"
    path.write_text(json.dumps(retrain_report))
    return path


@pytest.fixture
def metrics_v1(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"f1_macro": 0.85, "data_version": "v1"}))
    return path


# ---------------------------------------------------------------------------
# read_drift_report
# ---------------------------------------------------------------------------

class TestReadDriftReport:

    def test_reads_valid_report(self, drift_report_file_no_retrain):
        from src.monitoring.retrain_trigger import read_drift_report
        report = read_drift_report(drift_report_file_no_retrain)
        assert report["retrain_required"] is False

    def test_missing_report_exits_with_code_2(self, tmp_path):
        from src.monitoring.retrain_trigger import read_drift_report
        with pytest.raises(SystemExit) as exc:
            read_drift_report(tmp_path / "nonexistent.json")
        assert exc.value.code == 2

    def test_returns_dict(self, drift_report_file_no_retrain):
        from src.monitoring.retrain_trigger import read_drift_report
        assert isinstance(read_drift_report(drift_report_file_no_retrain), dict)

    def test_retrain_required_key_present(self, drift_report_file_retrain):
        from src.monitoring.retrain_trigger import read_drift_report
        assert "retrain_required" in read_drift_report(drift_report_file_retrain)

    def test_status_key_present(self, drift_report_file_retrain):
        from src.monitoring.retrain_trigger import read_drift_report
        assert "status" in read_drift_report(drift_report_file_retrain)

    def test_checks_key_present(self, drift_report_file_retrain):
        from src.monitoring.retrain_trigger import read_drift_report
        assert "checks" in read_drift_report(drift_report_file_retrain)

    def test_retrain_true_when_drift(self, drift_report_file_retrain):
        from src.monitoring.retrain_trigger import read_drift_report
        assert read_drift_report(drift_report_file_retrain)["retrain_required"] is True


# ---------------------------------------------------------------------------
# get_next_version
# ---------------------------------------------------------------------------

class TestGetNextVersion:

    def test_v1_when_no_metrics(self, tmp_path):
        from src.monitoring.retrain_trigger import get_next_version
        assert get_next_version(tmp_path / "nonexistent.json") == "v1"

    def test_increments_v1_to_v2(self, tmp_path):
        from src.monitoring.retrain_trigger import get_next_version
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"data_version": "v1"}))
        assert get_next_version(path) == "v2"

    def test_increments_v3_to_v4(self, tmp_path):
        from src.monitoring.retrain_trigger import get_next_version
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"data_version": "v3"}))
        assert get_next_version(path) == "v4"

    def test_increments_v10_to_v11(self, tmp_path):
        from src.monitoring.retrain_trigger import get_next_version
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"data_version": "v10"}))
        assert get_next_version(path) == "v11"

    def test_handles_missing_data_version_key(self, tmp_path):
        from src.monitoring.retrain_trigger import get_next_version
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"f1_macro": 0.8}))
        assert get_next_version(path).startswith("v")

    def test_handles_non_numeric_version(self, tmp_path):
        from src.monitoring.retrain_trigger import get_next_version
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"data_version": "bad_version"}))
        assert get_next_version(path) == "v_new"

    def test_v0_increments_to_v1(self, tmp_path):
        from src.monitoring.retrain_trigger import get_next_version
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"data_version": "v0"}))
        assert get_next_version(path) == "v1"


# ---------------------------------------------------------------------------
# main — no retrain needed  (patch read_drift_report directly)
# ---------------------------------------------------------------------------

class TestMainNoRetrain:

    def test_exits_0_when_no_retrain_needed(self, no_retrain_report):
        from src.monitoring import retrain_trigger
        with patch.object(retrain_trigger, "read_drift_report", return_value=no_retrain_report), \
             pytest.raises(SystemExit) as exc:
            retrain_trigger.main(dry_run=False)
        assert exc.value.code == 0

    def test_dry_run_exits_0_even_when_retrain_needed(self, retrain_report):
        from src.monitoring import retrain_trigger
        with patch.object(retrain_trigger, "read_drift_report", return_value=retrain_report), \
             pytest.raises(SystemExit) as exc:
            retrain_trigger.main(dry_run=True)
        assert exc.value.code == 0

    def test_run_retrain_not_called_when_no_drift(self, no_retrain_report):
        from src.monitoring import retrain_trigger
        with patch.object(retrain_trigger, "read_drift_report", return_value=no_retrain_report), \
             patch.object(retrain_trigger, "run_retrain") as mock_retrain, \
             pytest.raises(SystemExit):
            retrain_trigger.main(dry_run=False)
        mock_retrain.assert_not_called()

    def test_run_retrain_not_called_on_dry_run(self, retrain_report):
        from src.monitoring import retrain_trigger
        with patch.object(retrain_trigger, "read_drift_report", return_value=retrain_report), \
             patch.object(retrain_trigger, "run_retrain") as mock_retrain, \
             pytest.raises(SystemExit):
            retrain_trigger.main(dry_run=True)
        mock_retrain.assert_not_called()


# ---------------------------------------------------------------------------
# main — retrain triggered
# ---------------------------------------------------------------------------

class TestMainRetrain:

    def test_calls_run_retrain_when_drift_detected(self, retrain_report):
        from src.monitoring import retrain_trigger
        with patch.object(retrain_trigger, "read_drift_report", return_value=retrain_report), \
             patch.object(retrain_trigger, "run_retrain", return_value=True) as mock_retrain, \
             pytest.raises(SystemExit):
            retrain_trigger.main(dry_run=False, gold_version="v99")
        mock_retrain.assert_called_once_with("v99")

    def test_exits_0_when_retrain_succeeds(self, retrain_report):
        from src.monitoring import retrain_trigger
        with patch.object(retrain_trigger, "read_drift_report", return_value=retrain_report), \
             patch.object(retrain_trigger, "run_retrain", return_value=True), \
             pytest.raises(SystemExit) as exc:
            retrain_trigger.main(dry_run=False, gold_version="v2")
        assert exc.value.code == 0

    def test_exits_1_when_retrain_fails_quality_gate(self, retrain_report):
        from src.monitoring import retrain_trigger
        with patch.object(retrain_trigger, "read_drift_report", return_value=retrain_report), \
             patch.object(retrain_trigger, "run_retrain", return_value=False), \
             pytest.raises(SystemExit) as exc:
            retrain_trigger.main(dry_run=False, gold_version="v2")
        assert exc.value.code == 1

    def test_uses_get_next_version_when_no_version_given(self, retrain_report, metrics_v1):
        from src.monitoring import retrain_trigger
        with patch.object(retrain_trigger, "read_drift_report", return_value=retrain_report), \
             patch.object(retrain_trigger, "get_next_version", return_value="v2") as mock_ver, \
             patch.object(retrain_trigger, "run_retrain", return_value=True) as mock_retrain, \
             pytest.raises(SystemExit):
            retrain_trigger.main(dry_run=False, gold_version=None)
        mock_ver.assert_called_once()
        mock_retrain.assert_called_once_with("v2")

    def test_explicit_version_skips_get_next_version(self, retrain_report):
        from src.monitoring import retrain_trigger
        with patch.object(retrain_trigger, "read_drift_report", return_value=retrain_report), \
             patch.object(retrain_trigger, "get_next_version") as mock_ver, \
             patch.object(retrain_trigger, "run_retrain", return_value=True), \
             pytest.raises(SystemExit):
            retrain_trigger.main(dry_run=False, gold_version="v5")
        mock_ver.assert_not_called()


# ---------------------------------------------------------------------------
# run_retrain — unit tests (subprocess mocked)
# ---------------------------------------------------------------------------

class TestRunRetrain:

    def _make_metrics(self, tmp_path, f1=0.85, version="v2"):
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"f1_macro": f1, "data_version": version}))
        return path

    def _mock_gold_paths(self, tmp_path):
        fake_meta = tmp_path / "meta.json"
        fake_meta.write_text("{}")
        fake_baseline = tmp_path / "baseline_meta.json"
        return fake_meta, fake_baseline

    def test_returns_true_when_quality_gate_passes(self, tmp_path):
        from src.monitoring import retrain_trigger
        metrics = self._make_metrics(tmp_path)
        _, fake_baseline = self._mock_gold_paths(tmp_path)
        mock_ok = MagicMock(returncode=0)
        with patch("src.monitoring.retrain_trigger.METRICS_PATH", metrics), \
             patch("subprocess.run", return_value=mock_ok), \
             patch("src.monitoring.monitor.BASELINE_PATH", fake_baseline), \
             patch("src.monitoring.monitor.GOLD_DIR", tmp_path):
            assert retrain_trigger.run_retrain("v2") is True

    def test_returns_false_when_pipeline_step_fails(self, tmp_path):
        from src.monitoring import retrain_trigger
        mock_fail = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_fail):
            assert retrain_trigger.run_retrain("v2") is False

    def test_returns_false_when_f1_below_threshold(self, tmp_path):
        from src.monitoring import retrain_trigger
        metrics = self._make_metrics(tmp_path, f1=0.40)
        mock_ok = MagicMock(returncode=0)
        with patch("src.monitoring.retrain_trigger.METRICS_PATH", metrics), \
             patch("subprocess.run", return_value=mock_ok):
            assert retrain_trigger.run_retrain("v2") is False

    def test_returns_false_when_no_metrics_written(self, tmp_path):
        from src.monitoring import retrain_trigger
        mock_ok = MagicMock(returncode=0)
        with patch("src.monitoring.retrain_trigger.METRICS_PATH", tmp_path / "missing.json"), \
             patch("subprocess.run", return_value=mock_ok):
            assert retrain_trigger.run_retrain("v2") is False

    def test_runs_five_pipeline_steps(self, tmp_path):
        from src.monitoring import retrain_trigger
        metrics = self._make_metrics(tmp_path)
        _, fake_baseline = self._mock_gold_paths(tmp_path)
        mock_ok = MagicMock(returncode=0)
        with patch("src.monitoring.retrain_trigger.METRICS_PATH", metrics), \
             patch("subprocess.run", return_value=mock_ok) as mock_sub, \
             patch("src.monitoring.monitor.BASELINE_PATH", fake_baseline), \
             patch("src.monitoring.monitor.GOLD_DIR", tmp_path):
            retrain_trigger.run_retrain("v2")
        assert mock_sub.call_count == 5

    def test_gold_version_passed_to_train_step(self, tmp_path):
        from src.monitoring import retrain_trigger
        metrics = self._make_metrics(tmp_path)
        _, fake_baseline = self._mock_gold_paths(tmp_path)
        calls_seen = []
        mock_ok = MagicMock(returncode=0)
        mock_ok.returncode = 0
        def capture(cmd, **kwargs):
            calls_seen.append(cmd)
            return mock_ok
        with patch("src.monitoring.retrain_trigger.METRICS_PATH", metrics), \
             patch("subprocess.run", side_effect=capture), \
             patch("src.monitoring.monitor.BASELINE_PATH", fake_baseline), \
             patch("src.monitoring.monitor.GOLD_DIR", tmp_path):
            retrain_trigger.run_retrain("v99")
        train_cmd = next(c for c in calls_seen if "train.py" in " ".join(c))
        assert "v99" in train_cmd

    def test_stops_on_first_failed_step(self, tmp_path):
        from src.monitoring import retrain_trigger
        mock_fail = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_fail) as mock_sub:
            retrain_trigger.run_retrain("v2")
        # Should stop after first failure, not run all 5
        assert mock_sub.call_count == 1

    def test_f1_exactly_at_threshold_passes(self, tmp_path):
        from src.monitoring import retrain_trigger
        metrics = self._make_metrics(tmp_path, f1=0.60)
        _, fake_baseline = self._mock_gold_paths(tmp_path)
        mock_ok = MagicMock(returncode=0)
        with patch("src.monitoring.retrain_trigger.METRICS_PATH", metrics), \
             patch("subprocess.run", return_value=mock_ok), \
             patch("src.monitoring.monitor.BASELINE_PATH", fake_baseline), \
             patch("src.monitoring.monitor.GOLD_DIR", tmp_path):
            assert retrain_trigger.run_retrain("v2") is True