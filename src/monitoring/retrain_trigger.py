"""
src/monitoring/retrain_trigger.py  —  Automatic Retraining Trigger

Reads drift_report.json and decides whether to kick off a retrain.
In CI/CD this is called after the monitor stage; locally it can
be polled by a cron job or a file-watcher.

Exit codes:
    0  — no retrain needed (or retrain completed successfully)
    1  — retrain was needed but failed the quality gate
    2  — drift_report.json missing (run monitor.py first)

Usage:
    python src/monitoring/retrain_trigger.py
    python src/monitoring/retrain_trigger.py --dry-run   # check only, no train
    python src/monitoring/retrain_trigger.py --gold-version v3
"""
import json
import logging
import argparse
import subprocess
import sys
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[2]
MONITORING_DIR = ROOT / "monitoring"
DRIFT_REPORT   = MONITORING_DIR / "drift_report.json"
METRICS_PATH   = ROOT / "metrics.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def read_drift_report(path: Path = DRIFT_REPORT) -> dict:
    if not path.exists():
        log.error(f"Drift report not found: {path}\nRun monitor.py first.")
        sys.exit(2)
    return json.loads(path.read_text())


def get_next_version(metrics_path: Path = METRICS_PATH) -> str:
    """Derive next version string from metrics.json, e.g. v1 → v2."""
    if not metrics_path.exists():
        return "v1"
    data = json.loads(metrics_path.read_text())
    current = data.get("data_version", "v0")
    try:
        n = int(current.lstrip("v"))
        return f"v{n + 1}"
    except ValueError:
        return "v_new"


def run_retrain(gold_version: str) -> bool:
    """
    Run the full training pipeline for the given gold version.
    Returns True if quality gate passes, False otherwise.
    """
    log.info(f"🚀 Triggering retrain for gold_version={gold_version}")

    steps = [
        ["python", "src/data/ingest.py"],
        ["python", "src/data/clean.py"],
        ["python", "src/data/featurize.py"],
        ["python", "src/models/train.py", "--gold-version", gold_version],
        ["python", "src/models/evaluate.py"],
    ]

    for cmd in steps:
        log.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
        if result.returncode != 0:
            log.error(f"  ❌ Step failed: {' '.join(cmd)}")
            return False

    # Check quality gate
    if not METRICS_PATH.exists():
        log.error("metrics.json not written after training — something went wrong.")
        return False

    metrics = json.loads(METRICS_PATH.read_text())
    f1      = metrics.get("f1_macro", 0.0)
    THRESHOLD = 0.60

    if f1 >= THRESHOLD:
        log.info(f"  ✅ Quality gate PASSED (F1={f1:.4f} ≥ {THRESHOLD})")

        # Update baseline now that retrain succeeded
        from src.monitoring.monitor import BASELINE_PATH, GOLD_DIR
        meta_path = GOLD_DIR / "meta.json"
        if meta_path.exists():
            import shutil
            shutil.copy(meta_path, BASELINE_PATH)
            log.info(f"  Baseline updated → {BASELINE_PATH}")

        return True
    else:
        log.warning(f"  ❌ Quality gate FAILED (F1={f1:.4f} < {THRESHOLD})")
        return False


def main(dry_run: bool = False, gold_version: str = None) -> None:
    log.info("=== Retrain Trigger ===")

    report = read_drift_report()
    log.info(f"Drift status : {report['status']}")
    log.info(f"Retrain needed: {report['retrain_required']}")

    # Print which checks fired
    for check_name, result in report.get("checks", {}).items():
        fired = result.get("drift_detected") or result.get("new_batch_detected", False)
        if fired:
            log.info(f"  ⚠️  {check_name} triggered retrain")

    if not report["retrain_required"]:
        log.info("No retrain needed. Exiting cleanly.")
        sys.exit(0)

    if dry_run:
        log.info("--dry-run: would have triggered retrain but skipping.")
        sys.exit(0)

    # Derive version if not provided
    version = gold_version or get_next_version()
    log.info(f"Starting retrain as {version} ...")

    success = run_retrain(version)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic retrain trigger")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Check drift but do not actually retrain")
    parser.add_argument("--gold-version", type=str, default=None,
                        help="Override gold version tag (e.g. v3)")
    args = parser.parse_args()

    main(dry_run=args.dry_run, gold_version=args.gold_version)