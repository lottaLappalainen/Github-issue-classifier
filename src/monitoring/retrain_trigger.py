import json, logging, argparse, subprocess, sys
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[2]
MONITORING_DIR = ROOT / "monitoring"
DRIFT_REPORT   = MONITORING_DIR / "drift_report.json"
METRICS_PATH   = ROOT / "metrics.json"
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def read_drift_report(path=DRIFT_REPORT):
    if not Path(path).exists():
        log.error(f"Drift report not found: {path}")
        sys.exit(2)
    return json.loads(Path(path).read_text())

def get_next_version(metrics_path=METRICS_PATH):
    if not Path(metrics_path).exists():
        return "v1"
    data = json.loads(Path(metrics_path).read_text())
    current = data.get("data_version", "v0")
    try:
        n = int(current.lstrip("v"))
        return f"v{n + 1}"
    except ValueError:
        return "v_new"

def run_retrain(gold_version):
    log.info(f"Triggering retrain for gold_version={gold_version}")
    import os
    env = os.environ.copy()   
    steps = [
        ["python", "src/data/ingest.py", "--pages", "5"],
        ["python", "src/data/clean.py"],
        ["python", "src/data/featurize.py"],
        ["python", "src/models/train.py", "--gold-version", gold_version],
        ["python", "src/models/evaluate.py"],
    ]
    for cmd in steps:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=False, env=env)
        if result.returncode != 0:
            log.error(f"Step failed: {' '.join(cmd)}")
            return False
    if not Path(METRICS_PATH).exists():
        return False
    metrics   = json.loads(Path(METRICS_PATH).read_text())
    f1        = metrics.get("f1_macro", 0.0)
    THRESHOLD = 0.60
    if f1 >= THRESHOLD:
        log.info(f"Quality gate PASSED (F1={f1:.4f})")
        from src.monitoring.monitor import BASELINE_PATH, GOLD_DIR
        meta_path = GOLD_DIR / "meta.json"
        if meta_path.exists():
            import shutil
            shutil.copy(meta_path, BASELINE_PATH)
        return True
    log.warning(f"Quality gate FAILED (F1={f1:.4f})")
    return False

def main(dry_run=False, gold_version=None):
    log.info("=== Retrain Trigger ===")
    report = read_drift_report()
    if not report["retrain_required"]:
        log.info("No retrain needed.")
        sys.exit(0)
    if dry_run:
        log.info("--dry-run: skipping retrain.")
        sys.exit(0)
    version = gold_version or get_next_version()
    success = run_retrain(version)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gold-version", type=str, default=None)
    args = parser.parse_args()
    main(dry_run=args.dry_run, gold_version=args.gold_version)