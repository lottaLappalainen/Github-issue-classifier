"""
src/monitoring/monitor.py  —  Data Drift + Class Balance + Confidence Monitor

Compares the current Gold meta.json against a saved baseline to detect:
  1. Class distribution drift  (chi-square test on label counts)
  2. Dataset size drift        (relative change in n_train)
  3. Class balance drift       (largest imbalance ratio between classes)
  4. Gold version change       (new data batch detected)
  5. Prediction confidence drift  (mean confidence from prediction_log.db
                                   dropped below threshold)

Usage:
    python src/monitoring/monitor.py
    python src/monitoring/monitor.py --fail-on-drift
"""
import json, sqlite3, logging, argparse, sys
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.stats import chisquare

ROOT            = Path(__file__).resolve().parents[2]
GOLD_DIR        = ROOT / "data" / "gold"
MONITORING_DIR  = ROOT / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_PATH   = MONITORING_DIR / "baseline_meta.json"
DRIFT_REPORT    = MONITORING_DIR / "drift_report.json"
PRED_LOG_DB     = MONITORING_DIR / "prediction_log.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CHI2_P_THRESHOLD     = 0.05
SIZE_DRIFT_THRESHOLD = 0.20
BALANCE_RATIO_MAX    = 2.0
CONFIDENCE_MIN       = 0.70
CONFIDENCE_WINDOW    = 100


def load_meta(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Meta file not found: {path}")
    return json.loads(path.read_text())


def check_class_distribution(baseline, current):
    labels = sorted(set(baseline["class_distribution"]) | set(current["class_distribution"]))
    base_counts    = np.array([baseline["class_distribution"].get(l, 0) for l in labels], dtype=float)
    current_counts = np.array([current["class_distribution"].get(l, 0) for l in labels], dtype=float)
    base_props = base_counts / base_counts.sum()
    expected   = base_props * current_counts.sum()
    mask = expected > 0
    chi2_stat, p_value = chisquare(f_obs=current_counts[mask], f_exp=expected[mask])
    drift_detected = bool(p_value < CHI2_P_THRESHOLD)
    log.info(f"  Class distribution — chi2={chi2_stat:.4f}, p={p_value:.4f}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {"chi2_stat": round(float(chi2_stat),4), "p_value": round(float(p_value),4),
            "threshold": CHI2_P_THRESHOLD, "drift_detected": drift_detected,
            "baseline_dist": baseline["class_distribution"], "current_dist": current["class_distribution"]}


def check_dataset_size(baseline, current):
    base_n    = baseline.get("n_train", 0)
    current_n = current.get("n_train", 0)
    rel_change = abs(current_n - base_n) / max(base_n, 1)
    drift_detected = rel_change > SIZE_DRIFT_THRESHOLD
    log.info(f"  Dataset size — base={base_n:,}, current={current_n:,}, Δ={rel_change:.1%}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {"baseline_n_train": base_n, "current_n_train": current_n,
            "relative_change": round(rel_change,4), "threshold": SIZE_DRIFT_THRESHOLD,
            "drift_detected": drift_detected}


def check_class_balance(current):
    counts = list(current["class_distribution"].values())
    if not counts or min(counts) == 0:
        return {"drift_detected": True, "reason": "zero-count class detected"}
    ratio = max(counts) / min(counts)
    drift_detected = ratio > BALANCE_RATIO_MAX
    log.info(f"  Class balance — ratio={ratio:.2f}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {"imbalance_ratio": round(ratio,4), "threshold": BALANCE_RATIO_MAX,
            "drift_detected": drift_detected, "class_counts": current["class_distribution"]}


def check_new_data_version(baseline, current):
    base_v    = baseline.get("gold_version", "unknown")
    current_v = current.get("gold_version", "unknown")
    new_batch = base_v != current_v
    log.info(f"  Gold version — baseline={base_v}, current={current_v}  {'🆕 NEW BATCH' if new_batch else '✅ same'}")
    return {"baseline_version": base_v, "current_version": current_v, "new_batch_detected": new_batch}


def check_prediction_confidence(db_path=PRED_LOG_DB, window=CONFIDENCE_WINDOW, threshold=CONFIDENCE_MIN):
    """
    Read the last `window` predictions from prediction_log.db (written by serve.py).
    If mean confidence drops below `threshold`, the model is likely degrading in
    production — a signal of concept drift or distribution shift.
    Skipped gracefully if the API has not served any predictions yet.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        log.info("  Confidence drift — no prediction log found  ✅ skipped")
        return {"drift_detected": False, "mean_confidence": None, "threshold": threshold,
                "n_predictions": 0, "note": "prediction_log.db not found yet"}
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT confidence FROM predictions ORDER BY id DESC LIMIT ?", (window,)
            ).fetchall()
    except Exception as exc:
        log.warning(f"  Could not read prediction log: {exc}")
        return {"drift_detected": False, "error": str(exc), "n_predictions": 0}

    if not rows:
        log.info("  Confidence drift — empty log  ✅ skipped")
        return {"drift_detected": False, "mean_confidence": None, "threshold": threshold,
                "n_predictions": 0, "note": "No predictions logged yet"}

    confidences    = [r[0] for r in rows]
    mean_conf      = sum(confidences) / len(confidences)
    drift_detected = mean_conf < threshold
    log.info(f"  Confidence drift — mean={mean_conf:.4f} (last {len(confidences)} preds, threshold={threshold})  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {"drift_detected": drift_detected, "mean_confidence": round(mean_conf,4),
            "min_confidence": round(min(confidences),4), "max_confidence": round(max(confidences),4),
            "threshold": threshold, "n_predictions": len(confidences), "window": window}


def run_monitoring(baseline_path=BASELINE_PATH, meta_path=None, fail_on_drift=False):
    log.info("=== Drift Monitoring ===")
    meta_path = Path(meta_path) if meta_path else (GOLD_DIR / "meta.json")
    current   = load_meta(meta_path)

    if not Path(baseline_path).exists():
        log.info("No baseline found — saving current meta as baseline.")
        Path(baseline_path).write_text(json.dumps(current, indent=2))
        report = {"status": "baseline_created", "retrain_required": False, "checks": {}}
        DRIFT_REPORT.write_text(json.dumps(report, indent=2))
        return report

    baseline = load_meta(baseline_path)
    log.info(f"Comparing current Gold vs baseline ({Path(baseline_path).name})")

    checks = {
        "class_distribution":    check_class_distribution(baseline, current),
        "dataset_size":          check_dataset_size(baseline, current),
        "class_balance":         check_class_balance(current),
        "gold_version":          check_new_data_version(baseline, current),
        "prediction_confidence": check_prediction_confidence(),
    }

    any_drift = any(v.get("drift_detected") or v.get("new_batch_detected") for v in checks.values())
    report = {"status": "drift_detected" if any_drift else "no_drift",
              "retrain_required": any_drift, "checks": checks}

    DRIFT_REPORT.write_text(json.dumps(report, indent=2))
    log.info(f"\nDrift report → {DRIFT_REPORT}")
    log.info(f"Overall: {'⚠️  RETRAIN REQUIRED' if any_drift else '✅ No retrain needed'}")

    if not any_drift:
        Path(baseline_path).write_text(json.dumps(current, indent=2))
        log.info("Baseline updated.")

    if fail_on_drift and any_drift:
        sys.exit(1)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    run_monitoring(baseline_path=args.baseline, meta_path=args.meta, fail_on_drift=args.fail_on_drift)