"""
src/monitoring/monitor.py  —  Data Drift + Class Balance Monitor

Compares the current Gold meta.json against a saved baseline to detect:
  1. Class distribution drift  (chi-square test on label counts)
  2. Dataset size drift        (relative change in n_train)
  3. Class balance drift       (largest imbalance ratio between classes)

Writes drift_report.json and exits with code 1 if drift is detected,
so it can be used as a DVC/CI gate.

Usage:
    python src/monitoring/monitor.py
    python src/monitoring/monitor.py --baseline monitoring/baseline_meta.json
    python src/monitoring/monitor.py --fail-on-drift   # exit 1 if drift found
"""
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import chisquare

# ── paths ──────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[2]
GOLD_DIR        = ROOT / "data" / "gold"
MONITORING_DIR  = ROOT / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH   = MONITORING_DIR / "baseline_meta.json"
DRIFT_REPORT    = MONITORING_DIR / "drift_report.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── thresholds ─────────────────────────────────────────────────────────────
# Chi-square p-value below this → distribution has shifted significantly
CHI2_P_THRESHOLD      = 0.05
# Relative change in dataset size above this → flag as drift
SIZE_DRIFT_THRESHOLD  = 0.20   # 20 %
# Imbalance ratio above this → flag as balance drift (majority / minority)
BALANCE_RATIO_MAX     = 2.0


# ── helpers ────────────────────────────────────────────────────────────────

def load_meta(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Meta file not found: {path}")
    return json.loads(path.read_text())


def check_class_distribution(
    baseline: dict,
    current: dict,
) -> dict:
    """
    Chi-square test: does the current label distribution differ from baseline?
    Works on the 'class_distribution' key in meta.json.
    """
    labels = sorted(set(baseline["class_distribution"]) | set(current["class_distribution"]))

    base_counts    = np.array([baseline["class_distribution"].get(l, 0) for l in labels], dtype=float)
    current_counts = np.array([current["class_distribution"].get(l, 0)  for l in labels], dtype=float)

    # Normalise baseline to expected proportions scaled to current total
    base_props = base_counts / base_counts.sum()
    expected   = base_props * current_counts.sum()

    # Avoid division-by-zero for zero-expected cells
    mask     = expected > 0
    chi2_stat, p_value = chisquare(
        f_obs=current_counts[mask],
        f_exp=expected[mask],
    )

    drift_detected = bool(p_value < CHI2_P_THRESHOLD)
    result = {
        "chi2_stat":      round(float(chi2_stat), 4),
        "p_value":        round(float(p_value), 4),
        "threshold":      CHI2_P_THRESHOLD,
        "drift_detected": drift_detected,
        "baseline_dist":  baseline["class_distribution"],
        "current_dist":   current["class_distribution"],
    }
    log.info(f"  Class distribution — chi2={chi2_stat:.4f}, p={p_value:.4f}  "
             f"{'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return result


def check_dataset_size(baseline: dict, current: dict) -> dict:
    """Flag if the training set size changed by more than SIZE_DRIFT_THRESHOLD."""
    base_n    = baseline.get("n_train", 0)
    current_n = current.get("n_train", 0)
    rel_change = abs(current_n - base_n) / max(base_n, 1)

    drift_detected = rel_change > SIZE_DRIFT_THRESHOLD
    result = {
        "baseline_n_train": base_n,
        "current_n_train":  current_n,
        "relative_change":  round(rel_change, 4),
        "threshold":        SIZE_DRIFT_THRESHOLD,
        "drift_detected":   drift_detected,
    }
    log.info(f"  Dataset size — base={base_n:,}, current={current_n:,}, "
             f"Δ={rel_change:.1%}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return result


def check_class_balance(current: dict) -> dict:
    """
    Check that no class dominates by more than BALANCE_RATIO_MAX.
    This catches silent balance failures in featurize.py.
    """
    counts = list(current["class_distribution"].values())
    if not counts or min(counts) == 0:
        return {"drift_detected": True, "reason": "zero-count class detected"}

    ratio = max(counts) / min(counts)
    drift_detected = ratio > BALANCE_RATIO_MAX
    result = {
        "imbalance_ratio": round(ratio, 4),
        "threshold":       BALANCE_RATIO_MAX,
        "drift_detected":  drift_detected,
        "class_counts":    current["class_distribution"],
    }
    log.info(f"  Class balance — ratio={ratio:.2f}  "
             f"{'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return result


def check_new_data_version(baseline: dict, current: dict) -> dict:
    """
    Simple version-bump check: if gold_version in meta changed, always retrain.
    featurize.py writes 'gold_version' into meta.json.
    """
    base_v    = baseline.get("gold_version", "unknown")
    current_v = current.get("gold_version", "unknown")
    new_batch = base_v != current_v
    result = {
        "baseline_version": base_v,
        "current_version":  current_v,
        "new_batch_detected": new_batch,
    }
    log.info(f"  Gold version — baseline={base_v}, current={current_v}  "
             f"{'🆕 NEW BATCH' if new_batch else '✅ same'}")
    return result


# ── main ───────────────────────────────────────────────────────────────────

def run_monitoring(
    baseline_path: Path = BASELINE_PATH,
    meta_path: Optional[Path] = None,
    fail_on_drift: bool = False,
) -> dict:
    log.info("=== Drift Monitoring ===")

    meta_path = meta_path or (GOLD_DIR / "meta.json")
    current   = load_meta(meta_path)

    # ── First run: save baseline and exit clean ────────────────────────────
    if not baseline_path.exists():
        log.info("No baseline found — saving current meta as baseline. No drift check.")
        baseline_path.write_text(json.dumps(current, indent=2))
        report = {
            "status":           "baseline_created",
            "retrain_required": False,
            "checks":           {},
        }
        DRIFT_REPORT.write_text(json.dumps(report, indent=2))
        log.info(f"Baseline saved → {baseline_path}")
        return report

    baseline = load_meta(baseline_path)
    log.info(f"Comparing current Gold vs baseline ({baseline_path.name})")

    # ── Run all checks ─────────────────────────────────────────────────────
    checks = {
        "class_distribution": check_class_distribution(baseline, current),
        "dataset_size":       check_dataset_size(baseline, current),
        "class_balance":      check_class_balance(current),
        "gold_version":       check_new_data_version(baseline, current),
    }

    any_drift = any(v.get("drift_detected") or v.get("new_batch_detected")
                    for v in checks.values())

    # Retrain is required if ANY check fires
    retrain_required = any_drift

    report = {
        "status":           "drift_detected" if any_drift else "no_drift",
        "retrain_required": retrain_required,
        "checks":           checks,
    }

    DRIFT_REPORT.write_text(json.dumps(report, indent=2))
    log.info(f"\nDrift report written → {DRIFT_REPORT}")
    log.info(f"Overall: {'⚠️  RETRAIN REQUIRED' if retrain_required else '✅ No retrain needed'}")

    # ── Update baseline only when no drift (stable state) ─────────────────
    # If drift was detected, keep old baseline so the gap stays measurable
    if not any_drift:
        baseline_path.write_text(json.dumps(current, indent=2))
        log.info("Baseline updated to current Gold meta.")

    if fail_on_drift and retrain_required:
        log.warning("--fail-on-drift set: exiting with code 1")
        sys.exit(1)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Gold data drift")
    parser.add_argument("--baseline",      type=Path, default=BASELINE_PATH)
    parser.add_argument("--meta",          type=Path, default=None,
                        help="Path to current Gold meta.json")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="Exit code 1 if drift/retrain is required")
    args = parser.parse_args()

    run_monitoring(
        baseline_path=args.baseline,
        meta_path=args.meta,
        fail_on_drift=args.fail_on_drift,
    )