"""
src/monitoring/monitor.py  —  Data Drift + Concept Drift + Confidence Monitor

Parameters are read from params.yaml (monitor section) so thresholds
are tracked by DVC and version-controlled alongside code.

Six checks, ordered from data layer to model layer:
  1. class_distribution    — chi-square test on Gold label counts
  2. dataset_size          — relative change in n_train
  3. class_balance         — majority/minority class ratio
  4. gold_version          — new data batch detected
  5. prediction_confidence — mean API confidence (concept drift proxy)
  6. text_vocabulary       — Jaccard overlap of top-N tokens (input drift)

On FIRST RUN (no baseline exists):
  - Saves meta.json as baseline_meta.json
  - Saves top-N tokens as baseline_vocab.json
  - Writes drift_report.json with status=baseline_created
  - Exits cleanly (no false drift alarm)

Usage:
    python src/monitoring/monitor.py
    python src/monitoring/monitor.py --text-drift
    python src/monitoring/monitor.py --fail-on-drift
"""
import json
import sqlite3
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
from scipy.stats import chisquare

ROOT            = Path(__file__).resolve().parents[2]
GOLD_DIR        = ROOT / "data" / "gold"
SILVER_DIR      = ROOT / "data" / "silver"
MONITORING_DIR  = ROOT / "monitoring"
PARAMS_PATH     = ROOT / "params.yaml"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH       = MONITORING_DIR / "baseline_meta.json"
BASELINE_VOCAB_PATH = MONITORING_DIR / "baseline_vocab.json"
DRIFT_REPORT        = MONITORING_DIR / "drift_report.json"
PRED_LOG_DB         = MONITORING_DIR / "prediction_log.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _load_params() -> dict:
    """Load monitor thresholds from params.yaml, falling back to safe defaults."""
    defaults = {
        "chi2_p_threshold":     0.05,
        "size_drift_threshold": 0.20,
        "balance_ratio_max":    2.0,
        "confidence_min":       0.70,
        "confidence_window":    100,
        "vocab_overlap_min":    0.60,
        "vocab_top_n":          500,
    }
    if not PARAMS_PATH.exists():
        log.warning("params.yaml not found — using default monitor thresholds")
        return defaults
    with open(PARAMS_PATH) as f:
        all_params = yaml.safe_load(f)
    params = {**defaults, **all_params.get("monitor", {})}
    log.info(f"Loaded monitor params from params.yaml: {params}")
    return params


def load_meta(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Meta file not found: {path}")
    return json.loads(path.read_text())


# ── Check 1 ────────────────────────────────────────────────────────────────

def check_class_distribution(baseline: dict, current: dict, p_threshold: float) -> dict:
    labels = sorted(
        set(baseline["class_distribution"]) | set(current["class_distribution"])
    )
    base_counts    = np.array([baseline["class_distribution"].get(l, 0) for l in labels], dtype=float)
    current_counts = np.array([current["class_distribution"].get(l, 0)  for l in labels], dtype=float)
    base_props = base_counts / base_counts.sum()
    expected   = base_props * current_counts.sum()
    mask       = expected > 0
    chi2_stat, p_value = chisquare(f_obs=current_counts[mask], f_exp=expected[mask])
    drift_detected = bool(p_value < p_threshold)
    log.info(f"  Class distribution — chi2={chi2_stat:.4f}, p={p_value:.4f}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {
        "chi2_stat": round(float(chi2_stat), 4), "p_value": round(float(p_value), 4),
        "threshold": p_threshold, "drift_detected": drift_detected,
        "baseline_dist": baseline["class_distribution"], "current_dist": current["class_distribution"],
    }


# ── Check 2 ────────────────────────────────────────────────────────────────

def check_dataset_size(baseline: dict, current: dict, size_threshold: float) -> dict:
    base_n     = baseline.get("n_train", 0)
    current_n  = current.get("n_train", 0)
    rel_change = abs(current_n - base_n) / max(base_n, 1)
    drift_detected = rel_change > size_threshold
    log.info(f"  Dataset size — base={base_n:,}, current={current_n:,}, Δ={rel_change:.1%}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {
        "baseline_n_train": base_n, "current_n_train": current_n,
        "relative_change": round(rel_change, 4), "threshold": size_threshold,
        "drift_detected": drift_detected,
    }


# ── Check 3 ────────────────────────────────────────────────────────────────

def check_class_balance(current: dict, ratio_max: float) -> dict:
    counts = list(current["class_distribution"].values())
    if not counts or min(counts) == 0:
        return {"drift_detected": True, "reason": "zero-count class detected"}
    ratio = max(counts) / min(counts)
    drift_detected = ratio > ratio_max
    log.info(f"  Class balance — ratio={ratio:.2f}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {
        "imbalance_ratio": round(ratio, 4), "threshold": ratio_max,
        "drift_detected": drift_detected, "class_counts": current["class_distribution"],
    }


# ── Check 4 ────────────────────────────────────────────────────────────────

def check_new_data_version(baseline: dict, current: dict) -> dict:
    base_v    = baseline.get("gold_version", "unknown")
    current_v = current.get("gold_version", "unknown")
    new_batch = base_v != current_v
    log.info(f"  Gold version — baseline={base_v}, current={current_v}  {'🆕 NEW BATCH' if new_batch else '✅ same'}")
    return {"baseline_version": base_v, "current_version": current_v, "new_batch_detected": new_batch}


# ── Check 5 ────────────────────────────────────────────────────────────────

def check_prediction_confidence(
    db_path=PRED_LOG_DB,
    window: int = 100,
    threshold: float = 0.70,
) -> dict:
    """
    Read the last `window` predictions from prediction_log.db.
    Concept drift proxy: if the model becomes uncertain about production
    requests (mean confidence drops), the input-output relationship may
    have shifted — even if the vocabulary looks familiar.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        log.info("  Confidence drift — no prediction log found  ✅ skipped")
        return {"drift_detected": False, "mean_confidence": None,
                "threshold": threshold, "n_predictions": 0,
                "note": "prediction_log.db not found yet"}
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
        return {"drift_detected": False, "mean_confidence": None,
                "threshold": threshold, "n_predictions": 0,
                "note": "No predictions logged yet"}

    confidences    = [r[0] for r in rows]
    mean_conf      = sum(confidences) / len(confidences)
    drift_detected = mean_conf < threshold
    log.info(f"  Confidence drift — mean={mean_conf:.4f} (last {len(confidences)}, threshold={threshold})  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {
        "drift_detected": drift_detected, "mean_confidence": round(mean_conf, 4),
        "min_confidence": round(min(confidences), 4), "max_confidence": round(max(confidences), 4),
        "threshold": threshold, "n_predictions": len(confidences), "window": window,
    }


# ── Check 6 ────────────────────────────────────────────────────────────────

def _extract_top_tokens(silver_path: Path, top_n: int) -> set:
    """Extract top-N tokens by frequency from the Silver text columns."""
    try:
        from collections import Counter
        import pandas as pd
        df   = pd.read_parquet(silver_path)
        text = (df["title"].fillna("") + " " + df["body"].fillna("")).str.lower()
        counts = Counter()
        for row in text:
            counts.update(row.split())
        return set(tok for tok, _ in counts.most_common(top_n))
    except Exception as exc:
        log.warning(f"  Could not extract tokens from {silver_path}: {exc}")
        return set()


def build_vocab_baseline(
    silver_path: Path = SILVER_DIR / "issues_clean.parquet",
    vocab_path:  Path = BASELINE_VOCAB_PATH,
    top_n:       int  = 500,
) -> None:
    """Save baseline vocabulary. Called automatically on first monitor run."""
    tokens = _extract_top_tokens(Path(silver_path), top_n)
    Path(vocab_path).write_text(json.dumps(sorted(tokens), indent=2))
    log.info(f"Baseline vocabulary saved ({len(tokens)} tokens) → {vocab_path}")


def check_text_drift(
    silver_path: Path = SILVER_DIR / "issues_clean.parquet",
    vocab_path:  Path = BASELINE_VOCAB_PATH,
    top_n:       int  = 500,
    min_overlap: float = 0.60,
) -> dict:
    """
    Jaccard similarity between baseline and current top-N token sets.
    Input feature drift: vocabulary has shifted, meaning the TF-IDF
    feature space the model was trained on is becoming stale.
    This is distinct from concept drift (check 5): here the WORDS have
    changed, not just the label relationships.
    """
    vocab_path  = Path(vocab_path)
    silver_path = Path(silver_path)

    if not vocab_path.exists():
        log.info("  Text drift — no baseline vocabulary  ✅ skipped")
        return {"drift_detected": False, "jaccard_similarity": None,
                "threshold": min_overlap, "note": "baseline_vocab.json not found"}
    if not silver_path.exists():
        log.info("  Text drift — Silver file not found  ✅ skipped")
        return {"drift_detected": False, "jaccard_similarity": None,
                "threshold": min_overlap, "note": "Silver parquet not found"}

    baseline_tokens = set(json.loads(vocab_path.read_text()))
    current_tokens  = _extract_top_tokens(silver_path, top_n)
    if not current_tokens:
        return {"drift_detected": False, "jaccard_similarity": None,
                "note": "Could not extract tokens from current Silver data"}

    intersection   = baseline_tokens & current_tokens
    union          = baseline_tokens | current_tokens
    jaccard        = len(intersection) / len(union) if union else 1.0
    drift_detected = jaccard < min_overlap

    log.info(
        f"  Text vocabulary — Jaccard={jaccard:.4f} "
        f"({len(intersection)}/{len(union)} tokens, threshold={min_overlap})  "
        f"{'⚠️  DRIFT' if drift_detected else '✅ OK'}"
    )
    return {
        "drift_detected":     drift_detected,
        "jaccard_similarity": round(jaccard, 4),
        "baseline_vocab_size": len(baseline_tokens),
        "current_vocab_size":  len(current_tokens),
        "overlap_tokens":      len(intersection),
        "threshold":           min_overlap,
    }


# ── main ───────────────────────────────────────────────────────────────────

def run_monitoring(
    baseline_path:  Path = BASELINE_PATH,
    meta_path:      Optional[Path] = None,
    fail_on_drift:  bool = False,
    run_text_drift: bool = False,
) -> dict:
    log.info("=== Drift Monitoring ===")

    # ── Read thresholds from params.yaml ───────────────────────────────────
    params = _load_params()

    meta_path = Path(meta_path) if meta_path else (GOLD_DIR / "meta.json")
    current   = load_meta(meta_path)

    # ── First run: save BOTH baselines, exit cleanly ───────────────────────
    if not Path(baseline_path).exists():
        log.info("No baseline found — first run. Saving baselines, no drift check.")
        Path(baseline_path).write_text(json.dumps(current, indent=2))
        log.info(f"  Saved meta baseline → {baseline_path}")

        # Auto-init vocabulary baseline so text drift check works next run
        silver_path = SILVER_DIR / "issues_clean.parquet"
        if silver_path.exists():
            build_vocab_baseline(
                silver_path=silver_path,
                vocab_path=BASELINE_VOCAB_PATH,
                top_n=params["vocab_top_n"],
            )
        else:
            log.warning("  Silver file not found — skipping vocab baseline init")

        report = {"status": "baseline_created", "retrain_required": False, "checks": {}}
        DRIFT_REPORT.write_text(json.dumps(report, indent=2))
        return report

    baseline = load_meta(baseline_path)
    log.info(f"Comparing current Gold vs baseline ({Path(baseline_path).name})")

    # ── Run all checks using thresholds from params.yaml ───────────────────
    checks = {
        "class_distribution":    check_class_distribution(
            baseline, current, params["chi2_p_threshold"]
        ),
        "dataset_size":          check_dataset_size(
            baseline, current, params["size_drift_threshold"]
        ),
        "class_balance":         check_class_balance(
            current, params["balance_ratio_max"]
        ),
        "gold_version":          check_new_data_version(baseline, current),
        "prediction_confidence": check_prediction_confidence(
            window=params["confidence_window"],
            threshold=params["confidence_min"],
        ),
    }

    if run_text_drift:
        checks["text_vocabulary"] = check_text_drift(
            top_n=params["vocab_top_n"],
            min_overlap=params["vocab_overlap_min"],
        )

    any_drift = any(
        v.get("drift_detected") or v.get("new_batch_detected")
        for v in checks.values()
    )

    report = {
        "status":           "drift_detected" if any_drift else "no_drift",
        "retrain_required": any_drift,
        "checks":           checks,
    }

    DRIFT_REPORT.write_text(json.dumps(report, indent=2))
    log.info(f"\nDrift report → {DRIFT_REPORT}")
    log.info(f"Overall: {'⚠️  RETRAIN REQUIRED' if any_drift else '✅ No retrain needed'}")

    if not any_drift:
        Path(baseline_path).write_text(json.dumps(current, indent=2))
        log.info("Baseline updated to current Gold meta.")

    if fail_on_drift and any_drift:
        sys.exit(1)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Gold data and model drift")
    parser.add_argument("--baseline",    type=Path, default=BASELINE_PATH)
    parser.add_argument("--meta",        type=Path, default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--text-drift",  action="store_true",
                        help="Also run vocabulary Jaccard check")
    args = parser.parse_args()
    run_monitoring(
        baseline_path=args.baseline,
        meta_path=args.meta,
        fail_on_drift=args.fail_on_drift,
        run_text_drift=args.text_drift,
    )