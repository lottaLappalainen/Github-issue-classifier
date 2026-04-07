"""
src/monitoring/monitor.py  —  Data Drift + Concept Drift + Confidence Monitor

Five checks, ordered from data layer to model layer:

  1. class_distribution   — chi-square test on Gold label counts
                            (label distribution drift)
  2. dataset_size         — relative change in n_train
  3. class_balance        — majority/minority class ratio
  4. gold_version         — new data batch detected
  5. prediction_confidence— mean confidence of recent API predictions
                            (proxy for concept drift in production)

Plus one text-feature check (run separately via --text-drift):

  6. text_vocabulary      — Jaccard overlap between TF-IDF top-N token
                            sets of baseline vs current Silver data.
                            Low overlap = input distribution has shifted,
                            which can CAUSE concept drift.
                            (Concept drift = same words → different labels.
                             Text drift = different words altogether.)

Usage:
    python src/monitoring/monitor.py
    python src/monitoring/monitor.py --fail-on-drift
    python src/monitoring/monitor.py --text-drift  # also run vocabulary check
"""
import json
import sqlite3
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import chisquare

ROOT            = Path(__file__).resolve().parents[2]
GOLD_DIR        = ROOT / "data" / "gold"
SILVER_DIR      = ROOT / "data" / "silver"
MONITORING_DIR  = ROOT / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH        = MONITORING_DIR / "baseline_meta.json"
BASELINE_VOCAB_PATH  = MONITORING_DIR / "baseline_vocab.json"
DRIFT_REPORT         = MONITORING_DIR / "drift_report.json"
PRED_LOG_DB          = MONITORING_DIR / "prediction_log.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CHI2_P_THRESHOLD     = 0.05
SIZE_DRIFT_THRESHOLD = 0.20
BALANCE_RATIO_MAX    = 2.0
CONFIDENCE_MIN       = 0.70
CONFIDENCE_WINDOW    = 100
VOCAB_OVERLAP_MIN    = 0.60   # Jaccard similarity below this → text drift
VOCAB_TOP_N          = 500    # compare top-N tokens by frequency


def load_meta(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Meta file not found: {path}")
    return json.loads(path.read_text())


# ── Check 1: label distribution ────────────────────────────────────────────

def check_class_distribution(baseline, current):
    labels = sorted(
        set(baseline["class_distribution"]) | set(current["class_distribution"])
    )
    base_counts    = np.array([baseline["class_distribution"].get(l, 0) for l in labels], dtype=float)
    current_counts = np.array([current["class_distribution"].get(l, 0) for l in labels], dtype=float)
    base_props     = base_counts / base_counts.sum()
    expected       = base_props * current_counts.sum()
    mask           = expected > 0
    chi2_stat, p_value = chisquare(f_obs=current_counts[mask], f_exp=expected[mask])
    drift_detected = bool(p_value < CHI2_P_THRESHOLD)
    log.info(f"  Class distribution — chi2={chi2_stat:.4f}, p={p_value:.4f}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {
        "chi2_stat": round(float(chi2_stat), 4), "p_value": round(float(p_value), 4),
        "threshold": CHI2_P_THRESHOLD, "drift_detected": drift_detected,
        "baseline_dist": baseline["class_distribution"], "current_dist": current["class_distribution"],
    }


# ── Check 2: dataset size ──────────────────────────────────────────────────

def check_dataset_size(baseline, current):
    base_n    = baseline.get("n_train", 0)
    current_n = current.get("n_train", 0)
    rel_change = abs(current_n - base_n) / max(base_n, 1)
    drift_detected = rel_change > SIZE_DRIFT_THRESHOLD
    log.info(f"  Dataset size — base={base_n:,}, current={current_n:,}, Δ={rel_change:.1%}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {
        "baseline_n_train": base_n, "current_n_train": current_n,
        "relative_change": round(rel_change, 4), "threshold": SIZE_DRIFT_THRESHOLD,
        "drift_detected": drift_detected,
    }


# ── Check 3: class balance ─────────────────────────────────────────────────

def check_class_balance(current):
    counts = list(current["class_distribution"].values())
    if not counts or min(counts) == 0:
        return {"drift_detected": True, "reason": "zero-count class detected"}
    ratio = max(counts) / min(counts)
    drift_detected = ratio > BALANCE_RATIO_MAX
    log.info(f"  Class balance — ratio={ratio:.2f}  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {
        "imbalance_ratio": round(ratio, 4), "threshold": BALANCE_RATIO_MAX,
        "drift_detected": drift_detected, "class_counts": current["class_distribution"],
    }


# ── Check 4: gold version ──────────────────────────────────────────────────

def check_new_data_version(baseline, current):
    base_v    = baseline.get("gold_version", "unknown")
    current_v = current.get("gold_version", "unknown")
    new_batch = base_v != current_v
    log.info(f"  Gold version — baseline={base_v}, current={current_v}  {'🆕 NEW BATCH' if new_batch else '✅ same'}")
    return {"baseline_version": base_v, "current_version": current_v, "new_batch_detected": new_batch}


# ── Check 5: prediction confidence (concept drift proxy) ──────────────────

def check_prediction_confidence(db_path=PRED_LOG_DB, window=CONFIDENCE_WINDOW, threshold=CONFIDENCE_MIN):
    """
    Read the last `window` predictions from prediction_log.db (written by serve.py).

    WHY this detects concept drift:
      Concept drift means the relationship between input features and output
      labels has changed — the same issue text now implies a different priority.
      When this happens, the model's softmax probabilities become more uniform
      (less certain), causing mean confidence to drop.

      This is a production-side proxy: we don't need ground-truth labels to
      detect it. If the model trained on v1 data becomes uncertain on new
      production requests, that's a signal the world has changed.

      Compare: text_vocabulary check (below) detects input drift BEFORE
      production; this check detects degradation AFTER deployment.
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
    log.info(f"  Confidence drift — mean={mean_conf:.4f} (last {len(confidences)}, threshold={threshold})  {'⚠️  DRIFT' if drift_detected else '✅ OK'}")
    return {
        "drift_detected": drift_detected, "mean_confidence": round(mean_conf, 4),
        "min_confidence": round(min(confidences), 4), "max_confidence": round(max(confidences), 4),
        "threshold": threshold, "n_predictions": len(confidences), "window": window,
    }


# ── Check 6: text vocabulary / input feature drift ─────────────────────────

def _extract_top_tokens(silver_path: Path, top_n: int = VOCAB_TOP_N) -> set:
    """
    Load Silver parquet and extract the top-N tokens by frequency from
    the combined text column. Uses simple whitespace tokenisation
    (no sklearn dependency) so this runs fast as a monitoring check.
    """
    try:
        import pandas as pd
        df   = pd.read_parquet(silver_path)
        text = (df["title"].fillna("") + " " + df["body"].fillna("")).str.lower()
        from collections import Counter
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
    top_n:       int  = VOCAB_TOP_N,
) -> None:
    """Save the baseline vocabulary to JSON. Call once after first featurize."""
    tokens = _extract_top_tokens(silver_path, top_n)
    vocab_path = Path(vocab_path)
    vocab_path.write_text(json.dumps(sorted(tokens), indent=2))
    log.info(f"Baseline vocabulary saved ({len(tokens)} tokens) → {vocab_path}")


def check_text_drift(
    silver_path:  Path = SILVER_DIR / "issues_clean.parquet",
    vocab_path:   Path = BASELINE_VOCAB_PATH,
    top_n:        int  = VOCAB_TOP_N,
    min_overlap:  float = VOCAB_OVERLAP_MIN,
) -> dict:
    """
    Compare the top-N vocabulary of the current Silver data against the
    saved baseline vocabulary using Jaccard similarity.

    WHY this matters (from the DataOps/ModelOps lectures):
      Data drift = shift in input feature distribution.
      If new GitHub issues use completely different terminology (e.g., a
      new framework becomes popular and its bug reports use new jargon),
      the TF-IDF features the model was trained on become stale — the
      feature space has changed.

      This is distinct from concept drift: here the WORDS have changed,
      not just the label relationships. Low vocabulary overlap is an early
      warning that the model will degrade even before confidence drops.

    Jaccard similarity = |intersection| / |union|
    Score of 1.0 = identical vocabulary, 0.0 = no overlap at all.
    """
    vocab_path  = Path(vocab_path)
    silver_path = Path(silver_path)

    if not vocab_path.exists():
        log.info("  Text drift — no baseline vocabulary found  ✅ skipped (run build_vocab_baseline first)")
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

    intersection    = baseline_tokens & current_tokens
    union           = baseline_tokens | current_tokens
    jaccard         = len(intersection) / len(union) if union else 1.0
    drift_detected  = jaccard < min_overlap

    log.info(
        f"  Text vocabulary — Jaccard={jaccard:.4f} "
        f"({len(intersection)}/{len(union)} tokens overlap, threshold={min_overlap})  "
        f"{'⚠️  DRIFT' if drift_detected else '✅ OK'}"
    )
    return {
        "drift_detected":    drift_detected,
        "jaccard_similarity": round(jaccard, 4),
        "baseline_vocab_size": len(baseline_tokens),
        "current_vocab_size":  len(current_tokens),
        "overlap_tokens":      len(intersection),
        "threshold":           min_overlap,
        "interpretation": (
            "Input feature drift: the vocabulary of incoming issues has shifted. "
            "This can CAUSE concept drift if the model's TF-IDF features no longer "
            "represent the current issue space."
        ),
    }


# ── main ───────────────────────────────────────────────────────────────────

def run_monitoring(
    baseline_path: Path = BASELINE_PATH,
    meta_path:     Optional[Path] = None,
    fail_on_drift: bool = False,
    run_text_drift: bool = False,
) -> dict:
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

    if run_text_drift:
        checks["text_vocabulary"] = check_text_drift()

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
        log.info("Baseline updated.")

    if fail_on_drift and any_drift:
        sys.exit(1)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Gold data and model drift")
    parser.add_argument("--baseline",    type=Path, default=BASELINE_PATH)
    parser.add_argument("--meta",        type=Path, default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--text-drift",  action="store_true",
                        help="Also run vocabulary Jaccard check (input feature drift)")
    args = parser.parse_args()
    run_monitoring(
        baseline_path=args.baseline,
        meta_path=args.meta,
        fail_on_drift=args.fail_on_drift,
        run_text_drift=args.text_drift,
    )