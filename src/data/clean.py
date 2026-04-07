"""
src/data/clean.py  —  Silver Layer
Cleans Bronze issues: deduplicates, assigns priority labels, fills nulls.

Usage:
    python src/data/clean.py
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
BRONZE_PATH = ROOT / "data" / "bronze" / "issues_raw.parquet"
SILVER_DIR  = ROOT / "data" / "silver"
SILVER_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── label maps ─────────────────────────────────────────────────────────────
HIGH_KEYWORDS   = {"bug", "critical", "crash", "priority:high", "severity:high",
                   "security", "regression", "blocker", "urgent"}
MEDIUM_KEYWORDS = {"enhancement", "feature", "feature-request", "priority:medium",
                   "improvement", "performance", "request"}
LOW_KEYWORDS    = {"documentation", "docs", "good first issue", "help wanted",
                   "priority:low", "question", "wontfix", "duplicate"}


# ── public helpers (imported by tests) ─────────────────────────────────────

def assign_priority(labels_str: str) -> Optional[str]:
    """
    Map a comma-separated GitHub labels string to high / medium / low.
    Returns None if no known label is found.
    """
    if not labels_str or not labels_str.strip():
        return None

    labels = {lbl.strip().lower() for lbl in labels_str.split(",")}

    if labels & HIGH_KEYWORDS:
        return "high"
    if labels & MEDIUM_KEYWORDS:
        return "medium"
    if labels & LOW_KEYWORDS:
        return "low"
    return None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bronze → Silver transformation.
    - Deduplicates on (repo, number)
    - Fills null bodies with ""
    - Drops empty titles
    - Assigns priority labels
    - Drops rows with no assignable priority
    - Creates combined `text` column
    """
    log.info(f"  input rows: {len(df):,}")

    # 1. Deduplicate
    df = df.drop_duplicates(subset=["repo", "number"]).copy()
    log.info(f"  after dedup: {len(df):,}")

    # 2. Fill nulls
    df["body"]  = df["body"].fillna("").str.strip()
    df["title"] = df["title"].fillna("").str.strip()

    # 3. Drop empty titles
    df = df[df["title"] != ""]
    log.info(f"  after dropping empty titles: {len(df):,}")

    # 4. Assign priority
    df["priority"] = df["labels"].apply(assign_priority)

    # 5. Drop unlabelled rows
    df = df[df["priority"].notna()].copy()
    log.info(f"  after dropping unlabelled: {len(df):,}")

    # 6. Create combined text field
    df["text"] = (df["title"] + " " + df["body"]).str.strip()

    log.info(f"  priority distribution:\n{df['priority'].value_counts().to_string()}")
    return df.reset_index(drop=True)


def save_silver(df: pd.DataFrame, path: Path) -> None:
    """Write Silver dataframe to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info(f"  saved {len(df):,} rows → {path}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Silver Layer: Cleaning ===")

    if not BRONZE_PATH.exists():
        raise FileNotFoundError(f"Bronze file not found: {BRONZE_PATH}\nRun ingest.py first.")

    df = pd.read_parquet(BRONZE_PATH)
    log.info(f"Loaded {len(df):,} rows from Bronze")

    silver = clean(df)
    out_path = SILVER_DIR / "issues_clean.parquet"
    save_silver(silver, out_path)
    log.info(f"\nDone. {len(silver):,} clean issues saved to {out_path}")


if __name__ == "__main__":
    main()