"""
src/data/featurize.py  —  Gold Layer
Reads clean Silver parquet, combines text, balances classes,
splits train/test, and saves Gold artifacts as parquet.

Parameters are read from params.yaml (featurize section) so that
DVC can detect changes and retrigger this stage automatically.

Outputs:
    data/gold/train.parquet  — columns: text, priority
    data/gold/test.parquet   — columns: text, priority
    data/gold/meta.json      — dataset statistics + gold_version

Usage:
    python src/data/featurize.py
    python src/data/featurize.py --gold-version v2
"""
import json
import hashlib
import logging
import argparse
from pathlib import Path

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
SILVER_PATH = ROOT / "data" / "silver" / "issues_clean.parquet"
GOLD_DIR    = ROOT / "data" / "gold"
PARAMS_PATH = ROOT / "params.yaml"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _load_params() -> dict:
    """
    Load featurize parameters from params.yaml.
    Falls back to hardcoded defaults if the file is missing,
    so tests that don't have the file on disk still work.
    """
    defaults = {"test_size": 0.2, "random_state": 42}
    if not PARAMS_PATH.exists():
        log.warning(f"params.yaml not found at {PARAMS_PATH} — using defaults {defaults}")
        return defaults
    with open(PARAMS_PATH) as f:
        all_params = yaml.safe_load(f)
    params = all_params.get("featurize", defaults)
    log.info(f"Loaded featurize params from params.yaml: {params}")
    return params


# ── public helpers (importable by tests) ───────────────────────────────────

def combine_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine title + body into a single text field.
    Title is repeated 3x to give it more weight than body.
    """
    title = df["title"].fillna("").str.strip()
    body  = df["body"].fillna("").str.strip()
    return (title + " " + title + " " + title + " " + body).str.strip()


def balance_classes(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Upsample minority classes to match the majority class size.
    Prevents the model from just predicting the dominant class.
    """
    max_count = df["priority"].value_counts().max()
    parts = []
    for priority in df["priority"].unique():
        subset = df[df["priority"] == priority]
        if len(subset) < max_count:
            subset = resample(
                subset,
                replace=True,
                n_samples=max_count,
                random_state=random_state,
            )
        parts.append(subset)
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def featurize(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Silver → Gold transformation.
    Returns (train_df, test_df) each with columns: text, priority.
    test_size and random_state come from params.yaml via _load_params().
    """
    log.info(f"  input rows: {len(df):,}")
    log.info(f"  class distribution (before balance):\n{df['priority'].value_counts().to_string()}")

    df = df.copy()
    df["text"] = combine_text(df)
    df = df[df["text"].str.len() > 0]
    log.info(f"  after dropping empty text: {len(df):,}")

    df_balanced = balance_classes(df[["text", "priority"]], random_state=random_state)
    log.info(f"  after balancing: {len(df_balanced):,}")
    log.info(f"  class distribution (after balance):\n{df_balanced['priority'].value_counts().to_string()}")

    train_df, test_df = train_test_split(
        df_balanced,
        test_size=test_size,
        random_state=random_state,
        stratify=df_balanced["priority"],
    )
    log.info(f"  train: {len(train_df):,} | test: {len(test_df):,}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _compute_silver_hash(silver_path: Path) -> str:
    """SHA-256 of the Silver file — used to auto-detect a new data batch."""
    h = hashlib.sha256()
    with open(silver_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def save_gold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gold_dir: Path,
    silver_path: Path,
    gold_version: str,
) -> None:
    """Save Gold train/test splits and enriched metadata."""
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(gold_dir / "train.parquet", index=False)
    test_df.to_parquet(gold_dir  / "test.parquet",  index=False)
    log.info(f"  saved train → {gold_dir / 'train.parquet'}")
    log.info(f"  saved test  → {gold_dir / 'test.parquet'}")

    meta = {
        "gold_version":       gold_version,
        "silver_hash":        _compute_silver_hash(silver_path),
        "n_train":            len(train_df),
        "n_test":             len(test_df),
        "class_distribution": train_df["priority"].value_counts().to_dict(),
        "class_proportions":  train_df["priority"].value_counts(normalize=True).round(4).to_dict(),
        "test_size":          float(test_df.__len__() / (len(train_df) + len(test_df))),
        "columns":            list(train_df.columns),
    }
    meta_path = gold_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"  saved meta  → {meta_path}")
    log.info(f"  gold_version={gold_version}  silver_hash={meta['silver_hash']}")


def _next_version(gold_dir: Path) -> str:
    meta_path = Path(gold_dir) / "meta.json"
    if not meta_path.exists():
        return "v1"
    try:
        existing = json.loads(meta_path.read_text())
        v = existing.get("gold_version", "v0")
        n = int(v.lstrip("v"))
        return f"v{n + 1}"
    except (ValueError, KeyError):
        return "v1"


def main(silver_path: Path, gold_dir: Path, gold_version: str = None) -> None:
    log.info("=== Gold Layer: Featurization ===")

    if not Path(silver_path).exists():
        raise FileNotFoundError(
            f"Silver file not found: {silver_path}\nRun clean.py first."
        )

    # ── Read params from params.yaml ───────────────────────────────────────
    params       = _load_params()
    test_size    = params["test_size"]
    random_state = params["random_state"]
    log.info(f"Params: test_size={test_size}, random_state={random_state}")

    df = pd.read_parquet(silver_path)
    log.info(f"Loaded {len(df):,} rows from Silver")

    train_df, test_df = featurize(df, test_size=test_size, random_state=random_state)
    version = gold_version or _next_version(gold_dir)
    save_gold(train_df, test_df, gold_dir, silver_path, version)
    log.info(f"\nDone. Gold data saved to {Path(gold_dir).resolve()}  (version={version})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize Silver → Gold layer")
    parser.add_argument("--silver-path",  type=Path, default=SILVER_PATH)
    parser.add_argument("--gold-dir",     type=Path, default=GOLD_DIR)
    parser.add_argument("--gold-version", type=str,  default=None)
    args = parser.parse_args()
    main(
        silver_path=args.silver_path,
        gold_dir=args.gold_dir,
        gold_version=args.gold_version,
    )