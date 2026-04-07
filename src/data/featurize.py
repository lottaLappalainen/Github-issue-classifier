"""
featurize.py — Gold Layer
Reads clean Silver CSV, engineers TF-IDF features,
balances classes, splits train/test, and saves Gold artifacts.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

SILVER_FILE = Path("data/silver/issues_clean.csv")
GOLD_DIR    = Path("data/gold")

TEST_SIZE    = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000   # TF-IDF vocabulary size


# ── Text combination ───────────────────────────────────────────────────────────

def combine_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine title + body into a single text field.
    Title is repeated 3x to give it more weight than body.
    """
    return df["title"] * 3 + " " + df["body"].fillna("")


# ── Class balancing ────────────────────────────────────────────────────────────

def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Upsample minority classes to match the majority class.
    This prevents the model from just predicting 'low' for everything.
    """
    max_count = df["priority"].value_counts().max()
    balanced_parts = []

    for priority in df["priority"].unique():
        subset = df[df["priority"] == priority]
        if len(subset) < max_count:
            subset = resample(
                subset,
                replace=True,
                n_samples=max_count,
                random_state=RANDOM_STATE,
            )
        balanced_parts.append(subset)

    return pd.concat(balanced_parts).sample(frac=1, random_state=RANDOM_STATE)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔄 Starting Gold featurization...")

    df = pd.read_csv(SILVER_FILE)
    print(f"  Loaded {len(df)} issues from Silver")

    # Combine text
    df["text"] = combine_text(df)

    # Balance classes
    df_balanced = balance_classes(df)
    print(f"  After balancing: {len(df_balanced)} issues")
    print(f"  Class distribution: {df_balanced['priority'].value_counts().to_dict()}")

    # Encode labels: high=0, medium=1, low=2
    le = LabelEncoder()
    le.fit(["high", "medium", "low"])
    df_balanced["label"] = le.transform(df_balanced["priority"])

    # Train / test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df_balanced["text"],
        df_balanced["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_balanced["label"],
    )

    # TF-IDF vectorization
    # Fit ONLY on training data to prevent data leakage
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),     # unigrams + bigrams
        sublinear_tf=True,      # apply log normalization
        strip_accents="unicode",
        analyzer="word",
        min_df=2,               # ignore very rare terms
    )

    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)

    print(f"  Feature matrix: {X_train.shape[1]} TF-IDF features")

    # Save Gold artifacts
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_train,    GOLD_DIR / "X_train.joblib")
    joblib.dump(X_test,     GOLD_DIR / "X_test.joblib")
    joblib.dump(y_train,    GOLD_DIR / "y_train.joblib")
    joblib.dump(y_test,     GOLD_DIR / "y_test.joblib")
    joblib.dump(vectorizer, GOLD_DIR / "vectorizer.joblib")
    joblib.dump(le,         GOLD_DIR / "label_encoder.joblib")

    # Also save a small metadata file for reference
    meta = {
        "n_train":       int(X_train.shape[0]),
        "n_test":        int(X_test.shape[0]),
        "n_features":    int(X_train.shape[1]),
        "classes":       list(le.classes_),
        "class_distribution": df_balanced["priority"].value_counts().to_dict(),
    }
    import json
    with open(GOLD_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"✅ Gold featurization complete.")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"   Output: {GOLD_DIR.resolve()}")


if __name__ == "__main__":
    main()