"""
clean.py — Silver Layer
Reads raw Bronze JSON files, cleans and normalises them,
assigns a single priority label per issue, and saves as CSV.
"""

import json
import pandas as pd
from pathlib import Path

BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
OUTPUT_FILE = SILVER_DIR / "issues_clean.csv"

# ── Label mapping ─────────────────────────────────────────────────────────────
# Maps GitHub label names → our priority classes.
# High beats medium beats low if multiple labels match.

LABEL_MAP = {
    "high": [
        "bug", "critical", "priority:high", "severity:high",
        "Priority: High", "type: bug", "kind/bug",
    ],
    "medium": [
        "enhancement", "feature", "priority:medium",
        "Priority: Medium", "type: enhancement", "kind/feature",
    ],
    "low": [
        "documentation", "good first issue", "priority:low",
        "Priority: Low", "help wanted", "type: documentation",
    ],
}

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def assign_priority(labels: list[dict]) -> str | None:
    """
    Given a list of GitHub label objects, return the highest priority class.
    Returns None if no relevant label is found.
    """
    label_names = [l["name"].lower() for l in labels]
    assigned = []

    for priority, keywords in LABEL_MAP.items():
        if any(k.lower() in label_names for k in keywords):
            assigned.append(priority)

    if not assigned:
        return None

    # Return the highest priority found
    return min(assigned, key=lambda p: PRIORITY_ORDER[p])


def clean_text(text: str | None) -> str:
    """Basic text cleaning — strip whitespace, handle nulls."""
    if not text:
        return ""
    return " ".join(text.strip().split())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔄 Starting Silver cleaning...")

    bronze_files = list(BRONZE_DIR.glob("*.json"))
    if not bronze_files:
        print("❌ No bronze files found. Run ingest.py first.")
        return

    all_rows = []

    for filepath in bronze_files:
        print(f"  Processing {filepath.name}...")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        repo = data["repo"]
        fetched_at = data["fetched_at"]

        for issue in data["issues"]:
            priority = assign_priority(issue.get("labels", []))

            if priority is None:
                continue  # skip issues we can't label

            title = clean_text(issue.get("title"))
            body = clean_text(issue.get("body"))

            # Skip if title is empty (unusable)
            if not title:
                continue

            all_rows.append({
                "id":           issue["number"],
                "repo":         repo,
                "title":        title,
                "body":         body[:2000],  # cap body length
                "priority":     priority,
                "state":        issue.get("state", ""),
                "created_at":   issue.get("created_at", ""),
                "fetched_at":   fetched_at,
                "label_names":  ", ".join(l["name"] for l in issue.get("labels", [])),
            })

    df = pd.DataFrame(all_rows)

    # Remove duplicates (same repo + issue number)
    before = len(df)
    df = df.drop_duplicates(subset=["repo", "id"])
    after = len(df)
    if before != after:
        print(f"  Removed {before - after} duplicates")

    # Print class distribution so we can spot imbalance
    print()
    print("  Class distribution:")
    dist = df["priority"].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"    {label:8s}: {count:5d} ({pct:.1f}%)")

    # Save
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"✅ Silver cleaning complete.")
    print(f"   Total issues: {len(df)}")
    print(f"   Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()