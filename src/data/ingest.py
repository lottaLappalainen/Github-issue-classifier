"""
ingest.py — Bronze Layer
Fetches raw GitHub issues from the API and saves them as-is.
No cleaning, no transformation. Just raw data with metadata.
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

REPOS = [
    "microsoft/vscode",
    "facebook/react",
    "scikit-learn/scikit-learn",
]

# Labels we care about — issues without ANY of these are skipped
RELEVANT_LABELS = {
    "high":   ["bug", "critical", "priority:high", "severity:high", "Priority: High"],
    "medium": ["enhancement", "feature", "priority:medium", "Priority: Medium"],
    "low":    ["documentation", "good first issue", "priority:low", "Priority: Low", "help wanted"],
}

ALL_RELEVANT = [l for labels in RELEVANT_LABELS.values() for l in labels]

MAX_ISSUES_PER_REPO = 500   # keep it manageable
OUTPUT_DIR = Path("data/bronze")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_headers():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("⚠️  No GITHUB_TOKEN set. You will hit rate limits quickly.")
        print("    Get a free token at https://github.com/settings/tokens")
        return {"Accept": "application/vnd.github+json"}
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def fetch_issues(repo: str, headers: dict) -> list[dict]:
    """Fetch issues from a single repo, paginated."""
    issues = []
    page = 1
    print(f"  Fetching {repo}...")

    while len(issues) < MAX_ISSUES_PER_REPO:
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {
            "state": "all",        # open + closed
            "per_page": 100,
            "page": page,
        }

        response = requests.get(url, headers=headers, params=params)

        # Handle rate limiting
        if response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_time - int(time.time()), 0) + 5
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue

        if response.status_code != 200:
            print(f"  Error {response.status_code} on page {page}: {response.text}")
            break

        batch = response.json()
        if not batch:
            break  # no more pages

        # Filter out pull requests (GitHub API returns them mixed with issues)
        batch = [i for i in batch if "pull_request" not in i]

        # Filter: only keep issues that have at least one relevant label
        batch = [
            i for i in batch
            if any(
                label["name"] in ALL_RELEVANT
                for label in i.get("labels", [])
            )
        ]

        issues.extend(batch)
        page += 1

        # Respect rate limits — 1 request per second is safe
        time.sleep(1)

    return issues[:MAX_ISSUES_PER_REPO]


def save_bronze(repo: str, issues: list[dict]):
    """Save raw issues to bronze layer as JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize repo name for filename: microsoft/vscode → microsoft_vscode
    safe_name = repo.replace("/", "_")
    filepath = OUTPUT_DIR / f"{safe_name}.json"

    # Wrap with metadata so we always know when and where data came from
    payload = {
        "repo": repo,
        "fetched_at": datetime.utcnow().isoformat(),
        "issue_count": len(issues),
        "issues": issues,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved {len(issues)} issues → {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔄 Starting Bronze ingestion...")
    print(f"   Repos: {REPOS}")
    print(f"   Max issues per repo: {MAX_ISSUES_PER_REPO}")
    print()

    headers = get_headers()
    total = 0

    for repo in REPOS:
        issues = fetch_issues(repo, headers)
        save_bronze(repo, issues)
        total += len(issues)

    print()
    print(f"✅ Bronze ingestion complete. Total issues: {total}")
    print(f"   Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()