"""
src/data/ingest.py  —  Bronze Layer
Fetches GitHub issues via REST API and saves raw parquet to data/bronze/.

Usage:
    python src/data/ingest.py
    python src/data/ingest.py --repos microsoft/vscode facebook/react --pages 5
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = ROOT / "data" / "bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── default repos ──────────────────────────────────────────────────────────
DEFAULT_REPOS = [
    "microsoft/vscode",
    "facebook/react",
    "scikit-learn/scikit-learn",
]


# ── public helpers (imported by tests) ─────────────────────────────────────

def parse_issue(issue: dict, repo: str) -> dict:
    """Flatten a raw GitHub API issue dict into a flat record."""
    labels = ",".join(lbl["name"] for lbl in issue.get("labels", []))
    reactions = issue.get("reactions", {})
    return {
        "number":          issue["number"],
        "title":           issue.get("title") or "",
        "body":            issue.get("body") or "",
        "state":           issue.get("state", ""),
        "created_at":      issue.get("created_at", ""),
        "updated_at":      issue.get("updated_at", ""),
        "closed_at":       issue.get("closed_at"),
        "comments":        issue.get("comments", 0),
        "labels":          labels,
        "author":          (issue.get("user") or {}).get("login", ""),
        "url":             issue.get("html_url", ""),
        "milestone":       (issue.get("milestone") or {}).get("title", ""),
        "reactions_total": reactions.get("total_count", 0),
        "reactions_plus1": reactions.get("+1", 0),
        "repo":            repo,
    }


def fetch_issues(
    repo: str,
    max_pages: int = 10,
    token: Optional[str] = None,
    state: str = "all",
) -> list[dict]:
    """
    Fetch issues from a GitHub repo.
    Returns a list of raw issue dicts (not yet parsed).
    Raises on 403 / 401. Returns [] on empty responses.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    all_issues: list[dict] = []
    page = 1

    while page <= max_pages:
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": state, "per_page": 100, "page": page}

        log.info(f"  fetching {repo} page {page}/{max_pages} …")
        response = requests.get(url, headers=headers, params=params)

        if response.status_code in (401, 403):
            raise Exception(
                f"GitHub API error {response.status_code}: {response.json().get('message', '')}"
            )
        if response.status_code != 200:
            log.warning(f"  unexpected status {response.status_code} — stopping pagination")
            break

        data = response.json()
        if not data:
            break

        # Filter out pull requests (GitHub returns them in /issues endpoint)
        issues = [i for i in data if "pull_request" not in i]
        all_issues.extend(issues)

        # Check for next page via Link header
        link = response.headers.get("Link", "")
        if 'rel="next"' not in link:
            break

        page += 1
        time.sleep(0.5)  # be polite to the API

    return all_issues


def save_bronze(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe to parquet at the given path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info(f"  saved {len(df):,} rows → {path}")


# ── main ───────────────────────────────────────────────────────────────────

def main(repos: list[str], max_pages: int, token: Optional[str]) -> None:
    log.info("=== Bronze Layer: GitHub Issue Ingestion ===")

    all_records: list[dict] = []
    for repo in repos:
        log.info(f"Fetching: {repo}")
        try:
            raw = fetch_issues(repo, max_pages=max_pages, token=token)
            records = [parse_issue(issue, repo=repo) for issue in raw]
            all_records.extend(records)
            log.info(f"  → {len(records):,} issues collected")
        except Exception as e:
            log.error(f"  failed for {repo}: {e}")

    if not all_records:
        log.error("No issues collected. Check your token and network.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    out_path = BRONZE_DIR / "issues_raw.parquet"
    save_bronze(df, out_path)
    log.info(f"\nDone. Total: {len(df):,} issues saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest GitHub issues → Bronze layer")
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    parser.add_argument("--pages", type=int, default=10)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.warning("GITHUB_TOKEN not set — you may hit rate limits")

    main(repos=args.repos, max_pages=args.pages, token=token)