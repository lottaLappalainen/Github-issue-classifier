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

# ── repos ──────────────────────────────────────────────────────────────────
# Chosen for high issue volume AND rich, consistent labelling across all
# three priority tiers (bug/critical → high, enhancement → medium, docs → low).
DEFAULT_REPOS = [
    # ── High label signal (bug-heavy, well triaged) ──
    "microsoft/vscode",           # ~180k issues, excellent label discipline
    "microsoft/TypeScript",       # strong bug / enhancement split
    "golang/go",                  # priority labels, very clean
    "rust-lang/rust",             # I-prioritize-high / I-enhancement etc.
    "kubernetes/kubernetes",      # priority/critical-urgent labels
    "ansible/ansible",            # bug / feature / docs well used

    # ── Enhancement / feature heavy ──
    "facebook/react",             # enhancement, feature-request
    "vuejs/vue",                  # feature labels
    "sveltejs/svelte",            # enhancement-heavy
    "vitejs/vite",                # feature requests well labelled
    "vercel/next.js",             # enhancement / bug / docs
    "shadcn-ui/ui",               # good first issue / enhancement

    # ── Documentation / good-first-issue heavy ──
    "scikit-learn/scikit-learn",  # docs, good first issue, enhancement
    "huggingface/transformers",   # docs, good first issue
    "pytorch/pytorch",            # docs / enhancement / bug
    "tensorflow/tensorflow",      # type:bug / type:feature / type:docs
    "django/django",              # Bug / New feature / Documentation
    "pallets/flask",              # bug / enhancement / documentation
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
    max_pages: int = 30,
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

        # Respect rate limit headers
        remaining = int(response.headers.get("X-RateLimit-Remaining", 999))
        if remaining < 10:
            reset_ts  = int(response.headers.get("X-RateLimit-Reset", 0))
            wait_secs = max(0, reset_ts - int(time.time())) + 5
            log.warning(f"  Rate limit low ({remaining} left) — sleeping {wait_secs}s")
            time.sleep(wait_secs)

        # Check for next page via Link header
        link = response.headers.get("Link", "")
        if 'rel="next"' not in link:
            break

        page += 1
        time.sleep(0.3)  # be polite to the API

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
    log.info(f"Repos: {len(repos)} | Pages per repo: {max_pages} | Est. max issues: {len(repos) * max_pages * 100:,}")

    all_records: list[dict] = []
    failed_repos: list[str] = []

    for repo in repos:
        log.info(f"\nFetching: {repo}")
        try:
            raw = fetch_issues(repo, max_pages=max_pages, token=token)
            records = [parse_issue(issue, repo=repo) for issue in raw]
            all_records.extend(records)
            log.info(f"  → {len(records):,} issues collected from {repo}")
        except Exception as e:
            log.error(f"  failed for {repo}: {e}")
            failed_repos.append(repo)

    if failed_repos:
        log.warning(f"\nFailed repos: {failed_repos}")

    if not all_records:
        log.error("No issues collected. Check your token and network.")
        sys.exit(1)

    df = pd.DataFrame(all_records)

    log.info(f"\nTotal collected: {len(df):,} issues from {df['repo'].nunique()} repos")
    log.info(f"Repo breakdown:\n{df['repo'].value_counts().to_string()}")

    out_path = BRONZE_DIR / "issues_raw.parquet"
    save_bronze(df, out_path)
    log.info(f"\nDone. {len(df):,} issues saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest GitHub issues → Bronze layer")
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS,
                        help="GitHub repos to fetch (default: 18 curated repos)")
    parser.add_argument("--pages", type=int, default=30,
                        help="Max pages per repo (100 issues/page, default: 30)")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.warning("GITHUB_TOKEN not set — you will hit rate limits quickly with 18 repos")
    else:
        log.info("GITHUB_TOKEN found ✓")

    main(repos=args.repos, max_pages=args.pages, token=token)