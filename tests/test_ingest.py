"""
Tests for src/data/ingest.py
Run with: python -m pytest tests/test_ingest.py -v
"""
import os
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ISSUE = {
    "number": 1,
    "title": "App crashes on startup",
    "body": "When I open the app it crashes immediately.",
    "state": "open",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-02T00:00:00Z",
    "closed_at": None,
    "comments": 3,
    "labels": [{"name": "bug"}, {"name": "priority:high"}],
    "user": {"login": "testuser"},
    "html_url": "https://github.com/test/repo/issues/1",
    "milestone": None,
    "reactions": {"+1": 2, "-1": 0, "total_count": 2},
}


@pytest.fixture
def sample_issue():
    return SAMPLE_ISSUE.copy()


@pytest.fixture
def sample_issues_list():
    issues = []
    for i in range(5):
        issue = SAMPLE_ISSUE.copy()
        issue["number"] = i + 1
        issue["title"] = f"Issue {i+1}"
        issues.append(issue)
    return issues


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestParseIssue:
    """Test the issue parsing / flattening logic."""

    def test_labels_are_flattened(self, sample_issue):
        from src.data.ingest import parse_issue
        parsed = parse_issue(sample_issue, repo="test/repo")
        assert parsed["labels"] == "bug,priority:high"

    def test_missing_body_becomes_empty_string(self, sample_issue):
        from src.data.ingest import parse_issue
        sample_issue["body"] = None
        parsed = parse_issue(sample_issue, repo="test/repo")
        assert parsed["body"] == ""

    def test_repo_field_is_added(self, sample_issue):
        from src.data.ingest import parse_issue
        parsed = parse_issue(sample_issue, repo="microsoft/vscode")
        assert parsed["repo"] == "microsoft/vscode"

    def test_required_fields_present(self, sample_issue):
        from src.data.ingest import parse_issue
        parsed = parse_issue(sample_issue, repo="test/repo")
        required = ["number", "title", "body", "state", "created_at",
                    "labels", "repo", "comments"]
        for field in required:
            assert field in parsed, f"Missing field: {field}"

    def test_reactions_extracted(self, sample_issue):
        from src.data.ingest import parse_issue
        parsed = parse_issue(sample_issue, repo="test/repo")
        assert parsed["reactions_total"] == 2


class TestSaveToParquet:
    """Test that bronze output is written correctly."""

    def test_parquet_written(self, tmp_path, sample_issues_list):
        from src.data.ingest import save_bronze
        out = tmp_path / "issues_raw.parquet"
        df = pd.DataFrame(sample_issues_list)
        save_bronze(df, out)
        assert out.exists()

    def test_parquet_readable(self, tmp_path, sample_issues_list):
        from src.data.ingest import save_bronze
        out = tmp_path / "issues_raw.parquet"
        df = pd.DataFrame(sample_issues_list)
        save_bronze(df, out)
        loaded = pd.read_parquet(out)
        assert len(loaded) == len(sample_issues_list)


class TestGitHubAPICall:
    """Test API call behavior (mocked — no real HTTP)."""

    @patch("src.data.ingest.requests.get")
    def test_api_called_with_token(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.headers = {"Link": ""}
        mock_get.return_value = mock_response

        from src.data.ingest import fetch_issues
        fetch_issues("test/repo", max_pages=1, token="fake-token")

        call_headers = mock_get.call_args[1]["headers"]
        assert "Authorization" in call_headers

    @patch("src.data.ingest.requests.get")
    def test_empty_response_returns_empty_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.headers = {"Link": ""}
        mock_get.return_value = mock_response

        from src.data.ingest import fetch_issues
        result = fetch_issues("test/repo", max_pages=1, token="fake-token")
        assert result == []

    @patch("src.data.ingest.requests.get")
    def test_rate_limit_raises(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"message": "rate limit exceeded"}
        mock_get.return_value = mock_response

        from src.data.ingest import fetch_issues
        with pytest.raises(Exception):
            fetch_issues("test/repo", max_pages=1, token="fake-token")