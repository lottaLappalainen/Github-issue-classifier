import os
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

SAMPLE_ISSUE = {
    "number": 1, "title": "App crashes on startup",
    "body": "When I open the app it crashes immediately.",
    "state": "open", "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-02T00:00:00Z", "closed_at": None,
    "comments": 3, "labels": [{"name": "bug"}, {"name": "priority:high"}],
    "user": {"login": "testuser"},
    "html_url": "https://github.com/test/repo/issues/1",
    "milestone": None, "reactions": {"+1": 2, "-1": 0, "total_count": 2},
}
PULL_REQUEST_ISSUE = {**SAMPLE_ISSUE, "number": 99, "pull_request": {"url": "..."}}

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

def make_mock_response(data, status=200, link=""):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = data
    mock.headers = {"Link": link, "X-RateLimit-Remaining": "100", "X-RateLimit-Reset": "9999999999"}
    return mock

class TestParseIssueFields:

    def test_labels_are_flattened(self, sample_issue):
        from src.data.ingest import parse_issue
        assert parse_issue(sample_issue, repo="test/repo")["labels"] == "bug,priority:high"

    def test_empty_labels_produce_empty_string(self, sample_issue):
        from src.data.ingest import parse_issue
        sample_issue["labels"] = []
        assert parse_issue(sample_issue, repo="test/repo")["labels"] == ""

    def test_single_label(self, sample_issue):
        from src.data.ingest import parse_issue
        sample_issue["labels"] = [{"name": "enhancement"}]
        assert parse_issue(sample_issue, repo="test/repo")["labels"] == "enhancement"

    def test_missing_body_becomes_empty_string(self, sample_issue):
        from src.data.ingest import parse_issue
        sample_issue["body"] = None
        assert parse_issue(sample_issue, repo="test/repo")["body"] == ""

    def test_missing_title_becomes_empty_string(self, sample_issue):
        from src.data.ingest import parse_issue
        sample_issue["title"] = None
        assert parse_issue(sample_issue, repo="test/repo")["title"] == ""

    def test_repo_field_is_added(self, sample_issue):
        from src.data.ingest import parse_issue
        assert parse_issue(sample_issue, repo="microsoft/vscode")["repo"] == "microsoft/vscode"

    def test_required_fields_present(self, sample_issue):
        from src.data.ingest import parse_issue
        parsed = parse_issue(sample_issue, repo="test/repo")
        for field in ["number", "title", "body", "state", "created_at", "labels", "repo", "comments"]:
            assert field in parsed, f"Missing field: {field}"

    def test_reactions_total_extracted(self, sample_issue):
        from src.data.ingest import parse_issue
        assert parse_issue(sample_issue, repo="test/repo")["reactions_total"] == 2

    def test_reactions_plus1_extracted(self, sample_issue):
        from src.data.ingest import parse_issue
        assert parse_issue(sample_issue, repo="test/repo")["reactions_plus1"] == 2

    def test_zero_reactions(self, sample_issue):
        from src.data.ingest import parse_issue
        sample_issue["reactions"] = {}
        parsed = parse_issue(sample_issue, repo="test/repo")
        assert parsed["reactions_total"] == 0
        assert parsed["reactions_plus1"] == 0

    def test_author_extracted(self, sample_issue):
        from src.data.ingest import parse_issue
        assert parse_issue(sample_issue, repo="test/repo")["author"] == "testuser"

    def test_null_user_handled(self, sample_issue):
        from src.data.ingest import parse_issue
        sample_issue["user"] = None
        assert parse_issue(sample_issue, repo="test/repo")["author"] == ""

    def test_milestone_extracted(self, sample_issue):
        from src.data.ingest import parse_issue
        sample_issue["milestone"] = {"title": "v2.0"}
        assert parse_issue(sample_issue, repo="test/repo")["milestone"] == "v2.0"

    def test_null_milestone_becomes_empty(self, sample_issue):
        from src.data.ingest import parse_issue
        assert parse_issue(sample_issue, repo="test/repo")["milestone"] == ""

    def test_comments_count_extracted(self, sample_issue):
        from src.data.ingest import parse_issue
        assert parse_issue(sample_issue, repo="test/repo")["comments"] == 3

    def test_state_extracted(self, sample_issue):
        from src.data.ingest import parse_issue
        assert parse_issue(sample_issue, repo="test/repo")["state"] == "open"

    def test_url_extracted(self, sample_issue):
        from src.data.ingest import parse_issue
        assert "github.com" in parse_issue(sample_issue, repo="test/repo")["url"]

    def test_returns_flat_dict(self, sample_issue):
        from src.data.ingest import parse_issue
        parsed = parse_issue(sample_issue, repo="test/repo")
        assert isinstance(parsed, dict)
        for k, v in parsed.items():
            assert not isinstance(v, dict), f"Field '{k}' is still a dict — not flattened"

    def test_number_is_integer(self, sample_issue):
        from src.data.ingest import parse_issue
        parsed = parse_issue(sample_issue, repo="test/repo")
        assert isinstance(parsed["number"], int)

    def test_different_repos_produce_different_repo_fields(self, sample_issue):
        from src.data.ingest import parse_issue
        r1 = parse_issue(sample_issue, repo="a/b")
        r2 = parse_issue(sample_issue, repo="c/d")
        assert r1["repo"] != r2["repo"]


class TestSaveBronze:

    def test_parquet_file_created(self, tmp_path, sample_issues_list):
        from src.data.ingest import save_bronze
        out = tmp_path / "issues_raw.parquet"
        save_bronze(pd.DataFrame(sample_issues_list), out)
        assert out.exists()

    def test_parquet_readable(self, tmp_path, sample_issues_list):
        from src.data.ingest import save_bronze
        out = tmp_path / "issues_raw.parquet"
        save_bronze(pd.DataFrame(sample_issues_list), out)
        assert len(pd.read_parquet(out)) == len(sample_issues_list)

    def test_row_count_preserved(self, tmp_path, sample_issues_list):
        from src.data.ingest import save_bronze
        out = tmp_path / "out.parquet"
        save_bronze(pd.DataFrame(sample_issues_list), out)
        assert len(pd.read_parquet(out)) == 5

    def test_creates_parent_dirs(self, tmp_path):
        from src.data.ingest import save_bronze
        out = tmp_path / "nested" / "deep" / "out.parquet"
        save_bronze(pd.DataFrame([{"a": 1}]), out)
        assert out.exists()

    def test_string_path_accepted(self, tmp_path):
        from src.data.ingest import save_bronze
        out = str(tmp_path / "out.parquet")
        save_bronze(pd.DataFrame([{"a": 1}]), out)
        assert Path(out).exists()

    def test_empty_dataframe_saved(self, tmp_path):
        from src.data.ingest import save_bronze
        out = tmp_path / "empty.parquet"
        save_bronze(pd.DataFrame(columns=["number", "title"]), out)
        assert out.exists()
        assert len(pd.read_parquet(out)) == 0

    def test_columns_preserved(self, tmp_path, sample_issues_list):
        from src.data.ingest import save_bronze, parse_issue
        records = [parse_issue(i, repo="test/repo") for i in sample_issues_list]
        out = tmp_path / "out.parquet"
        df = pd.DataFrame(records)
        save_bronze(df, out)
        loaded = pd.read_parquet(out)
        assert set(loaded.columns) == set(df.columns)

    def test_overwrite_existing_file(self, tmp_path):
        from src.data.ingest import save_bronze
        out = tmp_path / "out.parquet"
        save_bronze(pd.DataFrame([{"x": 1}]), out)
        save_bronze(pd.DataFrame([{"x": 2}, {"x": 3}]), out)
        assert len(pd.read_parquet(out)) == 2


class TestFetchIssuesAPI:

    @patch("src.data.ingest.requests.get")
    def test_authorization_header_sent(self, mock_get):
        mock_get.return_value = make_mock_response([])
        from src.data.ingest import fetch_issues
        fetch_issues("test/repo", max_pages=1, token="fake-token")
        headers = mock_get.call_args[1]["headers"]
        assert "Authorization" in headers
        assert "fake-token" in headers["Authorization"]

    @patch("src.data.ingest.requests.get")
    def test_no_token_no_auth_header(self, mock_get):
        mock_get.return_value = make_mock_response([])
        from src.data.ingest import fetch_issues
        fetch_issues("test/repo", max_pages=1, token=None)
        headers = mock_get.call_args[1]["headers"]
        assert "Authorization" not in headers

    @patch("src.data.ingest.requests.get")
    def test_empty_response_returns_empty_list(self, mock_get):
        mock_get.return_value = make_mock_response([])
        from src.data.ingest import fetch_issues
        assert fetch_issues("test/repo", max_pages=1, token="t") == []

    @patch("src.data.ingest.requests.get")
    def test_issues_returned(self, mock_get):
        mock_get.return_value = make_mock_response([SAMPLE_ISSUE])
        from src.data.ingest import fetch_issues
        assert len(fetch_issues("test/repo", max_pages=1, token="t")) == 1

    @patch("src.data.ingest.requests.get")
    def test_pull_requests_filtered_out(self, mock_get):
        mock_get.return_value = make_mock_response([SAMPLE_ISSUE, PULL_REQUEST_ISSUE])
        from src.data.ingest import fetch_issues
        result = fetch_issues("test/repo", max_pages=1, token="t")
        assert all("pull_request" not in i for i in result)
        assert len(result) == 1

    @patch("src.data.ingest.requests.get")
    def test_403_raises_exception(self, mock_get):
        mock_get.return_value = make_mock_response({"message": "rate limit"}, status=403)
        from src.data.ingest import fetch_issues
        with pytest.raises(Exception, match="403"):
            fetch_issues("test/repo", max_pages=1, token="t")

    @patch("src.data.ingest.requests.get")
    def test_401_raises_exception(self, mock_get):
        mock_get.return_value = make_mock_response({"message": "unauthorized"}, status=401)
        from src.data.ingest import fetch_issues
        with pytest.raises(Exception):
            fetch_issues("test/repo", max_pages=1, token="bad")

    @patch("src.data.ingest.requests.get")
    def test_pagination_stops_without_next_link(self, mock_get):
        mock_get.return_value = make_mock_response([SAMPLE_ISSUE], link="")
        from src.data.ingest import fetch_issues
        fetch_issues("test/repo", max_pages=5, token="t")
        assert mock_get.call_count == 1

    @patch("src.data.ingest.requests.get")
    def test_pagination_follows_next_link(self, mock_get):
        next_link = '<https://api.github.com/repos/test/repo/issues?page=2>; rel="next"'
        mock_get.side_effect = [
            make_mock_response([SAMPLE_ISSUE], link=next_link),
            make_mock_response([SAMPLE_ISSUE], link=""),
        ]
        from src.data.ingest import fetch_issues
        result = fetch_issues("test/repo", max_pages=5, token="t")
        assert mock_get.call_count == 2
        assert len(result) == 2

    @patch("src.data.ingest.requests.get")
    def test_max_pages_respected(self, mock_get):
        next_link = '<https://api.github.com/repos/test/repo/issues?page=2>; rel="next"'
        mock_get.return_value = make_mock_response([SAMPLE_ISSUE], link=next_link)
        from src.data.ingest import fetch_issues
        fetch_issues("test/repo", max_pages=3, token="t")
        assert mock_get.call_count == 3

    @patch("src.data.ingest.requests.get")
    def test_500_stops_pagination(self, mock_get):
        mock_get.return_value = make_mock_response({}, status=500)
        from src.data.ingest import fetch_issues
        result = fetch_issues("test/repo", max_pages=5, token="t")
        assert isinstance(result, list)

    @patch("src.data.ingest.requests.get")
    def test_correct_repo_in_url(self, mock_get):
        mock_get.return_value = make_mock_response([])
        from src.data.ingest import fetch_issues
        fetch_issues("facebook/react", max_pages=1, token="t")
        assert "facebook/react" in mock_get.call_args[0][0]

    @patch("src.data.ingest.requests.get")
    def test_state_param_passed(self, mock_get):
        mock_get.return_value = make_mock_response([])
        from src.data.ingest import fetch_issues
        fetch_issues("test/repo", max_pages=1, token="t", state="closed")
        assert mock_get.call_args[1]["params"]["state"] == "closed"

    @patch("src.data.ingest.requests.get")
    def test_per_page_is_100(self, mock_get):
        mock_get.return_value = make_mock_response([])
        from src.data.ingest import fetch_issues
        fetch_issues("test/repo", max_pages=1, token="t")
        assert mock_get.call_args[1]["params"]["per_page"] == 100

    @patch("src.data.ingest.requests.get")
    def test_multiple_issues_on_page_all_returned(self, mock_get):
        issues = [dict(SAMPLE_ISSUE, number=i) for i in range(10)]
        mock_get.return_value = make_mock_response(issues)
        from src.data.ingest import fetch_issues
        result = fetch_issues("test/repo", max_pages=1, token="t")
        assert len(result) == 10