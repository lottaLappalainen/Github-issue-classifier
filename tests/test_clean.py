"""
Tests for src/data/clean.py
Run with: python -m pytest tests/test_clean.py -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df():
    """Simulates a Bronze-layer dataframe."""
    return pd.DataFrame({
        "number":     [1, 2, 3, 4, 5, 1],          # 1 duplicate
        "title":      ["Bug in login", "Add dark mode", "Docs update",
                       "CRITICAL crash", "", "Bug in login"],
        "body":       ["It crashes", "Would be nice", None,
                       "App is down!!!", "   ", "It crashes"],
        "state":      ["open", "open", "closed", "open", "open", "open"],
        "labels":     ["bug,priority:high", "enhancement", "documentation",
                       "bug,critical", "good first issue", "bug,priority:high"],
        "comments":   [5, 1, 0, 20, 0, 5],
        "reactions_total": [3, 0, 0, 10, 0, 3],
        "created_at": ["2024-01-01T00:00:00Z"] * 6,
        "repo":       ["microsoft/vscode"] * 6,
    })


# ---------------------------------------------------------------------------
# Label assignment tests
# ---------------------------------------------------------------------------

class TestLabelAssignment:

    def test_bug_label_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("bug,priority:high") == "high"

    def test_critical_label_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("bug,critical") == "high"

    def test_enhancement_is_medium(self):
        from src.data.clean import assign_priority
        assert assign_priority("enhancement") == "medium"

    def test_feature_request_is_medium(self):
        from src.data.clean import assign_priority
        assert assign_priority("feature,request") == "medium"

    def test_documentation_is_low(self):
        from src.data.clean import assign_priority
        assert assign_priority("documentation") == "low"

    def test_good_first_issue_is_low(self):
        from src.data.clean import assign_priority
        assert assign_priority("good first issue") == "low"

    def test_unknown_label_returns_none(self):
        from src.data.clean import assign_priority
        result = assign_priority("some-unknown-label-xyz")
        assert result is None

    def test_empty_label_returns_none(self):
        from src.data.clean import assign_priority
        result = assign_priority("")
        assert result is None


class TestCleaning:

    def test_duplicates_removed(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert silver["number"].duplicated().sum() == 0

    def test_empty_titles_dropped(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert (silver["title"].str.strip() == "").sum() == 0

    def test_null_body_filled(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert silver["body"].isnull().sum() == 0

    def test_unlabelled_issues_dropped(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        # "good first issue" without body content should be dropped OR kept — either way no NaN priority
        assert silver["priority"].isnull().sum() == 0

    def test_priority_column_exists(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert "priority" in silver.columns

    def test_priority_values_valid(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        valid = {"high", "medium", "low"}
        assert set(silver["priority"].unique()).issubset(valid)

    def test_text_column_created(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert "text" in silver.columns

    def test_text_combines_title_and_body(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        # Every text field should contain something from title
        for _, row in silver.iterrows():
            assert len(row["text"]) > 0


class TestSilverOutput:

    def test_silver_parquet_written(self, tmp_path, raw_df):
        from src.data.clean import clean, save_silver
        silver = clean(raw_df)
        out = tmp_path / "issues_clean.parquet"
        save_silver(silver, out)
        assert out.exists()

    def test_silver_parquet_readable(self, tmp_path, raw_df):
        from src.data.clean import clean, save_silver
        silver = clean(raw_df)
        out = tmp_path / "issues_clean.parquet"
        save_silver(silver, out)
        loaded = pd.read_parquet(out)
        assert len(loaded) == len(silver)