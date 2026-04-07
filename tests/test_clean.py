import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "number":          [1, 2, 3, 4, 5, 1],
        "title":           ["Bug in login", "Add dark mode", "Docs update", "CRITICAL crash", "", "Bug in login"],
        "body":            ["It crashes", "Would be nice", None, "App is down!!!", "   ", "It crashes"],
        "state":           ["open", "open", "closed", "open", "open", "open"],
        "labels":          ["bug,priority:high", "enhancement", "documentation", "bug,critical", "good first issue", "bug,priority:high"],
        "comments":        [5, 1, 0, 20, 0, 5],
        "reactions_total": [3, 0, 0, 10, 0, 3],
        "created_at":      ["2024-01-01T00:00:00Z"] * 6,
        "repo":            ["microsoft/vscode"] * 6,
    })

@pytest.fixture
def multi_repo_df():
    return pd.DataFrame({
        "number": [1, 1, 2],
        "title":  ["Bug A", "Bug B", "Enhancement C"],
        "body":   ["crash", "crash too", "nice feature"],
        "state":  ["open"] * 3,
        "labels": ["bug", "bug", "enhancement"],
        "comments": [1, 2, 3],
        "reactions_total": [0, 0, 0],
        "created_at": ["2024-01-01T00:00:00Z"] * 3,
        "repo": ["microsoft/vscode", "facebook/react", "facebook/react"],
    })


# ---------------------------------------------------------------------------
# assign_priority
# ---------------------------------------------------------------------------

class TestLabelAssignment:

    # ── High priority ──────────────────────────────────────────────────────

    def test_bug_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("bug") == "high"

    def test_bug_with_priority_high_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("bug,priority:high") == "high"

    def test_critical_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("critical") == "high"

    def test_bug_and_critical_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("bug,critical") == "high"

    def test_crash_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("crash") == "high"

    def test_regression_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("regression") == "high"

    def test_security_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("security") == "high"

    def test_p0_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("p0") == "high"

    def test_p1_is_high(self):
        from src.data.clean import assign_priority
        assert assign_priority("p1") == "high"

    # ── Medium priority ────────────────────────────────────────────────────

    def test_enhancement_is_medium(self):
        from src.data.clean import assign_priority
        assert assign_priority("enhancement") == "medium"

    def test_feature_is_medium(self):
        from src.data.clean import assign_priority
        assert assign_priority("feature") == "medium"

    def test_feature_request_is_medium(self):
        from src.data.clean import assign_priority
        assert assign_priority("feature,request") == "medium"

    def test_performance_is_medium(self):
        from src.data.clean import assign_priority
        assert assign_priority("performance") == "medium"

    def test_improvement_is_medium(self):
        from src.data.clean import assign_priority
        assert assign_priority("improvement") == "medium"

    def test_p2_is_medium(self):
        from src.data.clean import assign_priority
        assert assign_priority("p2") == "medium"

    # ── Low priority ───────────────────────────────────────────────────────

    def test_documentation_is_low(self):
        from src.data.clean import assign_priority
        assert assign_priority("documentation") == "low"

    def test_docs_is_low(self):
        from src.data.clean import assign_priority
        assert assign_priority("docs") == "low"

    def test_good_first_issue_is_low(self):
        from src.data.clean import assign_priority
        assert assign_priority("good first issue") == "low"

    def test_help_wanted_is_low(self):
        from src.data.clean import assign_priority
        assert assign_priority("help wanted") == "low"

    def test_question_is_low(self):
        from src.data.clean import assign_priority
        assert assign_priority("question") == "low"

    def test_duplicate_is_low(self):
        from src.data.clean import assign_priority
        assert assign_priority("duplicate") == "low"

    def test_p3_is_low(self):
        from src.data.clean import assign_priority
        assert assign_priority("p3") == "low"

    # ── Priority ordering: HIGH wins ──────────────────────────────────────

    def test_bug_beats_enhancement(self):
        from src.data.clean import assign_priority
        assert assign_priority("bug,enhancement") == "high"

    def test_bug_beats_documentation(self):
        from src.data.clean import assign_priority
        assert assign_priority("bug,documentation") == "high"

    def test_enhancement_beats_documentation(self):
        from src.data.clean import assign_priority
        assert assign_priority("enhancement,documentation") == "medium"

    def test_case_insensitive(self):
        from src.data.clean import assign_priority
        assert assign_priority("BUG") == "high"
        assert assign_priority("Enhancement") == "medium"
        assert assign_priority("DOCUMENTATION") == "low"

    def test_whitespace_trimmed_in_labels(self):
        from src.data.clean import assign_priority
        assert assign_priority("  bug  ,  enhancement  ") == "high"

    # ── None / unknown ────────────────────────────────────────────────────

    def test_unknown_label_returns_none(self):
        from src.data.clean import assign_priority
        assert assign_priority("some-unknown-label-xyz") is None

    def test_empty_string_returns_none(self):
        from src.data.clean import assign_priority
        assert assign_priority("") is None

    def test_whitespace_only_returns_none(self):
        from src.data.clean import assign_priority
        assert assign_priority("   ") is None

    def test_multiple_unknown_labels_returns_none(self):
        from src.data.clean import assign_priority
        assert assign_priority("foo,bar,baz") is None


# ---------------------------------------------------------------------------
# clean()
# ---------------------------------------------------------------------------

class TestCleanDeduplication:

    def test_duplicates_removed_same_repo(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert silver["number"].duplicated().sum() == 0

    def test_same_number_different_repos_both_kept(self, multi_repo_df):
        from src.data.clean import clean
        silver = clean(multi_repo_df)
        # number=1 appears in vscode AND react — both should survive
        assert len(silver[silver["number"] == 1]) == 2

    def test_row_count_decreases_with_duplicates(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert len(silver) < len(raw_df)


class TestCleanNullHandling:

    def test_null_body_filled_with_empty_string(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert silver["body"].isnull().sum() == 0

    def test_null_body_filled_not_nan(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert "" in silver["body"].values or silver["body"].notna().all()

    def test_whitespace_body_stripped(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        # The "   " body entry should be stripped to ""
        assert not any(v.startswith(" ") for v in silver["body"].dropna())


class TestCleanTitleFiltering:

    def test_empty_titles_dropped(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert (silver["title"].str.strip() == "").sum() == 0

    def test_valid_titles_kept(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert "Bug in login" in silver["title"].values


class TestCleanPriorityColumn:

    def test_priority_column_exists(self, raw_df):
        from src.data.clean import clean
        assert "priority" in clean(raw_df).columns

    def test_priority_values_are_valid(self, raw_df):
        from src.data.clean import clean
        assert set(clean(raw_df)["priority"].unique()).issubset({"high", "medium", "low"})

    def test_no_null_priorities(self, raw_df):
        from src.data.clean import clean
        assert clean(raw_df)["priority"].isnull().sum() == 0

    def test_unlabelled_rows_dropped(self):
        from src.data.clean import clean
        df = pd.DataFrame({
            "number": [1, 2],
            "title": ["Unknown issue", "Bug found"],
            "body": ["no idea", "crash"],
            "state": ["open", "open"],
            "labels": ["xyz-unknown", "bug"],
            "comments": [0, 1],
            "reactions_total": [0, 0],
            "created_at": ["2024-01-01"] * 2,
            "repo": ["test/repo"] * 2,
        })
        silver = clean(df)
        assert len(silver) == 1
        assert silver.iloc[0]["priority"] == "high"


class TestCleanTextColumn:

    def test_text_column_created(self, raw_df):
        from src.data.clean import clean
        assert "text" in clean(raw_df).columns

    def test_text_is_non_empty_for_all_rows(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert (silver["text"].str.len() > 0).all()

    def test_text_contains_title(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        for _, row in silver.iterrows():
            assert row["title"].lower() in row["text"].lower() or len(row["title"]) == 0

    def test_text_is_string_type(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert silver["text"].dtype == object


class TestCleanOutputShape:

    def test_result_is_dataframe(self, raw_df):
        from src.data.clean import clean
        assert isinstance(clean(raw_df), pd.DataFrame)

    def test_index_is_reset(self, raw_df):
        from src.data.clean import clean
        silver = clean(raw_df)
        assert list(silver.index) == list(range(len(silver)))


# ---------------------------------------------------------------------------
# save_silver
# ---------------------------------------------------------------------------

class TestSaveSilver:

    def test_parquet_file_created(self, tmp_path, raw_df):
        from src.data.clean import clean, save_silver
        out = tmp_path / "silver.parquet"
        save_silver(clean(raw_df), out)
        assert out.exists()

    def test_parquet_readable(self, tmp_path, raw_df):
        from src.data.clean import clean, save_silver
        silver = clean(raw_df)
        out = tmp_path / "silver.parquet"
        save_silver(silver, out)
        assert len(pd.read_parquet(out)) == len(silver)

    def test_columns_preserved(self, tmp_path, raw_df):
        from src.data.clean import clean, save_silver
        silver = clean(raw_df)
        out = tmp_path / "silver.parquet"
        save_silver(silver, out)
        loaded = pd.read_parquet(out)
        assert set(loaded.columns) == set(silver.columns)

    def test_priority_values_survive_roundtrip(self, tmp_path, raw_df):
        from src.data.clean import clean, save_silver
        silver = clean(raw_df)
        out = tmp_path / "silver.parquet"
        save_silver(silver, out)
        loaded = pd.read_parquet(out)
        assert set(loaded["priority"].unique()).issubset({"high", "medium", "low"})

    def test_creates_parent_directories(self, tmp_path, raw_df):
        from src.data.clean import clean, save_silver
        out = tmp_path / "nested" / "dir" / "silver.parquet"
        save_silver(clean(raw_df), out)
        assert out.exists()

    def test_string_path_accepted(self, tmp_path, raw_df):
        from src.data.clean import clean, save_silver
        out = str(tmp_path / "silver.parquet")
        save_silver(clean(raw_df), out)
        assert Path(out).exists()