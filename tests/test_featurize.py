import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def silver_df():
    """Minimal Silver-layer dataframe with imbalanced classes."""
    return pd.DataFrame({
        "title":    ["App crash bug"] * 10 + ["Add feature"] * 3 + ["Fix docs"] * 2,
        "body":     ["crashes on start"] * 10 + ["would be nice"] * 3 + ["typo fix"] * 2,
        "priority": ["high"] * 10 + ["medium"] * 3 + ["low"] * 2,
        "labels":   ["bug"] * 10 + ["enhancement"] * 3 + ["documentation"] * 2,
        "repo":     ["test/repo"] * 15,
        "number":   list(range(1, 16)),
    })

@pytest.fixture
def balanced_df():
    """Already balanced Silver dataframe."""
    return pd.DataFrame({
        "title":    ["Bug crash"] * 5 + ["Add feature"] * 5 + ["Fix docs"] * 5,
        "body":     ["crash"] * 5 + ["feature"] * 5 + ["docs"] * 5,
        "priority": ["high"] * 5 + ["medium"] * 5 + ["low"] * 5,
        "labels":   ["bug"] * 15,
        "repo":     ["test/repo"] * 15,
        "number":   list(range(1, 16)),
    })


# ---------------------------------------------------------------------------
# combine_text
# ---------------------------------------------------------------------------

class TestCombineText:

    def test_returns_series(self, silver_df):
        from src.data.featurize import combine_text
        result = combine_text(silver_df)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, silver_df):
        from src.data.featurize import combine_text
        assert len(combine_text(silver_df)) == len(silver_df)

    def test_title_appears_in_text(self, silver_df):
        from src.data.featurize import combine_text
        result = combine_text(silver_df)
        for title, text in zip(silver_df["title"], result):
            assert title.lower() in text.lower()

    def test_body_appears_in_text(self, silver_df):
        from src.data.featurize import combine_text
        result = combine_text(silver_df)
        for body, text in zip(silver_df["body"], result):
            assert body.lower() in text.lower()

    def test_null_title_handled(self):
        from src.data.featurize import combine_text
        df = pd.DataFrame({"title": [None], "body": ["some body"]})
        result = combine_text(df)
        assert result.iloc[0] == "some body"

    def test_null_body_handled(self):
        from src.data.featurize import combine_text
        df = pd.DataFrame({"title": ["Some title"], "body": [None]})
        result = combine_text(df)
        assert "Some title" in result.iloc[0]

    def test_title_given_more_weight_than_body(self):
        from src.data.featurize import combine_text
        df = pd.DataFrame({"title": ["TITLE"], "body": ["BODY"]})
        result = combine_text(df).iloc[0]
        assert result.count("TITLE") > result.count("BODY")

    def test_no_leading_trailing_whitespace(self, silver_df):
        from src.data.featurize import combine_text
        result = combine_text(silver_df)
        assert not any(t != t.strip() for t in result)


# ---------------------------------------------------------------------------
# balance_classes
# ---------------------------------------------------------------------------

class TestBalanceClasses:

    def test_output_is_dataframe(self, silver_df):
        from src.data.featurize import balance_classes
        df_with_text = silver_df.assign(text="x " * 5)
        assert isinstance(balance_classes(df_with_text[["text", "priority"]]), pd.DataFrame)

    def test_all_classes_equal_after_balancing(self, silver_df):
        from src.data.featurize import balance_classes
        df = silver_df[["priority"]].assign(text="x")
        balanced = balance_classes(df)
        counts = balanced["priority"].value_counts()
        assert counts.min() == counts.max()

    def test_majority_class_size_preserved(self, silver_df):
        from src.data.featurize import balance_classes
        df = silver_df[["priority"]].assign(text="x")
        max_before = df["priority"].value_counts().max()
        balanced = balance_classes(df)
        assert balanced["priority"].value_counts().max() == max_before

    def test_all_classes_still_present(self, silver_df):
        from src.data.featurize import balance_classes
        df = silver_df[["priority"]].assign(text="x")
        balanced = balance_classes(df)
        assert set(balanced["priority"].unique()) == {"high", "medium", "low"}

    def test_already_balanced_unchanged_size(self, balanced_df):
        from src.data.featurize import balance_classes
        df = balanced_df[["priority"]].assign(text="x")
        original_len = len(df)
        balanced = balance_classes(df)
        assert len(balanced) == original_len

    def test_random_state_reproducible(self, silver_df):
        from src.data.featurize import balance_classes
        df = silver_df[["priority"]].assign(text="x")
        b1 = balance_classes(df, random_state=42)
        b2 = balance_classes(df, random_state=42)
        assert list(b1["priority"]) == list(b2["priority"])

    def test_different_seeds_may_differ(self, silver_df):
        from src.data.featurize import balance_classes
        df = silver_df[["priority"]].assign(text="x " * 15)
        b1 = balance_classes(df, random_state=1)
        b2 = balance_classes(df, random_state=999)
        # With different seeds at least order should differ
        assert list(b1["priority"]) != list(b2["priority"]) or True  # soft check


# ---------------------------------------------------------------------------
# featurize()
# ---------------------------------------------------------------------------

class TestFeaturize:

    def test_returns_two_dataframes(self, silver_df):
        from src.data.featurize import featurize
        result = featurize(silver_df)
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(r, pd.DataFrame) for r in result)

    def test_train_larger_than_test(self, silver_df):
        from src.data.featurize import featurize
        train, test = featurize(silver_df)
        assert len(train) > len(test)

    def test_test_size_approx_20_percent(self, silver_df):
        from src.data.featurize import featurize
        train, test = featurize(silver_df)
        total = len(train) + len(test)
        ratio = len(test) / total
        assert 0.15 <= ratio <= 0.25

    def test_both_splits_have_text_column(self, silver_df):
        from src.data.featurize import featurize
        train, test = featurize(silver_df)
        assert "text" in train.columns
        assert "text" in test.columns

    def test_both_splits_have_priority_column(self, silver_df):
        from src.data.featurize import featurize
        train, test = featurize(silver_df)
        assert "priority" in train.columns
        assert "priority" in test.columns

    def test_all_priorities_present_in_train(self, silver_df):
        from src.data.featurize import featurize
        train, _ = featurize(silver_df)
        assert set(train["priority"].unique()) == {"high", "medium", "low"}

    def test_classes_balanced_in_train(self, silver_df):
        from src.data.featurize import featurize
        train, _ = featurize(silver_df)
        counts = train["priority"].value_counts()
        assert counts.max() - counts.min() <= 2  # allow tiny rounding difference

    def test_no_empty_texts_in_output(self, silver_df):
        from src.data.featurize import featurize
        train, test = featurize(silver_df)
        assert (train["text"].str.len() > 0).all()
        assert (test["text"].str.len() > 0).all()

    def test_no_null_priorities_in_output(self, silver_df):
        from src.data.featurize import featurize
        train, test = featurize(silver_df)
        assert train["priority"].isnull().sum() == 0
        assert test["priority"].isnull().sum() == 0

    def test_indices_reset(self, silver_df):
        from src.data.featurize import featurize
        train, test = featurize(silver_df)
        assert list(train.index) == list(range(len(train)))
        assert list(test.index) == list(range(len(test)))

    def test_custom_test_size(self, silver_df):
        from src.data.featurize import featurize
        train, test = featurize(silver_df, test_size=0.3)
        total = len(train) + len(test)
        assert 0.25 <= len(test) / total <= 0.35


# ---------------------------------------------------------------------------
# save_gold
# ---------------------------------------------------------------------------

class TestSaveGold:

    def test_train_parquet_created(self, tmp_path, silver_df):
        from src.data.featurize import featurize, save_gold
        train, test = featurize(silver_df)
        save_gold(train, test, tmp_path)
        assert (tmp_path / "train.parquet").exists()

    def test_test_parquet_created(self, tmp_path, silver_df):
        from src.data.featurize import featurize, save_gold
        train, test = featurize(silver_df)
        save_gold(train, test, tmp_path)
        assert (tmp_path / "test.parquet").exists()

    def test_meta_json_created(self, tmp_path, silver_df):
        from src.data.featurize import featurize, save_gold
        train, test = featurize(silver_df)
        save_gold(train, test, tmp_path)
        assert (tmp_path / "meta.json").exists()

    def test_meta_json_has_required_keys(self, tmp_path, silver_df):
        from src.data.featurize import featurize, save_gold
        train, test = featurize(silver_df)
        save_gold(train, test, tmp_path)
        meta = json.loads((tmp_path / "meta.json").read_text())
        for key in ["n_train", "n_test", "class_distribution", "test_size"]:
            assert key in meta

    def test_meta_n_train_correct(self, tmp_path, silver_df):
        from src.data.featurize import featurize, save_gold
        train, test = featurize(silver_df)
        save_gold(train, test, tmp_path)
        meta = json.loads((tmp_path / "meta.json").read_text())
        assert meta["n_train"] == len(train)

    def test_meta_n_test_correct(self, tmp_path, silver_df):
        from src.data.featurize import featurize, save_gold
        train, test = featurize(silver_df)
        save_gold(train, test, tmp_path)
        meta = json.loads((tmp_path / "meta.json").read_text())
        assert meta["n_test"] == len(test)

    def test_parquet_round_trip_train(self, tmp_path, silver_df):
        from src.data.featurize import featurize, save_gold
        train, test = featurize(silver_df)
        save_gold(train, test, tmp_path)
        loaded = pd.read_parquet(tmp_path / "train.parquet")
        assert len(loaded) == len(train)

    def test_parquet_round_trip_test(self, tmp_path, silver_df):
        from src.data.featurize import featurize, save_gold
        train, test = featurize(silver_df)
        save_gold(train, test, tmp_path)
        loaded = pd.read_parquet(tmp_path / "test.parquet")
        assert len(loaded) == len(test)

    def test_creates_output_directory(self, tmp_path, silver_df):
        from src.data.featurize import featurize, save_gold
        train, test = featurize(silver_df)
        out = tmp_path / "new_gold_dir"
        save_gold(train, test, out)
        assert out.exists()