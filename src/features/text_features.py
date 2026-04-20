"""
src/features/text_features.py  —  Shared Text Feature Logic

Single source of truth for how raw issue text is combined into the
feature string fed to the TF-IDF vectoriser.

WHY this file exists:
  The same combination logic must be used at training time (featurize.py)
  and at serving time (serve.py). If these diverge — for example the
  title weight is changed in one file but not the other — the model is
  evaluated on different features than it serves, which is training-serving
  skew. This is a well-known form of technical debt in ML systems
  (Sculley et al., 2015, "Hidden Technical Debt in Machine Learning Systems").

  By importing from this single module, both featurize.py and serve.py
  are guaranteed to use identical logic.

Usage:
    from src.features.text_features import combine_issue_text

    # At training time (featurize.py) — works on a DataFrame column
    df["text"] = combine_issue_text(df["title"], df["body"])

    # At serving time (serve.py) — works on a single issue
    text = combine_issue_text_single(title="App crashes", body="Null pointer")
"""

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
# Title is repeated TITLE_WEIGHT times to give it more influence than body.
# Rationale: issue titles are written to be concise and descriptive — they
# carry the priority signal in far fewer words than the body, which often
# contains reproduction steps, logs, and environment details that add noise.
TITLE_WEIGHT = 3


def combine_issue_text(title: pd.Series, body: pd.Series) -> pd.Series:
    """
    Combine title and body Series into a single text Series for TF-IDF.

    Title is repeated TITLE_WEIGHT times so the vectoriser sees it more
    often and assigns higher weight to its tokens.

    Args:
        title: Series of issue titles (may contain NaN)
        body:  Series of issue bodies (may contain NaN)

    Returns:
        Series of combined text strings, stripped of leading/trailing whitespace.

    Example:
        >>> import pandas as pd
        >>> t = pd.Series(["App crashes on startup"])
        >>> b = pd.Series(["Steps to reproduce: run the app"])
        >>> combine_issue_text(t, b).iloc[0]
        'App crashes on startup App crashes on startup App crashes on startup Steps to reproduce: run the app'
    """
    title_clean = title.fillna("").str.strip()
    body_clean  = body.fillna("").str.strip()
    repeated    = " ".join(["{title}"] * TITLE_WEIGHT)  # "title title title"
    return (
        (title_clean + " ") * TITLE_WEIGHT + body_clean
    ).str.strip()


def combine_issue_text_single(title: str, body: str = "") -> str:
    """
    Combine a single issue's title and body into a feature string.
    Used at serving time in serve.py.

    Args:
        title: Issue title string
        body:  Issue body string (optional, defaults to empty)

    Returns:
        Combined feature string with title repeated TITLE_WEIGHT times.

    Example:
        >>> combine_issue_text_single("App crashes", "Null pointer on launch")
        'App crashes App crashes App crashes Null pointer on launch'
    """
    title_clean = (title or "").strip()
    body_clean  = (body  or "").strip()
    return ((title_clean + " ") * TITLE_WEIGHT + body_clean).strip()