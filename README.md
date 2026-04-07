# GitHub Issue Priority Classifier — MLOps Pipeline

![CI/CD](https://github.com/lottaLappalainen/github-issue-classifier/actions/workflows/train.yml/badge.svg)

An end-to-end MLOps pipeline that automatically classifies GitHub issues as **high / medium / low** priority using NLP, with full CI/CD, data versioning, and model monitoring.

Built as part of the MLOps Course at Tampere University 2026.

---

## Architecture

```
GitHub API
    │
    ▼
[Bronze Layer]  ── raw issues as fetched
    │
    ▼
[Silver Layer]  ── cleaned, labelled, deduplicated
    │
    ▼
[Gold Layer]    ── TF-IDF features, balanced, train/test split
    │
    ▼
[Model Training] ── scikit-learn + MLflow experiment tracking
    │
    ▼
[FastAPI Service] ── REST endpoint serving predictions
    │
    ▼
[CI/CD] ── GitHub Actions: retrain on data change, redeploy if better
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data versioning | DVC + DagsHub |
| ML + tracking | scikit-learn + MLflow |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Serving | FastAPI |
| Data source | GitHub REST API |

---

## Medallion Architecture

| Layer | Location | Description |
|---|---|---|
| Bronze | `data/bronze/` | Raw GitHub issues, untouched |
| Silver | `data/silver/` | Cleaned, labelled, deduplicated |
| Gold | `data/gold/` | ML-ready features, train/test split |

---

## Project Structure

```
github-issue-classifier/
│
├── data/
│   ├── bronze/         # Raw API data (DVC tracked)
│   ├── silver/         # Cleaned data (DVC tracked)
│   └── gold/           # ML-ready data (DVC tracked)
│
├── src/
│   ├── data/
│   │   ├── ingest.py       # Fetch from GitHub API → Bronze
│   │   ├── clean.py        # Bronze → Silver
│   │   └── featurize.py    # Silver → Gold
│   ├── models/
│   │   ├── train.py        # Train + log to MLflow
│   │   └── evaluate.py     # Evaluate + compare versions
│   └── api/
│       └── serve.py        # FastAPI prediction endpoint
│
├── notebooks/
│   └── eda.ipynb           # Exploratory Data Analysis
│
├── tests/
│   ├── test_ingest.py
│   ├── test_clean.py
│   └── test_model.py
│
├── .github/
│   └── workflows/
│       └── train.yml       # CI/CD pipeline
│
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Hyperparameters
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/github-issue-classifier
cd github-issue-classifier
pip install -r requirements.txt

# Set your GitHub token
export GITHUB_TOKEN=your_token_here

# Run the full pipeline
dvc repro

# Or run steps manually
python src/data/ingest.py
python src/data/clean.py
python src/data/featurize.py
python src/models/train.py

# Serve the model
docker-compose up
```

---

## KPIs

| Metric | Target | Achieved |
|---|---|---|
| F1-score (macro) | > 0.70 | TBD |
| Training time | < 2 min | TBD |
| Data drift score | monitored | TBD |

---

## MLOps Maturity

This project targets **Google MLOps Level 1** (ML pipeline automation) with elements of Level 2 (CI/CD pipeline automation).

---

## Dataset

GitHub issues fetched from high-quality public repositories:
- [microsoft/vscode](https://github.com/microsoft/vscode)
- [facebook/react](https://github.com/facebook/react)
- [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)

Priority labels are derived from existing GitHub labels:
- **High**: `bug`, `critical`, `priority:high`, `severity:high`
- **Medium**: `enhancement`, `feature`, `priority:medium`
- **Low**: `documentation`, `good first issue`, `priority:low`