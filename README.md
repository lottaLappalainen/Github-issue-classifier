# GitHub Issue Priority Classifier

![CI/CD](https://github.com/lottaLappalainen/github-issue-classifier/actions/workflows/train.yml/badge.svg)

An end-to-end MLOps pipeline that automatically classifies GitHub issues as **high / medium / low** priority using NLP. Covers the full lifecycle: data ingestion, cleaning, feature engineering, model training, experiment tracking, REST API serving, and CI/CD automation.

Built as part of the MLOps Course at Tampere University 2026.

---

## Architecture

```
GitHub REST API (18 repos)
        │
        ▼
[Bronze Layer]  ── raw issues as parquet, untouched
        │
        ▼
[Silver Layer]  ── deduplicated, labelled, nulls filled
        │
        ▼
[Gold Layer]    ── combined text, class-balanced, train/test split
        │
        ▼
[Model Training] ── TF-IDF + sklearn classifiers, MLflow tracking
        │
        ▼
[FastAPI Service] ── /predict endpoint, /health, /docs
        │
        ▼
[CI/CD] ── GitHub Actions: test → train → quality gate → Docker
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data versioning | DVC (local remote) |
| ML + tracking | scikit-learn + MLflow |
| CI/CD | GitHub Actions |
| Containerization | Docker + docker-compose |
| API serving | FastAPI + uvicorn |
| Data source | GitHub REST API |

---

## Project Structure

```
github-issue-classifier/
│
├── data/
│   ├── bronze/             # Raw API data (DVC tracked)
│   ├── silver/             # Cleaned, labelled data (DVC tracked)
│   └── gold/               # ML-ready train/test splits (DVC tracked)
│
├── src/
│   ├── data/
│   │   ├── ingest.py       # GitHub API → Bronze (18 repos)
│   │   ├── clean.py        # Bronze → Silver (dedup, label, validate)
│   │   └── featurize.py    # Silver → Gold (text combine, balance, split)
│   ├── models/
│   │   ├── train.py        # Train 3 classifiers, log to MLflow
│   │   └── evaluate.py     # Evaluate, compare runs, run inference
│   └── api/
│       └── serve.py        # FastAPI prediction endpoint
│
├── tests/
│   ├── conftest.py
│   ├── test_ingest.py      # 20 tests — API mocking, parsing, parquet
│   ├── test_clean.py       # 30 tests — labelling, dedup, cleaning
│   ├── test_featurize.py   # 25 tests — text combine, balance, split
│   ├── test_model.py       # 25 tests — pipeline, training, persistence
│   └── test_evaluate.py    # 20 tests — metrics, inference, quality gate
│
├── .github/
│   └── workflows/
│       └── train.yml       # CI/CD: lint → test → train → docker
│
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                # DVC pipeline (5 stages)
├── params.yaml             # Hyperparameters tracked by DVC
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/lottaLappalainen/github-issue-classifier
cd github-issue-classifier
pip install -r requirements.txt
```

### 2. Set your GitHub token

**Windows CMD:**
```bat
set GITHUB_TOKEN=your_token_here
```

**Linux / macOS:**
```bash
export GITHUB_TOKEN=your_token_here
```

A token is required to avoid GitHub API rate limits (60 req/hr unauthenticated vs 5000/hr authenticated). Generate one at [github.com/settings/tokens](https://github.com/settings/tokens) — no scopes needed for public repos.

### 3. Run the full pipeline

```bash
# Option A: DVC (recommended — tracks versions automatically)
dvc repro

# Option B: Manual step by step
python src/data/ingest.py
python src/data/clean.py
python src/data/featurize.py
python src/models/train.py
python src/models/evaluate.py
```

### 4. Serve the API

```bash
# With Docker (recommended)
docker-compose up

# Or directly
uvicorn src.api.serve:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## Data Pipeline (Medallion Architecture)

### Bronze — Raw Ingestion

`src/data/ingest.py` fetches issues from 18 curated public repos via the GitHub REST API. Repos were selected for high issue volume and consistent label discipline across all three priority tiers.

Pull requests are filtered out (GitHub returns them in the `/issues` endpoint). Data is saved as parquet to `data/bronze/issues_raw.parquet`.

```bash
# Quick test run (3 repos, 5 pages each)
python src/data/ingest.py --repos microsoft/vscode facebook/react django/django --pages 5

# Full run (18 repos, 30 pages each — ~10-20 min)
python src/data/ingest.py
```

### Silver — Cleaning & Labelling

`src/data/clean.py` transforms Bronze → Silver:

- Deduplicates on `(repo, number)`
- Fills null bodies with `""`
- Drops empty titles
- Maps GitHub labels to `high / medium / low` using an expanded keyword vocabulary covering repo-specific conventions (`kind/bug`, `type:feature`, `p0/p1/p2`, etc.)
- Drops issues with no assignable priority
- Creates combined `text` column (`title + body`)

**Priority mapping (first match wins):**

| Priority | Label examples |
|---|---|
| High | `bug`, `critical`, `crash`, `regression`, `p0`, `p1`, `security`, `blocker` |
| Medium | `enhancement`, `feature`, `improvement`, `performance`, `p2` |
| Low | `documentation`, `good first issue`, `help wanted`, `question`, `p3`, `p4` |

### Gold — Feature Engineering

`src/data/featurize.py` transforms Silver → Gold:

- Combines title (×3 weight) + body into a single `text` field
- Upsamples minority classes to balance the dataset (prevents majority-class bias)
- Stratified 80/20 train/test split
- Saves `data/gold/train.parquet`, `data/gold/test.parquet`, `data/gold/meta.json`

---

## Model Training

`src/models/train.py` trains three classifiers and logs all runs to MLflow:

| Classifier | Config |
|---|---|
| Logistic Regression | C=0.5, max_features=5k |
| Logistic Regression | C=1.0, max_features=10k |
| Random Forest | n_estimators=100, max_features=10k |

All classifiers use a TF-IDF vectorizer with unigrams + bigrams (`ngram_range=(1,2)`).

For each run, MLflow logs:
- Parameters (classifier, C, max_features, etc.)
- Test set metrics (F1 macro, accuracy, per-class F1)
- 3-fold cross-validation F1 (mean ± std)
- Full classification report as artifact
- `data_version` tag linking model → data version

The best model (by F1 macro) is saved to `models/best_model.joblib` and `metrics.json` is written for the CI/CD quality gate.

### View MLflow runs

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Evaluation

```bash
# Evaluate best model on Gold test split
python src/models/evaluate.py

# Run inference on a CSV file
python src/models/evaluate.py --input data/my_issues.csv --output data/predictions.csv

# Print MLflow run comparison table
python src/models/evaluate.py --compare-runs
```

Input CSV format for inference:

```csv
title,body
App crashes on startup,Steps to reproduce: delete config and run
Add dark mode support,Would improve usability
```

### KPIs

| Metric | Target |
|---|---|
| F1 macro | > 0.70 |
| Training time | < 2 min |
| CI/CD quality gate | F1 > 0.60 |

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "App crashes on startup", "body": "Null pointer exception on launch"}'
```

Response:
```json
{
  "priority": "high",
  "confidence": 0.91,
  "probabilities": {"high": 0.91, "medium": 0.07, "low": 0.02},
  "f1_macro": 0.74
}
```

### `GET /health`

Returns `{"status": "healthy"}` if model is loaded, `503` otherwise.

### `GET /`

Returns service info and current model F1 score.

---

## DVC Workflow

This project uses DVC with a local remote for data versioning. Each `dvc repro` run is committed as a new data/model version via `dvc.lock`.

### Setup (first time)

```bash
dvc init
dvc remote add -d local_remote /path/to/dvc-storage
git add .dvc/config
git commit -m "chore: add local DVC remote"
```

### Daily workflow

```bash
# Run the full pipeline
dvc repro

# Push data to local remote
dvc push

# Commit the version snapshot
git add dvc.lock
git commit -m "data: ingest v2 — added pytorch/pytorch"
```

### Reproduce any version

```bash
git checkout <commit-hash>
dvc checkout
```

### Pipeline stages

```
ingest → clean → featurize → train → evaluate
```

Defined in `dvc.yaml`. Parameters tracked in `params.yaml` — changing `test_size` or `C` triggers only the affected downstream stages.

---

## CI/CD

GitHub Actions runs on every push to `main` or `develop`:

| Job | Trigger | Steps |
|---|---|---|
| `test` | every push | flake8 lint, 100+ pytest tests, coverage report |
| `train` | main only | ingest → clean → featurize → train → evaluate → quality gate |
| `docker` | main only, after train | build image, smoke test `/health` |

The quality gate fails the pipeline if F1 macro < 0.60, preventing a regression from being deployed.

Add your GitHub token as a repository secret (`GITHUB_TOKEN` is automatic in Actions).

---

## Docker

```bash
# Build and run
docker-compose up

# Services:
#   API:    http://localhost:8000
#   MLflow: http://localhost:5000
```

The `models/` directory is volume-mounted so you can hot-swap the model without rebuilding the image.

---

## Limitations & Assumptions

- **Label coverage**: Only issues with recognisable priority labels are used (~40-70% of fetched issues depending on repo). Unlabelled issues are dropped.
- **Label noise**: Priority labels are assigned by repo maintainers with varying consistency. Some repos use `bug` for both trivial and critical issues.
- **Class balance**: Minority classes are upsampled with replacement, which may cause overfitting on small datasets. A larger fetch (`--pages 30`) mitigates this.
- **No issue history**: The model uses only title and body text. Comments, reactions, and time-to-close are available in Bronze/Silver but not used in Gold features.
- **Static model**: The pipeline does not retrain automatically on new data without a manual `dvc repro` or CI/CD trigger.