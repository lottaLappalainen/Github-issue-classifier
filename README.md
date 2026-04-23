# GitHub Issue Priority Classifier

![CI/CD](https://github.com/lottaLappalainen/Github-issue-classifier/actions/workflows/train.yml/badge.svg)

An end-to-end MLOps pipeline that automatically classifies GitHub issues as **high / medium / low** priority using NLP. Covers the full lifecycle: data ingestion, cleaning, feature engineering, model training, experiment tracking, drift monitoring, automatic retraining, REST API serving with prediction logging, MLflow Model Registry governance, and CI/CD automation.

Built as part of the MLOps Course at Tampere University 2026.

---

## Architecture

```
GitHub REST API (16 repos)
        │
        ▼
[Bronze Layer]  ── raw issues as parquet, untouched
        │
        ▼
[Silver Layer]  ── deduplicated, labelled, nulls filled
        │
        ▼
[Gold Layer]    ── combined text, class-balanced, train/test split
        │  └── meta.json (gold_version, silver_hash, class distribution)
        ▼
[Drift Monitor] ── 6 checks: label drift, size drift, balance drift,
        │           version change, confidence drift, vocabulary drift
        │           → monitoring/drift_report.json
        ▼
[Retrain Trigger] ── reads drift_report.json, kicks off full pipeline
        │             if retrain_required = true
        ▼
[Model Training] ── TF-IDF + 3 sklearn classifiers, MLflow tracking
        │            → MLflow Model Registry (aliases: production/staging/archived)
        ▼
[FastAPI Service] ── /predict, /health, /stats, /log
        │            → every prediction logged to prediction_log.db
        ▼
[CI/CD] ── GitHub Actions: lint → test → train → quality gate → Docker
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data versioning | DVC + DagsHub |
| ML tracking | scikit-learn + MLflow |
| Model registry | MLflow Model Registry (aliases) |
| Monitoring | Custom chi-square + Jaccard + SQLite |
| CI/CD | GitHub Actions |
| Containerisation | Docker + docker-compose |
| API serving | FastAPI + uvicorn |
| Data source | GitHub REST API (16 repos) |

---

## Project Structure

```
github-issue-classifier/
│
├── data/
│   ├── bronze/                 # Raw API data (DVC tracked)
│   ├── silver/                 # Cleaned, labelled (DVC tracked)
│   └── gold/                   # ML-ready splits + meta.json (DVC tracked)
│
├── src/
│   ├── data/
│   │   ├── ingest.py           # GitHub API → Bronze (16 repos)
│   │   ├── clean.py            # Bronze → Silver (dedup, label, validate)
│   │   └── featurize.py        # Silver → Gold (text, balance, split, version)
│   ├── features/
│   │   └── text_features.py    # Shared text combination logic (prevents training-serving skew)
│   ├── models/
│   │   ├── train.py            # Train 3 classifiers, MLflow + Registry
│   │   └── evaluate.py         # Evaluate, compare runs, version comparison table
│   ├── monitoring/
│   │   ├── monitor.py          # 6-check drift detector → drift_report.json
│   │   └── retrain_trigger.py  # Reads drift_report, triggers retraining
│   └── api/
│       └── serve.py            # FastAPI: /predict /health /stats /log
│
├── monitoring/
│   ├── baseline_meta.json      # Gold metadata snapshot (drift baseline)
│   ├── baseline_vocab.json     # Top-500 token baseline (vocabulary drift)
│   ├── drift_report.json       # Latest drift check results
│   ├── version_comparison.json # MLflow run comparison across data versions
│   └── prediction_log.db       # SQLite log of every API prediction
│
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis across all three layers
│
├── evidence/                   # Generated graphs and screenshots (20 PNG files)
│
├── tests/
│   ├── test_ingest.py          # 35 tests
│   ├── test_clean.py           # 46 tests
│   ├── test_featurize.py       # 38 tests
│   ├── test_model.py           # 46 tests
│   ├── test_evaluate.py        # 38 tests
│   ├── test_monitor.py         # 38 tests
│   └── test_retrain_trigger.py # 30 tests
│
├── .github/workflows/
│   ├── train.yml               # CI/CD: all three jobs run on every push
│   └── scheduled_retrain.yml   # Weekly Monday 06:00 UTC retrain
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                    # Pipeline stages
├── dvc.lock                    # Committed version snapshot
├── params.yaml                 # All thresholds and hyperparameters — DVC tracked
├── setup.py                    # Makes src/ importable as a package
├── generate_evidence.py        # Generates all 20 evidence graphs
├── metrics.json                # Latest F1, run ID, registry version
└── requirements.txt
```

**Total test coverage: 289 tests across 7 files.**

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/lottaLappalainen/Github-issue-classifier
cd Github-issue-classifier
pip install -r requirements.txt
pip install -e .
```

### 2. Set your GitHub token

**Windows CMD:**
```cmd
set GITHUB_TOKEN=your_token_here
```

**Linux / macOS:**
```bash
export GITHUB_TOKEN=your_token_here
```

Generate a token at [github.com/settings/tokens](https://github.com/settings/tokens) — no scopes needed for public repos.

### 3. Run the full pipeline

```bash
# Option A: DVC (recommended)
dvc repro

# Option B: Manual
set PYTHONPATH=.
python src/data/ingest.py --pages 3
python src/data/clean.py
python src/data/featurize.py
python src/monitoring/monitor.py --text-drift
python src/models/train.py
python src/models/evaluate.py
```

### 4. Serve the API

```bash
docker-compose up
# API: http://localhost:8000/docs
# MLflow: http://localhost:5000
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

### 6. Generate evidence graphs

```bash
python generate_evidence.py
```

---

## Data Pipeline (Medallion Architecture)

### Bronze — Raw Ingestion

Fetches issues from 16 curated public repos.

### Silver — Cleaning & Labelling

| Priority | Label examples |
|---|---|
| High | `bug`, `critical`, `crash`, `regression`, `p0`, `p1`, `security`, `blocker` |
| Medium | `enhancement`, `feature`, `improvement`, `performance`, `p2` |
| Low | `documentation`, `good first issue`, `help wanted`, `question`, `p3`, `p4` |

### Gold — Feature Engineering

Text combination logic is centralised in `src/features/text_features.py` — used by both featurize and the API to prevent training-serving skew. Title is repeated ×3 before concatenation with body. All parameters read from `params.yaml`.

**Current data statistics:**
- Bronze: 1,555 raw issues from 16 repos
- Silver: 282 labelled issues (18% label hit rate)
- Gold: 441 balanced training examples, 111 test examples (37 per class)

---

## Model Training & MLflow Registry

All hyperparameters are read from `params.yaml`.

| Classifier | Test F1 | CV F1 (σ) |
|---|---|---|
| LR C=0.5, TF-IDF 5k | 0.9458 | 0.947 (0.023) |
| **LR C=1.0, TF-IDF 10k** | **0.9549** | 0.948 (0.023) |
| Random Forest 100est | 0.9547 | 0.950 (0.017) |

### MLflow Model Registry

Uses **aliases** (MLflow ≥ 2.9 compatible):

- F1 ≥ 0.70 → alias `production` (previous production → `archived-vN`)
- F1 < 0.70 → alias `staging` for manual review

Every version tagged with `data_version` and `f1_macro`.

---

## Drift Monitoring

| Check | What it detects | Threshold |
|---|---|---|
| `class_distribution` | Label proportions shifted | Chi-square p < 0.05 |
| `dataset_size` | Training set changed | Relative change > 20% |
| `class_balance` | One class dominates | Ratio > 2.0 |
| `gold_version` | New data batch | Version changed |
| `prediction_confidence` | Concept drift proxy | Mean confidence < 0.70 |
| `text_vocabulary` | Input feature drift | Jaccard < 0.60 |

**Data drift** (vocabulary Jaccard) detects input distribution shift. **Concept drift** (prediction confidence) detects relationship shift between inputs and outputs — without requiring labelled production feedback.

---

## Automatic Retraining

`retrain_trigger.py` reads `drift_report.json`. If retrain is required, runs the full pipeline with `--pages 5`, passes `GITHUB_TOKEN` through to avoid rate limits, and evaluates against the quality gate (F1 ≥ 0.60).

```bash
python src/monitoring/retrain_trigger.py           # retrain if needed
python src/monitoring/retrain_trigger.py --dry-run # log decision only
```

---

## API Reference

| Endpoint | Description |
|---|---|
| `POST /predict` | Predict priority for an issue title + body |
| `GET /stats` | Last 100 predictions summary (feeds confidence drift check) |
| `GET /log` | Raw prediction rows for debugging |
| `GET /health` | Health check — 503 if model not loaded |
| `GET /` | Service info, current F1, data version, registry alias |

---

## DVC + DagsHub

Data and experiments visible at:
`https://dagshub.com/lottaLappalainen/Github-issue-classifier`

```bash
dvc repro      # run pipeline, skip unchanged stages
dvc push       # push data to DagsHub
dvc checkout   # restore data for any git commit
```

---

## CI/CD

| Job | Steps |
|---|---|
| `Lint & Unit Tests` | flake8, 289 pytest tests, coverage |
| `Train & Evaluate` | full pipeline, quality gate, artifacts |
| `Build Docker Image` | build + smoke test `/health` |

Scheduled retrain every Monday. Both workflows use `PYTHONPATH: "."`.

---

## Docker

Model is **not baked into the image** — volume-mounted at runtime. `HEALTHCHECK` polls `/health` every 30 seconds.

```bash
docker-compose up
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

---

## Limitations

- **Label coverage**: ~18% of fetched issues have mappable labels. The rest are dropped.
- **Label noise**: Labelling consistency varies across repos. Some distinctive vocabulary tokens are repo-specific artefacts such as usernames or project names rather than genuine priority signals.
- **Removed repos**: rust-lang/rust and facebook/react removed after EDA showed 0% label hit rate.
- **Concept drift detection**: Confidence-based check is a proxy — cannot explain degradation without labelled production data.
- **Class balance**: Upsampling with replacement may cause mild overfitting on small datasets.
- **Text features only**: Comments, reactions, and assignees not used in Gold features.
- **No online learning**: Full retrain from scratch on each trigger.