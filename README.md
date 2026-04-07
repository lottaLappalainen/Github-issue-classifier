# GitHub Issue Priority Classifier

![CI/CD](https://github.com/lottaLappalainen/github-issue-classifier/actions/workflows/train.yml/badge.svg)

An end-to-end MLOps pipeline that automatically classifies GitHub issues as **high / medium / low** priority using NLP. Covers the full lifecycle: data ingestion, cleaning, feature engineering, model training, experiment tracking, drift monitoring, automatic retraining, REST API serving with prediction logging, MLflow Model Registry governance, and CI/CD automation.

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
        │            → MLflow Model Registry (Staging → Production)
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
| Data versioning | DVC + dvc.lock |
| ML tracking | scikit-learn + MLflow |
| Model registry | MLflow Model Registry |
| Monitoring | Custom chi-square + Jaccard + SQLite |
| CI/CD | GitHub Actions |
| Containerisation | Docker + docker-compose |
| API serving | FastAPI + uvicorn |
| Data source | GitHub REST API (18 repos) |

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
│   │   ├── ingest.py           # GitHub API → Bronze (18 repos)
│   │   ├── clean.py            # Bronze → Silver (dedup, label, validate)
│   │   └── featurize.py        # Silver → Gold (text, balance, split, version)
│   ├── models/
│   │   ├── train.py            # Train 3 classifiers, MLflow + Registry
│   │   └── evaluate.py         # Evaluate, compare runs, run inference
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
│   └── prediction_log.db       # SQLite log of every API prediction
│
├── tests/
│   ├── conftest.py
│   ├── test_ingest.py          # 35 tests
│   ├── test_clean.py           # 46 tests
│   ├── test_featurize.py       # 38 tests
│   ├── test_model.py           # 46 tests
│   ├── test_evaluate.py        # 38 tests
│   ├── test_monitor.py         # 38 tests — drift checks + registry
│   └── test_retrain_trigger.py # 30 tests
│
├── .github/workflows/train.yml  # CI/CD: all jobs run on every push
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                     # Pipeline: ingest→clean→featurize→monitor→train→evaluate
├── dvc.lock                     # Committed version snapshot
├── params.yaml                  # Tracked hyperparameters
├── metrics.json                 # Latest F1, run ID, registry version
└── requirements.txt
```

**Total test coverage: 289 tests across 7 files.**

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
# Option A: DVC (recommended — tracks all versions automatically)
dvc repro

# Option B: Manual step by step
python src/data/ingest.py
python src/data/clean.py
python src/data/featurize.py
python src/monitoring/monitor.py
python src/models/train.py
python src/models/evaluate.py
```

### 4. Serve the API

```bash
# With Docker (recommended — starts API + MLflow UI)
docker-compose up

# Or directly
uvicorn src.api.serve:app --reload --port 8000
```

API docs at `http://localhost:8000/docs` | MLflow UI at `http://localhost:5000`

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## Data Pipeline (Medallion Architecture)

### Bronze — Raw Ingestion

`src/data/ingest.py` fetches issues from 18 curated public repos via the GitHub REST API. Repos were selected for high issue volume and consistent label discipline across all three priority tiers. Pull requests are filtered out at ingestion time.

```bash
# Quick run (3 repos, 5 pages)
python src/data/ingest.py --repos microsoft/vscode facebook/react django/django --pages 5

# Full run (18 repos, 30 pages each — ~15-20 min)
python src/data/ingest.py
```

### Silver — Cleaning & Labelling

`src/data/clean.py` transforms Bronze → Silver:

- Deduplicates on `(repo, number)`
- Fills null bodies, drops empty titles
- Maps GitHub labels → `high / medium / low` using an expanded keyword vocabulary covering repo-specific conventions (`kind/bug`, `type:feature`, `p0/p1/p2`, `i-prioritize-high`, etc.)
- Drops issues with no assignable priority
- Creates combined `text` column

| Priority | Label examples |
|---|---|
| High | `bug`, `critical`, `crash`, `regression`, `p0`, `p1`, `security`, `blocker` |
| Medium | `enhancement`, `feature`, `improvement`, `performance`, `p2` |
| Low | `documentation`, `good first issue`, `help wanted`, `question`, `p3`, `p4` |

### Gold — Feature Engineering

`src/data/featurize.py` transforms Silver → Gold:

- Combines title (×3 weight) + body into a single `text` field — title repetition gives the model a stronger signal on the most important part
- Upsamples minority classes to balance the dataset
- Stratified 80/20 train/test split
- Saves `train.parquet`, `test.parquet`, and enriched `meta.json`

`meta.json` is the key file for the monitoring system. It records the `gold_version` (auto-incremented: v1, v2, …), a SHA-256 hash of the Silver file, class distribution, and split config. The monitor compares this file against a saved baseline to detect drift.

```bash
# Auto-increment version
python src/data/featurize.py

# Explicit version tag
python src/data/featurize.py --gold-version v2
```

---

## Model Training & MLflow Registry

`src/models/train.py` trains three classifiers and implements the full MLflow Model Registry lifecycle.

### Classifiers trained

| Classifier | Config |
|---|---|
| Logistic Regression | C=0.5, TF-IDF 5k features |
| Logistic Regression | C=1.0, TF-IDF 10k features |
| Random Forest | 100 estimators, TF-IDF 10k features |

All use TF-IDF with unigrams + bigrams and `sublinear_tf=True`.

### MLflow tracking

For every run, MLflow logs parameters, test set metrics (F1 macro, accuracy, per-class F1), 3-fold cross-validation F1 (mean ± std), a full classification report artifact, and a `data_version` tag linking the model to the exact Gold dataset version it was trained on.

### MLflow Model Registry

After training, the best run is registered in the **MLflow Model Registry** under the name `github-issue-priority`. The registry implements governance workflow:

```
Run logged → Registered as new version
                    │
                    ├── F1 ≥ 0.70 → transition to Production
                    │              (previous Production → Archived)
                    │
                    └── F1 < 0.70 → left in Staging for manual review
```

Each registered version is tagged with `data_version` and `f1_macro` for full traceability — you can see exactly which dataset version produced which model version.

```bash
# Train and register
python src/models/train.py --gold-version v2

# View registry
mlflow ui   # → http://localhost:5000 → Models tab
```

### Primary KPI
**F1 macro** — macro-averaged F1 score across all three classes. Chosen because the classes are balanced after featurization, and macro averaging treats each class equally regardless of size. Target: > 0.70. CI/CD quality gate: > 0.60.

### Secondary KPI
**Cross-validation F1 stability** (mean ± std over 3 folds) — measures whether the model generalises consistently or overfits to a particular split. A high mean with low std indicates a robust model. Logged to MLflow for every run.

---

## Drift Monitoring

`src/monitoring/monitor.py` runs six checks comparing the current Gold `meta.json` against a saved baseline. The results are written to `monitoring/drift_report.json`. Any check that fires sets `retrain_required: true`.

### The six checks

| Check | What it detects | Method |
|---|---|---|
| `class_distribution` | Label distribution shifted | Chi-square test (p < 0.05) |
| `dataset_size` | Training set grew or shrank significantly | Relative change > 20% |
| `class_balance` | One class now dominates | Majority/minority ratio > 2.0 |
| `gold_version` | New data batch ingested | Version string changed |
| `prediction_confidence` | Model degrading in production | Mean API confidence < 0.70 |
| `text_vocabulary` | Input feature space shifted | Jaccard similarity < 0.60 |

### Data drift vs concept drift

**Data drift** (input distribution shift) is caught by `text_vocabulary`: if the vocabulary of incoming GitHub issues has changed significantly — for example, a new framework becomes popular and its bug reports use new terminology the model never saw — the top-500 token sets will diverge. Low Jaccard similarity is an early warning that the feature space has changed.

**Concept drift** (input-output relationship shift) is caught by `prediction_confidence`: even when the words are familiar, if the same kinds of issues are now labelled differently by maintainers, the model's softmax probabilities become more uniform. A drop in mean confidence across recent API predictions is a production-side signal that the world has changed in a way the model no longer understands.

```bash
# Standard run (5 checks)
python src/monitoring/monitor.py

# Include vocabulary check (requires baseline_vocab.json)
python src/monitoring/monitor.py --text-drift

# Build vocabulary baseline (run once after first featurize)
python -c "from src.monitoring.monitor import build_vocab_baseline; build_vocab_baseline()"

# Exit with code 1 if retrain is required (for CI gates)
python src/monitoring/monitor.py --fail-on-drift
```

---

## Automatic Retraining

`src/monitoring/retrain_trigger.py` reads `drift_report.json` and, if `retrain_required` is true, runs the full pipeline automatically:

```
ingest → clean → featurize → train → evaluate
```

The new model version is evaluated against the quality gate (F1 ≥ 0.60). If it passes, `baseline_meta.json` is updated to the new Gold version. If it fails, the old model stays in production and the trigger exits with code 1.

```bash
# Check drift and retrain if needed
python src/monitoring/retrain_trigger.py

# Check only — do not retrain
python src/monitoring/retrain_trigger.py --dry-run

# Override the version tag
python src/monitoring/retrain_trigger.py --gold-version v3
```

---

## API Reference

`src/api/serve.py` — every prediction is logged to `monitoring/prediction_log.db` (SQLite) for observability. The monitor reads this database to detect confidence drift.

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "App crashes on startup", "body": "Null pointer on launch"}'
```

```json
{
  "priority": "high",
  "confidence": 0.91,
  "probabilities": {"high": 0.91, "medium": 0.07, "low": 0.02},
  "f1_macro": 0.9458,
  "data_version": "v1"
}
```

### `GET /stats`

Returns a summary of the last 100 predictions — used by `monitor.py` to track confidence drift over time.

```json
{
  "n_predictions": 100,
  "mean_confidence": 0.887,
  "min_confidence": 0.612,
  "max_confidence": 0.997,
  "priority_distribution": {"high": 41, "medium": 35, "low": 24}
}
```

### `GET /log?limit=20`

Returns the last N raw prediction rows for debugging.

### `GET /health`

Returns `{"status": "healthy"}` if model is loaded, `503` otherwise.

### `GET /`

Returns service info, current model F1, and data version.

---

## DVC Pipeline

The full pipeline is defined in `dvc.yaml` with five stages: `ingest → clean → featurize → monitor → train → evaluate`. Parameters in `params.yaml` (`test_size`, `max_features`, `ngram_range`, `C`) are tracked — changing a parameter only reruns downstream stages.

```bash
# Run full pipeline (skips unchanged stages)
dvc repro

# Commit the new version snapshot
git add dvc.lock
git commit -m "data: Gold v2 — added pytorch/pytorch, drift detected, retrained"
```

Each committed `dvc.lock` is a reproducible snapshot of the exact data and model versions. Any previous state can be restored with `git checkout <hash> && dvc checkout`.

---

## CI/CD

GitHub Actions runs on every push to every branch — no branch filter. Three sequential jobs:

| Job | Runs when | Steps |
|---|---|---|
| `test` | Every push | flake8 lint, 289 pytest tests, coverage report |
| `train` | After `test` passes | ingest (3 pages) → clean → featurize → monitor → train → quality gate |
| `docker` | After `train` passes | Build image, smoke-test `/health` endpoint |

The quality gate (F1 ≥ 0.60) blocks the Docker build if the model regressed.

---

## Docker

```bash
docker-compose up

# Services:
#   API:    http://localhost:8000  (predict, stats, log, health)
#   MLflow: http://localhost:5000  (experiments, registry)
```

`models/` and `metrics.json` are volume-mounted so you can hot-swap the model without rebuilding the image.

---

## Limitations & Assumptions

- **Label coverage**: Only issues with recognisable priority labels are used (~40–70% of fetched issues depending on repo). Unlabelled issues are dropped.
- **Label noise**: Priority labels are assigned by maintainers with varying consistency. Some repos use `bug` for both trivial and critical issues.
- **Class balance**: Minority classes are upsampled with replacement, which may cause overfitting on small datasets. A larger fetch (`--pages 30`) mitigates this.
- **Concept drift detection**: The confidence-based check is a proxy — it detects that the model is degrading but cannot explain why without labelled production data. True concept drift confirmation requires human review of low-confidence predictions.
- **Vocabulary drift threshold**: The 0.60 Jaccard threshold is heuristic. Repos with fast-moving terminology (e.g., AI frameworks) may trigger false positives; stable repos (e.g., Linux kernel) may never reach this threshold even with real drift.
- **Text features only**: Comments, reactions, assignees, milestones, and time-to-close are available in Bronze/Silver but not used in Gold. These could improve precision for the `high` class specifically.
- **No online learning**: The pipeline retrains from scratch on each trigger. Incremental or online learning would reduce retraining cost on large datasets.