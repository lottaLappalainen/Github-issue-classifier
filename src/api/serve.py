"""
src/api/serve.py — FastAPI Prediction Service
Loads the trained model pipeline and serves predictions via REST API.
Every prediction is logged to monitoring/prediction_log.db (SQLite)
so that monitor.py can detect confidence drift over time.

Usage:
    python src/api/serve.py
    uvicorn src.api.serve:app --reload --port 8000
"""

import json
import sqlite3
import joblib
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── paths ──────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parents[2]
MODEL_DIR      = ROOT / "models"
MONITORING_DIR = ROOT / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)
PRED_LOG_DB    = MONITORING_DIR / "prediction_log.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="GitHub Issue Priority Classifier",
    description=(
        "Predicts the priority (high/medium/low) of a GitHub issue. "
        "Every prediction is logged for drift monitoring."
    ),
    version="1.0.0",
)

# ── state ──────────────────────────────────────────────────────────────────
pipeline   = None
model_meta = {}


# ── prediction log (SQLite) ───────────────────────────────────────────────

def _init_db() -> None:
    """Create the prediction log table if it does not exist."""
    with sqlite3.connect(PRED_LOG_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                title             TEXT,
                body              TEXT,
                predicted_priority TEXT   NOT NULL,
                confidence        REAL    NOT NULL,
                prob_high         REAL,
                prob_medium       REAL,
                prob_low          REAL,
                model_f1          REAL,
                data_version      TEXT
            )
        """)
        conn.commit()
    log.info(f"Prediction log DB ready → {PRED_LOG_DB}")


def _log_prediction(
    title: str,
    body: str,
    priority: str,
    confidence: float,
    probabilities: dict,
    model_f1: float,
    data_version: str,
) -> None:
    """Append one prediction row to the SQLite log."""
    try:
        with sqlite3.connect(PRED_LOG_DB) as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (timestamp, title, body, predicted_priority,
                     confidence, prob_high, prob_medium, prob_low,
                     model_f1, data_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    title[:500],          # cap length to avoid runaway rows
                    body[:1000],
                    priority,
                    round(confidence, 4),
                    round(probabilities.get("high",   0.0), 4),
                    round(probabilities.get("medium", 0.0), 4),
                    round(probabilities.get("low",    0.0), 4),
                    round(model_f1, 4),
                    data_version,
                ),
            )
            conn.commit()
    except Exception as exc:
        # Never let logging crash the API
        log.warning(f"Failed to log prediction: {exc}")


# ── startup ───────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup() -> None:
    global pipeline, model_meta

    model_path = MODEL_DIR / "best_model.joblib"
    meta_path  = ROOT / "metrics.json"

    if not model_path.exists():
        log.warning(f"No model found at {model_path} — run train.py first.")
        return

    pipeline = joblib.load(model_path)
    log.info(f"Model loaded ✓  ({model_path})")

    if meta_path.exists():
        model_meta = json.loads(meta_path.read_text())
        log.info(f"Model meta: F1={model_meta.get('f1_macro')}  "
                 f"version={model_meta.get('data_version')}")

    _init_db()


# ── schemas ───────────────────────────────────────────────────────────────

class IssueRequest(BaseModel):
    title: str
    body: str = ""

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "App crashes on startup when config file is missing",
                "body":  "Steps to reproduce: delete config.json and run the app.",
            }
        }
    }


class PredictionResponse(BaseModel):
    priority:      str
    confidence:    float
    probabilities: dict[str, float]
    f1_macro:      float
    data_version:  str


# ── endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":      "GitHub Issue Priority Classifier",
        "status":       "running" if pipeline is not None else "model not loaded",
        "f1_macro":     model_meta.get("f1_macro"),
        "data_version": model_meta.get("data_version"),
        "docs":         "/docs",
        "prediction_log": str(PRED_LOG_DB),
    }


@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(issue: IssueRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first")

    # Same text combination as featurize.py (title weighted 3x)
    text = (issue.title + " ") * 3 + issue.body

    priority   = pipeline.predict([text])[0]
    proba_dict = {}
    confidence = 1.0

    if hasattr(pipeline, "predict_proba"):
        probas     = pipeline.predict_proba([text])[0]
        classes    = pipeline.classes_
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probas)}
        confidence = float(max(probas))
    else:
        proba_dict = {"high": 0.0, "medium": 0.0, "low": 0.0}
        proba_dict[priority] = 1.0

    f1           = model_meta.get("f1_macro", 0.0)
    data_version = model_meta.get("data_version", "unknown")

    # ── Log every prediction for observability ─────────────────────────
    _log_prediction(
        title        = issue.title,
        body         = issue.body,
        priority     = priority,
        confidence   = confidence,
        probabilities= proba_dict,
        model_f1     = f1,
        data_version = data_version,
    )

    return PredictionResponse(
        priority      = priority,
        confidence    = round(confidence, 4),
        probabilities = proba_dict,
        f1_macro      = f1,
        data_version  = data_version,
    )


@app.get("/stats")
def prediction_stats():
    """
    Summarise recent prediction log — used by monitor.py to detect
    confidence drift without reading the DB directly.
    Returns counts, mean confidence, and priority distribution
    for the last 100 predictions.
    """
    if not PRED_LOG_DB.exists():
        return {"message": "No predictions logged yet."}

    try:
        with sqlite3.connect(PRED_LOG_DB) as conn:
            rows = conn.execute("""
                SELECT predicted_priority, confidence
                FROM predictions
                ORDER BY id DESC
                LIMIT 100
            """).fetchall()

        if not rows:
            return {"message": "No predictions logged yet."}

        confidences  = [r[1] for r in rows]
        priorities   = [r[0] for r in rows]
        dist         = {}
        for p in priorities:
            dist[p] = dist.get(p, 0) + 1

        return {
            "n_predictions":       len(rows),
            "mean_confidence":     round(sum(confidences) / len(confidences), 4),
            "min_confidence":      round(min(confidences), 4),
            "max_confidence":      round(max(confidences), 4),
            "priority_distribution": dist,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/log")
def prediction_log(limit: int = 20):
    """Return the last `limit` raw prediction rows for debugging."""
    if not PRED_LOG_DB.exists():
        return {"rows": []}

    try:
        with sqlite3.connect(PRED_LOG_DB) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return {"rows": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── run directly ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.serve:app", host="0.0.0.0", port=8000, reload=True)