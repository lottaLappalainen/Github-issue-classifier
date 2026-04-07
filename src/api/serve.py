"""
serve.py — FastAPI Prediction Service
Loads the trained model pipeline and serves predictions via REST API.

Usage:
    python src/api/serve.py
    uvicorn src.api.serve:app --reload --port 8000
"""

import json
import joblib
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="GitHub Issue Priority Classifier",
    description="Predicts the priority (high/medium/low) of a GitHub issue.",
    version="1.0.0",
)

# ── state ─────────────────────────────────────────────────────────────────────
pipeline   = None   # full sklearn Pipeline (TF-IDF + classifier)
model_meta = {}


# ── load on startup ───────────────────────────────────────────────────────────

@app.on_event("startup")
def load_model():
    global pipeline, model_meta

    model_path = MODEL_DIR / "best_model.joblib"
    meta_path  = ROOT / "metrics.json"

    if not model_path.exists():
        log.warning(f"No model found at {model_path} — run train.py first.")
        return

    pipeline = joblib.load(model_path)
    log.info(f"Model loaded from {model_path}")

    if meta_path.exists():
        model_meta = json.loads(meta_path.read_text())
        log.info(f"F1 macro: {model_meta.get('f1_macro', 'N/A')}")


# ── schemas ───────────────────────────────────────────────────────────────────

class IssueRequest(BaseModel):
    title: str
    body: str = ""

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "App crashes on startup when config file is missing",
                "body": "Steps to reproduce: delete config.json and run the app."
            }
        }
    }


class PredictionResponse(BaseModel):
    priority:      str
    confidence:    float
    probabilities: dict[str, float]
    f1_macro:      float


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":  "GitHub Issue Priority Classifier",
        "status":   "running" if pipeline is not None else "model not loaded",
        "f1_macro": model_meta.get("f1_macro"),
        "docs":     "/docs",
    }


@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(issue: IssueRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first")

    # Same text combination as featurize.py
    text = (issue.title + " ") * 3 + issue.body

    preds    = pipeline.predict([text])
    priority = preds[0]

    proba_dict = {}
    confidence = 1.0

    if hasattr(pipeline, "predict_proba"):
        probas  = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probas)}
        confidence = float(max(probas))
    else:
        proba_dict = {"high": 0.0, "medium": 0.0, "low": 0.0}
        proba_dict[priority] = 1.0

    return PredictionResponse(
        priority      = priority,
        confidence    = round(confidence, 4),
        probabilities = proba_dict,
        f1_macro      = model_meta.get("f1_macro", 0.0),
    )


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.serve:app", host="0.0.0.0", port=8000, reload=True)