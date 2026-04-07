"""
serve.py — FastAPI Prediction Service
Loads the trained model and serves predictions via REST API.
"""

import json
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path("models")

app = FastAPI(
    title="GitHub Issue Priority Classifier",
    description="Predicts the priority (high/medium/low) of a GitHub issue.",
    version="1.0.0",
)

# ── Load model on startup ─────────────────────────────────────────────────────

model       = None
vectorizer  = None
label_encoder = None
model_meta  = {}

@app.on_event("startup")
def load_model():
    global model, vectorizer, label_encoder, model_meta

    try:
        model         = joblib.load(MODEL_DIR / "model.joblib")
        vectorizer    = joblib.load(MODEL_DIR / "vectorizer.joblib")
        label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

        with open(MODEL_DIR / "meta.json") as f:
            model_meta = json.load(f)

        print(f"✅ Model loaded: {model_meta.get('best_model')} (F1={model_meta.get('f1_macro', 'N/A'):.4f})")

    except FileNotFoundError:
        print("⚠️  No model found. Run train.py first.")


# ── Request / Response schemas ────────────────────────────────────────────────

class IssueRequest(BaseModel):
    title: str
    body: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "title": "App crashes on startup when config file is missing",
                "body": "Steps to reproduce: delete config.json and run the app. Expected: helpful error message. Actual: unhandled exception."
            }
        }


class PredictionResponse(BaseModel):
    priority:    str
    confidence:  float
    probabilities: dict[str, float]
    model_version: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "GitHub Issue Priority Classifier",
        "status":  "running",
        "model":   model_meta.get("best_model", "not loaded"),
        "f1_macro": model_meta.get("f1_macro"),
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(issue: IssueRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Combine title + body (same logic as featurize.py)
    text = issue.title * 3 + " " + issue.body

    # Vectorize
    X = vectorizer.transform([text])

    # Predict
    label_idx   = model.predict(X)[0]
    label_name  = label_encoder.inverse_transform([label_idx])[0]

    # Probabilities (not all models support this, fallback gracefully)
    proba_dict = {}
    confidence = 1.0

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[0]
        proba_dict = {
            label_encoder.classes_[i]: round(float(p), 4)
            for i, p in enumerate(probas)
        }
        confidence = float(max(probas))
    else:
        proba_dict = {c: 0.0 for c in label_encoder.classes_}
        proba_dict[label_name] = 1.0

    return PredictionResponse(
        priority      = label_name,
        confidence    = round(confidence, 4),
        probabilities = proba_dict,
        model_version = model_meta.get("best_model", "unknown"),
    )


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)