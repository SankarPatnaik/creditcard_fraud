from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/creditcard_xgb_v1.joblib")
THRESHOLD = float(os.getenv("THRESHOLD", "0.4"))

app = FastAPI(title="Fraud Detection API", version="1.0")
model = None

class Transaction(BaseModel):
    features: dict = Field(..., description="Key-value map of model features")

@app.on_event("startup")
def _load_model():
    global model
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(tx: Transaction):
    X = pd.DataFrame([tx.features])
    proba = model.predict_proba(X)[:, 1][0]
    decision = int(proba >= THRESHOLD)
    return {"probability": float(proba), "is_fraud": decision, "threshold": THRESHOLD}
