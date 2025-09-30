from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd, os
from pycaret.classification import load_model, predict_model
MODEL_BASE = os.getenv('PYCARET_MODEL','models/pycaret_fraud_v1'); THRESHOLD = float(os.getenv('THRESHOLD','0.5'))
app = FastAPI(title='Fraud Detection (PyCaret)', version='1.0'); model = load_model(MODEL_BASE)
class Transaction(BaseModel): features: dict = Field(..., description='Feature map (V1..V28, Amount, Time)')
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/predict')
def predict(tx: Transaction):
    X = pd.DataFrame([tx.features])
    out = predict_model(model, data=X)
    score = float(out.loc[0,'Score']) if 'Score' in out.columns else None
    label = int(out.loc[0,'Label']) if 'Label' in out.columns else None
    is_fraud = int((score is not None and score >= THRESHOLD) or (label == 1))
    return {'probability':score,'label':label,'is_fraud':is_fraud,'threshold':THRESHOLD}
