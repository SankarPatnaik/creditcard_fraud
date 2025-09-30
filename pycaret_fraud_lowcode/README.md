# Credit Card Fraud Detection — PyCaret (Low/No Code)

## Train
```bash
python scripts/train_pycaret.py --data data/raw/tiny_transactions.csv --out models/pycaret_fraud_v1
```
## Batch Predict
```bash
python scripts/batch_predict.py --model models/pycaret_fraud_v1 --data data/raw/tiny_transactions.csv --out predictions.csv --drop-target Class
```
## Serve API
```bash
uvicorn api.serve_api:app --reload --port 8000
```
