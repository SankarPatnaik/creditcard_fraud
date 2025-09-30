# Credit Card Fraud Detection — Student Starter (v2)

End-to-end skeleton for a fraud detection project: EDA, preprocessing, training, explainability (SHAP),
and a FastAPI service.

## Quickstart

### 1) Create & activate venv
```bash
python3 -m venv venv
source venv/bin/activate     # Windows: .\venv\Scripts\activate
```

### 2) Install deps
```bash
pip install -r requirements.txt
```

### 3) Pick a dataset
- Default: tiny synthetic CSV at `data/raw/tiny_transactions.csv`
- Recommended: Kaggle "Credit Card Fraud Detection" dataset (`creditcard.csv`)

#### Switch active dataset (writes to `data/raw/transactions.csv`)
```bash
# use Kaggle CSV (after you download it locally)
python scripts/switch_dataset.py --use kaggle --kaggle-path /absolute/path/to/creditcard.csv

# switch back to tiny demo
python scripts/switch_dataset.py --use tiny
```
> All notebooks and scripts read from `data/raw/transactions.csv`.

### 4) Run EDA notebook
```bash
jupyter notebook notebooks/01_exploration.ipynb
```

### 5) Train a model (with optional SHAP plot)
```bash
python -m src.models.train --data data/raw/transactions.csv --out models/creditcard_xgb_v1.joblib --shap
# SHAP summary saved to reports/shap_summary.png
```

### 6) Explain a specific row
```bash
python -m src.models.explain --model models/creditcard_xgb_v1.joblib --data data/raw/transactions.csv --index 0 --outdir reports
```

### 7) Batch predict (CLI demo)
```bash
# we provide a small batch file
python -m src.models.predict --model models/creditcard_xgb_v1.joblib --data data/raw/tiny_batch.csv --out predictions.json --pretty
```

### 8) Serve API
```bash
uvicorn src.api.app:app --reload --port 8000
```
Then POST JSON to `http://127.0.0.1:8000/predict`:
```json
{
  "features": {"V1": 0.1, "V2": -1.2, "Amount": 125.0, "Time": 10000, "V3": 0.0}
}
```

### 9) Docker (optional)
```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

## Repo layout
```
creditcard-fraud/
├─ data/
│  ├─ raw/                 # original datasets (CSV)
│  └─ processed/           # train/test CSVs
├─ notebooks/
│  └─ 01_exploration.ipynb
├─ reports/                # SHAP plots, metrics
├─ src/
│  ├─ data/preprocess.py
│  ├─ features/feature_engineering.py
│  ├─ models/train.py
│  ├─ models/predict.py
│  ├─ models/explain.py
│  └─ api/app.py
├─ scripts/switch_dataset.py
├─ tests/
├─ Dockerfile
├─ requirements.txt
└─ README.md
```
