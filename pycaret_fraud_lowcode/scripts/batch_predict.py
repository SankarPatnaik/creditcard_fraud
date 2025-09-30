from __future__ import annotations
import argparse, pandas as pd
from pycaret.classification import load_model, predict_model
p = argparse.ArgumentParser()
p.add_argument('--model', required=True); p.add_argument('--data', required=True)
p.add_argument('--out', default='predictions.csv'); p.add_argument('--drop-target', default=None)
p.add_argument('--keep-target', action='store_true'); a = p.parse_args()
model = load_model(a.model); df = pd.read_csv(a.data)
if a.drop_target and not a.keep_target and a.drop_target in df.columns: df = df.drop(columns=[a.drop_target])
preds = predict_model(model, data=df); preds.to_csv(a.out, index=False); print(f'Wrote {a.out}')
