from __future__ import annotations
import argparse, json
import pandas as pd
import joblib

def parse_args():
    p = argparse.ArgumentParser(description="Batch predict fraud probabilities")
    p.add_argument("--model", type=str, default="models/creditcard_xgb_v1.joblib")
    p.add_argument("--data", type=str, required=True, help="CSV with features (no target)")
    p.add_argument("--out", type=str, default="predictions.json")
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--pretty", action="store_true", help="Pretty-print first few predictions to stdout")
    return p.parse_args()

def main():
    args = parse_args()
    model = joblib.load(args.model)
    X = pd.read_csv(args.data)
    prob = model.predict_proba(X)[:,1]
    is_fraud = (prob >= args.threshold).astype(int)
    out = [{"probability": float(p), "is_fraud": int(i)} for p, i in zip(prob, is_fraud)]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} predictions to {args.out}")
    if args.pretty:
        from pprint import pprint
        pprint(out[:5])

if __name__ == "__main__":
    main()
