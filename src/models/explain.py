from __future__ import annotations
import argparse, os
import joblib, shap
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Generate SHAP plots for a trained model and a data sample.")
    p.add_argument("--model", type=str, default="models/creditcard_xgb_v1.joblib")
    p.add_argument("--data", type=str, default="data/raw/transactions.csv", help="CSV with features + optional Class")
    p.add_argument("--index", type=int, default=0, help="Row index (after filtering) for individual plot")
    p.add_argument("--outdir", type=str, default="reports", help="Where to save plots")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])
    idx = min(max(args.index, 0), len(df)-1)
    x_row = df.iloc[[idx]]

    booster = model.named_steps.get('xgb', None)
    if booster is None:
        raise ValueError("Model pipeline must contain a step named 'xgb' (XGBClassifier)")
    explainer = shap.TreeExplainer(booster)

    # Summary
    sample = df.sample(n=min(1000, len(df)), random_state=42)
    sv = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(sv, sample, show=False)
    plt.tight_layout()
    sum_path = os.path.join(args.outdir, "shap_summary.png")
    plt.savefig(sum_path, dpi=200)
    plt.close()

    # One-row plot
    sv_row = explainer.shap_values(x_row)
    try:
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, sv_row[0], feature_names=x_row.columns, max_display=15, show=False)
        ind_path = os.path.join(args.outdir, f"shap_row_{idx}.png")
        plt.tight_layout()
        plt.savefig(ind_path, dpi=200)
        plt.close()
    except Exception:
        ind_path = None

    print(f"Saved summary: {sum_path}")
    if ind_path:
        print(f"Saved row plot: {ind_path}")

if __name__ == "__main__":
    main()
