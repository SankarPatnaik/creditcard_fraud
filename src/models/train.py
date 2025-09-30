from __future__ import annotations
import argparse, os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.data.preprocess import load_csv, basic_splits
from src.features.feature_engineering import add_basic_time_features
from src.utils.metrics import evaluate

def parse_args():
    p = argparse.ArgumentParser(description="Train fraud detection model")
    p.add_argument("--data", type=str, default="data/raw/transactions.csv", help="Path to CSV with target col 'Class'")
    p.add_argument("--out", type=str, default="models/creditcard_xgb_v1.joblib", help="Output path for model artifact")
    p.add_argument("--threshold", type=float, default=0.4, help="Decision threshold for binary flag")
    p.add_argument("--shap", action="store_true", help="Generate SHAP summary plot to reports/shap_summary.png")
    return p.parse_args()

def main():
    args = parse_args()
    df = load_csv(args.data)
    df = add_basic_time_features(df)

    X_train, X_test, y_train, y_test = basic_splits(df, target_col="Class", test_size=0.2, random_state=42)

    # Resample only training set
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=4
    )

    pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("xgb", clf)])
    pipe.fit(X_res, y_res)

    # Evaluate
    y_score = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_score, threshold=args.threshold)
    print("== Evaluation ==")
    for k, v in metrics.items():
        if k == "report":
            print("\nClassification Report:\n", v)
        else:
            print(f"{k}: {v}")

    # Optional SHAP explainability
    if args.shap:
        import shap, matplotlib.pyplot as plt
        os.makedirs("reports", exist_ok=True)
        sample = min(1000, len(X_test))
        X_plot = X_test.sample(n=sample, random_state=42)
        booster = pipe.named_steps['xgb']
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_plot)
        plt.figure()
        shap.summary_plot(shap_values, X_plot, show=False)
        plt.tight_layout()
        plt.savefig("reports/shap_summary.png", dpi=200)
        plt.close()
        print("Saved SHAP summary to reports/shap_summary.png")

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"Model saved to {args.out}")

if __name__ == "__main__":
    main()
