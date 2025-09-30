from __future__ import annotations
import argparse, os, pandas as pd
from pycaret.classification import setup, compare_models, tune_model, blend_models, finalize_model, save_model, pull
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--out', default='models/pycaret_fraud_v1')
    p.add_argument('--optimize', default='AUC'); p.add_argument('--folds', type=int, default=5)
    return p.parse_args()
def main():
    a = parse_args(); df = pd.read_csv(a.data)
    setup(data=df, target='Class', session_id=42, train_size=0.8, fold=a.folds,
          fold_strategy='stratifiedkfold', normalize=True, remove_multicollinearity=True,
          fix_imbalance=True, log_experiment=False, silent=True, html=False)
    top3 = compare_models(n_select=3, sort=a.optimize)
    tuned = [tune_model(m, optimize=a.optimize) for m in top3]
    ens = blend_models(estimator_list=tuned, optimize=a.optimize)
    best = finalize_model(ens); os.makedirs('reports', exist_ok=True)
    pull().to_csv('reports/pycaret_compare_metrics.csv', index=False)
    save_model(best, a.out); print(f'Saved model to {a.out}.pkl')
if __name__ == '__main__': main()
