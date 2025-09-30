from __future__ import annotations
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, confusion_matrix

def evaluate(y_true, y_score, threshold: float = 0.5):
    import numpy as np
    y_pred = (y_score >= threshold).astype(int)
    ap = average_precision_score(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": threshold,
        "average_precision": ap,
        "roc_auc": roc,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }
