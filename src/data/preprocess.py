from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def basic_splits(df: pd.DataFrame, target_col: str = "Class", test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def scale_amount(X_train, X_test, amount_col: str = "Amount"):
    scaler = StandardScaler()
    if amount_col in X_train.columns:
        X_train[amount_col] = scaler.fit_transform(X_train[[amount_col]])
        X_test[amount_col] = scaler.transform(X_test[[amount_col]])
    return X_train, X_test, scaler
