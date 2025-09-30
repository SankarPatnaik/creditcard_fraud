from __future__ import annotations
import pandas as pd

def add_basic_time_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    if time_col in df.columns:
        df = df.copy()
        hour = (df[time_col] % (24*3600)) // 3600
        df["hour"] = hour
    return df
