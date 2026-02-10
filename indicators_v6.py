"""Indicators for v6 — adds EMA-20 to the existing compute_all pipeline."""

import pandas as pd
from indicators import compute_all as _compute_all_v5
from config_v6 import EMA_TREND_PERIOD


def ema(df: pd.DataFrame, period: int = EMA_TREND_PERIOD) -> pd.DataFrame:
    df = df.copy()
    df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def compute_all_v6(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all v5 indicators + EMA-20 for trend following."""
    df = _compute_all_v5(df)
    df = ema(df)
    return df
