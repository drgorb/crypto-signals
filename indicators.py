"""Technical indicator calculations (no ta-lib dependency)."""

import pandas as pd
import numpy as np
from config import (BOLLINGER_PERIOD, BOLLINGER_STD, RSI_PERIOD, SMA_TREND_PERIOD,
                    ATR_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
                    ADX_PERIOD, MARKET_STRUCTURE_LOOKBACK, SWING_PIVOT_WINDOW,
                    CORRELATION_LOOKBACK,
                    MOMENTUM_ROC_SHORT, MOMENTUM_ROC_LONG,
                    MOMENTUM_RSI_SLOPE_WINDOW, MOMENTUM_MACD_SLOPE_WINDOW,
                    REVERSION_TARGET_PCT, REVERSION_ADVERSE_PCT)


def bollinger_bands(df: pd.DataFrame, period: int = BOLLINGER_PERIOD, std_dev: float = BOLLINGER_STD) -> pd.DataFrame:
    close = df["close"]
    df = df.copy()
    df["bb_mid"] = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    df["bb_upper"] = df["bb_mid"] + std_dev * rolling_std
    df["bb_lower"] = df["bb_mid"] - std_dev * rolling_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def volume_profile(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["vol_sma"] = df["volume"].rolling(window=lookback).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"].replace(0, np.nan)
    return df


def sma(df: pd.DataFrame, period: int = SMA_TREND_PERIOD) -> pd.DataFrame:
    df = df.copy()
    df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
    return df


def vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cum_vol = df["volume"].cumsum()
    cum_vp = (df["close"] * df["volume"]).cumsum()
    df["vwap"] = cum_vp / cum_vol.replace(0, np.nan)
    return df


def macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW,
         signal: int = MACD_SIGNAL) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    df = df.copy()
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=period).mean()
    return df


def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Calculate ADX, +DI, and -DI.

    Adds columns: plus_di, minus_di, adx
    """
    df = df.copy()
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    # Smoothed averages (Wilder's smoothing)
    atr_smooth = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period).mean()

    df["plus_di"] = 100 * (plus_dm_smooth / atr_smooth.replace(0, np.nan))
    df["minus_di"] = 100 * (minus_dm_smooth / atr_smooth.replace(0, np.nan))

    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1/period, min_periods=period).mean()

    return df


def detect_market_structure(df: pd.DataFrame, lookback: int = MARKET_STRUCTURE_LOOKBACK,
                            pivot_window: int = SWING_PIVOT_WINDOW) -> str:
    """Analyze swing highs/lows to determine market structure.

    Returns: "uptrend", "downtrend", or "ranging"
    """
    if len(df) < lookback:
        return "ranging"

    recent = df.tail(lookback).reset_index(drop=True)
    highs = recent["high"].values
    lows = recent["low"].values
    n = len(recent)

    swing_highs = []
    swing_lows = []

    for i in range(pivot_window, n - pivot_window):
        # Swing high: higher than pivot_window candles on each side
        if all(highs[i] > highs[i - j] for j in range(1, pivot_window + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, pivot_window + 1)):
            swing_highs.append((i, highs[i]))

        # Swing low: lower than pivot_window candles on each side
        if all(lows[i] < lows[i - j] for j in range(1, pivot_window + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, pivot_window + 1)):
            swing_lows.append((i, lows[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "ranging"

    # Check last 2 swing highs and lows
    hh = swing_highs[-1][1] > swing_highs[-2][1]  # higher high
    hl = swing_lows[-1][1] > swing_lows[-2][1]     # higher low
    lh = swing_highs[-1][1] < swing_highs[-2][1]   # lower high
    ll = swing_lows[-1][1] < swing_lows[-2][1]     # lower low

    if hh and hl:
        return "uptrend"
    if lh and ll:
        return "downtrend"
    return "ranging"


def btc_eth_correlation(btc_df: pd.DataFrame, eth_df: pd.DataFrame,
                        lookback: int = CORRELATION_LOOKBACK) -> float | None:
    """Compute rolling correlation between BTC and ETH returns.

    Returns the latest correlation value, or None if insufficient data.
    """
    if len(btc_df) < lookback + 1 or len(eth_df) < lookback + 1:
        return None

    btc_ret = btc_df["close"].pct_change().dropna().tail(lookback)
    eth_ret = eth_df["close"].pct_change().dropna().tail(lookback)

    if len(btc_ret) < lookback or len(eth_ret) < lookback:
        return None

    # Align by taking last N values
    btc_vals = btc_ret.values[-lookback:]
    eth_vals = eth_ret.values[-lookback:]

    corr = np.corrcoef(btc_vals, eth_vals)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def regime_filter(df: pd.DataFrame, bw_lookback: int = 100) -> pd.DataFrame:
    df = df.copy()
    if "bb_width" not in df.columns:
        df = bollinger_bands(df)
    bw = df["bb_width"]
    df["bb_bw_pctile"] = bw.rolling(window=bw_lookback, min_periods=20).rank(pct=True)
    df["regime_ok"] = df["bb_bw_pctile"] <= 0.75
    return df


def time_of_day_boost(df: pd.DataFrame) -> pd.DataFrame:
    from config import HIGH_VOL_WINDOWS, LOW_VOL_WINDOWS
    df = df.copy()
    hours = df["timestamp"].dt.hour

    high = pd.Series(False, index=df.index)
    for start, end in HIGH_VOL_WINDOWS:
        if start < end:
            high = high | ((hours >= start) & (hours < end))
        else:
            high = high | ((hours >= start) | (hours < end))
    df["tod_high_vol"] = high

    low = pd.Series(False, index=df.index)
    for start, end in LOW_VOL_WINDOWS:
        if start < end:
            low = low | ((hours >= start) & (hours < end))
        else:
            low = low | ((hours >= start) | (hours < end))
    df["tod_low_vol"] = low
    return df


def momentum_outlook(df: pd.DataFrame) -> dict:
    """Compute forward-looking momentum direction score.

    Returns dict with:
        - macd_hist_slope: slope of MACD histogram over last N bars
        - roc_short / roc_long: Rate of Change over short/long periods
        - rsi_slope: RSI slope over last N bars
        - direction: 'accelerating_bullish', 'decelerating_bullish',
                     'accelerating_bearish', 'decelerating_bearish', or 'neutral'
        - score: -1.0 to +1.0
    """
    result = {"macd_hist_slope": 0.0, "roc_short": 0.0, "roc_long": 0.0,
              "rsi_slope": 0.0, "direction": "neutral", "score": 0.0}

    if len(df) < max(MOMENTUM_ROC_LONG, MOMENTUM_MACD_SLOPE_WINDOW, MOMENTUM_RSI_SLOPE_WINDOW) + 1:
        return result

    close = df["close"].values
    # ROC
    if len(close) > MOMENTUM_ROC_SHORT:
        result["roc_short"] = (close[-1] - close[-MOMENTUM_ROC_SHORT]) / close[-MOMENTUM_ROC_SHORT]
    if len(close) > MOMENTUM_ROC_LONG:
        result["roc_long"] = (close[-1] - close[-MOMENTUM_ROC_LONG]) / close[-MOMENTUM_ROC_LONG]

    # MACD histogram slope
    if "macd_hist" in df.columns:
        hist = df["macd_hist"].dropna().values
        if len(hist) >= MOMENTUM_MACD_SLOPE_WINDOW:
            recent_hist = hist[-MOMENTUM_MACD_SLOPE_WINDOW:]
            result["macd_hist_slope"] = float(np.polyfit(range(len(recent_hist)), recent_hist, 1)[0])

    # RSI slope
    if "rsi" in df.columns:
        rsi_vals = df["rsi"].dropna().values
        if len(rsi_vals) >= MOMENTUM_RSI_SLOPE_WINDOW:
            recent_rsi = rsi_vals[-MOMENTUM_RSI_SLOPE_WINDOW:]
            result["rsi_slope"] = float(np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0])

    # Combine into score: normalize each component to roughly -1..+1 and average
    components = []
    # ROC: already a fraction, scale by 100 and clip
    components.append(np.clip(result["roc_short"] * 100, -1, 1))
    components.append(np.clip(result["roc_long"] * 50, -1, 1))
    # MACD slope: normalize by abs value range (heuristic)
    components.append(np.clip(result["macd_hist_slope"] * 10, -1, 1))
    # RSI slope: positive slope = bullish momentum
    components.append(np.clip(result["rsi_slope"] / 5.0, -1, 1))

    score = float(np.mean(components))
    result["score"] = round(np.clip(score, -1, 1), 4)

    # Determine direction
    hist_val = df["macd_hist"].iloc[-1] if "macd_hist" in df.columns else 0
    if pd.isna(hist_val):
        hist_val = 0
    slope = result["macd_hist_slope"]

    if hist_val >= 0:
        result["direction"] = "accelerating_bullish" if slope > 0 else "decelerating_bullish"
    else:
        # Key insight: negative histogram but positive slope = momentum shifting bullish
        result["direction"] = "decelerating_bearish" if slope > 0 else "accelerating_bearish"

    return result


def estimate_reversion_time(df: pd.DataFrame, touch_type: str = "lower") -> dict:
    """Estimate mean-reversion timing from historical BB touches.

    Args:
        df: DataFrame with bb_lower, bb_upper, close columns
        touch_type: 'lower' for buy signals, 'upper' for sell signals

    Returns dict with:
        - median_candles: median candles to 1% reversion
        - success_rate: fraction of touches that achieved 1% reversion before 1% adverse
        - sample_size: number of historical touches analyzed
    """
    result = {"median_candles": None, "success_rate": None, "sample_size": 0}

    bb_col = "bb_lower" if touch_type == "lower" else "bb_upper"
    if bb_col not in df.columns or len(df) < 30:
        return result

    close = df["close"].values
    bb = df[bb_col].values
    n = len(df)

    candles_to_revert = []
    successes = 0
    total = 0

    for i in range(20, n - 5):  # skip edges
        if pd.isna(bb[i]):
            continue
        # Detect touch
        if touch_type == "lower" and close[i] <= bb[i]:
            target = close[i] * (1 + REVERSION_TARGET_PCT)
            adverse = close[i] * (1 - REVERSION_ADVERSE_PCT)
            for j in range(i + 1, min(i + 50, n)):
                if close[j] >= target:
                    candles_to_revert.append(j - i)
                    successes += 1
                    break
                if close[j] <= adverse:
                    break
            total += 1
        elif touch_type == "upper" and close[i] >= bb[i]:
            target = close[i] * (1 - REVERSION_TARGET_PCT)
            adverse = close[i] * (1 + REVERSION_ADVERSE_PCT)
            for j in range(i + 1, min(i + 50, n)):
                if close[j] <= target:
                    candles_to_revert.append(j - i)
                    successes += 1
                    break
                if close[j] >= adverse:
                    break
            total += 1

    result["sample_size"] = total
    if total > 0:
        result["success_rate"] = round(successes / total, 3)
    if candles_to_revert:
        result["median_candles"] = int(np.median(candles_to_revert))
    return result


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicators to a DataFrame."""
    df = bollinger_bands(df)
    df = rsi(df)
    df = volume_profile(df)
    df = sma(df)
    df = vwap(df)
    df = macd(df)
    df = atr(df)
    df = adx(df)
    df = regime_filter(df)
    df = time_of_day_boost(df)
    return df
