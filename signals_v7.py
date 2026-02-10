"""Signal generation v7 — strict quality filters, only STRONG + TREND_FOLLOW."""

import pandas as pd
import numpy as np
from config_v7 import (
    V7_RSI_OVERSOLD, V7_RSI_OVERBOUGHT, V7_VOLUME_THRESHOLD,
    V7_ADX_TREND_THRESHOLD, V7_EMA_PULLBACK_PCT, V7_MIN_ATR_PCT,
    V7_HTF_SMA_PERIOD, V7_STRONG_LEVERAGE, V7_TREND_LEVERAGE,
    EMA_TREND_PERIOD, SYMBOL_NAMES,
)


def generate_signals_v7(symbol: str, df: pd.DataFrame, timeframe: str = "15m",
                        htf_df: pd.DataFrame = None) -> list[dict]:
    """Generate v7 signals: only STRONG mean-reversion and TREND_FOLLOW.
    
    Args:
        symbol: e.g. "BTCUSDT"
        df: DataFrame with indicators (bb_lower, bb_upper, rsi, vol_ratio, adx, atr, etc.)
        timeframe: "15m" only in v7
        htf_df: 1h DataFrame with SMA for multi-timeframe confirmation
    
    Returns list of signal dicts (0 or 1 typically).
    """
    if len(df) < 30:
        return []
    
    latest = df.iloc[-1]
    price = latest["close"]
    rsi_val = latest.get("rsi")
    bb_lower = latest.get("bb_lower")
    bb_upper = latest.get("bb_upper")
    bb_pct = latest.get("bb_pct")
    vol_ratio = latest.get("vol_ratio", 1.0)
    adx_val = latest.get("adx")
    plus_di = latest.get("plus_di")
    minus_di = latest.get("minus_di")
    atr_val = latest.get("atr")
    macd_hist = latest.get("macd_hist")
    
    if pd.isna(rsi_val) or pd.isna(bb_lower) or pd.isna(bb_upper):
        return []
    
    # ===== MINIMUM ATR FILTER =====
    # Don't trade in low volatility — moves too small to overcome fees
    if atr_val is None or pd.isna(atr_val):
        return []
    atr_pct = atr_val / price
    if atr_pct < V7_MIN_ATR_PCT:
        return []
    
    # ===== HIGHER TIMEFRAME CONFIRMATION =====
    # Get 1h 50-SMA direction
    htf_bullish = None  # None = no data, don't filter
    if htf_df is not None and len(htf_df) > V7_HTF_SMA_PERIOD:
        htf_sma_col = f"sma_{V7_HTF_SMA_PERIOD}"
        if htf_sma_col in htf_df.columns:
            htf_latest = htf_df.iloc[-1]
            htf_sma = htf_latest.get(htf_sma_col)
            htf_price = htf_latest["close"]
            if not pd.isna(htf_sma):
                htf_bullish = htf_price > htf_sma
    
    display = SYMBOL_NAMES.get(symbol, symbol)
    signals = []
    
    bb_width = bb_upper - bb_lower
    if bb_width <= 0:
        return []
    
    # ADX / DI for trend
    adx_strong = adx_val is not None and not pd.isna(adx_val) and adx_val > V7_ADX_TREND_THRESHOLD
    trend_bullish = (plus_di is not None and minus_di is not None and
                     not pd.isna(plus_di) and not pd.isna(minus_di) and plus_di > minus_di)
    
    ema_col = f"ema_{EMA_TREND_PERIOD}"
    ema_val = latest.get(ema_col)
    
    # ============ TREND-FOLLOWING MODE ============
    if adx_strong and ema_val is not None and not pd.isna(ema_val):
        ema_dist = abs(price - ema_val) / price
        near_ema = ema_dist <= V7_EMA_PULLBACK_PCT
        
        if near_ema and trend_bullish and htf_bullish is not False:
            # Pullback to EMA in uptrend, 1h confirms bullish
            sig = _build_signal("BUY", "TREND_FOLLOW", display, price, rsi_val, bb_pct,
                                bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                                atr_val, macd_hist, timeframe, ema_val)
            signals.append(sig)
        
        elif near_ema and not trend_bullish and htf_bullish is not True:
            # Rally to EMA in downtrend, 1h confirms bearish
            sig = _build_signal("SELL", "TREND_FOLLOW", display, price, rsi_val, bb_pct,
                                bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                                atr_val, macd_hist, timeframe, ema_val)
            signals.append(sig)
    
    # STRONG mean-reversion signals removed — trend-follow only (Config 6)
    
    return signals


def _build_signal(direction, conviction, display, price, rsi_val, bb_pct,
                  bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                  atr_val, macd_hist, timeframe, ema_val):
    leverage = V7_STRONG_LEVERAGE if conviction == "STRONG" else V7_TREND_LEVERAGE
    return {
        "type": direction,
        "conviction": conviction,
        "symbol": display,
        "price": price,
        "rsi": round(rsi_val, 1),
        "bb_pct": round(bb_pct, 3) if bb_pct is not None and not pd.isna(bb_pct) else None,
        "bb_lower": round(bb_lower, 2),
        "bb_upper": round(bb_upper, 2),
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None and not pd.isna(vol_ratio) else None,
        "adx": round(adx_val, 1) if adx_val is not None and not pd.isna(adx_val) else None,
        "plus_di": round(plus_di, 1) if plus_di is not None and not pd.isna(plus_di) else None,
        "minus_di": round(minus_di, 1) if minus_di is not None and not pd.isna(minus_di) else None,
        "atr": round(atr_val, 2) if atr_val is not None and not pd.isna(atr_val) else None,
        "macd_hist": round(macd_hist, 4) if macd_hist is not None and not pd.isna(macd_hist) else None,
        "timeframe": timeframe,
        "ema_20": round(ema_val, 2) if ema_val is not None and not pd.isna(ema_val) else None,
        "leverage": leverage,
    }
