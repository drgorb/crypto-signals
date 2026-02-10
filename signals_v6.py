"""Signal generation v6 — standard/strong signals + trend-following mode."""

import pandas as pd
import numpy as np
from config_v6 import (
    V6_BB_PROXIMITY_PCT, V6_RSI_OVERSOLD, V6_RSI_OVERBOUGHT, V6_VOLUME_THRESHOLD,
    V6_STRONG_RSI_OVERSOLD, V6_STRONG_RSI_OVERBOUGHT, V6_STRONG_VOLUME_THRESHOLD,
    V6_ADX_TREND_THRESHOLD, V6_EMA_PULLBACK_PCT, EMA_TREND_PERIOD,
    SYMBOL_NAMES, ADX_WEAK_THRESHOLD,
)


def generate_signals_v6(symbol: str, df: pd.DataFrame, timeframe: str = "15m",
                        higher_tf_direction: str | None = None) -> list[dict]:
    """Generate v6 signals: STANDARD, STRONG, and TREND_FOLLOW.
    
    Args:
        symbol: e.g. "BTCUSDT"
        df: DataFrame with indicators computed (bb_lower, bb_upper, rsi, vol_ratio, adx, etc.)
        timeframe: "5m" or "15m"
        higher_tf_direction: direction from higher TF ("BUY", "SELL", None) — used to filter 5m conflicts
    
    Returns list of signal dicts.
    """
    if len(df) < 30:
        return []
    
    latest = df.iloc[-1]
    price = latest["close"]
    rsi_val = latest.get("rsi")
    bb_lower = latest.get("bb_lower")
    bb_upper = latest.get("bb_upper")
    bb_mid = latest.get("bb_mid")
    bb_pct = latest.get("bb_pct")
    vol_ratio = latest.get("vol_ratio", 1.0)
    adx_val = latest.get("adx")
    plus_di = latest.get("plus_di")
    minus_di = latest.get("minus_di")
    atr_val = latest.get("atr")
    macd_hist = latest.get("macd_hist")
    
    if pd.isna(rsi_val) or pd.isna(bb_lower) or pd.isna(bb_upper):
        return []
    
    display = SYMBOL_NAMES.get(symbol, symbol)
    signals = []
    
    bb_width = bb_upper - bb_lower
    if bb_width <= 0:
        return []
    
    # Distance from bands as fraction of band width
    dist_from_lower = (price - bb_lower) / bb_width
    dist_from_upper = (bb_upper - price) / bb_width
    
    volume_ok_standard = not pd.isna(vol_ratio) and vol_ratio >= V6_VOLUME_THRESHOLD
    volume_ok_strong = not pd.isna(vol_ratio) and vol_ratio >= V6_STRONG_VOLUME_THRESHOLD
    
    # ADX info (soft modifier)
    adx_strong = adx_val is not None and not pd.isna(adx_val) and adx_val > V6_ADX_TREND_THRESHOLD
    trend_bullish = (plus_di is not None and minus_di is not None and 
                     not pd.isna(plus_di) and not pd.isna(minus_di) and plus_di > minus_di)
    
    # EMA for trend-following
    ema_col = f"ema_{EMA_TREND_PERIOD}"
    ema_val = latest.get(ema_col)
    
    # ============ TREND-FOLLOWING MODE ============
    if adx_strong and ema_val is not None and not pd.isna(ema_val):
        ema_dist = abs(price - ema_val) / price
        near_ema = ema_dist <= V6_EMA_PULLBACK_PCT
        
        if near_ema and trend_bullish:
            # Pullback to EMA in uptrend → BUY
            if higher_tf_direction != "SELL":
                sig = _build_v6_signal("BUY", "TREND_FOLLOW", display, price, rsi_val, bb_pct,
                                       bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                                       atr_val, macd_hist, timeframe, ema_val)
                signals.append(sig)
        
        elif near_ema and not trend_bullish:
            # Rally to EMA in downtrend → SELL
            if higher_tf_direction != "BUY":
                sig = _build_v6_signal("SELL", "TREND_FOLLOW", display, price, rsi_val, bb_pct,
                                       bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                                       atr_val, macd_hist, timeframe, ema_val)
                signals.append(sig)
    
    # ============ MEAN-REVERSION SIGNALS ============
    
    # --- BUY signals ---
    bb_buy_standard = dist_from_lower <= V6_BB_PROXIMITY_PCT  # within 10% of lower band
    bb_buy_strong = price <= bb_lower  # actually touching/below lower band
    rsi_buy_standard = rsi_val < V6_RSI_OVERSOLD
    rsi_buy_strong = rsi_val < V6_STRONG_RSI_OVERSOLD
    
    if bb_buy_strong and rsi_buy_strong and volume_ok_strong:
        # STRONG BUY — full v5 criteria met
        if higher_tf_direction != "SELL":
            sig = _build_v6_signal("BUY", "STRONG", display, price, rsi_val, bb_pct,
                                   bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                                   atr_val, macd_hist, timeframe, ema_val)
            signals.append(sig)
    elif bb_buy_standard and rsi_buy_standard and volume_ok_standard:
        # STANDARD BUY — looser criteria
        if higher_tf_direction != "SELL":
            sig = _build_v6_signal("BUY", "STANDARD", display, price, rsi_val, bb_pct,
                                   bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                                   atr_val, macd_hist, timeframe, ema_val)
            signals.append(sig)
    
    # --- SELL signals ---
    bb_sell_standard = dist_from_upper <= V6_BB_PROXIMITY_PCT
    bb_sell_strong = price >= bb_upper
    rsi_sell_standard = rsi_val > V6_RSI_OVERBOUGHT
    rsi_sell_strong = rsi_val > V6_STRONG_RSI_OVERBOUGHT
    
    if bb_sell_strong and rsi_sell_strong and volume_ok_strong:
        if higher_tf_direction != "BUY":
            sig = _build_v6_signal("SELL", "STRONG", display, price, rsi_val, bb_pct,
                                   bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                                   atr_val, macd_hist, timeframe, ema_val)
            signals.append(sig)
    elif bb_sell_standard and rsi_sell_standard and volume_ok_standard:
        if higher_tf_direction != "BUY":
            sig = _build_v6_signal("SELL", "STANDARD", display, price, rsi_val, bb_pct,
                                   bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                                   atr_val, macd_hist, timeframe, ema_val)
            signals.append(sig)
    
    return signals


def _build_v6_signal(direction, conviction, display, price, rsi_val, bb_pct,
                     bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
                     atr_val, macd_hist, timeframe, ema_val):
    return {
        "type": direction,
        "conviction": conviction,  # "STANDARD", "STRONG", or "TREND_FOLLOW"
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
        "position_size": 2.0 if conviction == "STRONG" else 1.0,
    }
