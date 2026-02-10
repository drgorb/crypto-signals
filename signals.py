"""Signal generation logic — v5 with multi-timeframe trend, ADX, market structure, cross-asset correlation, composite trend score."""

import pandas as pd
import numpy as np
from config import (RSI_OVERSOLD, RSI_OVERBOUGHT, PROFIT_TARGET, STOP_LOSS,
                    SYMBOL_NAMES, SMA_TREND_PERIOD, VOLUME_SPIKE_THRESHOLD,
                    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
                    FUNDING_RATE_BEARISH, ORDERBOOK_BID_RATIO_BULLISH,
                    RSI_OVERSOLD_RANGING, RSI_OVERBOUGHT_RANGING,
                    RSI_OVERSOLD_NORMAL, RSI_OVERBOUGHT_NORMAL,
                    FUNDING_RATE_CONTRARIAN_BULLISH, FUNDING_RATE_SELL_BOOST,
                    ADX_SUPPRESS_MEAN_REVERSION, ADX_WEAK_THRESHOLD,
                    HOURLY_SMA_PERIOD, DAILY_SMA_PERIOD,
                    MTF_MIN_BULLISH_FOR_BUY, MTF_MIN_BEARISH_FOR_SELL,
                    CORRELATION_SUPPRESS_THRESHOLD,
                    TREND_SCORE_DAILY_SMA_WEIGHT, TREND_SCORE_HOURLY_SMA_WEIGHT,
                    TREND_SCORE_ADX_DI_WEIGHT, TREND_SCORE_MARKET_STRUCTURE_WEIGHT,
                    TREND_SCORE_FUNDING_WEIGHT, TREND_SCORE_BTC_CORR_WEIGHT,
                    TREND_SCORE_BUY_MIN, TREND_SCORE_SELL_MAX,
                    TREND_SCORE_MOMENTUM_WEIGHT, TREND_SCORE_PREDICTION_MARKET_WEIGHT,
                    TREND_SCORE_DERIVATIVES_WEIGHT, REVERSION_MIN_SUCCESS_RATE)
from indicators import detect_market_structure, btc_eth_correlation, momentum_outlook, estimate_reversion_time


def _get_dynamic_rsi_thresholds(row):
    """Return (oversold, overbought) thresholds based on BB bandwidth percentile (regime)."""
    bb_bw_pctile = row.get("bb_bw_pctile")
    if pd.isna(bb_bw_pctile) if bb_bw_pctile is not None else True:
        return RSI_OVERSOLD_NORMAL, RSI_OVERBOUGHT_NORMAL
    if bb_bw_pctile < 0.25:
        return RSI_OVERSOLD_RANGING, RSI_OVERBOUGHT_RANGING
    return RSI_OVERSOLD_NORMAL, RSI_OVERBOUGHT_NORMAL


def _compute_mtf_trend(price: float, daily_df: pd.DataFrame | None,
                       hourly_df: pd.DataFrame | None, adx_val: float | None):
    """Compute multi-timeframe trend signals.

    Returns dict with keys: daily_bullish, hourly_bullish, daily_neutral, hourly_neutral,
    bullish_count, bearish_count (out of 2 higher TFs; 15m is entry timing only).
    """
    result = {"daily_bullish": None, "hourly_bullish": None,
              "daily_neutral": False, "hourly_neutral": False,
              "bullish_count": 0, "bearish_count": 0}

    # Daily 20-SMA
    if daily_df is not None and len(daily_df) >= DAILY_SMA_PERIOD:
        daily_sma = daily_df["close"].rolling(DAILY_SMA_PERIOD).mean().iloc[-1]
        if not pd.isna(daily_sma):
            result["daily_bullish"] = price >= daily_sma

    # Hourly 50-SMA
    if hourly_df is not None and len(hourly_df) >= HOURLY_SMA_PERIOD:
        hourly_sma = hourly_df["close"].rolling(HOURLY_SMA_PERIOD).mean().iloc[-1]
        if not pd.isna(hourly_sma):
            result["hourly_bullish"] = price >= hourly_sma

    # ADX neutrality: if ADX < weak threshold, the timeframe is "neutral" (doesn't count against)
    adx_neutral = adx_val is not None and not pd.isna(adx_val) and adx_val < ADX_WEAK_THRESHOLD

    # Count bullish/bearish/neutral across daily + hourly
    for key in ["daily_bullish", "hourly_bullish"]:
        val = result[key]
        neutral_key = key.replace("_bullish", "_neutral")
        if val is None or adx_neutral:
            # Neutral — counts as favorable for both directions
            result[neutral_key] = True
            result["bullish_count"] += 1
            result["bearish_count"] += 1
        elif val:
            result["bullish_count"] += 1
        else:
            result["bearish_count"] += 1

    return result


def compute_trend_score(price: float, daily_df: pd.DataFrame | None,
                        hourly_df: pd.DataFrame | None,
                        adx_val: float | None, plus_di: float | None, minus_di: float | None,
                        market_structure: str,
                        funding_rate: float | None,
                        is_eth: bool = False, btc_bearish: bool = False,
                        btc_eth_corr: float | None = None,
                        momentum_score: float = 0.0,
                        prediction_market_score: float = 0.0,
                        derivatives_score: float = 0.0) -> int:
    """Compute composite trend score from -100 to +100."""
    score = 0

    # 1. Daily SMA direction (±20)
    if daily_df is not None and len(daily_df) >= DAILY_SMA_PERIOD:
        daily_sma = daily_df["close"].rolling(DAILY_SMA_PERIOD).mean().iloc[-1]
        if not pd.isna(daily_sma):
            score += TREND_SCORE_DAILY_SMA_WEIGHT if price >= daily_sma else -TREND_SCORE_DAILY_SMA_WEIGHT

    # 2. Hourly SMA direction (±15)
    if hourly_df is not None and len(hourly_df) >= HOURLY_SMA_PERIOD:
        hourly_sma = hourly_df["close"].rolling(HOURLY_SMA_PERIOD).mean().iloc[-1]
        if not pd.isna(hourly_sma):
            score += TREND_SCORE_HOURLY_SMA_WEIGHT if price >= hourly_sma else -TREND_SCORE_HOURLY_SMA_WEIGHT

    # 3. ADX + DI direction (±20, scaled by ADX strength)
    if adx_val is not None and not pd.isna(adx_val) and \
       plus_di is not None and not pd.isna(plus_di) and \
       minus_di is not None and not pd.isna(minus_di):
        adx_scale = min(adx_val / 40.0, 1.0)  # scale 0-1 based on ADX strength
        direction = 1 if plus_di > minus_di else -1
        score += int(direction * TREND_SCORE_ADX_DI_WEIGHT * adx_scale)

    # 4. Market structure (±25)
    if market_structure == "uptrend":
        score += TREND_SCORE_MARKET_STRUCTURE_WEIGHT
    elif market_structure == "downtrend":
        score -= TREND_SCORE_MARKET_STRUCTURE_WEIGHT

    # 5. Funding rate (±10)
    if funding_rate is not None:
        if funding_rate < -0.0003:
            score += TREND_SCORE_FUNDING_WEIGHT  # negative funding = bullish contrarian
        elif funding_rate > 0.0003:
            score -= TREND_SCORE_FUNDING_WEIGHT  # positive funding = bearish contrarian

    # 6. BTC correlation penalty for ETH (±10)
    if is_eth and btc_eth_corr is not None and btc_eth_corr > CORRELATION_SUPPRESS_THRESHOLD:
        if btc_bearish:
            score -= TREND_SCORE_BTC_CORR_WEIGHT
        else:
            score += TREND_SCORE_BTC_CORR_WEIGHT

    # 7. Momentum outlook (±10)
    if momentum_score != 0.0:
        score += int(momentum_score * TREND_SCORE_MOMENTUM_WEIGHT)

    # 8. Prediction market forward-looking score (±15)
    if prediction_market_score != 0.0:
        score += int(prediction_market_score * TREND_SCORE_PREDICTION_MARKET_WEIGHT)

    # 9. Derivatives sentiment score (±15)
    if derivatives_score != 0.0:
        score += int(derivatives_score * TREND_SCORE_DERIVATIVES_WEIGHT)

    return max(-100, min(100, score))


def generate_signals(symbol: str, df: pd.DataFrame, sentiment: dict,
                     funding_rate: float | None = None,
                     ob_ratio: float | None = None,
                     daily_df: pd.DataFrame | None = None,
                     hourly_df: pd.DataFrame | None = None,
                     btc_df_15m: pd.DataFrame | None = None,
                     btc_daily_df: pd.DataFrame | None = None,
                     btc_hourly_df: pd.DataFrame | None = None,
                     derivatives_data: dict | None = None) -> list[dict]:
    """Generate buy and sell signals from indicator data."""
    signals = []

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    price = latest["close"]
    rsi_val = latest.get("rsi")
    bb_lower = latest.get("bb_lower")
    bb_upper = latest.get("bb_upper")
    bb_mid = latest.get("bb_mid")
    bb_pct = latest.get("bb_pct")
    vol_ratio = latest.get("vol_ratio", 1.0)
    sma_col = f"sma_{SMA_TREND_PERIOD}"
    sma_val = latest.get(sma_col)
    atr_val = latest.get("atr")
    macd_hist = latest.get("macd_hist")
    prev_macd_hist = prev.get("macd_hist") if len(df) > 1 else None
    vwap_val = latest.get("vwap")
    adx_val = latest.get("adx")
    plus_di = latest.get("plus_di")
    minus_di = latest.get("minus_di")

    # Time-of-day flags
    tod_high = latest.get("tod_high_vol", False)
    tod_low = latest.get("tod_low_vol", False)
    if pd.isna(tod_high):
        tod_high = False
    if pd.isna(tod_low):
        tod_low = False

    if pd.isna(rsi_val) or pd.isna(bb_lower):
        return signals

    display = SYMBOL_NAMES.get(symbol, symbol)
    is_eth = symbol == "ETHUSDT"

    # --- ADX: suppress mean-reversion if trend too strong ---
    adx_suppressed = (adx_val is not None and not pd.isna(adx_val)
                      and adx_val > ADX_SUPPRESS_MEAN_REVERSION)

    # --- Market structure ---
    market_structure = detect_market_structure(df)

    # --- Multi-timeframe trend ---
    mtf = _compute_mtf_trend(price, daily_df, hourly_df, adx_val)

    # --- BTC trend info for cross-asset correlation ---
    btc_bearish = False
    btc_eth_corr = None
    if is_eth:
        if btc_daily_df is not None and len(btc_daily_df) >= DAILY_SMA_PERIOD:
            btc_daily_sma = btc_daily_df["close"].rolling(DAILY_SMA_PERIOD).mean().iloc[-1]
            if not pd.isna(btc_daily_sma):
                btc_bearish = btc_daily_df["close"].iloc[-1] < btc_daily_sma
        if btc_df_15m is not None:
            btc_eth_corr = btc_eth_correlation(btc_df_15m, df)

    # --- Momentum outlook ---
    mom = momentum_outlook(df)

    # --- Mean-reversion timing ---
    reversion_buy = estimate_reversion_time(df, "lower")
    reversion_sell = estimate_reversion_time(df, "upper")

    # --- Derivatives score ---
    deriv_score = 0.0
    if derivatives_data is not None:
        deriv_score = derivatives_data.get("score", 0.0)

    # --- Prediction market score ---
    pred_score = sentiment.get("prediction_market_score", 0.0)

    # --- Composite trend score ---
    trend_score = compute_trend_score(
        price, daily_df, hourly_df, adx_val, plus_di, minus_di,
        market_structure, funding_rate,
        is_eth=is_eth, btc_bearish=btc_bearish, btc_eth_corr=btc_eth_corr,
        momentum_score=mom["score"],
        prediction_market_score=pred_score,
        derivatives_score=deriv_score)

    # --- Dynamic RSI thresholds ---
    rsi_oversold, rsi_overbought = _get_dynamic_rsi_thresholds(latest)

    # Funding rate: extreme negative lowers RSI threshold for BUY
    if funding_rate is not None and funding_rate < FUNDING_RATE_CONTRARIAN_BULLISH:
        rsi_oversold = max(rsi_oversold, 35)

    # Low-vol hours: tighten thresholds
    if tod_low:
        rsi_oversold = max(rsi_oversold - 5, 15)
        rsi_overbought = min(rsi_overbought + 5, 90)

    # --- ATR-based dynamic TP/SL ---
    if not pd.isna(atr_val) and atr_val > 0:
        tp_pct = (atr_val * ATR_TP_MULTIPLIER) / price
        sl_pct = (atr_val * ATR_SL_MULTIPLIER) / price
        tp_source = "ATR"
    else:
        tp_pct = PROFIT_TARGET
        sl_pct = STOP_LOSS
        tp_source = "fixed"

    # --- Regime filter ---
    regime_ok = latest.get("regime_ok", True)
    if pd.isna(regime_ok):
        regime_ok = True

    # --- Common conditions ---
    volume_confirmed = (not pd.isna(vol_ratio) and vol_ratio >= VOLUME_SPIKE_THRESHOLD)

    # MACD info
    macd_positive = not pd.isna(macd_hist) and macd_hist > 0
    macd_improving = False
    if not pd.isna(macd_hist) and prev_macd_hist is not None and not pd.isna(prev_macd_hist):
        macd_improving = macd_hist > prev_macd_hist

    # --- Sentiment suppression ---
    suppress_buy = sentiment.get("is_bearish", False)
    suppress_sell = sentiment.get("is_bullish", False)

    # --- Cross-asset correlation suppression ---
    corr_suppress_buy = (is_eth and btc_bearish and btc_eth_corr is not None
                         and btc_eth_corr > CORRELATION_SUPPRESS_THRESHOLD)

    # === BUY SIGNAL ===
    # Suppress if historical reversion success rate too low
    reversion_suppressed_buy = (reversion_buy["success_rate"] is not None
                                and reversion_buy["success_rate"] < REVERSION_MIN_SUCCESS_RATE)

    core_buy = (price <= bb_lower and rsi_val < rsi_oversold
                and volume_confirmed and regime_ok
                and not suppress_buy
                and not adx_suppressed
                and not corr_suppress_buy
                and not reversion_suppressed_buy
                and market_structure != "downtrend"
                and mtf["bullish_count"] >= MTF_MIN_BULLISH_FOR_BUY
                and trend_score > TREND_SCORE_BUY_MIN)

    if core_buy:
        funding_ok = (funding_rate is None or funding_rate < FUNDING_RATE_BEARISH)
        ob_ok = (ob_ratio is None or ob_ratio >= ORDERBOOK_BID_RATIO_BULLISH)

        if funding_ok and ob_ok:
            # Trend-aligned BUY (bullish trend) gets 2x SL for more room
            trade_aligned = trend_score > 0
            effective_sl = sl_pct * 1.75 if trade_aligned else sl_pct
            tp = round(price * (1 + tp_pct), 2)
            sl = round(price * (1 - effective_sl), 2)

            strength_score = _signal_strength_score(
                rsi_val, bb_pct, vol_ratio, "buy",
                macd_positive=macd_positive, macd_improving=macd_improving,
                ob_ratio=ob_ratio, is_btc=(symbol == "BTCUSDT"))

            if funding_rate is not None and funding_rate < 0:
                strength_score += 1
            if tod_high:
                strength_score += 1

            signals.append(_build_signal(
                "BUY", display, price, rsi_val, bb_pct, bb_lower, bb_upper,
                tp, sl, tp_source, atr_val, vol_ratio, sentiment,
                funding_rate, ob_ratio, macd_hist, macd_positive, macd_improving,
                vwap_val, sma_val, strength_score, trend_score, market_structure,
                adx_val, plus_di, minus_di, btc_eth_corr,
                mom, reversion_buy, deriv_score, pred_score))

    # === SELL SIGNAL ===
    reversion_suppressed_sell = (reversion_sell["success_rate"] is not None
                                 and reversion_sell["success_rate"] < REVERSION_MIN_SUCCESS_RATE)

    core_sell = (price >= bb_upper and rsi_val > rsi_overbought
                 and volume_confirmed and regime_ok
                 and not suppress_sell
                 and not adx_suppressed
                 and not reversion_suppressed_sell
                 and market_structure != "uptrend"
                 and mtf["bearish_count"] >= MTF_MIN_BEARISH_FOR_SELL
                 and trend_score < TREND_SCORE_SELL_MAX)

    if core_sell:
        # Trend-aligned SELL (bearish trend) gets 2x SL for more room
        trade_aligned = trend_score < 0
        effective_sl = sl_pct * 1.75 if trade_aligned else sl_pct
        tp = round(price * (1 - tp_pct), 2)
        sl = round(price * (1 + effective_sl), 2)

        strength_score = _signal_strength_score(
            rsi_val, bb_pct, vol_ratio, "sell",
            macd_positive=macd_positive, macd_improving=macd_improving,
            ob_ratio=ob_ratio, is_btc=(symbol == "BTCUSDT"))

        if funding_rate is not None and funding_rate > FUNDING_RATE_SELL_BOOST:
            strength_score += 1
        if tod_high:
            strength_score += 1

        signals.append(_build_signal(
            "SELL", display, price, rsi_val, bb_pct, bb_lower, bb_upper,
            tp, sl, tp_source, atr_val, vol_ratio, sentiment,
            funding_rate, ob_ratio, macd_hist, macd_positive, macd_improving,
            vwap_val, sma_val, strength_score, trend_score, market_structure,
            adx_val, plus_di, minus_di, btc_eth_corr,
            mom, reversion_sell, deriv_score, pred_score))

    return signals


def _build_signal(sig_type, display, price, rsi_val, bb_pct, bb_lower, bb_upper,
                  tp, sl, tp_source, atr_val, vol_ratio, sentiment,
                  funding_rate, ob_ratio, macd_hist, macd_positive, macd_improving,
                  vwap_val, sma_val, strength_score, trend_score, market_structure,
                  adx_val, plus_di, minus_di, btc_eth_corr,
                  momentum_data=None, reversion_data=None,
                  derivatives_score=0.0, prediction_market_score=0.0):
    return {
        "type": sig_type,
        "symbol": display,
        "price": price,
        "rsi": round(rsi_val, 1),
        "bb_pct": round(bb_pct, 3) if bb_pct is not None and not pd.isna(bb_pct) else None,
        "bb_lower": round(bb_lower, 2),
        "bb_upper": round(bb_upper, 2),
        "take_profit": tp,
        "stop_loss": sl,
        "tp_sl_source": tp_source,
        "atr": round(atr_val, 2) if atr_val is not None and not pd.isna(atr_val) else None,
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None and not pd.isna(vol_ratio) else None,
        "sentiment_score": sentiment.get("score", 0),
        "funding_rate": funding_rate,
        "ob_ratio": round(ob_ratio, 2) if ob_ratio is not None else None,
        "macd_hist": round(macd_hist, 4) if macd_hist is not None and not pd.isna(macd_hist) else None,
        "macd_positive": macd_positive,
        "macd_improving": macd_improving,
        "vwap": round(vwap_val, 2) if vwap_val is not None and not pd.isna(vwap_val) else None,
        "strength": _score_to_label(strength_score),
        "strength_score": strength_score,
        "above_sma": True,
        "volume_confirmed": True,
        # New v5 fields
        "trend_score": trend_score,
        "market_structure": market_structure,
        "adx": round(adx_val, 1) if adx_val is not None and not pd.isna(adx_val) else None,
        "plus_di": round(plus_di, 1) if plus_di is not None and not pd.isna(plus_di) else None,
        "minus_di": round(minus_di, 1) if minus_di is not None and not pd.isna(minus_di) else None,
        "btc_eth_corr": round(btc_eth_corr, 3) if btc_eth_corr is not None else None,
        # v6 prediction fields
        "momentum_direction": momentum_data["direction"] if momentum_data else None,
        "momentum_score": momentum_data["score"] if momentum_data else None,
        "est_reversion_candles": reversion_data["median_candles"] if reversion_data else None,
        "est_success_rate": reversion_data["success_rate"] if reversion_data else None,
        "reversion_sample_size": reversion_data["sample_size"] if reversion_data else None,
        "derivatives_score": derivatives_score,
        "prediction_market_score": prediction_market_score,
    }


def _signal_strength_score(rsi, bb_pct, vol_ratio, direction,
                           macd_positive=False, macd_improving=False,
                           ob_ratio=None, is_btc=False):
    score = 0
    if direction == "buy":
        if rsi < 25: score += 2
        elif rsi < 30: score += 1
        if bb_pct is not None and not pd.isna(bb_pct) and bb_pct < -0.05: score += 1
        if macd_positive: score += 1
        elif macd_improving: score += 0.5
        if ob_ratio is not None and ob_ratio > 1.5: score += 1
    else:
        if rsi > 75: score += 2
        elif rsi > 70: score += 1
        if bb_pct is not None and not pd.isna(bb_pct) and bb_pct > 1.05: score += 1
        if not macd_positive: score += 1
        if ob_ratio is not None and ob_ratio < 0.7: score += 1

    if vol_ratio is not None and not pd.isna(vol_ratio) and vol_ratio > 1.5:
        score += 1
    if is_btc:
        score += 1
    return score


def _score_to_label(score):
    if score >= 4: return "STRONG"
    if score >= 2: return "MODERATE"
    return "WEAK"
