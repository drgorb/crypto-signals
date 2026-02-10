"""Backtest for CryptoSignals v4 — BUY+SELL signals, dynamic RSI, TOD filter, funding boost."""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from data import fetch_klines_paginated, fetch_daily_klines
from indicators import compute_all
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT, PROFIT_TARGET, STOP_LOSS, SYMBOL_NAMES,
    SMA_TREND_PERIOD, VOLUME_SPIKE_THRESHOLD,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    RSI_OVERSOLD_RANGING, RSI_OVERBOUGHT_RANGING,
    RSI_OVERSOLD_NORMAL, RSI_OVERBOUGHT_NORMAL,
    FUNDING_RATE_CONTRARIAN_BULLISH,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DAYS = 90
TIMEOUT_CANDLES = 96  # 24h
WARMUP = 55


def get_daily_sma_at_time(daily_df, ts, period=20):
    if daily_df is None or len(daily_df) < period:
        return None
    mask = daily_df["timestamp"] <= ts
    relevant = daily_df[mask]
    if len(relevant) < period:
        return None
    return relevant["close"].iloc[-period:].mean()


def get_dynamic_rsi_thresholds(row):
    bb_bw_pctile = row.get("bb_bw_pctile")
    if pd.isna(bb_bw_pctile) if bb_bw_pctile is not None else True:
        return RSI_OVERSOLD_NORMAL, RSI_OVERBOUGHT_NORMAL
    if bb_bw_pctile < 0.25:
        return RSI_OVERSOLD_RANGING, RSI_OVERBOUGHT_RANGING
    return RSI_OVERSOLD_NORMAL, RSI_OVERBOUGHT_NORMAL


# ── v3 signal check (BUY only, unchanged from original) ──

def check_signal_v3(row, daily_sma_val):
    price = row["close"]
    rsi_val = row.get("rsi")
    bb_lower = row.get("bb_lower")
    vol_ratio = row.get("vol_ratio", 1.0)
    regime_ok = row.get("regime_ok", True)
    if pd.isna(regime_ok): regime_ok = True
    if pd.isna(rsi_val) or pd.isna(bb_lower): return None
    if not regime_ok: return None
    if daily_sma_val is not None and price < daily_sma_val: return None
    if pd.isna(vol_ratio) or vol_ratio < VOLUME_SPIKE_THRESHOLD: return None
    if price <= bb_lower and rsi_val < RSI_OVERSOLD:
        return "BUY"
    return None


# ── v4 signal check (BUY + SELL, dynamic RSI, TOD filter) ──

def check_signal_v4(row, daily_sma_val):
    price = row["close"]
    rsi_val = row.get("rsi")
    bb_lower = row.get("bb_lower")
    bb_upper = row.get("bb_upper")
    vol_ratio = row.get("vol_ratio", 1.0)
    regime_ok = row.get("regime_ok", True)
    tod_high = row.get("tod_high_vol", False)
    tod_low = row.get("tod_low_vol", False)
    if pd.isna(regime_ok): regime_ok = True
    if pd.isna(tod_high): tod_high = False
    if pd.isna(tod_low): tod_low = False

    if pd.isna(rsi_val) or pd.isna(bb_lower) or pd.isna(bb_upper):
        return None, 0
    if not regime_ok:
        return None, 0
    if pd.isna(vol_ratio) or vol_ratio < VOLUME_SPIKE_THRESHOLD:
        return None, 0

    rsi_os, rsi_ob = get_dynamic_rsi_thresholds(row)

    # Low-vol hours: tighten
    if tod_low:
        rsi_os = max(rsi_os - 5, 15)
        rsi_ob = min(rsi_ob + 5, 90)

    strength_boost = 1 if tod_high else 0

    # BUY
    if daily_sma_val is None or price >= daily_sma_val:
        if price <= bb_lower and rsi_val < rsi_os:
            return "BUY", strength_boost

    # SELL: price above upper BB, RSI overbought, price above daily SMA (overextended)
    if daily_sma_val is None or price >= daily_sma_val:
        if price >= bb_upper and rsi_val > rsi_ob:
            return "SELL", strength_boost

    return None, 0


# ── Trade simulation ──

def simulate_trade(df, entry_idx, tp_pct, sl_pct, direction="BUY"):
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]

    if direction == "BUY":
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # SELL
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    for j in range(entry_idx + 1, min(entry_idx + 1 + TIMEOUT_CANDLES, len(df))):
        candle = df.iloc[j]
        if direction == "BUY":
            if candle["low"] <= sl_price:
                return _trade_result(entry_row, candle, sl_price, -sl_pct * 100, "SL", j - entry_idx, direction)
            if candle["high"] >= tp_price:
                return _trade_result(entry_row, candle, tp_price, tp_pct * 100, "TP", j - entry_idx, direction)
        else:  # SELL
            if candle["high"] >= sl_price:
                return _trade_result(entry_row, candle, sl_price, -sl_pct * 100, "SL", j - entry_idx, direction)
            if candle["low"] <= tp_price:
                return _trade_result(entry_row, candle, tp_price, tp_pct * 100, "TP", j - entry_idx, direction)

    last_idx = min(entry_idx + TIMEOUT_CANDLES, len(df) - 1)
    exit_row = df.iloc[last_idx]
    exit_price = exit_row["close"]
    if direction == "BUY":
        pnl = (exit_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - exit_price) / entry_price * 100
    return _trade_result(entry_row, exit_row, exit_price, round(pnl, 4), "TIMEOUT", last_idx - entry_idx, direction)


def _trade_result(entry_row, exit_row, exit_price, pnl_pct, outcome, candles, direction):
    return {
        "entry_time": entry_row["timestamp"],
        "entry_price": entry_row["close"],
        "exit_time": exit_row["timestamp"],
        "exit_price": exit_price,
        "pnl_pct": pnl_pct,
        "outcome": outcome,
        "candles_held": candles,
        "direction": direction,
    }


def simulate_trade_atr(df, entry_idx, direction="BUY"):
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]
    atr_val = entry_row.get("atr")
    if pd.isna(atr_val) or atr_val <= 0:
        return simulate_trade(df, entry_idx, PROFIT_TARGET, STOP_LOSS, direction)
    tp_pct = (atr_val * ATR_TP_MULTIPLIER) / entry_price
    sl_pct = (atr_val * ATR_SL_MULTIPLIER) / entry_price
    return simulate_trade(df, entry_idx, tp_pct, sl_pct, direction)


# ── Run strategies ──

def run_v3(df, daily_df):
    trades = []
    in_trade_until = -1
    for i in range(WARMUP, len(df)):
        if i <= in_trade_until: continue
        row = df.iloc[i]
        daily_sma = get_daily_sma_at_time(daily_df, row["timestamp"])
        sig = check_signal_v3(row, daily_sma)
        if sig:
            trade = simulate_trade_atr(df, i, "BUY")
            trades.append(trade)
            in_trade_until = i + trade["candles_held"]
    return trades


def run_v4(df, daily_df):
    trades = []
    in_trade_until = -1
    for i in range(WARMUP, len(df)):
        if i <= in_trade_until: continue
        row = df.iloc[i]
        daily_sma = get_daily_sma_at_time(daily_df, row["timestamp"])
        sig, boost = check_signal_v4(row, daily_sma)
        if sig:
            trade = simulate_trade_atr(df, i, sig)
            trade["tod_boost"] = boost
            trades.append(trade)
            in_trade_until = i + trade["candles_held"]
    return trades


# ── Reporting ──

def summarize(trades, label=""):
    if not trades:
        return {"total": 0, "buy": 0, "sell": 0, "tp": 0, "sl": 0, "to": 0,
                "wr": 0, "cum_pnl": 0, "avg_pnl": 0}
    tdf = pd.DataFrame(trades)
    total = len(tdf)
    buy = int((tdf["direction"] == "BUY").sum()) if "direction" in tdf.columns else total
    sell = int((tdf["direction"] == "SELL").sum()) if "direction" in tdf.columns else 0
    tp = int((tdf["outcome"] == "TP").sum())
    sl = int((tdf["outcome"] == "SL").sum())
    to = int((tdf["outcome"] == "TIMEOUT").sum())
    return {"total": total, "buy": buy, "sell": sell, "tp": tp, "sl": sl, "to": to,
            "wr": tp / total * 100, "cum_pnl": tdf["pnl_pct"].sum(), "avg_pnl": tdf["pnl_pct"].mean()}


def print_table(stats_dict, title):
    print(f"\n  {title}")
    print(f"  {'Strategy':<22} {'Trades':>6} {'BUY':>5} {'SELL':>5} {'TP':>4} {'SL':>4} {'TO':>4} {'WR':>7} {'Cum P&L':>10} {'Avg P&L':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*7} {'-'*10} {'-'*10}")
    for label, s in stats_dict.items():
        if s["total"] == 0:
            print(f"  {label:<22} {0:>6}     —     —    —    —    —      —          —          —")
        else:
            print(f"  {label:<22} {s['total']:>6} {s['buy']:>5} {s['sell']:>5} {s['tp']:>4} {s['sl']:>4} {s['to']:>4} {s['wr']:>6.1f}% {s['cum_pnl']:>+9.2f}% {s['avg_pnl']:>+9.4f}%")


def monthly_breakdown(trades, label):
    if not trades:
        print(f"  {label}: no trades")
        return
    tdf = pd.DataFrame(trades)
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["month"] = tdf["entry_time"].dt.to_period("M")
    print(f"\n  {label} — Monthly Breakdown:")
    print(f"  {'Month':<12} {'Trades':>6} {'BUY':>5} {'SELL':>5} {'WR':>7} {'P&L':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*10}")
    for month, grp in tdf.groupby("month"):
        tp = int((grp["outcome"] == "TP").sum())
        total = len(grp)
        wr = tp / total * 100
        cum = grp["pnl_pct"].sum()
        buy_ct = int((grp["direction"] == "BUY").sum()) if "direction" in grp.columns else total
        sell_ct = int((grp["direction"] == "SELL").sum()) if "direction" in grp.columns else 0
        print(f"  {str(month):<12} {total:>6} {buy_ct:>5} {sell_ct:>5} {wr:>6.1f}% {cum:>+9.2f}%")


def main():
    start_ms = int((time.time() - DAYS * 86400) * 1000)

    print(f"\n{'='*90}")
    print(f"  CryptoSignals v3 vs v4 BACKTEST ({DAYS}-day, 15m candles)")
    print(f"{'='*90}")
    print(f"  v3: BB + RSI<30 + VolSpike + regime + daily 20-SMA (BUY only)")
    print(f"  v4: + SELL signals, dynamic RSI, TOD filter, funding boost logic")
    print(f"  Config: ATR_TP={ATR_TP_MULTIPLIER}x, ATR_SL={ATR_SL_MULTIPLIER}x")

    grand_v3, grand_v4 = [], []
    per_sym = {}

    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        print(f"\n{'='*90}")
        print(f"  Fetching {sym}...")

        df = fetch_klines_paginated(sym, "15m", start_ms)
        df = compute_all(df)
        print(f"  15m data: {df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']}  ({len(df)} candles)")

        daily_df = fetch_daily_klines(sym, days=DAYS + 60)
        print(f"  Daily data: {len(daily_df)} candles")

        v3_trades = run_v3(df, daily_df)
        v4_trades = run_v4(df, daily_df)

        stats = {
            "v3 (BUY only)": summarize(v3_trades),
            "v4 (BUY+SELL)": summarize(v4_trades),
        }
        print_table(stats, display)
        monthly_breakdown(v3_trades, f"{display} v3")
        monthly_breakdown(v4_trades, f"{display} v4")

        per_sym[sym] = {"v3": v3_trades, "v4": v4_trades}
        grand_v3.extend(v3_trades)
        grand_v4.extend(v4_trades)

    # Grand total
    print(f"\n{'='*90}")
    print_table({
        "v3 (BUY only)": summarize(grand_v3),
        "v4 (BUY+SELL)": summarize(grand_v4),
    }, "GRAND TOTAL (BTC + ETH)")

    # BTC vs ETH
    print(f"\n{'='*90}")
    print(f"  BTC vs ETH — Per-asset breakdown:")
    for strat in ["v3", "v4"]:
        row_data = {}
        for sym in SYMBOLS:
            row_data[f"{SYMBOL_NAMES.get(sym, sym)} {strat}"] = summarize(per_sym[sym][strat])
        print_table(row_data, f"{strat} by asset")

    # v4 BUY vs SELL breakdown
    print(f"\n{'='*90}")
    print(f"  v4 BUY vs SELL breakdown:")
    buy_trades = [t for t in grand_v4 if t.get("direction") == "BUY"]
    sell_trades = [t for t in grand_v4 if t.get("direction") == "SELL"]
    print_table({
        "v4 BUY trades": summarize(buy_trades),
        "v4 SELL trades": summarize(sell_trades),
    }, "v4 Direction Breakdown")

    print()


if __name__ == "__main__":
    main()
