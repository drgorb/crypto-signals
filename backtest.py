"""Backtest for CryptoSignals v3 — walks 90 days of 15m candles and simulates trades."""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from data import fetch_klines_paginated, fetch_daily_klines
from indicators import compute_all, regime_filter
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT, PROFIT_TARGET, STOP_LOSS, SYMBOL_NAMES,
    SMA_TREND_PERIOD, VOLUME_SPIKE_THRESHOLD,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER
)

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DAYS = 90
TIMEOUT_CANDLES = 96  # 24h
WARMUP = 55


def get_daily_sma_at_time(daily_df, ts, period=20):
    """Get the daily 50-SMA value valid at a given 15m candle timestamp."""
    if daily_df is None or len(daily_df) < period:
        return None
    # Use only daily candles that closed before this 15m candle
    mask = daily_df["timestamp"] <= ts
    relevant = daily_df[mask]
    if len(relevant) < period:
        return None
    return relevant["close"].iloc[-period:].mean()


def check_signal_v3(row, daily_sma_val):
    """v3: BB + RSI<30 + VolSpike + regime_ok + daily 20-SMA trend"""
    price = row["close"]
    rsi_val = row.get("rsi")
    bb_lower = row.get("bb_lower")
    vol_ratio = row.get("vol_ratio", 1.0)
    regime_ok = row.get("regime_ok", True)
    if pd.isna(regime_ok):
        regime_ok = True

    if pd.isna(rsi_val) or pd.isna(bb_lower):
        return None
    if not regime_ok:
        return None
    # Daily SMA trend filter
    if daily_sma_val is not None and price < daily_sma_val:
        return None
    if pd.isna(vol_ratio) or vol_ratio < VOLUME_SPIKE_THRESHOLD:
        return None
    if price <= bb_lower and rsi_val < RSI_OVERSOLD:
        return "BUY"
    return None


def check_signal_v3_loose(row):
    """v3-loose: BB + RSI<30 + VolSpike + regime_ok only (no daily SMA, no MACD)"""
    price = row["close"]
    rsi_val = row.get("rsi")
    bb_lower = row.get("bb_lower")
    vol_ratio = row.get("vol_ratio", 1.0)
    regime_ok = row.get("regime_ok", True)
    if pd.isna(regime_ok):
        regime_ok = True

    if pd.isna(rsi_val) or pd.isna(bb_lower):
        return None
    if not regime_ok:
        return None
    if pd.isna(vol_ratio) or vol_ratio < VOLUME_SPIKE_THRESHOLD:
        return None
    if price <= bb_lower and rsi_val < RSI_OVERSOLD:
        return "BUY"
    return None


def simulate_trade(df, entry_idx, tp_pct, sl_pct):
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)
    for j in range(entry_idx + 1, min(entry_idx + 1 + TIMEOUT_CANDLES, len(df))):
        candle = df.iloc[j]
        if candle["low"] <= sl_price:
            return {"entry_time": entry_row["timestamp"], "entry_price": entry_price,
                    "exit_time": candle["timestamp"], "exit_price": sl_price,
                    "pnl_pct": -sl_pct * 100, "outcome": "SL", "candles_held": j - entry_idx}
        if candle["high"] >= tp_price:
            return {"entry_time": entry_row["timestamp"], "entry_price": entry_price,
                    "exit_time": candle["timestamp"], "exit_price": tp_price,
                    "pnl_pct": tp_pct * 100, "outcome": "TP", "candles_held": j - entry_idx}
    last_idx = min(entry_idx + TIMEOUT_CANDLES, len(df) - 1)
    exit_row = df.iloc[last_idx]
    exit_price = exit_row["close"]
    pnl = (exit_price - entry_price) / entry_price * 100
    return {"entry_time": entry_row["timestamp"], "entry_price": entry_price,
            "exit_time": exit_row["timestamp"], "exit_price": exit_price,
            "pnl_pct": round(pnl, 4), "outcome": "TIMEOUT", "candles_held": last_idx - entry_idx}


def simulate_trade_atr(df, entry_idx):
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]
    atr_val = entry_row.get("atr")
    if pd.isna(atr_val) or atr_val <= 0:
        return simulate_trade(df, entry_idx, PROFIT_TARGET, STOP_LOSS)
    tp_pct = (atr_val * ATR_TP_MULTIPLIER) / entry_price
    sl_pct = (atr_val * ATR_SL_MULTIPLIER) / entry_price
    return simulate_trade(df, entry_idx, tp_pct, sl_pct)


def run_v3(df, daily_df, use_atr=True):
    trades = []
    in_trade_until = -1
    for i in range(WARMUP, len(df)):
        if i <= in_trade_until:
            continue
        row = df.iloc[i]
        daily_sma = get_daily_sma_at_time(daily_df, row["timestamp"])
        sig = check_signal_v3(row, daily_sma)
        if sig:
            trade = simulate_trade_atr(df, i) if use_atr else simulate_trade(df, i, PROFIT_TARGET, STOP_LOSS)
            trades.append(trade)
            in_trade_until = i + trade["candles_held"]
    return trades


def run_v3_loose(df, use_atr=True):
    trades = []
    in_trade_until = -1
    for i in range(WARMUP, len(df)):
        if i <= in_trade_until:
            continue
        if check_signal_v3_loose(df.iloc[i]):
            trade = simulate_trade_atr(df, i) if use_atr else simulate_trade(df, i, PROFIT_TARGET, STOP_LOSS)
            trades.append(trade)
            in_trade_until = i + trade["candles_held"]
    return trades


def summarize(trades):
    if not trades:
        return {"total": 0, "tp": 0, "sl": 0, "to": 0, "wr": 0, "cum_pnl": 0, "avg_pnl": 0}
    tdf = pd.DataFrame(trades)
    total = len(tdf)
    tp = int((tdf["outcome"] == "TP").sum())
    sl = int((tdf["outcome"] == "SL").sum())
    to = int((tdf["outcome"] == "TIMEOUT").sum())
    return {"total": total, "tp": tp, "sl": sl, "to": to,
            "wr": tp / total * 100, "cum_pnl": tdf["pnl_pct"].sum(), "avg_pnl": tdf["pnl_pct"].mean()}


def print_table(stats_dict, title):
    print(f"\n  {title}")
    print(f"  {'Strategy':<28} {'Trades':>6} {'TP':>4} {'SL':>4} {'TO':>4} {'WR':>7} {'Cum P&L':>10} {'Avg P&L':>10}")
    print(f"  {'-'*28} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*7} {'-'*10} {'-'*10}")
    for label, s in stats_dict.items():
        if s["total"] == 0:
            print(f"  {label:<28} {0:>6}    —    —    —      —          —          —")
        else:
            print(f"  {label:<28} {s['total']:>6} {s['tp']:>4} {s['sl']:>4} {s['to']:>4} {s['wr']:>6.1f}% {s['cum_pnl']:>+9.2f}% {s['avg_pnl']:>+9.4f}%")


def monthly_breakdown(trades, label):
    if not trades:
        print(f"  {label}: no trades")
        return
    tdf = pd.DataFrame(trades)
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["month"] = tdf["entry_time"].dt.to_period("M")
    print(f"\n  {label} — Monthly Breakdown:")
    print(f"  {'Month':<12} {'Trades':>6} {'WR':>7} {'P&L':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*10}")
    for month, grp in tdf.groupby("month"):
        tp = int((grp["outcome"] == "TP").sum())
        total = len(grp)
        wr = tp / total * 100
        cum = grp["pnl_pct"].sum()
        print(f"  {str(month):<12} {total:>6} {wr:>6.1f}% {cum:>+9.2f}%")


def main():
    start_ms = int((time.time() - DAYS * 86400) * 1000)

    print(f"\n{'='*80}")
    print(f"  CryptoSignals v3 BACKTEST ({DAYS}-day, 15m candles)")
    print(f"{'='*80}")
    print(f"  v3:       BB + RSI<30 + VolSpike + regime filter + daily 20-SMA, ATR TP/SL")
    print(f"  v3-loose: BB + RSI<30 + VolSpike + regime filter only, ATR TP/SL")
    print(f"  Config: SL=1%, ATR_TP={ATR_TP_MULTIPLIER}x, ATR_SL={ATR_SL_MULTIPLIER}x")

    grand = {"v3": [], "v3-loose": []}
    per_sym = {}

    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        print(f"\n{'='*80}")
        print(f"  Fetching {sym}...")

        df = fetch_klines_paginated(sym, "15m", start_ms)
        df = compute_all(df)
        print(f"  15m data: {df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']}  ({len(df)} candles)")

        daily_df = fetch_daily_klines(sym, days=DAYS + 60)  # extra for SMA warmup
        print(f"  Daily data: {len(daily_df)} candles")

        v3_trades = run_v3(df, daily_df)
        v3l_trades = run_v3_loose(df)

        stats = {
            "v3 (full)": summarize(v3_trades),
            "v3-loose (regime only)": summarize(v3l_trades),
        }
        print_table(stats, display)
        monthly_breakdown(v3_trades, f"{display} v3")
        monthly_breakdown(v3l_trades, f"{display} v3-loose")

        per_sym[sym] = {"v3": v3_trades, "v3-loose": v3l_trades}
        grand["v3"].extend(v3_trades)
        grand["v3-loose"].extend(v3l_trades)

    # Grand total
    print(f"\n{'='*80}")
    print_table({
        "v3 (full)": summarize(grand["v3"]),
        "v3-loose (regime only)": summarize(grand["v3-loose"]),
    }, "GRAND TOTAL (BTC + ETH)")

    # BTC vs ETH comparison
    print(f"\n{'='*80}")
    print(f"  BTC vs ETH — Per-asset breakdown:")
    for strat in ["v3", "v3-loose"]:
        row_data = {}
        for sym in SYMBOLS:
            row_data[f"{SYMBOL_NAMES.get(sym, sym)} {strat}"] = summarize(per_sym[sym][strat])
        print_table(row_data, f"{strat} by asset")
    print()


if __name__ == "__main__":
    main()
