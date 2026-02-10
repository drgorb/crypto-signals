"""Backtest for CryptoSignals v5 — full signal pipeline with MTF, ADX, market structure, trend score."""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from data import fetch_klines_paginated
from indicators import compute_all
from signals import generate_signals
from config import SYMBOL_NAMES

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DAYS = 90
TIMEOUT_CANDLES = 96  # 24h
WARMUP = 55  # candles needed for indicators to stabilize

# Stub sentiment: neutral (no suppression, no prediction market bias)
NEUTRAL_SENTIMENT = {
    "is_bearish": False,
    "is_bullish": False,
    "score": 0.0,
    "bullish_score": 0.0,
    "prediction_market_score": 0.0,
    "markets": [],
    "reason": "backtest stub"
}

# Stub derivatives: neutral
NEUTRAL_DERIVATIVES = {"score": 0.0}


def fetch_all_data():
    """Fetch 90 days of 15m, 1h, daily candles for BTC and ETH."""
    start_ms = int((time.time() - DAYS * 86400) * 1000)
    # Extra history for indicator warmup
    warmup_start_ms = int((time.time() - (DAYS + 30) * 86400) * 1000)
    
    data = {}
    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        print(f"  Fetching {display} 15m candles...")
        df_15m = fetch_klines_paginated(sym, "15m", warmup_start_ms)
        print(f"    Got {len(df_15m)} candles: {df_15m.iloc[0]['timestamp']} → {df_15m.iloc[-1]['timestamp']}")
        
        print(f"  Fetching {display} 1h candles...")
        df_1h = fetch_klines_paginated(sym, "1h", warmup_start_ms)
        print(f"    Got {len(df_1h)} candles")
        
        print(f"  Fetching {display} daily candles...")
        df_1d = fetch_klines_paginated(sym, "1d", int((time.time() - (DAYS + 90) * 86400) * 1000))
        print(f"    Got {len(df_1d)} candles")
        
        data[sym] = {"15m": df_15m, "1h": df_1h, "1d": df_1d}
    
    return data, start_ms


def simulate_trade_from_signal(df_15m, entry_idx, signal):
    """Simulate a trade using the signal's TP/SL prices directly."""
    entry_price = signal["price"]
    tp_price = signal["take_profit"]
    sl_price = signal["stop_loss"]
    direction = signal["type"]
    
    # Calculate pct from prices
    if direction == "BUY":
        tp_pct = (tp_price - entry_price) / entry_price * 100
        sl_pct = (entry_price - sl_price) / entry_price * 100
    else:
        tp_pct = (entry_price - tp_price) / entry_price * 100
        sl_pct = (sl_price - entry_price) / entry_price * 100
    
    for j in range(entry_idx + 1, min(entry_idx + 1 + TIMEOUT_CANDLES, len(df_15m))):
        candle = df_15m.iloc[j]
        if direction == "BUY":
            if candle["low"] <= sl_price:
                return _result(signal, df_15m, entry_idx, j, sl_price, -sl_pct, "SL")
            if candle["high"] >= tp_price:
                return _result(signal, df_15m, entry_idx, j, tp_price, tp_pct, "TP")
        else:
            if candle["high"] >= sl_price:
                return _result(signal, df_15m, entry_idx, j, sl_price, -sl_pct, "SL")
            if candle["low"] <= tp_price:
                return _result(signal, df_15m, entry_idx, j, tp_price, tp_pct, "TP")
    
    # Timeout
    last_idx = min(entry_idx + TIMEOUT_CANDLES, len(df_15m) - 1)
    exit_price = df_15m.iloc[last_idx]["close"]
    if direction == "BUY":
        pnl = (exit_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - exit_price) / entry_price * 100
    return _result(signal, df_15m, entry_idx, last_idx, exit_price, round(pnl, 4), "TIMEOUT")


def _result(signal, df, entry_idx, exit_idx, exit_price, pnl_pct, outcome):
    trend_score = signal.get("trend_score", 0)
    direction = signal["type"]
    # Trend-aligned: BUY when trend_score > 0, SELL when trend_score < 0
    if direction == "BUY":
        trend_aligned = trend_score > 0
    else:
        trend_aligned = trend_score < 0
    
    return {
        "symbol": signal["symbol"],
        "direction": direction,
        "entry_time": df.iloc[entry_idx]["timestamp"],
        "entry_price": signal["price"],
        "exit_price": exit_price,
        "exit_time": df.iloc[exit_idx]["timestamp"],
        "pnl_pct": pnl_pct,
        "outcome": outcome,
        "candles_held": exit_idx - entry_idx,
        "trend_score": trend_score,
        "trend_aligned": trend_aligned,
        "strength": signal.get("strength", "WEAK"),
        "market_structure": signal.get("market_structure", "ranging"),
    }


def run_backtest(data, start_ms):
    """Walk through 15m candles and generate signals using the full v5 pipeline."""
    all_trades = []
    
    # Pre-compute indicators on full datasets
    computed_15m = {}
    for sym in SYMBOLS:
        computed_15m[sym] = compute_all(data[sym]["15m"].copy())
    
    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        is_eth = sym == "ETHUSDT"
        
        df_15m = computed_15m[sym]
        df_1h = data[sym]["1h"]
        df_1d = data[sym]["1d"]
        
        # Pre-compute BTC 15m with indicators for ETH correlation
        btc_15m_full = computed_15m.get("BTCUSDT") if is_eth else None
        btc_1d = data["BTCUSDT"]["1d"] if is_eth else None
        btc_1h = data["BTCUSDT"]["1h"] if is_eth else None
        
        # Find start index
        start_ts = pd.Timestamp(start_ms, unit="ms")
        candidates = df_15m[df_15m["timestamp"] >= start_ts].index
        if len(candidates) == 0:
            print(f"  {display}: no data in backtest range")
            continue
        start_idx = max(candidates[0], WARMUP)
        
        total_candles = len(df_15m) - start_idx
        print(f"\n  Running {display}: {total_candles} candles to evaluate...")
        
        in_trade_until = -1
        sym_trades = 0
        
        for i in range(start_idx, len(df_15m)):
            if i <= in_trade_until:
                continue
            
            current_ts = df_15m.iloc[i]["timestamp"]
            
            # Use a fixed lookback window (200 candles is plenty for all indicators)
            lookback_start = max(0, i - 200)
            df_slice = df_15m.iloc[lookback_start:i+1]
            
            # Get most recent completed hourly/daily candles (use searchsorted for speed)
            h_idx = df_1h["timestamp"].searchsorted(current_ts, side="right")
            d_idx = df_1d["timestamp"].searchsorted(current_ts, side="right")
            hourly_slice = df_1h.iloc[:h_idx]
            daily_slice = df_1d.iloc[:d_idx]
            
            kwargs = {
                "daily_df": daily_slice if len(daily_slice) > 0 else None,
                "hourly_df": hourly_slice if len(hourly_slice) > 0 else None,
                "derivatives_data": NEUTRAL_DERIVATIVES,
            }
            
            if is_eth and btc_15m_full is not None:
                btc_i = btc_15m_full["timestamp"].searchsorted(current_ts, side="right")
                btc_start = max(0, btc_i - 200)
                kwargs["btc_df_15m"] = btc_15m_full.iloc[btc_start:btc_i]
                btc_d_idx = btc_1d["timestamp"].searchsorted(current_ts, side="right")
                kwargs["btc_daily_df"] = btc_1d.iloc[:btc_d_idx]
                btc_h_idx = btc_1h["timestamp"].searchsorted(current_ts, side="right")
                kwargs["btc_hourly_df"] = btc_1h.iloc[:btc_h_idx]
            
            signals = generate_signals(
                sym, df_slice, NEUTRAL_SENTIMENT,
                funding_rate=None, ob_ratio=None,
                **kwargs
            )
            
            if signals:
                sig = signals[0]
                trade = simulate_trade_from_signal(df_15m, i, sig)
                all_trades.append(trade)
                in_trade_until = i + trade["candles_held"]
                sym_trades += 1
        
        print(f"    {display}: {sym_trades} trades found")
    
    return all_trades


def summarize(trades, label=""):
    if not trades:
        return {"total": 0, "buy": 0, "sell": 0, "tp": 0, "sl": 0, "to": 0,
                "wr": 0, "cum_pnl": 0, "avg_pnl": 0}
    tdf = pd.DataFrame(trades)
    total = len(tdf)
    buy = int((tdf["direction"] == "BUY").sum())
    sell = int((tdf["direction"] == "SELL").sum())
    tp = int((tdf["outcome"] == "TP").sum())
    sl = int((tdf["outcome"] == "SL").sum())
    to = int((tdf["outcome"] == "TIMEOUT").sum())
    return {"total": total, "buy": buy, "sell": sell, "tp": tp, "sl": sl, "to": to,
            "wr": tp / total * 100 if total > 0 else 0,
            "cum_pnl": round(tdf["pnl_pct"].sum(), 2),
            "avg_pnl": round(tdf["pnl_pct"].mean(), 4)}


def print_table(stats_dict, title):
    print(f"\n  {title}")
    print(f"  {'Label':<30} {'Trades':>6} {'BUY':>5} {'SELL':>5} {'TP':>4} {'SL':>4} {'TO':>4} {'WR':>7} {'Cum P&L':>10} {'Avg P&L':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*7} {'-'*10} {'-'*10}")
    for label, s in stats_dict.items():
        if s["total"] == 0:
            print(f"  {label:<30} {0:>6}     —     —    —    —    —      —          —          —")
        else:
            print(f"  {label:<30} {s['total']:>6} {s['buy']:>5} {s['sell']:>5} {s['tp']:>4} {s['sl']:>4} {s['to']:>4} {s['wr']:>6.1f}% {s['cum_pnl']:>+9.2f}% {s['avg_pnl']:>+9.4f}%")


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
        buy_ct = int((grp["direction"] == "BUY").sum())
        sell_ct = int((grp["direction"] == "SELL").sum())
        print(f"  {str(month):<12} {total:>6} {buy_ct:>5} {sell_ct:>5} {wr:>6.1f}% {cum:>+9.2f}%")


def main():
    print(f"\n{'='*90}")
    print(f"  CryptoSignals v5 BACKTEST ({DAYS}-day, 15m candles)")
    print(f"{'='*90}")
    print(f"  v5: Full pipeline — BB+RSI+Vol + dynamic RSI + regime + ADX suppression")
    print(f"       + market structure + MTF agreement + trend score gates")
    print(f"       + asymmetric SL (trend-aligned 2x, contrarian 1x)")
    print(f"       + BTC-ETH correlation + momentum outlook + reversion success rate")
    print(f"  Stubs: derivatives_data=neutral, prediction_market=0, funding=None, OB=None")
    print(f"  (These filters pass through as neutral when None/0)")
    
    print(f"\n  Fetching data...")
    data, start_ms = fetch_all_data()
    
    print(f"\n  Running backtest...")
    all_trades = run_backtest(data, start_ms)
    
    if not all_trades:
        print("\n  ⚠ No trades generated!")
        return
    
    # === Overall ===
    print(f"\n{'='*90}")
    print_table({"v5 (all trades)": summarize(all_trades)}, "GRAND TOTAL")
    
    # === BUY vs SELL ===
    buy_trades = [t for t in all_trades if t["direction"] == "BUY"]
    sell_trades = [t for t in all_trades if t["direction"] == "SELL"]
    print(f"\n{'='*90}")
    print_table({
        "BUY trades": summarize(buy_trades),
        "SELL trades": summarize(sell_trades),
    }, "BUY vs SELL Breakdown")
    
    # === Trend-aligned vs Contrarian ===
    aligned = [t for t in all_trades if t["trend_aligned"]]
    contrarian = [t for t in all_trades if not t["trend_aligned"]]
    print(f"\n{'='*90}")
    print_table({
        "Trend-aligned": summarize(aligned),
        "Contrarian": summarize(contrarian),
    }, "Trend-Aligned vs Contrarian (asymmetric SL)")
    
    # === BTC vs ETH ===
    btc_trades = [t for t in all_trades if "BTC" in t["symbol"]]
    eth_trades = [t for t in all_trades if "ETH" in t["symbol"]]
    print(f"\n{'='*90}")
    print_table({
        "BTC/USDT": summarize(btc_trades),
        "ETH/USDT": summarize(eth_trades),
    }, "BTC vs ETH Breakdown")
    
    # === Monthly ===
    print(f"\n{'='*90}")
    monthly_breakdown(all_trades, "v5 All Trades")
    monthly_breakdown(buy_trades, "v5 BUY Trades")
    monthly_breakdown(sell_trades, "v5 SELL Trades")
    
    # === Comparison with v4 ===
    s = summarize(all_trades)
    print(f"\n{'='*90}")
    print(f"  v4 vs v5 Comparison:")
    print(f"  {'Metric':<25} {'v4':>12} {'v5':>12} {'Delta':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Total trades':<25} {'55':>12} {s['total']:>12}")
    print(f"  {'Win rate':<25} {'54.5%':>12} {s['wr']:>11.1f}%")
    print(f"  {'Cumulative P&L':<25} {'-0.59%':>12} {s['cum_pnl']:>+11.2f}%")
    print(f"  {'Avg P&L/trade':<25} {'-0.0107%':>12} {s['avg_pnl']:>+11.4f}%")
    
    # === Sample trades ===
    print(f"\n{'='*90}")
    print(f"  Sample trades (first 10):")
    for t in all_trades[:10]:
        ts = t["trend_score"]
        aligned_str = "aligned" if t["trend_aligned"] else "contrarian"
        print(f"    {t['direction']:4} {t['symbol']:<10} {t['entry_time']}  "
              f"entry={t['entry_price']:.2f}  exit={t['exit_price']:.2f}  "
              f"{t['outcome']:7} {t['pnl_pct']:+.2f}%  ts={ts:+d} ({aligned_str})")
    
    print()


if __name__ == "__main__":
    main()
