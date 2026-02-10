"""Backtest for CryptoSignals v6 — trailing stops, multi-TF, concurrent positions."""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from data import fetch_klines_paginated
from indicators_v6 import compute_all_v6
from signals_v6 import generate_signals_v6
from config_v6 import (
    SYMBOLS, SYMBOL_NAMES, TRAILING_15M, TRAILING_5M,
    V6_MAX_CONCURRENT_PER_ASSET, V6_SL_COOLDOWN_CANDLES,
)

WARMUP = 55


def fetch_all_data(days=90):
    """Fetch 15m, 5m, 1h, daily candles for all symbols."""
    end_ms = int(time.time() * 1000)
    start_ms = int((time.time() - days * 86400) * 1000)
    warmup_start_ms = int((time.time() - (days + 30) * 86400) * 1000)
    
    data = {}
    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        print(f"  Fetching {display} data...")
        
        df_15m = fetch_klines_paginated(sym, "15m", warmup_start_ms)
        print(f"    15m: {len(df_15m)} candles")
        
        df_5m = fetch_klines_paginated(sym, "5m", warmup_start_ms)
        print(f"    5m: {len(df_5m)} candles")
        
        df_1h = fetch_klines_paginated(sym, "1h", warmup_start_ms)
        print(f"    1h: {len(df_1h)} candles")
        
        df_1d = fetch_klines_paginated(sym, "1d", int((time.time() - (days + 90) * 86400) * 1000))
        print(f"    1d: {len(df_1d)} candles")
        
        data[sym] = {"5m": df_5m, "15m": df_15m, "1h": df_1h, "1d": df_1d}
    
    return data, start_ms


def simulate_trailing_stop(df, entry_idx, direction, entry_price, trail_cfg):
    """Walk candle-by-candle simulating trailing stop.
    
    Returns (exit_idx, exit_price, pnl_pct, outcome).
    """
    sl_price = (entry_price * (1 - trail_cfg["initial_sl_pct"]) if direction == "BUY"
                else entry_price * (1 + trail_cfg["initial_sl_pct"]))
    
    max_favorable = 0.0  # track best unrealized P&L
    max_hold = trail_cfg["max_hold_candles"]
    
    for j in range(entry_idx + 1, min(entry_idx + 1 + max_hold, len(df))):
        candle = df.iloc[j]
        high = candle["high"]
        low = candle["low"]
        close = candle["close"]
        
        # Check SL hit first (using high/low)
        if direction == "BUY":
            if low <= sl_price:
                pnl = (sl_price - entry_price) / entry_price
                return j, sl_price, pnl, "SL" if pnl < 0 else "TRAIL"
            
            # Update max favorable using high
            current_pnl = (high - entry_price) / entry_price
        else:
            if high >= sl_price:
                pnl = (entry_price - sl_price) / entry_price
                return j, sl_price, pnl, "SL" if pnl < 0 else "TRAIL"
            
            current_pnl = (entry_price - low) / entry_price
        
        max_favorable = max(max_favorable, current_pnl)
        
        # Update trailing stop based on max favorable excursion
        if max_favorable >= trail_cfg["trail_2_trigger"]:
            # Tight trail
            if direction == "BUY":
                new_sl = high * (1 - trail_cfg["trail_2_distance"])
                sl_price = max(sl_price, new_sl)
            else:
                new_sl = low * (1 + trail_cfg["trail_2_distance"])
                sl_price = min(sl_price, new_sl)
        elif max_favorable >= trail_cfg["trail_1_trigger"]:
            # Standard trail
            if direction == "BUY":
                new_sl = high * (1 - trail_cfg["trail_1_distance"])
                sl_price = max(sl_price, new_sl)
            else:
                new_sl = low * (1 + trail_cfg["trail_1_distance"])
                sl_price = min(sl_price, new_sl)
        elif max_favorable >= trail_cfg["breakeven_trigger"]:
            # Move to breakeven
            if direction == "BUY":
                sl_price = max(sl_price, entry_price)
            else:
                sl_price = min(sl_price, entry_price)
    
    # Timeout — exit at close
    last_idx = min(entry_idx + max_hold, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    if direction == "BUY":
        pnl = (exit_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exit_price) / entry_price
    return last_idx, exit_price, pnl, "TIMEOUT"


class PositionTracker:
    """Track open positions and cooldowns per symbol+direction."""
    
    def __init__(self):
        self.open_positions = []  # list of dicts with symbol, direction, entry_idx, exit_idx, timeframe
        self.sl_cooldowns = {}  # key: (symbol, direction) → candle_idx when cooldown expires
    
    def count_open(self, symbol, candle_idx):
        return sum(1 for p in self.open_positions 
                   if p["symbol"] == symbol and p["exit_idx"] > candle_idx)
    
    def has_conflicting(self, symbol, direction, candle_idx):
        """Check if there's an open position in the opposite direction."""
        opp = "SELL" if direction == "BUY" else "BUY"
        return any(p["symbol"] == symbol and p["direction"] == opp and p["exit_idx"] > candle_idx
                   for p in self.open_positions)
    
    def in_cooldown(self, symbol, direction, candle_idx):
        key = (symbol, direction)
        return candle_idx < self.sl_cooldowns.get(key, -1)
    
    def add_position(self, symbol, direction, entry_idx, exit_idx, timeframe):
        self.open_positions.append({
            "symbol": symbol, "direction": direction,
            "entry_idx": entry_idx, "exit_idx": exit_idx, "timeframe": timeframe
        })
    
    def set_cooldown(self, symbol, direction, candle_idx, cooldown_candles):
        key = (symbol, direction)
        self.sl_cooldowns[key] = candle_idx + cooldown_candles


def run_backtest(data, start_ms, days_label="90d"):
    """Run v6 backtest on both 5m and 15m timeframes."""
    all_trades = []
    
    # Pre-compute indicators
    computed = {}
    for sym in SYMBOLS:
        computed[sym] = {}
        for tf in ["5m", "15m"]:
            print(f"  Computing indicators for {SYMBOL_NAMES[sym]} {tf}...")
            computed[sym][tf] = compute_all_v6(data[sym][tf].copy())
    
    start_ts = pd.Timestamp(start_ms, unit="ms")
    
    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        tracker = PositionTracker()
        
        # Process 15m first (primary), then 5m
        for tf, trail_cfg in [("15m", TRAILING_15M), ("5m", TRAILING_5M)]:
            df = computed[sym][tf]
            
            candidates = df[df["timestamp"] >= start_ts].index
            if len(candidates) == 0:
                continue
            tf_start_idx = max(candidates[0], WARMUP)
            
            total_candles = len(df) - tf_start_idx
            print(f"\n  {display} {tf}: evaluating {total_candles} candles...")
            
            tf_trades = 0
            
            for i in range(tf_start_idx, len(df)):
                current_ts = df.iloc[i]["timestamp"]
                
                # Check position limits
                if tracker.count_open(sym, i) >= V6_MAX_CONCURRENT_PER_ASSET:
                    continue
                
                # Build lookback slice
                lookback_start = max(0, i - 200)
                df_slice = df.iloc[lookback_start:i+1]
                
                # Get higher TF direction for 5m signals (check 15m trend)
                higher_dir = None
                if tf == "5m":
                    df_15m = computed[sym]["15m"]
                    idx_15m = df_15m["timestamp"].searchsorted(current_ts, side="right") - 1
                    if idx_15m >= WARMUP:
                        slice_15m = df_15m.iloc[max(0, idx_15m-200):idx_15m+1]
                        sigs_15m = generate_signals_v6(sym, slice_15m, "15m")
                        if sigs_15m:
                            higher_dir = sigs_15m[0]["type"]
                
                signals = generate_signals_v6(sym, df_slice, tf, higher_dir)
                
                for sig in signals:
                    direction = sig["type"]
                    
                    # Check cooldown
                    if tracker.in_cooldown(sym, direction, i):
                        continue
                    
                    # Check conflicting positions
                    if tracker.has_conflicting(sym, direction, i):
                        continue
                    
                    # Check position limit again
                    if tracker.count_open(sym, i) >= V6_MAX_CONCURRENT_PER_ASSET:
                        break
                    
                    # Simulate trade with trailing stop
                    exit_idx, exit_price, pnl, outcome = simulate_trailing_stop(
                        df, i, direction, sig["price"], trail_cfg)
                    
                    pnl_pct = pnl * 100
                    
                    # Record position
                    tracker.add_position(sym, direction, i, exit_idx, tf)
                    
                    # Set cooldown if SL
                    if outcome == "SL":
                        tracker.set_cooldown(sym, direction, i, V6_SL_COOLDOWN_CANDLES)
                    
                    trade = {
                        "symbol": display,
                        "direction": direction,
                        "conviction": sig["conviction"],
                        "timeframe": tf,
                        "entry_time": df.iloc[i]["timestamp"],
                        "entry_price": sig["price"],
                        "exit_price": exit_price,
                        "exit_time": df.iloc[exit_idx]["timestamp"],
                        "pnl_pct": round(pnl_pct, 4),
                        "outcome": outcome,
                        "candles_held": exit_idx - i,
                        "rsi": sig["rsi"],
                        "adx": sig["adx"],
                        "vol_ratio": sig["vol_ratio"],
                        "position_size": sig["position_size"],
                    }
                    all_trades.append(trade)
                    tf_trades += 1
            
            print(f"    {display} {tf}: {tf_trades} trades")
    
    return all_trades


def summarize(trades, label=""):
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "wr": 0, "cum_pnl": 0,
                "weighted_pnl": 0, "avg_pnl": 0, "avg_win": 0, "avg_loss": 0,
                "best": 0, "worst": 0, "max_dd": 0}
    tdf = pd.DataFrame(trades)
    total = len(tdf)
    wins = int((tdf["pnl_pct"] > 0).sum())
    losses = int((tdf["pnl_pct"] <= 0).sum())
    cum = tdf["pnl_pct"].sum()
    
    # Weighted P&L (STRONG = 2x position)
    weighted = (tdf["pnl_pct"] * tdf["position_size"]).sum()
    
    win_trades = tdf[tdf["pnl_pct"] > 0]["pnl_pct"]
    loss_trades = tdf[tdf["pnl_pct"] <= 0]["pnl_pct"]
    
    # Max drawdown (sequential)
    equity = tdf["pnl_pct"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min()
    
    return {
        "total": total, "wins": wins, "losses": losses,
        "wr": wins / total * 100 if total > 0 else 0,
        "cum_pnl": round(cum, 2),
        "weighted_pnl": round(weighted, 2),
        "avg_pnl": round(tdf["pnl_pct"].mean(), 4),
        "avg_win": round(win_trades.mean(), 4) if len(win_trades) > 0 else 0,
        "avg_loss": round(loss_trades.mean(), 4) if len(loss_trades) > 0 else 0,
        "best": round(tdf["pnl_pct"].max(), 4),
        "worst": round(tdf["pnl_pct"].min(), 4),
        "max_dd": round(max_dd, 2),
    }


def print_summary(stats, title):
    print(f"\n  {title}")
    print(f"    Trades: {stats['total']} | Wins: {stats['wins']} | Losses: {stats['losses']} | WR: {stats['wr']:.1f}%")
    print(f"    Cum P&L: {stats['cum_pnl']:+.2f}% | Weighted P&L: {stats['weighted_pnl']:+.2f}%")
    print(f"    Avg P&L: {stats['avg_pnl']:+.4f}% | Avg Win: {stats['avg_win']:+.4f}% | Avg Loss: {stats['avg_loss']:+.4f}%")
    print(f"    Best: {stats['best']:+.4f}% | Worst: {stats['worst']:+.4f}% | Max DD: {stats['max_dd']:.2f}%")


def daily_breakdown(trades, label):
    if not trades:
        print(f"  {label}: no trades")
        return
    tdf = pd.DataFrame(trades)
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["date"] = tdf["entry_time"].dt.date
    print(f"\n  {label} — Daily P&L:")
    print(f"  {'Date':<12} {'Trades':>6} {'Wins':>5} {'P&L':>10} {'Weighted':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*10} {'-'*10}")
    for date, grp in tdf.groupby("date"):
        wins = int((grp["pnl_pct"] > 0).sum())
        cum = grp["pnl_pct"].sum()
        weighted = (grp["pnl_pct"] * grp["position_size"]).sum()
        print(f"  {str(date):<12} {len(grp):>6} {wins:>5} {cum:>+9.2f}% {weighted:>+9.2f}%")


def main():
    print(f"\n{'='*90}")
    print(f"  CryptoSignals v6 BACKTEST")
    print(f"{'='*90}")
    print(f"  v6: Loose entry (BB proximity + RSI 40/60 + Vol 1.2x)")
    print(f"       Trailing stops (breakeven → trail → tight trail)")
    print(f"       Multi-TF (5m scalping + 15m swing)")
    print(f"       Trend-following on ADX > 30")
    print(f"       Up to 2 concurrent positions per asset")
    
    print(f"\n  Fetching data...")
    data, start_ms_90 = fetch_all_data(days=90)
    
    # === 14-day backtest ===
    start_ms_14 = int((time.time() - 14 * 86400) * 1000)
    
    print(f"\n{'='*90}")
    print(f"  14-DAY BACKTEST (targeting 5% in 2 weeks)")
    print(f"{'='*90}")
    trades_14 = run_backtest(data, start_ms_14, "14d")
    
    if trades_14:
        print_summary(summarize(trades_14), "14-DAY OVERALL")
        
        # By conviction
        standard = [t for t in trades_14 if t["conviction"] == "STANDARD"]
        strong = [t for t in trades_14 if t["conviction"] == "STRONG"]
        trend_follow = [t for t in trades_14 if t["conviction"] == "TREND_FOLLOW"]
        print_summary(summarize(standard), "14-DAY STANDARD signals")
        print_summary(summarize(strong), "14-DAY STRONG signals")
        print_summary(summarize(trend_follow), "14-DAY TREND_FOLLOW signals")
        
        # By timeframe
        tf_5m = [t for t in trades_14 if t["timeframe"] == "5m"]
        tf_15m = [t for t in trades_14 if t["timeframe"] == "15m"]
        print_summary(summarize(tf_5m), "14-DAY 5m trades")
        print_summary(summarize(tf_15m), "14-DAY 15m trades")
        
        # By symbol
        for sym_name in ["BTC/USDT", "ETH/USDT"]:
            sym_trades = [t for t in trades_14 if t["symbol"] == sym_name]
            print_summary(summarize(sym_trades), f"14-DAY {sym_name}")
        
        # By outcome
        for outcome in ["SL", "TRAIL", "TIMEOUT"]:
            oc_trades = [t for t in trades_14 if t["outcome"] == outcome]
            if oc_trades:
                print_summary(summarize(oc_trades), f"14-DAY outcome={outcome}")
        
        daily_breakdown(trades_14, "14-DAY ALL")
    else:
        print("\n  ⚠ No trades in 14-day window!")
    
    # === 90-day backtest ===
    print(f"\n{'='*90}")
    print(f"  90-DAY BACKTEST")
    print(f"{'='*90}")
    trades_90 = run_backtest(data, start_ms_90, "90d")
    
    if trades_90:
        print_summary(summarize(trades_90), "90-DAY OVERALL")
        
        standard = [t for t in trades_90 if t["conviction"] == "STANDARD"]
        strong = [t for t in trades_90 if t["conviction"] == "STRONG"]
        trend_follow = [t for t in trades_90 if t["conviction"] == "TREND_FOLLOW"]
        print_summary(summarize(standard), "90-DAY STANDARD signals")
        print_summary(summarize(strong), "90-DAY STRONG signals")
        print_summary(summarize(trend_follow), "90-DAY TREND_FOLLOW signals")
        
        tf_5m = [t for t in trades_90 if t["timeframe"] == "5m"]
        tf_15m = [t for t in trades_90 if t["timeframe"] == "15m"]
        print_summary(summarize(tf_5m), "90-DAY 5m trades")
        print_summary(summarize(tf_15m), "90-DAY 15m trades")
        
        for sym_name in ["BTC/USDT", "ETH/USDT"]:
            sym_trades = [t for t in trades_90 if t["symbol"] == sym_name]
            print_summary(summarize(sym_trades), f"90-DAY {sym_name}")
        
        # Monthly breakdown
        tdf = pd.DataFrame(trades_90)
        tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
        tdf["month"] = tdf["entry_time"].dt.to_period("M")
        print(f"\n  90-DAY Monthly Breakdown:")
        print(f"  {'Month':<12} {'Trades':>6} {'Wins':>5} {'WR':>7} {'P&L':>10} {'Weighted':>10}")
        print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*7} {'-'*10} {'-'*10}")
        for month, grp in tdf.groupby("month"):
            wins = int((grp["pnl_pct"] > 0).sum())
            total = len(grp)
            wr = wins / total * 100
            cum = grp["pnl_pct"].sum()
            weighted = (grp["pnl_pct"] * grp["position_size"]).sum()
            print(f"  {str(month):<12} {total:>6} {wins:>5} {wr:>6.1f}% {cum:>+9.2f}% {weighted:>+9.2f}%")
    else:
        print("\n  ⚠ No trades in 90-day window!")
    
    # === Sample trades ===
    all_trades = trades_14 or trades_90
    if all_trades:
        print(f"\n{'='*90}")
        print(f"  Sample trades (first 15):")
        for t in (all_trades)[:15]:
            print(f"    {t['direction']:4} {t['conviction']:<12} {t['symbol']:<10} {t['timeframe']} "
                  f"{t['entry_time']}  entry={t['entry_price']:.2f}  exit={t['exit_price']:.2f}  "
                  f"{t['outcome']:7} {t['pnl_pct']:+.2f}%  RSI={t['rsi']} ADX={t['adx']}")
    
    print()


if __name__ == "__main__":
    main()
