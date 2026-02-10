"""Long-range v6 backtest: May 1, 2025 → Feb 10, 2026."""

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

# May 1, 2025 00:00 UTC
START_MS = 1746057600000
# 30 days warmup before that
WARMUP_START_MS = START_MS - 30 * 86400 * 1000
# Extra warmup for daily candles
DAILY_WARMUP_MS = START_MS - 120 * 86400 * 1000

END_MS = int(time.time() * 1000)


def fetch_all_data():
    data = {}
    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        print(f"  Fetching {display}...")
        
        t0 = time.time()
        df_15m = fetch_klines_paginated(sym, "15m", WARMUP_START_MS, END_MS)
        print(f"    15m: {len(df_15m)} candles ({time.time()-t0:.1f}s)")
        
        t0 = time.time()
        df_1h = fetch_klines_paginated(sym, "1h", WARMUP_START_MS, END_MS)
        print(f"    1h: {len(df_1h)} candles ({time.time()-t0:.1f}s)")
        
        t0 = time.time()
        df_1d = fetch_klines_paginated(sym, "1d", DAILY_WARMUP_MS, END_MS)
        print(f"    1d: {len(df_1d)} candles ({time.time()-t0:.1f}s)")
        
        # Try 5m but skip if too slow
        t0 = time.time()
        try:
            df_5m = fetch_klines_paginated(sym, "5m", WARMUP_START_MS, END_MS)
            elapsed = time.time() - t0
            print(f"    5m: {len(df_5m)} candles ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    5m: FAILED ({e}), skipping")
            df_5m = pd.DataFrame()
        
        data[sym] = {"5m": df_5m, "15m": df_15m, "1h": df_1h, "1d": df_1d}
    return data


def simulate_trailing_stop(df, entry_idx, direction, entry_price, trail_cfg):
    sl_price = (entry_price * (1 - trail_cfg["initial_sl_pct"]) if direction == "BUY"
                else entry_price * (1 + trail_cfg["initial_sl_pct"]))
    max_favorable = 0.0
    max_hold = trail_cfg["max_hold_candles"]
    
    for j in range(entry_idx + 1, min(entry_idx + 1 + max_hold, len(df))):
        high = df.iloc[j]["high"]
        low = df.iloc[j]["low"]
        
        if direction == "BUY":
            if low <= sl_price:
                pnl = (sl_price - entry_price) / entry_price
                return j, sl_price, pnl, "SL" if pnl < 0 else "TRAIL"
            current_pnl = (high - entry_price) / entry_price
        else:
            if high >= sl_price:
                pnl = (entry_price - sl_price) / entry_price
                return j, sl_price, pnl, "SL" if pnl < 0 else "TRAIL"
            current_pnl = (entry_price - low) / entry_price
        
        max_favorable = max(max_favorable, current_pnl)
        
        if max_favorable >= trail_cfg["trail_2_trigger"]:
            if direction == "BUY":
                sl_price = max(sl_price, high * (1 - trail_cfg["trail_2_distance"]))
            else:
                sl_price = min(sl_price, low * (1 + trail_cfg["trail_2_distance"]))
        elif max_favorable >= trail_cfg["trail_1_trigger"]:
            if direction == "BUY":
                sl_price = max(sl_price, high * (1 - trail_cfg["trail_1_distance"]))
            else:
                sl_price = min(sl_price, low * (1 + trail_cfg["trail_1_distance"]))
        elif max_favorable >= trail_cfg["breakeven_trigger"]:
            if direction == "BUY":
                sl_price = max(sl_price, entry_price)
            else:
                sl_price = min(sl_price, entry_price)
    
    last_idx = min(entry_idx + max_hold, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    pnl = ((exit_price - entry_price) / entry_price if direction == "BUY"
           else (entry_price - exit_price) / entry_price)
    return last_idx, exit_price, pnl, "TIMEOUT"


class PositionTracker:
    def __init__(self):
        self.open_positions = []
        self.sl_cooldowns = {}
    
    def count_open(self, symbol, candle_idx):
        return sum(1 for p in self.open_positions
                   if p["symbol"] == symbol and p["exit_idx"] > candle_idx)
    
    def has_conflicting(self, symbol, direction, candle_idx):
        opp = "SELL" if direction == "BUY" else "BUY"
        return any(p["symbol"] == symbol and p["direction"] == opp and p["exit_idx"] > candle_idx
                   for p in self.open_positions)
    
    def in_cooldown(self, symbol, direction, candle_idx):
        return candle_idx < self.sl_cooldowns.get((symbol, direction), -1)
    
    def add_position(self, symbol, direction, entry_idx, exit_idx, timeframe):
        self.open_positions.append({
            "symbol": symbol, "direction": direction,
            "entry_idx": entry_idx, "exit_idx": exit_idx, "timeframe": timeframe
        })
    
    def set_cooldown(self, symbol, direction, candle_idx, cooldown_candles):
        self.sl_cooldowns[(symbol, direction)] = candle_idx + cooldown_candles


def run_backtest(data, computed, timeframes=None):
    if timeframes is None:
        timeframes = ["15m", "5m"]
    
    all_trades = []
    start_ts = pd.Timestamp(START_MS, unit="ms")
    
    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        tracker = PositionTracker()
        
        for tf, trail_cfg in [("15m", TRAILING_15M), ("5m", TRAILING_5M)]:
            if tf not in timeframes:
                continue
            if tf not in computed[sym] or computed[sym][tf] is None:
                continue
            
            df = computed[sym][tf]
            candidates = df[df["timestamp"] >= start_ts].index
            if len(candidates) == 0:
                continue
            tf_start_idx = max(candidates[0], WARMUP)
            total_candles = len(df) - tf_start_idx
            print(f"  {display} {tf}: {total_candles} candles to evaluate...")
            
            tf_trades = 0
            for i in range(tf_start_idx, len(df)):
                if tracker.count_open(sym, i) >= V6_MAX_CONCURRENT_PER_ASSET:
                    continue
                
                lookback_start = max(0, i - 200)
                df_slice = df.iloc[lookback_start:i+1]
                
                higher_dir = None
                if tf == "5m" and "15m" in computed[sym] and computed[sym]["15m"] is not None:
                    current_ts = df.iloc[i]["timestamp"]
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
                    if tracker.in_cooldown(sym, direction, i):
                        continue
                    if tracker.has_conflicting(sym, direction, i):
                        continue
                    if tracker.count_open(sym, i) >= V6_MAX_CONCURRENT_PER_ASSET:
                        break
                    
                    exit_idx, exit_price, pnl, outcome = simulate_trailing_stop(
                        df, i, direction, sig["price"], trail_cfg)
                    
                    tracker.add_position(sym, direction, i, exit_idx, tf)
                    if outcome == "SL":
                        tracker.set_cooldown(sym, direction, i, V6_SL_COOLDOWN_CANDLES)
                    
                    all_trades.append({
                        "symbol": display, "direction": direction,
                        "conviction": sig["conviction"], "timeframe": tf,
                        "entry_time": df.iloc[i]["timestamp"],
                        "entry_price": sig["price"], "exit_price": exit_price,
                        "exit_time": df.iloc[exit_idx]["timestamp"],
                        "pnl_pct": round(pnl * 100, 4), "outcome": outcome,
                        "candles_held": exit_idx - i,
                        "rsi": sig["rsi"], "adx": sig["adx"],
                        "vol_ratio": sig["vol_ratio"],
                        "position_size": sig["position_size"],
                    })
                    tf_trades += 1
            
            print(f"    → {tf_trades} trades")
    
    return all_trades


def summarize(trades):
    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    total = len(tdf)
    wins = int((tdf["pnl_pct"] > 0).sum())
    cum = tdf["pnl_pct"].sum()
    weighted = (tdf["pnl_pct"] * tdf["position_size"]).sum()
    win_trades = tdf[tdf["pnl_pct"] > 0]["pnl_pct"]
    loss_trades = tdf[tdf["pnl_pct"] <= 0]["pnl_pct"]
    equity = tdf["pnl_pct"].cumsum()
    max_dd = (equity - equity.cummax()).min()
    
    return {
        "total": total, "wins": wins, "losses": total - wins,
        "wr": wins / total * 100,
        "cum_pnl": round(cum, 2), "weighted_pnl": round(weighted, 2),
        "avg_pnl": round(tdf["pnl_pct"].mean(), 4),
        "avg_win": round(win_trades.mean(), 4) if len(win_trades) > 0 else 0,
        "avg_loss": round(loss_trades.mean(), 4) if len(loss_trades) > 0 else 0,
        "best": round(tdf["pnl_pct"].max(), 4),
        "worst": round(tdf["pnl_pct"].min(), 4),
        "max_dd": round(max_dd, 2),
    }


def print_summary(stats, title):
    if stats is None:
        print(f"\n  {title}: no trades")
        return
    print(f"\n  {title}")
    print(f"    Trades: {stats['total']} | W: {stats['wins']} | L: {stats['losses']} | WR: {stats['wr']:.1f}%")
    print(f"    Cum P&L: {stats['cum_pnl']:+.2f}% | Weighted: {stats['weighted_pnl']:+.2f}%")
    print(f"    Avg: {stats['avg_pnl']:+.4f}% | Avg Win: {stats['avg_win']:+.4f}% | Avg Loss: {stats['avg_loss']:+.4f}%")
    print(f"    Best: {stats['best']:+.4f}% | Worst: {stats['worst']:+.4f}% | Max DD: {stats['max_dd']:.2f}%")


def main():
    print(f"\n{'='*90}")
    print(f"  CryptoSignals v6 LONG BACKTEST: May 1, 2025 → Feb 10, 2026")
    print(f"{'='*90}")
    
    print(f"\n  Fetching data (paginated)...")
    data = fetch_all_data()
    
    # Compute indicators
    computed = {}
    timeframes_to_run = []
    for sym in SYMBOLS:
        computed[sym] = {}
        for tf in ["15m", "5m"]:
            if len(data[sym].get(tf, pd.DataFrame())) > WARMUP:
                print(f"  Computing indicators: {SYMBOL_NAMES[sym]} {tf} ({len(data[sym][tf])} candles)...")
                computed[sym][tf] = compute_all_v6(data[sym][tf].copy())
                if tf not in timeframes_to_run:
                    timeframes_to_run.append(tf)
            else:
                computed[sym][tf] = None
                print(f"  Skipping {SYMBOL_NAMES[sym]} {tf}: insufficient data")
    
    print(f"\n  Running backtest on timeframes: {timeframes_to_run}")
    trades = run_backtest(data, computed, timeframes_to_run)
    
    if not trades:
        print("\n  ⚠ No trades generated!")
        return
    
    tdf = pd.DataFrame(trades)
    
    # ===== OVERALL =====
    print_summary(summarize(trades), "OVERALL")
    
    # ===== BY SIGNAL TYPE =====
    for conv in ["STANDARD", "STRONG", "TREND_FOLLOW"]:
        sub = [t for t in trades if t["conviction"] == conv]
        print_summary(summarize(sub), f"Signal: {conv}")
    
    # ===== BY ASSET =====
    for sym_name in ["BTC/USDT", "ETH/USDT"]:
        sub = [t for t in trades if t["symbol"] == sym_name]
        print_summary(summarize(sub), f"Asset: {sym_name}")
    
    # ===== BY TIMEFRAME =====
    for tf in ["15m", "5m"]:
        sub = [t for t in trades if t["timeframe"] == tf]
        print_summary(summarize(sub), f"Timeframe: {tf}")
    
    # ===== BY OUTCOME =====
    for oc in ["SL", "TRAIL", "TIMEOUT"]:
        sub = [t for t in trades if t["outcome"] == oc]
        if sub:
            print_summary(summarize(sub), f"Outcome: {oc}")
    
    # ===== MONTHLY BREAKDOWN =====
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["month"] = tdf["entry_time"].dt.to_period("M")
    print(f"\n  {'='*80}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"  {'Month':<10} {'Trades':>7} {'Wins':>6} {'WR':>7} {'P&L':>10} {'Weighted':>10} {'MaxDD':>8}")
    print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")
    for month, grp in tdf.groupby("month"):
        wins = int((grp["pnl_pct"] > 0).sum())
        total = len(grp)
        wr = wins / total * 100
        cum = grp["pnl_pct"].sum()
        weighted = (grp["pnl_pct"] * grp["position_size"]).sum()
        eq = grp["pnl_pct"].cumsum()
        dd = (eq - eq.cummax()).min()
        print(f"  {str(month):<10} {total:>7} {wins:>6} {wr:>6.1f}% {cum:>+9.2f}% {weighted:>+9.2f}% {dd:>+7.2f}%")
    
    # ===== SAMPLE TRADES =====
    print(f"\n  Sample trades (first 20):")
    for t in trades[:20]:
        print(f"    {t['direction']:4} {t['conviction']:<12} {t['symbol']:<10} {t['timeframe']} "
              f"{t['entry_time']}  {t['outcome']:7} {t['pnl_pct']:+.2f}%")
    
    print()


if __name__ == "__main__":
    main()
