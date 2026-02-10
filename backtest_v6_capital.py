"""Compounding capital backtest for CryptoSignals v6.

Simulates realistic portfolio growth with fees, slippage, leverage for STRONG signals,
and capital splitting for concurrent positions.

Period: May 1, 2025 → Feb 10, 2026
"""

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

# Dates
START_MS = 1746057600000  # May 1, 2025
WARMUP_START_MS = START_MS - 30 * 86400 * 1000
DAILY_WARMUP_MS = START_MS - 120 * 86400 * 1000
END_MS = 1770768000000  # Feb 10, 2026 ~23:59 UTC

# Capital config
STARTING_CAPITAL_PER_ASSET = 10000.0
FEE_PCT = 0.001        # 0.1% per side
SLIPPAGE_PCT = 0.0005  # 0.05% per side


def fetch_all_data():
    """Load data directly from cache CSV files."""
    import os
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    data = {}
    warmup_ts = pd.Timestamp(WARMUP_START_MS, unit="ms")
    end_ts = pd.Timestamp(END_MS, unit="ms")
    
    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        print(f"  Loading {display} from cache...")
        data[sym] = {}
        for tf in ["15m", "5m"]:
            path = os.path.join(cache_dir, f"{sym}_{tf}.csv")
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df = df[(df["timestamp"] >= warmup_ts) & (df["timestamp"] <= end_ts)].reset_index(drop=True)
            print(f"    {tf}: {len(df)} candles")
            data[sym][tf] = df
    return data


def simulate_trailing_stop(df, entry_idx, direction, entry_price, trail_cfg):
    """Same trailing stop logic as the original backtest."""
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


def run_capital_backtest(data, computed):
    """Run backtest with compounding capital tracking."""
    
    # Capital per asset
    capital = {"BTCUSDT": STARTING_CAPITAL_PER_ASSET, "ETHUSDT": STARTING_CAPITAL_PER_ASSET}
    
    # Track all trades with capital info
    all_trades = []
    # Equity snapshots: list of (timestamp, btc_equity, eth_equity)
    equity_snapshots = []
    # Track total fees
    total_fees = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    total_slippage = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    
    start_ts = pd.Timestamp(START_MS, unit="ms")
    
    # We need to process trades chronologically across all symbols/timeframes
    # First, collect all potential trades with their entry times
    raw_trades = []
    
    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        
        for tf, trail_cfg in [("15m", TRAILING_15M), ("5m", TRAILING_5M)]:
            if tf not in computed[sym] or computed[sym][tf] is None:
                continue
            
            df = computed[sym][tf]
            candidates = df[df["timestamp"] >= start_ts].index
            if len(candidates) == 0:
                continue
            tf_start_idx = max(candidates[0], WARMUP)
            total_candles = len(df) - tf_start_idx
            print(f"  Scanning {display} {tf}: {total_candles} candles...", flush=True)
            trade_count = 0
            
            for idx_counter, i in enumerate(range(tf_start_idx, len(df))):
                if idx_counter % 10000 == 0 and idx_counter > 0:
                    print(f"    ...{idx_counter}/{total_candles} ({trade_count} trades so far)", flush=True)
                
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
                    # Pre-compute exit
                    exit_idx, exit_price, gross_pnl, outcome = simulate_trailing_stop(
                        df, i, sig["type"], sig["price"], trail_cfg)
                    
                    trade_count += 1
                    raw_trades.append({
                        "symbol": sym,
                        "display": display,
                        "direction": sig["type"],
                        "conviction": sig["conviction"],
                        "timeframe": tf,
                        "entry_idx": i,
                        "exit_idx": exit_idx,
                        "entry_time": df.iloc[i]["timestamp"],
                        "exit_time": df.iloc[exit_idx]["timestamp"],
                        "entry_price": sig["price"],
                        "exit_price": exit_price,
                        "gross_pnl_pct": gross_pnl,
                        "outcome": outcome,
                        "candles_held": exit_idx - i,
                        "position_size": sig["position_size"],
                        "rsi": sig["rsi"],
                        "adx": sig["adx"],
                        "vol_ratio": sig["vol_ratio"],
                    })
            
            print(f"    → {trade_count} trades from {display} {tf}", flush=True)
    
    print(f"  Total raw trades: {len(raw_trades)}", flush=True)
    # Sort by entry time
    raw_trades.sort(key=lambda t: t["entry_time"])
    
    # Now simulate capital with position management
    # Track open positions per asset: list of {exit_time, capital_allocated}
    open_positions = {"BTCUSDT": [], "ETHUSDT": []}
    sl_cooldowns = {"BTCUSDT": {}, "ETHUSDT": {}}  # direction -> cooldown_until (timestamp)
    
    for trade in raw_trades:
        sym = trade["symbol"]
        direction = trade["direction"]
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        conviction = trade["conviction"]
        
        # Clean up expired positions
        open_positions[sym] = [p for p in open_positions[sym] if p["exit_time"] > entry_time]
        
        # Check max concurrent
        if len(open_positions[sym]) >= V6_MAX_CONCURRENT_PER_ASSET:
            continue
        
        # Check cooldown
        cooldown_until = sl_cooldowns[sym].get(direction)
        if cooldown_until is not None and entry_time < cooldown_until:
            continue
        
        # Check conflicting direction
        opp = "SELL" if direction == "BUY" else "BUY"
        has_conflict = any(p.get("direction") == opp for p in open_positions[sym])
        if has_conflict:
            continue
        
        # Determine capital to use
        available_capital = capital[sym]
        # If there's already an open position, we split capital
        # But the open position already locked some capital, so available = total - locked
        locked = sum(p["capital_allocated"] for p in open_positions[sym])
        available_capital = capital[sym] - locked
        
        if available_capital <= 0:
            continue
        
        # Position sizing: STRONG = 2x leverage
        leverage = 2.0 if conviction == "STRONG" else 1.0
        notional = available_capital * leverage
        
        # Apply slippage on entry
        if direction == "BUY":
            effective_entry = trade["entry_price"] * (1 + SLIPPAGE_PCT)
        else:
            effective_entry = trade["entry_price"] * (1 - SLIPPAGE_PCT)
        
        # Apply slippage on exit
        if direction == "BUY":
            effective_exit = trade["exit_price"] * (1 - SLIPPAGE_PCT)
        else:
            effective_exit = trade["exit_price"] * (1 + SLIPPAGE_PCT)
        
        # Compute net PnL after slippage
        if direction == "BUY":
            net_pnl_pct = (effective_exit - effective_entry) / effective_entry
        else:
            net_pnl_pct = (effective_entry - effective_exit) / effective_entry
        
        # Fees: 0.1% on entry notional + 0.1% on exit notional
        entry_fee = notional * FEE_PCT
        exit_notional = notional * (1 + net_pnl_pct)
        exit_fee = exit_notional * FEE_PCT
        total_fee = entry_fee + exit_fee
        
        # Dollar PnL from the trade (on notional)
        gross_dollar_pnl = notional * trade["gross_pnl_pct"]
        net_dollar_pnl = notional * net_pnl_pct - total_fee
        
        # Update capital
        capital[sym] += net_dollar_pnl
        total_fees[sym] += total_fee
        total_slippage[sym] += notional * SLIPPAGE_PCT * 2  # entry + exit slippage
        
        # Record
        open_positions[sym].append({
            "exit_time": exit_time,
            "capital_allocated": available_capital,
            "direction": direction,
        })
        
        if trade["outcome"] == "SL":
            # Approximate cooldown: use exit_time + some candles worth of time
            if trade["timeframe"] == "15m":
                cooldown_delta = pd.Timedelta(minutes=15 * V6_SL_COOLDOWN_CANDLES)
            else:
                cooldown_delta = pd.Timedelta(minutes=5 * V6_SL_COOLDOWN_CANDLES)
            sl_cooldowns[sym][direction] = exit_time + cooldown_delta
        
        all_trades.append({
            **trade,
            "capital_before": capital[sym] - net_dollar_pnl,
            "capital_after": capital[sym],
            "notional": round(notional, 2),
            "leverage": leverage,
            "gross_dollar_pnl": round(gross_dollar_pnl, 2),
            "net_dollar_pnl": round(net_dollar_pnl, 2),
            "fees_paid": round(total_fee, 2),
            "net_pnl_pct": round(net_pnl_pct * 100, 4),
        })
        
        # Snapshot equity at exit time
        equity_snapshots.append({
            "timestamp": exit_time,
            "btc_capital": capital["BTCUSDT"],
            "eth_capital": capital["ETHUSDT"],
            "total": capital["BTCUSDT"] + capital["ETHUSDT"],
        })
    
    return all_trades, equity_snapshots, capital, total_fees, total_slippage


def print_results(trades, snapshots, final_capital, total_fees, total_slippage):
    if not trades:
        print("\n  ⚠ No trades generated!")
        return
    
    tdf = pd.DataFrame(trades)
    total_starting = STARTING_CAPITAL_PER_ASSET * 2
    total_ending = final_capital["BTCUSDT"] + final_capital["ETHUSDT"]
    total_return = (total_ending - total_starting) / total_starting * 100
    
    print(f"\n{'='*90}")
    print(f"  CryptoSignals v6 COMPOUNDING CAPITAL BACKTEST")
    print(f"  Period: May 1, 2025 → Feb 10, 2026")
    print(f"{'='*90}")
    
    # === CAPITAL SUMMARY ===
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  CAPITAL SUMMARY                                    │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  BTC: ${STARTING_CAPITAL_PER_ASSET:>10,.2f} → ${final_capital['BTCUSDT']:>10,.2f}  ({(final_capital['BTCUSDT']/STARTING_CAPITAL_PER_ASSET-1)*100:+.2f}%)  │")
    print(f"  │  ETH: ${STARTING_CAPITAL_PER_ASSET:>10,.2f} → ${final_capital['ETHUSDT']:>10,.2f}  ({(final_capital['ETHUSDT']/STARTING_CAPITAL_PER_ASSET-1)*100:+.2f}%)  │")
    print(f"  │  ─────────────────────────────────────────────────  │")
    print(f"  │  TOTAL: ${total_starting:>10,.2f} → ${total_ending:>10,.2f}  ({total_return:+.2f}%)  │")
    print(f"  └─────────────────────────────────────────────────────┘")
    
    # === TRADE STATS ===
    total = len(tdf)
    wins = int((tdf["net_dollar_pnl"] > 0).sum())
    losses = total - wins
    wr = wins / total * 100 if total > 0 else 0
    
    gross_pnl = tdf["gross_dollar_pnl"].sum()
    net_pnl = tdf["net_dollar_pnl"].sum()
    fees = total_fees["BTCUSDT"] + total_fees["ETHUSDT"]
    slippage = total_slippage["BTCUSDT"] + total_slippage["ETHUSDT"]
    
    print(f"\n  TRADE STATISTICS")
    print(f"    Total trades:     {total}")
    print(f"    Wins:             {wins}")
    print(f"    Losses:           {losses}")
    print(f"    Win rate:         {wr:.1f}%")
    print(f"    Avg trade P&L:    ${tdf['net_dollar_pnl'].mean():+.2f}")
    print(f"    Avg winner:       ${tdf[tdf['net_dollar_pnl']>0]['net_dollar_pnl'].mean():+.2f}" if wins else "")
    print(f"    Avg loser:        ${tdf[tdf['net_dollar_pnl']<=0]['net_dollar_pnl'].mean():+.2f}" if losses else "")
    
    print(f"\n  GROSS vs NET P&L")
    print(f"    Gross P&L:        ${gross_pnl:+,.2f}")
    print(f"    Total fees:       ${fees:,.2f}")
    print(f"    Total slippage:   ${slippage:,.2f}")
    print(f"    Net P&L:          ${net_pnl:+,.2f}")
    print(f"    Fee+slip drag:    ${fees+slippage:,.2f} ({(fees+slippage)/total_starting*100:.2f}% of starting capital)")
    
    # === MAX DRAWDOWN ===
    sdf = pd.DataFrame(snapshots)
    if len(sdf) > 0:
        peak = sdf["total"].cummax()
        drawdown = sdf["total"] - peak
        max_dd_dollar = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        peak_at_dd = peak.iloc[max_dd_idx]
        max_dd_pct = max_dd_dollar / peak_at_dd * 100
        
        print(f"\n  MAX DRAWDOWN")
        print(f"    Dollar:           ${max_dd_dollar:,.2f}")
        print(f"    Percentage:       {max_dd_pct:.2f}%")
        print(f"    Peak equity:      ${peak_at_dd:,.2f}")
    
    # === BY CONVICTION ===
    print(f"\n  BY SIGNAL TYPE")
    for conv in ["STANDARD", "STRONG", "TREND_FOLLOW"]:
        sub = tdf[tdf["conviction"] == conv]
        if len(sub) == 0:
            continue
        sw = int((sub["net_dollar_pnl"] > 0).sum())
        print(f"    {conv:<14} {len(sub):>4} trades | WR: {sw/len(sub)*100:.1f}% | Net: ${sub['net_dollar_pnl'].sum():+,.2f} | Avg: ${sub['net_dollar_pnl'].mean():+.2f}")
    
    # === BY ASSET ===
    print(f"\n  BY ASSET")
    for sym_name in ["BTC/USDT", "ETH/USDT"]:
        sub = tdf[tdf["display"] == sym_name]
        if len(sub) == 0:
            continue
        sw = int((sub["net_dollar_pnl"] > 0).sum())
        print(f"    {sym_name:<10} {len(sub):>4} trades | WR: {sw/len(sub)*100:.1f}% | Net: ${sub['net_dollar_pnl'].sum():+,.2f}")
    
    # === MONTHLY ===
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["month"] = tdf["entry_time"].dt.to_period("M")
    
    print(f"\n  MONTHLY EQUITY SNAPSHOTS")
    print(f"  {'Month':<10} {'Trades':>7} {'WR':>7} {'Gross $':>12} {'Net $':>12} {'Fees $':>10} {'BTC Cap':>12} {'ETH Cap':>12} {'Total':>12}")
    print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    
    monthly_net = []
    for month, grp in tdf.groupby("month"):
        m_trades = len(grp)
        m_wins = int((grp["net_dollar_pnl"] > 0).sum())
        m_wr = m_wins / m_trades * 100
        m_gross = grp["gross_dollar_pnl"].sum()
        m_net = grp["net_dollar_pnl"].sum()
        m_fees = grp["fees_paid"].sum()
        # Get end-of-month capital from last trade that month
        last_trade = grp.iloc[-1]
        # Find snapshot closest to end of month
        month_snapshots = [s for s in snapshots if pd.Timestamp(s["timestamp"]).to_period("M") == month]
        if month_snapshots:
            last_snap = month_snapshots[-1]
            btc_cap = last_snap["btc_capital"]
            eth_cap = last_snap["eth_capital"]
            total_cap = last_snap["total"]
        else:
            btc_cap = eth_cap = total_cap = 0
        
        monthly_net.append((str(month), m_net))
        print(f"  {str(month):<10} {m_trades:>7} {m_wr:>6.1f}% ${m_gross:>+10,.2f} ${m_net:>+10,.2f} ${m_fees:>8,.2f} ${btc_cap:>10,.2f} ${eth_cap:>10,.2f} ${total_cap:>10,.2f}")
    
    if monthly_net:
        best_month = max(monthly_net, key=lambda x: x[1])
        worst_month = min(monthly_net, key=lambda x: x[1])
        print(f"\n  Best month:  {best_month[0]}  ${best_month[1]:+,.2f}")
        print(f"  Worst month: {worst_month[0]}  ${worst_month[1]:+,.2f}")
    
    # === SAMPLE TRADES ===
    print(f"\n  SAMPLE TRADES (first 15):")
    print(f"  {'Dir':4} {'Type':<12} {'Asset':<10} {'TF':3} {'Lev':>4} {'Notional':>10} {'Gross$':>10} {'Fees$':>8} {'Net$':>10} {'Outcome':>7}")
    for _, t in tdf.head(15).iterrows():
        print(f"  {t['direction']:4} {t['conviction']:<12} {t['display']:<10} {t['timeframe']:3} {t['leverage']:>3.0f}x ${t['notional']:>9,.0f} ${t['gross_dollar_pnl']:>+8,.2f} ${t['fees_paid']:>7,.2f} ${t['net_dollar_pnl']:>+8,.2f} {t['outcome']:>7}")
    
    print()


def main():
    print(f"\n{'='*90}")
    print(f"  CryptoSignals v6 COMPOUNDING CAPITAL BACKTEST")
    print(f"  May 1, 2025 → Feb 10, 2026 | $20,000 starting capital")
    print(f"{'='*90}")
    
    print(f"\n  Fetching data...")
    data = fetch_all_data()
    
    # Compute indicators
    computed = {}
    for sym in SYMBOLS:
        computed[sym] = {}
        for tf in ["15m", "5m"]:
            if len(data[sym].get(tf, pd.DataFrame())) > WARMUP:
                print(f"  Computing indicators: {SYMBOL_NAMES[sym]} {tf} ({len(data[sym][tf])} candles)...")
                computed[sym][tf] = compute_all_v6(data[sym][tf].copy())
            else:
                computed[sym][tf] = None
    
    print(f"\n  Running capital simulation...")
    trades, snapshots, final_capital, total_fees, total_slippage = run_capital_backtest(data, computed)
    
    print_results(trades, snapshots, final_capital, total_fees, total_slippage)


if __name__ == "__main__":
    main()
