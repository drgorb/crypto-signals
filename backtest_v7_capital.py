"""Compounding capital backtest for CryptoSignals v7.

Quality over quantity: strict filters, wider stops, 15m only.
Compares taker vs maker fee scenarios.

Period: May 1, 2025 → Feb 10, 2026
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from indicators_v6 import compute_all_v6
from indicators import sma as compute_sma
from signals_v7 import generate_signals_v7
from config_v7 import (
    SYMBOLS, SYMBOL_NAMES, TRAILING_15M,
    V7_MAX_CONCURRENT_PER_ASSET, V7_SL_COOLDOWN_CANDLES,
    V7_HTF_SMA_PERIOD,
    TAKER_FEE_PCT, MAKER_FEE_PCT, SLIPPAGE_PCT,
)

WARMUP = 55

# Dates
START_MS = 1746057600000   # May 1, 2025
WARMUP_START_MS = START_MS - 30 * 86400 * 1000
END_MS = 1770768000000     # Feb 10, 2026

STARTING_CAPITAL_PER_ASSET = 10000.0


def load_data():
    """Load 15m and 1h data from cache."""
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    data = {}
    warmup_ts = pd.Timestamp(WARMUP_START_MS, unit="ms")
    end_ts = pd.Timestamp(END_MS, unit="ms")

    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        data[sym] = {}
        for tf in ["15m", "1h"]:
            path = os.path.join(cache_dir, f"{sym}_{tf}.csv")
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df = df[(df["timestamp"] >= warmup_ts) & (df["timestamp"] <= end_ts)].reset_index(drop=True)
            print(f"  {display} {tf}: {len(df)} candles")
            data[sym][tf] = df
    return data


def compute_indicators(data):
    """Compute indicators for 15m and 1h data."""
    computed = {}
    for sym in SYMBOLS:
        computed[sym] = {}
        # 15m: full v6 indicators (BB, RSI, vol, SMA, EMA, MACD, ATR, ADX)
        df_15m = data[sym]["15m"].copy()
        if len(df_15m) > WARMUP:
            computed[sym]["15m"] = compute_all_v6(df_15m)
        else:
            computed[sym]["15m"] = None

        # 1h: just need SMA-50 for HTF confirmation
        df_1h = data[sym]["1h"].copy()
        if len(df_1h) > V7_HTF_SMA_PERIOD:
            computed[sym]["1h"] = compute_sma(df_1h, V7_HTF_SMA_PERIOD)
        else:
            computed[sym]["1h"] = None
    return computed


def simulate_trailing_stop(df, entry_idx, direction, entry_price, trail_cfg):
    """Walk candle-by-candle with v7 wider trailing stops."""
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

        # Update trailing stop
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


def collect_raw_trades(computed):
    """Scan all candles and collect potential trades with pre-computed exits."""
    start_ts = pd.Timestamp(START_MS, unit="ms")
    raw_trades = []

    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        df = computed[sym]["15m"]
        htf_df = computed[sym]["1h"]
        if df is None:
            continue

        candidates = df[df["timestamp"] >= start_ts].index
        if len(candidates) == 0:
            continue
        tf_start_idx = max(candidates[0], WARMUP)
        total_candles = len(df) - tf_start_idx
        print(f"  Scanning {display} 15m: {total_candles} candles...", flush=True)
        trade_count = 0

        for idx_counter, i in enumerate(range(tf_start_idx, len(df))):
            if idx_counter % 5000 == 0 and idx_counter > 0:
                print(f"    ...{idx_counter}/{total_candles} ({trade_count} trades)", flush=True)

            current_ts = df.iloc[i]["timestamp"]

            # Get 1h slice up to current time for HTF confirmation
            htf_slice = None
            if htf_df is not None:
                htf_idx = htf_df["timestamp"].searchsorted(current_ts, side="right")
                if htf_idx > V7_HTF_SMA_PERIOD:
                    htf_slice = htf_df.iloc[:htf_idx]

            lookback_start = max(0, i - 200)
            df_slice = df.iloc[lookback_start:i + 1]

            signals = generate_signals_v7(sym, df_slice, "15m", htf_slice)

            for sig in signals:
                exit_idx, exit_price, gross_pnl, outcome = simulate_trailing_stop(
                    df, i, sig["type"], sig["price"], TRAILING_15M)

                trade_count += 1
                raw_trades.append({
                    "symbol": sym,
                    "display": display,
                    "direction": sig["type"],
                    "conviction": sig["conviction"],
                    "entry_idx": i,
                    "exit_idx": exit_idx,
                    "entry_time": df.iloc[i]["timestamp"],
                    "exit_time": df.iloc[exit_idx]["timestamp"],
                    "entry_price": sig["price"],
                    "exit_price": exit_price,
                    "gross_pnl_pct": gross_pnl,
                    "outcome": outcome,
                    "candles_held": exit_idx - i,
                    "leverage": sig["leverage"],
                    "rsi": sig["rsi"],
                    "adx": sig["adx"],
                    "vol_ratio": sig["vol_ratio"],
                })

        print(f"    → {trade_count} trades from {display}", flush=True)

    raw_trades.sort(key=lambda t: t["entry_time"])
    print(f"  Total raw trades: {len(raw_trades)}", flush=True)
    return raw_trades


def simulate_capital(raw_trades, fee_per_side, label=""):
    """Run capital simulation with given fee model."""
    capital = {"BTCUSDT": STARTING_CAPITAL_PER_ASSET, "ETHUSDT": STARTING_CAPITAL_PER_ASSET}
    all_trades = []
    equity_snapshots = []
    total_fees = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}

    # Track open positions per asset
    open_pos = {"BTCUSDT": None, "ETHUSDT": None}  # only 1 at a time
    sl_cooldowns = {"BTCUSDT": {}, "ETHUSDT": {}}

    for trade in raw_trades:
        sym = trade["symbol"]
        direction = trade["direction"]
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]

        # Clear expired position
        if open_pos[sym] is not None and open_pos[sym]["exit_time"] <= entry_time:
            open_pos[sym] = None

        # Skip if position open
        if open_pos[sym] is not None:
            continue

        # Check cooldown
        cooldown_until = sl_cooldowns[sym].get(direction)
        if cooldown_until is not None and entry_time < cooldown_until:
            continue

        available = capital[sym]
        if available <= 0:
            continue

        leverage = trade["leverage"]
        notional = available * leverage

        # Slippage on entry/exit
        if direction == "BUY":
            eff_entry = trade["entry_price"] * (1 + SLIPPAGE_PCT)
            eff_exit = trade["exit_price"] * (1 - SLIPPAGE_PCT)
            net_pnl_pct = (eff_exit - eff_entry) / eff_entry
        else:
            eff_entry = trade["entry_price"] * (1 - SLIPPAGE_PCT)
            eff_exit = trade["exit_price"] * (1 + SLIPPAGE_PCT)
            net_pnl_pct = (eff_entry - eff_exit) / eff_entry

        # Fees
        entry_fee = notional * fee_per_side
        exit_notional = notional * (1 + net_pnl_pct)
        exit_fee = exit_notional * fee_per_side
        total_fee = entry_fee + exit_fee

        net_dollar_pnl = notional * net_pnl_pct - total_fee
        gross_dollar_pnl = notional * trade["gross_pnl_pct"]

        capital[sym] += net_dollar_pnl
        total_fees[sym] += total_fee

        open_pos[sym] = {"exit_time": exit_time, "direction": direction}

        if trade["outcome"] == "SL":
            cooldown_delta = pd.Timedelta(minutes=15 * V7_SL_COOLDOWN_CANDLES)
            sl_cooldowns[sym][direction] = exit_time + cooldown_delta

        all_trades.append({
            **trade,
            "capital_before": capital[sym] - net_dollar_pnl,
            "capital_after": capital[sym],
            "notional": round(notional, 2),
            "gross_dollar_pnl": round(gross_dollar_pnl, 2),
            "net_dollar_pnl": round(net_dollar_pnl, 2),
            "fees_paid": round(total_fee, 2),
            "net_pnl_pct": round(net_pnl_pct * 100, 4),
        })

        equity_snapshots.append({
            "timestamp": exit_time,
            "btc_capital": capital["BTCUSDT"],
            "eth_capital": capital["ETHUSDT"],
            "total": capital["BTCUSDT"] + capital["ETHUSDT"],
        })

    return all_trades, equity_snapshots, capital, total_fees


def print_results(trades, snapshots, final_capital, total_fees, fee_label):
    total_starting = STARTING_CAPITAL_PER_ASSET * 2
    total_ending = final_capital["BTCUSDT"] + final_capital["ETHUSDT"]
    total_return = (total_ending - total_starting) / total_starting * 100

    print(f"\n{'='*90}")
    print(f"  CryptoSignals v7 — {fee_label}")
    print(f"  Period: May 1, 2025 → Feb 10, 2026")
    print(f"{'='*90}")

    if not trades:
        print("  ⚠ No trades generated!")
        return

    tdf = pd.DataFrame(trades)

    # Capital summary
    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  CAPITAL SUMMARY                                         │")
    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │  BTC: ${STARTING_CAPITAL_PER_ASSET:>10,.2f} → ${final_capital['BTCUSDT']:>10,.2f}  ({(final_capital['BTCUSDT']/STARTING_CAPITAL_PER_ASSET-1)*100:+.2f}%)   │")
    print(f"  │  ETH: ${STARTING_CAPITAL_PER_ASSET:>10,.2f} → ${final_capital['ETHUSDT']:>10,.2f}  ({(final_capital['ETHUSDT']/STARTING_CAPITAL_PER_ASSET-1)*100:+.2f}%)   │")
    print(f"  │  ────────────────────────────────────────────────────    │")
    print(f"  │  TOTAL: ${total_starting:>10,.2f} → ${total_ending:>10,.2f}  ({total_return:+.2f}%)   │")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # Trade stats
    total = len(tdf)
    wins = int((tdf["net_dollar_pnl"] > 0).sum())
    losses = total - wins
    wr = wins / total * 100 if total > 0 else 0

    days_in_period = (pd.Timestamp(END_MS, unit="ms") - pd.Timestamp(START_MS, unit="ms")).days
    trades_per_day = total / days_in_period if days_in_period > 0 else 0

    fees_total = total_fees["BTCUSDT"] + total_fees["ETHUSDT"]
    gross_pnl = tdf["gross_dollar_pnl"].sum()
    net_pnl = tdf["net_dollar_pnl"].sum()

    print(f"\n  TRADE STATISTICS")
    print(f"    Total trades:       {total}")
    print(f"    Trades/day:         {trades_per_day:.1f}")
    print(f"    Wins:               {wins}")
    print(f"    Losses:             {losses}")
    print(f"    Win rate:           {wr:.1f}%")
    if wins:
        print(f"    Avg winner:         ${tdf[tdf['net_dollar_pnl']>0]['net_dollar_pnl'].mean():+.2f}")
    if losses:
        print(f"    Avg loser:          ${tdf[tdf['net_dollar_pnl']<=0]['net_dollar_pnl'].mean():+.2f}")
    print(f"    Avg trade:          ${tdf['net_dollar_pnl'].mean():+.2f}")
    if wins and losses:
        avg_w = tdf[tdf['net_dollar_pnl']>0]['net_dollar_pnl'].mean()
        avg_l = abs(tdf[tdf['net_dollar_pnl']<=0]['net_dollar_pnl'].mean())
        print(f"    Profit factor:      {(avg_w * wins) / (avg_l * losses):.2f}" if avg_l > 0 else "")

    print(f"\n  GROSS vs NET")
    print(f"    Gross P&L:          ${gross_pnl:+,.2f}")
    print(f"    Fees paid:          ${fees_total:,.2f}")
    print(f"    Net P&L:            ${net_pnl:+,.2f}")
    print(f"    Fee drag:           {fees_total/total_starting*100:.2f}% of starting capital")

    # Max drawdown
    sdf = pd.DataFrame(snapshots)
    if len(sdf) > 0:
        peak = sdf["total"].cummax()
        drawdown = sdf["total"] - peak
        max_dd_dollar = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        peak_at_dd = peak.iloc[max_dd_idx]
        max_dd_pct = max_dd_dollar / peak_at_dd * 100

        print(f"\n  MAX DRAWDOWN")
        print(f"    Dollar:             ${max_dd_dollar:,.2f}")
        print(f"    Percentage:         {max_dd_pct:.2f}%")

    # By conviction
    print(f"\n  BY SIGNAL TYPE")
    for conv in ["STRONG", "TREND_FOLLOW"]:
        sub = tdf[tdf["conviction"] == conv]
        if len(sub) == 0:
            continue
        sw = int((sub["net_dollar_pnl"] > 0).sum())
        print(f"    {conv:<14} {len(sub):>4} trades | WR: {sw/len(sub)*100:.1f}% | Net: ${sub['net_dollar_pnl'].sum():+,.2f} | Avg: ${sub['net_dollar_pnl'].mean():+.2f}")

    # By asset
    print(f"\n  BY ASSET")
    for sym_name in ["BTC/USDT", "ETH/USDT"]:
        sub = tdf[tdf["display"] == sym_name]
        if len(sub) == 0:
            continue
        sw = int((sub["net_dollar_pnl"] > 0).sum())
        print(f"    {sym_name:<10} {len(sub):>4} trades | WR: {sw/len(sub)*100:.1f}% | Net: ${sub['net_dollar_pnl'].sum():+,.2f}")

    # By outcome
    print(f"\n  BY OUTCOME")
    for outcome in ["SL", "TRAIL", "TIMEOUT"]:
        sub = tdf[tdf["outcome"] == outcome]
        if len(sub) == 0:
            continue
        sw = int((sub["net_dollar_pnl"] > 0).sum())
        print(f"    {outcome:<10} {len(sub):>4} trades | WR: {sw/len(sub)*100:.1f}% | Net: ${sub['net_dollar_pnl'].sum():+,.2f}")

    # Monthly breakdown
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["month"] = tdf["entry_time"].dt.to_period("M")

    print(f"\n  MONTHLY P&L")
    print(f"  {'Month':<10} {'Trades':>7} {'WR':>7} {'Net $':>12} {'Net %':>8} {'Fees $':>10} {'BTC':>12} {'ETH':>12} {'Total':>12}")
    print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    for month, grp in tdf.groupby("month"):
        m_trades = len(grp)
        m_wins = int((grp["net_dollar_pnl"] > 0).sum())
        m_wr = m_wins / m_trades * 100
        m_net = grp["net_dollar_pnl"].sum()
        m_fees = grp["fees_paid"].sum()
        # Get end-of-month equity
        month_snaps = [s for s in snapshots if pd.Timestamp(s["timestamp"]).to_period("M") == month]
        if month_snaps:
            ls = month_snaps[-1]
            btc_c, eth_c, tot_c = ls["btc_capital"], ls["eth_capital"], ls["total"]
        else:
            btc_c = eth_c = tot_c = 0
        # Net % relative to start-of-month capital
        start_cap = tot_c - m_net
        m_pct = m_net / start_cap * 100 if start_cap > 0 else 0
        print(f"  {str(month):<10} {m_trades:>7} {m_wr:>6.1f}% ${m_net:>+10,.2f} {m_pct:>+7.2f}% ${m_fees:>8,.2f} ${btc_c:>10,.2f} ${eth_c:>10,.2f} ${tot_c:>10,.2f}")

    # Sample trades
    print(f"\n  SAMPLE TRADES (first 20):")
    print(f"  {'Time':<20} {'Dir':4} {'Type':<12} {'Asset':<8} {'Lev':>4} {'Entry':>10} {'Exit':>10} {'Gross$':>9} {'Fees$':>8} {'Net$':>9} {'Out':>5} {'Held':>5}")
    for _, t in tdf.head(20).iterrows():
        print(f"  {str(t['entry_time'])[:16]:<20} {t['direction']:4} {t['conviction']:<12} {t['display']:<8} {t['leverage']:>3.1f}x "
              f"${t['entry_price']:>9,.1f} ${t['exit_price']:>9,.1f} ${t['gross_dollar_pnl']:>+7,.0f} ${t['fees_paid']:>6,.0f} ${t['net_dollar_pnl']:>+7,.0f} {t['outcome']:>5} {t['candles_held']:>5}")

    print()


def main():
    print(f"\n{'='*90}")
    print(f"  CryptoSignals v7 COMPOUNDING CAPITAL BACKTEST")
    print(f"  May 1, 2025 → Feb 10, 2026 | $20,000 starting capital")
    print(f"  15m only | Strict filters | Wider stops | Quality > Quantity")
    print(f"{'='*90}")

    print(f"\n  Loading data...")
    data = load_data()

    print(f"\n  Computing indicators...")
    computed = compute_indicators(data)

    print(f"\n  Collecting raw trades...")
    raw_trades = collect_raw_trades(computed)

    if not raw_trades:
        print("\n  ⚠ No raw trades found! Filters may be too strict.")
        return

    # === TAKER FEE SCENARIO ===
    print(f"\n  Running TAKER fee scenario (0.075% per side + 0.03% slippage)...")
    trades_t, snaps_t, cap_t, fees_t = simulate_capital(raw_trades, TAKER_FEE_PCT, "taker")
    print_results(trades_t, snaps_t, cap_t, fees_t, "TAKER FEES (0.075%/side + 0.03% slippage = 0.21% RT)")

    # === MAKER FEE SCENARIO ===
    print(f"\n  Running MAKER fee scenario (0.02% per side + 0.03% slippage)...")
    trades_m, snaps_m, cap_m, fees_m = simulate_capital(raw_trades, MAKER_FEE_PCT, "maker")
    print_results(trades_m, snaps_m, cap_m, fees_m, "MAKER FEES (0.02%/side + 0.03% slippage = 0.10% RT)")

    # === COMPARISON ===
    total_starting = STARTING_CAPITAL_PER_ASSET * 2
    t_end = cap_t["BTCUSDT"] + cap_t["ETHUSDT"]
    m_end = cap_m["BTCUSDT"] + cap_m["ETHUSDT"]
    t_ret = (t_end - total_starting) / total_starting * 100
    m_ret = (m_end - total_starting) / total_starting * 100
    t_fees = fees_t["BTCUSDT"] + fees_t["ETHUSDT"]
    m_fees = fees_m["BTCUSDT"] + fees_m["ETHUSDT"]

    print(f"\n{'='*90}")
    print(f"  TAKER vs MAKER COMPARISON")
    print(f"{'='*90}")
    print(f"  {'':20} {'TAKER':>15} {'MAKER':>15} {'Difference':>15}")
    print(f"  {'Final Capital':20} ${t_end:>13,.2f} ${m_end:>13,.2f} ${m_end-t_end:>+13,.2f}")
    print(f"  {'Total Return':20} {t_ret:>14.2f}% {m_ret:>14.2f}% {m_ret-t_ret:>+14.2f}%")
    print(f"  {'Total Fees':20} ${t_fees:>13,.2f} ${m_fees:>13,.2f} ${t_fees-m_fees:>+13,.2f}")
    print(f"  {'Trades':20} {len(trades_t):>15} {len(trades_m):>15}")
    print()


if __name__ == "__main__":
    main()
