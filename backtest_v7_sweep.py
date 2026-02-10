"""Parameter sweep for CryptoSignals v7 — test 6 configurations side by side.

Period: May 1, 2025 → Feb 10, 2026 | $20k starting capital | Maker fees
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from indicators_v6 import compute_all_v6
from indicators import sma as compute_sma
from config_v7 import (
    SYMBOLS, SYMBOL_NAMES, TRAILING_15M,
    V7_MAX_CONCURRENT_PER_ASSET, V7_SL_COOLDOWN_CANDLES,
    V7_HTF_SMA_PERIOD, EMA_TREND_PERIOD,
    MAKER_FEE_PCT, SLIPPAGE_PCT,
)

WARMUP = 55
START_MS = 1746057600000   # May 1, 2025
WARMUP_START_MS = START_MS - 30 * 86400 * 1000
END_MS = 1770768000000     # Feb 10, 2026
STARTING_CAPITAL_PER_ASSET = 10000.0
FEE_PER_SIDE = MAKER_FEE_PCT  # 0.02%

# ===================== CONFIG DEFINITIONS =====================

CONFIGS = {
    "1_baseline": {
        "label": "1. Current loosened",
        "rsi_oversold": 30, "rsi_overbought": 70,
        "vol_threshold": 1.5, "adx_threshold": 30,
        "ema_pullback_pct": 0.003, "min_atr_pct": 0.015,
        "signal_types": ["STRONG", "TREND_FOLLOW"],
        "timeframes": ["15m"],
        "trailing": TRAILING_15M,
        "strong_leverage": 1.5, "trend_leverage": 1.0,
    },
    "2_looser": {
        "label": "2. Much looser",
        "rsi_oversold": 35, "rsi_overbought": 65,
        "vol_threshold": 1.2, "adx_threshold": 25,
        "ema_pullback_pct": 0.003, "min_atr_pct": 0.015,
        "signal_types": ["STRONG", "TREND_FOLLOW"],
        "timeframes": ["15m"],
        "trailing": TRAILING_15M,
        "strong_leverage": 1.5, "trend_leverage": 1.0,
    },
    "3_looser_1h": {
        "label": "3. Looser + 1h signals",
        "rsi_oversold": 35, "rsi_overbought": 65,
        "vol_threshold": 1.2, "adx_threshold": 25,
        "ema_pullback_pct": 0.003, "min_atr_pct": 0.015,
        "signal_types": ["STRONG", "TREND_FOLLOW"],
        "timeframes": ["15m", "1h"],
        "trailing": TRAILING_15M,  # same trailing for now
        "strong_leverage": 1.5, "trend_leverage": 1.0,
    },
    "4_trend_only": {
        "label": "4. Trend-follow only",
        "rsi_oversold": 40, "rsi_overbought": 60,
        "vol_threshold": 1.2, "adx_threshold": 25,
        "ema_pullback_pct": 0.003, "min_atr_pct": 0.015,
        "signal_types": ["TREND_FOLLOW"],
        "timeframes": ["15m"],
        "trailing": TRAILING_15M,
        "strong_leverage": 1.5, "trend_leverage": 1.0,
    },
    "5_trend_wider_trail": {
        "label": "5. Trend + wider trail",
        "rsi_oversold": 40, "rsi_overbought": 60,
        "vol_threshold": 1.2, "adx_threshold": 25,
        "ema_pullback_pct": 0.003, "min_atr_pct": 0.015,
        "signal_types": ["TREND_FOLLOW"],
        "timeframes": ["15m"],
        "trailing": {
            **TRAILING_15M,
            "breakeven_trigger": 0.01,  # +1% instead of +1.5%
        },
        "strong_leverage": 1.5, "trend_leverage": 1.0,
    },
    "6_multi_tf_trend": {
        "label": "6. Multi-TF trend, no vol filter",
        "rsi_oversold": 40, "rsi_overbought": 60,
        "vol_threshold": 0.0,  # no volume filter
        "adx_threshold": 25,
        "ema_pullback_pct": 0.003, "min_atr_pct": 0.015,
        "signal_types": ["TREND_FOLLOW"],
        "timeframes": ["15m", "1h"],
        "trailing": TRAILING_15M,
        "strong_leverage": 1.5, "trend_leverage": 1.0,
    },
}

# ===================== DATA LOADING =====================

def load_data():
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
    computed = {}
    for sym in SYMBOLS:
        computed[sym] = {}
        for tf in ["15m", "1h"]:
            df = data[sym][tf].copy()
            if len(df) > WARMUP:
                df = compute_all_v6(df)
                # Also add SMA-50 for HTF
                df = compute_sma(df, V7_HTF_SMA_PERIOD)
                computed[sym][tf] = df
            else:
                computed[sym][tf] = None
    return computed

# ===================== SIGNAL GENERATION (parameterized) =====================

def generate_signals_parameterized(symbol, df, cfg, timeframe="15m", htf_df=None):
    """Generate signals with configurable parameters."""
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

    if atr_val is None or pd.isna(atr_val):
        return []
    atr_pct = atr_val / price
    if atr_pct < cfg["min_atr_pct"]:
        return []

    # HTF confirmation
    htf_bullish = None
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

    volume_ok = (cfg["vol_threshold"] <= 0 or
                 (not pd.isna(vol_ratio) and vol_ratio >= cfg["vol_threshold"]))
    adx_strong = (adx_val is not None and not pd.isna(adx_val) and
                  adx_val > cfg["adx_threshold"])
    trend_bullish = (plus_di is not None and minus_di is not None and
                     not pd.isna(plus_di) and not pd.isna(minus_di) and plus_di > minus_di)

    ema_col = f"ema_{EMA_TREND_PERIOD}"
    ema_val = latest.get(ema_col)

    # TREND_FOLLOW
    if "TREND_FOLLOW" in cfg["signal_types"] and adx_strong and ema_val is not None and not pd.isna(ema_val):
        ema_dist = abs(price - ema_val) / price
        near_ema = ema_dist <= cfg["ema_pullback_pct"]

        if near_ema and trend_bullish and htf_bullish is not False:
            signals.append(_build_sig("BUY", "TREND_FOLLOW", display, price, rsi_val,
                                      bb_pct, bb_lower, bb_upper, vol_ratio, adx_val,
                                      plus_di, minus_di, atr_val, macd_hist, timeframe,
                                      ema_val, cfg["trend_leverage"]))
        elif near_ema and not trend_bullish and htf_bullish is not True:
            signals.append(_build_sig("SELL", "TREND_FOLLOW", display, price, rsi_val,
                                      bb_pct, bb_lower, bb_upper, vol_ratio, adx_val,
                                      plus_di, minus_di, atr_val, macd_hist, timeframe,
                                      ema_val, cfg["trend_leverage"]))

    # STRONG mean-reversion
    if "STRONG" in cfg["signal_types"]:
        if (price <= bb_lower and rsi_val < cfg["rsi_oversold"] and volume_ok
                and htf_bullish is not False):
            signals.append(_build_sig("BUY", "STRONG", display, price, rsi_val,
                                      bb_pct, bb_lower, bb_upper, vol_ratio, adx_val,
                                      plus_di, minus_di, atr_val, macd_hist, timeframe,
                                      ema_val, cfg["strong_leverage"]))
        if (price >= bb_upper and rsi_val > cfg["rsi_overbought"] and volume_ok
                and htf_bullish is not True):
            signals.append(_build_sig("SELL", "STRONG", display, price, rsi_val,
                                      bb_pct, bb_lower, bb_upper, vol_ratio, adx_val,
                                      plus_di, minus_di, atr_val, macd_hist, timeframe,
                                      ema_val, cfg["strong_leverage"]))

    return signals


def _build_sig(direction, conviction, display, price, rsi_val, bb_pct,
               bb_lower, bb_upper, vol_ratio, adx_val, plus_di, minus_di,
               atr_val, macd_hist, timeframe, ema_val, leverage):
    return {
        "type": direction,
        "conviction": conviction,
        "symbol": display,
        "price": price,
        "rsi": round(rsi_val, 1),
        "timeframe": timeframe,
        "leverage": leverage,
    }

# ===================== TRAILING STOP =====================

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

# ===================== TRADE COLLECTION =====================

def collect_raw_trades(computed, cfg):
    """Collect trades for a given config."""
    start_ts = pd.Timestamp(START_MS, unit="ms")
    raw_trades = []

    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)

        for tf in cfg["timeframes"]:
            df = computed[sym][tf]
            if df is None:
                continue

            # For 1h signals, we use the 1h df as both signal source and price source
            # HTF confirmation: for 15m signals use 1h; for 1h signals use None (already on 1h)
            if tf == "15m":
                htf_df_full = computed[sym]["1h"]
                price_df = df  # simulate on 15m candles
            else:
                htf_df_full = None  # 1h signals don't need further HTF
                price_df = df  # simulate on 1h candles

            candidates = df[df["timestamp"] >= start_ts].index
            if len(candidates) == 0:
                continue
            tf_start_idx = max(candidates[0], WARMUP)

            for i in range(tf_start_idx, len(df)):
                current_ts = df.iloc[i]["timestamp"]

                htf_slice = None
                if htf_df_full is not None:
                    htf_idx = htf_df_full["timestamp"].searchsorted(current_ts, side="right")
                    if htf_idx > V7_HTF_SMA_PERIOD:
                        htf_slice = htf_df_full.iloc[:htf_idx]

                lookback_start = max(0, i - 200)
                df_slice = df.iloc[lookback_start:i + 1]

                signals = generate_signals_parameterized(sym, df_slice, cfg, tf, htf_slice)

                for sig in signals:
                    exit_idx, exit_price, gross_pnl, outcome = simulate_trailing_stop(
                        price_df, i, sig["type"], sig["price"], cfg["trailing"])

                    raw_trades.append({
                        "symbol": sym,
                        "display": display,
                        "direction": sig["type"],
                        "conviction": sig["conviction"],
                        "entry_idx": i,
                        "exit_idx": exit_idx,
                        "entry_time": df.iloc[i]["timestamp"],
                        "exit_time": price_df.iloc[exit_idx]["timestamp"],
                        "entry_price": sig["price"],
                        "exit_price": exit_price,
                        "gross_pnl_pct": gross_pnl,
                        "outcome": outcome,
                        "candles_held": exit_idx - i,
                        "leverage": sig["leverage"],
                        "timeframe": tf,
                    })

    raw_trades.sort(key=lambda t: t["entry_time"])
    return raw_trades

# ===================== CAPITAL SIMULATION =====================

def simulate_capital(raw_trades):
    capital = {"BTCUSDT": STARTING_CAPITAL_PER_ASSET, "ETHUSDT": STARTING_CAPITAL_PER_ASSET}
    all_trades = []
    equity_snapshots = []
    total_fees = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    open_pos = {"BTCUSDT": None, "ETHUSDT": None}
    sl_cooldowns = {"BTCUSDT": {}, "ETHUSDT": {}}

    for trade in raw_trades:
        sym = trade["symbol"]
        direction = trade["direction"]
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]

        if open_pos[sym] is not None and open_pos[sym]["exit_time"] <= entry_time:
            open_pos[sym] = None
        if open_pos[sym] is not None:
            continue

        cooldown_until = sl_cooldowns[sym].get(direction)
        if cooldown_until is not None and entry_time < cooldown_until:
            continue

        available = capital[sym]
        if available <= 0:
            continue

        leverage = trade["leverage"]
        notional = available * leverage

        if direction == "BUY":
            eff_entry = trade["entry_price"] * (1 + SLIPPAGE_PCT)
            eff_exit = trade["exit_price"] * (1 - SLIPPAGE_PCT)
            net_pnl_pct = (eff_exit - eff_entry) / eff_entry
        else:
            eff_entry = trade["entry_price"] * (1 - SLIPPAGE_PCT)
            eff_exit = trade["exit_price"] * (1 + SLIPPAGE_PCT)
            net_pnl_pct = (eff_entry - eff_exit) / eff_entry

        entry_fee = notional * FEE_PER_SIDE
        exit_notional = notional * (1 + net_pnl_pct)
        exit_fee = exit_notional * FEE_PER_SIDE
        total_fee = entry_fee + exit_fee

        net_dollar_pnl = notional * net_pnl_pct - total_fee
        capital[sym] += net_dollar_pnl
        total_fees[sym] += total_fee

        open_pos[sym] = {"exit_time": exit_time, "direction": direction}

        if trade["outcome"] == "SL":
            cooldown_delta = pd.Timedelta(minutes=15 * V7_SL_COOLDOWN_CANDLES)
            sl_cooldowns[sym][direction] = exit_time + cooldown_delta

        all_trades.append({
            **trade,
            "net_dollar_pnl": net_dollar_pnl,
            "fees_paid": total_fee,
            "notional": notional,
        })

        equity_snapshots.append({
            "timestamp": exit_time,
            "total": capital["BTCUSDT"] + capital["ETHUSDT"],
        })

    return all_trades, equity_snapshots, capital, total_fees

# ===================== METRICS =====================

def compute_metrics(trades, snapshots, final_capital, total_fees):
    total_starting = STARTING_CAPITAL_PER_ASSET * 2
    total_ending = final_capital["BTCUSDT"] + final_capital["ETHUSDT"]
    total_return = (total_ending - total_starting) / total_starting * 100
    n = len(trades)

    if n == 0:
        return {
            "trades": 0, "trades_mo": 0, "wr": 0, "final_cap": total_starting,
            "return_pct": 0, "max_dd_pct": 0, "profit_factor": 0,
            "fees_paid": 0, "avg_trade": 0,
        }

    tdf = pd.DataFrame(trades)
    wins = int((tdf["net_dollar_pnl"] > 0).sum())
    losses = n - wins
    wr = wins / n * 100

    days = (pd.Timestamp(END_MS, unit="ms") - pd.Timestamp(START_MS, unit="ms")).days
    months = days / 30.44
    trades_mo = n / months

    fees_total = total_fees["BTCUSDT"] + total_fees["ETHUSDT"]
    avg_trade = tdf["net_dollar_pnl"].mean()

    # Profit factor
    gross_wins = tdf[tdf["net_dollar_pnl"] > 0]["net_dollar_pnl"].sum()
    gross_losses = abs(tdf[tdf["net_dollar_pnl"] <= 0]["net_dollar_pnl"].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Max drawdown
    max_dd_pct = 0
    if snapshots:
        sdf = pd.DataFrame(snapshots)
        peak = sdf["total"].cummax()
        dd = (sdf["total"] - peak) / peak * 100
        max_dd_pct = dd.min()

    return {
        "trades": n,
        "trades_mo": round(trades_mo, 1),
        "wr": round(wr, 1),
        "final_cap": round(total_ending, 2),
        "return_pct": round(total_return, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "profit_factor": round(pf, 2),
        "fees_paid": round(fees_total, 2),
        "avg_trade": round(avg_trade, 2),
    }

# ===================== MAIN =====================

def main():
    print(f"\n{'='*100}")
    print(f"  CryptoSignals v7 PARAMETER SWEEP")
    print(f"  May 1, 2025 → Feb 10, 2026 | $20k start | Maker 0.02%/side + 0.03% slippage")
    print(f"{'='*100}")

    print(f"\n  Loading data...")
    data = load_data()

    print(f"\n  Computing indicators (15m + 1h)...")
    computed = compute_indicators(data)

    results = {}

    for cfg_key, cfg in CONFIGS.items():
        print(f"\n  {'─'*80}")
        print(f"  Testing: {cfg['label']}")
        print(f"    RSI {cfg['rsi_oversold']}/{cfg['rsi_overbought']}, Vol {cfg['vol_threshold']}x, "
              f"ADX {cfg['adx_threshold']}, TFs: {cfg['timeframes']}, "
              f"Types: {cfg['signal_types']}, BE trigger: {cfg['trailing']['breakeven_trigger']}")

        raw = collect_raw_trades(computed, cfg)
        print(f"    Raw signals: {len(raw)}")

        trades, snaps, cap, fees = simulate_capital(raw)
        m = compute_metrics(trades, snaps, cap, fees)
        results[cfg_key] = {"cfg": cfg, "metrics": m}

        profitable = m["return_pct"] > 0
        flag = "✅" if profitable else "❌ UNPROFITABLE"
        print(f"    → {m['trades']} trades, {m['trades_mo']}/mo, WR {m['wr']}%, "
              f"Return {m['return_pct']:+.2f}%, MaxDD {m['max_dd_pct']:.2f}%, "
              f"PF {m['profit_factor']:.2f}  {flag}")

    # ===================== SUMMARY TABLE =====================
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY — ALL CONFIGS (Maker fees 0.02%/side + 0.03% slippage)")
    print(f"{'='*100}")

    header = (f"  {'Config':<32} {'Trades':>6} {'T/mo':>5} {'WR%':>5} "
              f"{'Final $':>10} {'Ret%':>7} {'MaxDD%':>7} {'PF':>5} "
              f"{'Fees$':>8} {'Avg$':>8} {'':>4}")
    print(header)
    print(f"  {'─'*32} {'─'*6} {'─'*5} {'─'*5} {'─'*10} {'─'*7} {'─'*7} {'─'*5} {'─'*8} {'─'*8} {'─'*4}")

    for cfg_key, res in results.items():
        m = res["metrics"]
        cfg = res["cfg"]
        flag = "✅" if m["return_pct"] > 0 else "❌"
        print(f"  {cfg['label']:<32} {m['trades']:>6} {m['trades_mo']:>5} {m['wr']:>5} "
              f"${m['final_cap']:>9,.0f} {m['return_pct']:>+6.2f}% {m['max_dd_pct']:>6.2f}% {m['profit_factor']:>5.2f} "
              f"${m['fees_paid']:>7,.0f} ${m['avg_trade']:>7,.0f} {flag:>4}")

    print()
    print(f"  Starting capital: $20,000 | Period: 9.5 months")
    print(f"  Fee model: Maker 0.02%/side + 0.03% slippage = ~0.10% round-trip")
    print()

if __name__ == "__main__":
    main()
