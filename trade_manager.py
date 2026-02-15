#!/usr/bin/env python3
"""
Trade Manager — runs every 15 minutes.
1. Reads latest prediction from /tmp/tu_forecast.json
2. Fetches live prices + technical indicators from Binance
3. Manages positions: open, adjust, close
"""

import json
import csv
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import fetch_klines, get_current_price
from indicators import rsi, macd, atr, bollinger_bands, sma, volume_profile

# === CONFIG ===
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions.json")
EVENT_LOG_PATH = "/home/mroon/crypto-trades.csv"
FORECAST_PATH = "/tmp/tu_forecast.json"
SYMBOLS = {"BTC/USD": "BTCUSDT", "ETH/USD": "ETHUSDT"}

# Position sizing
RISK_PER_TRADE = 2000       # $2K risk per trade
MAX_POSITION = 100000       # $100K max position
STOP_LOSS_PCT = 2.0         # 2% stop loss

# Entry thresholds
MIN_24H_MOVE = 1.2          # Minimum predicted 24h move to trigger trade

# TA thresholds for exit/adjustment
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
TRADE_EXPIRY_HOURS = 24     # Close trades after 24h (prediction stale)

CSV_FIELDNAMES = [
    "timestamp", "symbol", "strategy", "action", "entry_price", "target_price",
    "shares", "position_value", "stop_loss_pct", "take_profit_pct",
    "stop_loss_price", "take_profit_price", "entry_reason", "target_change_pct",
    "risk_usd", "status", "exit_price", "exit_time", "pnl_usd"
]

EVENT_LOG_FIELDS = [
    "timestamp", "event_type", "symbol", "price", "pred_24h_price", "pred_24h_pct",
    "rsi", "macd_cross", "trend", "bb_pct", "vol_ratio",
    "action", "detail", "pnl_usd"
]


def log_event(event_type: str, symbol: str, price: float, ta: Dict,
              forecast: Optional[Dict] = None, action: str = "", detail: str = "", pnl: float = 0):
    """Append a row to the event log CSV."""
    h24_price = ""
    h24_pct = ""
    if forecast:
        f = forecast.get("forecasts", {}).get("24h", {})
        h24_price = f.get("price", "")
        h24_pct = f.get("change_pct", "")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "symbol": symbol,
        "price": round(price, 2) if price else "",
        "pred_24h_price": round(h24_price, 2) if isinstance(h24_price, (int, float)) else h24_price,
        "pred_24h_pct": round(h24_pct, 2) if isinstance(h24_pct, (int, float)) else h24_pct,
        "rsi": round(ta.get("rsi", 0), 1) if ta else "",
        "macd_cross": ta.get("macd_cross", "") if ta else "",
        "trend": ta.get("trend", "") if ta else "",
        "bb_pct": round(ta.get("bb_pct", 0), 3) if ta else "",
        "vol_ratio": round(ta.get("vol_ratio", 0), 2) if ta else "",
        "action": action,
        "detail": detail,
        "pnl_usd": round(pnl, 2) if pnl else "",
    }

    exists = os.path.exists(EVENT_LOG_PATH)
    with open(EVENT_LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# === DATA LOADING ===

def load_forecast() -> Optional[Dict]:
    if not os.path.exists(FORECAST_PATH):
        return None
    try:
        with open(FORECAST_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading forecast: {e}")
        return None


def load_trades() -> List[Dict]:
    if not os.path.exists(CSV_PATH):
        return []
    try:
        with open(CSV_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def save_trades(trades: List[Dict]):
    with open(CSV_PATH, 'w') as f:
        json.dump(trades, f, indent=2)


def append_trade(trade: Dict):
    """Add a new trade to the positions file (JSON list)."""
    trades = load_trades()
    trades.append(trade)
    save_trades(trades)


# === TECHNICAL ANALYSIS ===

def get_ta_snapshot(binance_symbol: str) -> Dict:
    """Fetch 1h klines and compute key indicators"""
    try:
        df = fetch_klines(binance_symbol, "1h", limit=100)
        if df is None or df.empty:
            return {}

        df = rsi(df)
        df = macd(df)
        df = atr(df)
        df = bollinger_bands(df)
        df = sma(df, period=20)
        df = volume_profile(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(latest["close"])

        # Trend: price vs SMA20
        sma20 = float(latest.get("sma_20", price))
        trend = "bullish" if price > sma20 else "bearish"

        # MACD crossover
        macd_val = float(latest.get("macd", 0))
        macd_sig = float(latest.get("macd_signal", 0))
        macd_prev = float(prev.get("macd", 0))
        macd_sig_prev = float(prev.get("macd_signal", 0))
        macd_cross = "bullish" if (macd_prev < macd_sig_prev and macd_val > macd_sig) else \
                     "bearish" if (macd_prev > macd_sig_prev and macd_val < macd_sig) else "none"

        return {
            "price": price,
            "rsi": float(latest.get("rsi", 50)),
            "macd": macd_val,
            "macd_signal": macd_sig,
            "macd_cross": macd_cross,
            "atr": float(latest.get("atr", 0)),
            "bb_pct": float(latest.get("bb_pct", 0.5)),  # 0=lower band, 1=upper band
            "sma20": sma20,
            "trend": trend,
            "vol_ratio": float(latest.get("vol_ratio", 1.0)),
        }
    except Exception as e:
        print(f"⚠️  TA error for {binance_symbol}: {e}")
        return {}


# === POSITION MANAGEMENT ===

def check_exit(trade: Dict, price: float, ta: Dict) -> Optional[str]:
    """
    Decide if a trade should be closed. Returns exit reason or None.
    """
    action = trade["action"]
    entry = float(trade["entry_price"])
    sl = float(trade["stop_loss_price"])
    tp = float(trade["take_profit_price"])

    # Hard stops
    if action == "BUY":
        if price <= sl:
            return "STOP_LOSS"
        if price >= tp:
            return "TAKE_PROFIT"
    elif action == "SELL":
        if price >= sl:
            return "STOP_LOSS"
        if price <= tp:
            return "TAKE_PROFIT"

    # Expiry
    try:
        trade_time = datetime.fromisoformat(trade["timestamp"].replace('Z', '+00:00'))
        age_h = (datetime.now(timezone.utc) - trade_time).total_seconds() / 3600
        if age_h > TRADE_EXPIRY_HOURS:
            return "EXPIRED"
    except:
        pass

    if not ta:
        return None

    rsi_val = ta.get("rsi", 50)
    macd_cross = ta.get("macd_cross", "none")

    # TA-based exits
    if action == "BUY":
        # Exit long if RSI overbought + MACD bearish cross
        if rsi_val > RSI_OVERBOUGHT and macd_cross == "bearish":
            return "TA_OVERBOUGHT"
        # Exit long if trend reversed and we're in profit
        if ta.get("trend") == "bearish" and price > entry:
            return "TA_TREND_REVERSAL"

    elif action == "SELL":
        # Exit short if RSI oversold + MACD bullish cross
        if rsi_val < RSI_OVERSOLD and macd_cross == "bullish":
            return "TA_OVERSOLD"
        # Exit short if trend reversed and we're in profit
        if ta.get("trend") == "bullish" and price < entry:
            return "TA_TREND_REVERSAL"

    return None


def should_tighten_stop(trade: Dict, price: float, ta: Dict) -> Optional[float]:
    """
    Check if we should move the stop loss closer (trailing stop behavior).
    Returns new SL price or None.
    """
    action = trade["action"]
    entry = float(trade["entry_price"])
    current_sl = float(trade["stop_loss_price"])
    atr_val = ta.get("atr", 0)

    if not atr_val:
        return None

    if action == "BUY":
        # If price moved significantly in our favor, trail the stop
        profit_pct = (price - entry) / entry * 100
        if profit_pct > 0.5:  # At least 0.5% in profit
            new_sl = price - (1.5 * atr_val)  # 1.5x ATR trailing stop
            if new_sl > current_sl:
                return round(new_sl, 2)

    elif action == "SELL":
        profit_pct = (entry - price) / entry * 100
        if profit_pct > 0.5:
            new_sl = price + (1.5 * atr_val)
            if new_sl < current_sl:
                return round(new_sl, 2)

    return None


def create_entry(symbol: str, forecast: Dict, ta: Dict) -> Optional[Dict]:
    """
    Decide if we should open a new trade based on 24h forecast + TA confirmation.
    """
    f = forecast.get("forecasts", {})
    h24 = f.get("24h", {})
    h24_change = h24.get("change_pct", 0)
    h24_target = h24.get("price", 0)
    price = ta.get("price", forecast.get("current_price", 0))

    if abs(h24_change) < MIN_24H_MOVE:
        return None

    rsi_val = ta.get("rsi", 50)
    trend = ta.get("trend", "neutral")
    macd_cross = ta.get("macd_cross", "none")

    context_1mo = f.get("1mo", {}).get("change_pct", 0)
    context_12mo = f.get("12mo", {}).get("change_pct", 0)

    action = None
    strategy = None
    confirmation = []

    if h24_change <= -MIN_24H_MOVE:
        # Bearish 24h prediction — confirm with TA
        if trend == "bearish":
            confirmation.append("trend↓")
        if rsi_val > 60:
            confirmation.append(f"RSI={rsi_val:.0f}(room to fall)")
        if macd_cross == "bearish":
            confirmation.append("MACD↓")

        # Need at least 1 TA confirmation, or strong prediction (>2%)
        if confirmation or abs(h24_change) > 2.0:
            action = "SELL"
            strategy = "24h Bearish Follow"

    elif h24_change >= MIN_24H_MOVE:
        # Bullish 24h prediction — confirm with TA
        if trend == "bullish":
            confirmation.append("trend↑")
        if rsi_val < 40:
            confirmation.append(f"RSI={rsi_val:.0f}(room to rise)")
        if macd_cross == "bullish":
            confirmation.append("MACD↑")

        if confirmation or abs(h24_change) > 2.0:
            action = "BUY"
            strategy = "24h Bullish Follow"

    if not action:
        return None

    # Position sizing
    sl_dollar = price * (STOP_LOSS_PCT / 100)
    shares = RISK_PER_TRADE / sl_dollar
    position_value = shares * price
    if position_value > MAX_POSITION:
        shares = MAX_POSITION / price
        position_value = MAX_POSITION

    tp_pct = abs(h24_change) * 0.8

    if action == "BUY":
        sl_price = round(price * (1 - STOP_LOSS_PCT / 100), 2)
        tp_price = round(price * (1 + tp_pct / 100), 2)
    else:
        sl_price = round(price * (1 + STOP_LOSS_PCT / 100), 2)
        tp_price = round(price * (1 - tp_pct / 100), 2)

    conf_str = ", ".join(confirmation) if confirmation else "strong prediction"
    reason = (f"24h→${h24_target:,.2f} ({h24_change:+.1f}%). "
              f"TA: {conf_str}. "
              f"Context: 1mo({context_1mo:+.1f}%), 12mo({context_12mo:+.1f}%)")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "strategy": strategy,
        "action": action,
        "entry_price": round(price, 2),
        "target_price": round(h24_target, 2),
        "shares": round(shares, 6),
        "position_value": round(position_value, 2),
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": round(tp_pct, 3),
        "stop_loss_price": sl_price,
        "take_profit_price": tp_price,
        "entry_reason": reason,
        "target_change_pct": h24_change,
        "risk_usd": round(min(RISK_PER_TRADE, position_value * STOP_LOSS_PCT / 100), 2),
        "status": "OPEN",
    }


# === MAIN ===

def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.isoformat()}] Trade Manager running...")

    # Load state
    forecast_data = load_forecast()
    forecasts = forecast_data.get("data", {}) if forecast_data else {}
    trades = load_trades()
    modified = False
    results = []  # For reporting

    # Step 1 & 2: Check existing positions
    for trade in trades:
        if trade["status"] != "OPEN":
            continue

        symbol = trade["symbol"]
        binance_sym = SYMBOLS.get(symbol)
        if not binance_sym:
            continue

        ta = get_ta_snapshot(binance_sym)
        price = ta.get("price", 0)
        if not price:
            try:
                price = get_current_price(binance_sym)
            except:
                continue

        forecast = forecasts.get(symbol)

        # Check exit
        exit_reason = check_exit(trade, price, ta)
        if exit_reason:
            entry = float(trade["entry_price"])
            shares = float(trade["shares"])
            pnl = (price - entry) * shares if trade["action"] == "BUY" else (entry - price) * shares
            trade["status"] = exit_reason
            trade["exit_price"] = str(round(price, 2))
            trade["exit_time"] = now.isoformat()
            trade["pnl_usd"] = str(round(pnl, 2))
            emoji = "💰" if pnl > 0 else "💸"
            results.append(f"{emoji} CLOSED {symbol} {trade['action']} — {exit_reason} @ ${price:,.2f} (P&L: ${pnl:+,.2f})")
            log_event("CLOSE", symbol, price, ta, forecast, exit_reason,
                      f"{trade['action']} closed. Entry=${entry:.2f}", pnl)
            modified = True
            continue

        # Check trailing stop tighten
        new_sl = should_tighten_stop(trade, price, ta)
        if new_sl:
            old_sl = trade["stop_loss_price"]
            trade["stop_loss_price"] = str(new_sl)
            results.append(f"🔧 {symbol} SL tightened: ${old_sl} → ${new_sl}")
            log_event("SL_ADJUST", symbol, price, ta, forecast, "TIGHTEN",
                      f"SL {old_sl} → {new_sl}")
            modified = True
        else:
            # Log a HOLD event — no action but we checked
            entry = float(trade["entry_price"])
            shares = float(trade["shares"])
            unrealized = (price - entry) * shares if trade["action"] == "BUY" else (entry - price) * shares
            log_event("HOLD", symbol, price, ta, forecast, trade["action"],
                      f"SL=${trade['stop_loss_price']} TP=${trade['take_profit_price']}", unrealized)

    # Save if modified
    if modified:
        save_trades(trades)

    # Step 3: Look for new entries
    open_symbols = {t["symbol"] for t in trades if t["status"] == "OPEN"}

    for symbol, binance_sym in SYMBOLS.items():
        if symbol in open_symbols:
            print(f"⏳ {symbol} has open position — skipping entry")
            continue

        if symbol not in forecasts:
            continue

        ta = get_ta_snapshot(binance_sym)
        if not ta:
            continue

        forecast = forecasts[symbol]
        new_trade = create_entry(symbol, forecast, ta)
        if new_trade:
            append_trade(new_trade)
            results.append(
                f"✅ NEW {new_trade['action']} {symbol} @ ${new_trade['entry_price']:,.2f} "
                f"| TP ${new_trade['take_profit_price']} | SL ${new_trade['stop_loss_price']} "
                f"| {new_trade['entry_reason']}"
            )
            log_event("OPEN", symbol, new_trade["entry_price"], ta, forecast,
                      new_trade["action"], new_trade["entry_reason"])
        else:
            price = ta.get("price", 0)
            h24 = forecast.get("forecasts", {}).get("24h", {})
            log_event("SKIP", symbol, price, ta, forecast, "NONE",
                      f"No signal. 24h pred={h24.get('change_pct', 0):+.1f}%")
            print(f"ℹ️  No signal for {symbol}")

    # Print summary
    if results:
        print("\n--- TRADE ACTIONS ---")
        for r in results:
            print(r)
    else:
        print("No actions taken")

    return results


if __name__ == "__main__":
    main()
