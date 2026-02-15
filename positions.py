"""Position tracking and trailing stop management for CryptoSignals v6."""

import os
import json
import time

POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "cache", "open_positions.json")

# Trailing stop config (optimized for Binance VIP2)
BREAKEVEN_TRIGGER = 0.004    # move SL to breakeven at +0.4% (covers VIP2 fees)
TRAIL_TRIGGER_1 = 0.015      # start trailing at +1.5%
TRAIL_PCT_1 = 0.008          # trail 0.8% behind
TRAIL_TRIGGER_2 = 0.025      # tighten trail at +2.5%
TRAIL_PCT_2 = 0.005          # trail 0.5% behind
MAX_HOLD_HOURS = 48


def load_positions() -> list[dict]:
    if not os.path.exists(POSITIONS_FILE):
        return []
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_positions(positions: list[dict]):
    os.makedirs(os.path.dirname(POSITIONS_FILE), exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def open_position(signal: dict) -> dict:
    """Create a new position from a signal."""
    direction = signal["type"]
    entry = signal["price"]
    tf = signal.get("timeframe", "15m")
    initial_sl_pct = 0.02 if tf == "15m" else 0.01    # Optimized for Binance VIP2 fees

    if direction == "BUY":
        sl_price = entry * (1 - initial_sl_pct)
    else:
        sl_price = entry * (1 + initial_sl_pct)

    return {
        "id": f"{signal['symbol']}_{direction}_{int(time.time())}",
        "symbol": signal.get("symbol", ""),
        "raw_symbol": signal.get("raw_symbol", signal.get("symbol", "").replace("/", "")),
        "direction": direction,
        "conviction": signal.get("conviction", "STANDARD"),
        "entry_price": entry,
        "sl_price": round(sl_price, 2),
        "current_sl_stage": "initial",  # initial, breakeven, trail_1, trail_2
        "high_water_mark": entry,
        "timeframe": tf,
        "opened_at": time.time(),
        "opened_at_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }


def check_trailing_stops(positions: list[dict], current_prices: dict) -> tuple[list[dict], list[dict]]:
    """Check all open positions against current prices.
    
    Args:
        positions: list of open position dicts
        current_prices: {"BTCUSDT": 97000.0, "ETHUSDT": 3200.0}
    
    Returns:
        (still_open, closed) — two lists of position dicts.
        Closed positions have 'exit_price', 'exit_reason', 'pnl_pct' added.
    """
    still_open = []
    closed = []

    for pos in positions:
        sym = pos["raw_symbol"].replace("/", "")
        price = current_prices.get(sym)
        if price is None:
            still_open.append(pos)
            continue

        direction = pos["direction"]
        entry = pos["entry_price"]

        # Calculate current P&L
        if direction == "BUY":
            pnl_pct = (price - entry) / entry
        else:
            pnl_pct = (entry - price) / entry

        # Update high water mark
        if pnl_pct > (pos.get("high_water_mark", entry) - entry) / entry if direction == "BUY" else pnl_pct > 0:
            if direction == "BUY":
                pos["high_water_mark"] = max(pos.get("high_water_mark", entry), price)
            else:
                pos["high_water_mark"] = min(pos.get("high_water_mark", entry), price)

        hwm = pos["high_water_mark"]
        if direction == "BUY":
            hwm_pnl = (hwm - entry) / entry
        else:
            hwm_pnl = (entry - hwm) / entry

        # Update trailing stop level
        old_sl = pos["sl_price"]

        if hwm_pnl >= TRAIL_TRIGGER_2:
            if direction == "BUY":
                new_sl = hwm * (1 - TRAIL_PCT_2)
            else:
                new_sl = hwm * (1 + TRAIL_PCT_2)
            pos["current_sl_stage"] = "trail_2"
        elif hwm_pnl >= TRAIL_TRIGGER_1:
            if direction == "BUY":
                new_sl = hwm * (1 - TRAIL_PCT_1)
            else:
                new_sl = hwm * (1 + TRAIL_PCT_1)
            pos["current_sl_stage"] = "trail_1"
        elif hwm_pnl >= BREAKEVEN_TRIGGER:
            new_sl = entry  # breakeven
            pos["current_sl_stage"] = "breakeven"
        else:
            new_sl = old_sl

        # SL can only move in favorable direction
        if direction == "BUY":
            pos["sl_price"] = round(max(old_sl, new_sl), 2)
        else:
            pos["sl_price"] = round(min(old_sl, new_sl), 2)

        # Check if SL hit
        sl_hit = False
        if direction == "BUY" and price <= pos["sl_price"]:
            sl_hit = True
        elif direction == "SELL" and price >= pos["sl_price"]:
            sl_hit = True

        # Check timeout
        timed_out = (time.time() - pos["opened_at"]) > MAX_HOLD_HOURS * 3600

        if sl_hit or timed_out:
            pos["exit_price"] = round(price, 2)
            pos["exit_reason"] = "timeout" if timed_out else f"trailing_stop ({pos['current_sl_stage']})"
            pos["pnl_pct"] = round(pnl_pct * 100, 2)
            pos["closed_at_utc"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            closed.append(pos)
        else:
            still_open.append(pos)

    return still_open, closed


def format_close_message(pos: dict) -> str:
    """Format a position close alert for Mattermost."""
    pnl = pos["pnl_pct"]
    emoji = "✅" if pnl >= 0 else "❌"
    direction_emoji = "🟢" if pos["direction"] == "BUY" else "🔴"
    
    # Duration
    duration_h = (time.time() - pos["opened_at"]) / 3600
    if duration_h < 1:
        duration_str = f"{duration_h * 60:.0f}m"
    else:
        duration_str = f"{duration_h:.1f}h"

    msg = (
        f"{emoji} {direction_emoji} **CLOSE {pos['direction']} {pos['symbol']}** — {pos['conviction']}\n\n"
        f"- **Entry:** ${pos['entry_price']:,.2f}\n"
        f"- **Exit:** ${pos['exit_price']:,.2f}\n"
        f"- **P&L:** {pnl:+.2f}%\n"
        f"- **Reason:** {pos['exit_reason']}\n"
        f"- **Duration:** {duration_str}\n"
        f"- **Opened:** {pos['opened_at_utc']}\n\n"
        f"_Closed {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_"
    )
    return msg


def format_status_message(positions: list[dict], current_prices: dict) -> str:
    """Format a status summary of all open positions."""
    if not positions:
        return "📭 No open positions."
    
    lines = [f"📊 **{len(positions)} open position(s):**\n"]
    for pos in positions:
        sym = pos["raw_symbol"].replace("/", "")
        price = current_prices.get(sym, 0)
        entry = pos["entry_price"]
        if pos["direction"] == "BUY":
            pnl = (price - entry) / entry * 100 if price else 0
        else:
            pnl = (entry - price) / entry * 100 if price else 0
        
        emoji = "🟢" if pos["direction"] == "BUY" else "🔴"
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        duration_h = (time.time() - pos["opened_at"]) / 3600
        
        lines.append(
            f"- {emoji} **{pos['direction']} {pos['symbol']}** ({pos['conviction']}) "
            f"— entry ${entry:,.2f}, now ${price:,.2f} "
            f"{pnl_emoji} {pnl:+.2f}% | SL: ${pos['sl_price']:,.2f} ({pos['current_sl_stage']}) "
            f"| {duration_h:.1f}h"
        )
    
    return "\n".join(lines)
