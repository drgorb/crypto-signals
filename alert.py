"""CryptoSignals v7 — Live alert runner with position tracking.

Config 6: Trend-follow only on 15m + 1h, no volume filter, ADX 25.
Runs every 15 minutes via cron.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from data import fetch_klines_paginated, fetch_daily_klines, fetch_hourly_klines, get_current_price
from indicators import compute_all
from signals_v7 import generate_signals_v7
from positions import (load_positions, save_positions, open_position,
                       check_trailing_stops, format_close_message, format_status_message)
import time

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SYMBOL_NAMES = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"}
MAX_POSITIONS_PER_ASSET = 1


def format_signal(sig):
    """Format a signal as a Mattermost message."""
    emoji = "🟢" if sig["type"] == "BUY" else "🔴"
    conviction = sig.get("conviction", "TREND_FOLLOW")
    tf = sig.get("timeframe", "15m")
    sl_pct = "2%"

    msg = (
        f"{emoji} 📈 **{sig['type']} {sig['symbol']}** — {conviction} ({tf})\n\n"
        f"- **Entry:** ${sig['price']:,.2f}\n"
        f"- **Initial SL:** {sl_pct}\n"
        f"- **RSI:** {sig.get('rsi', 'N/A')}\n"
        f"- **ADX:** {sig.get('adx', 'N/A')}\n"
        f"- **EMA-20:** ${sig.get('ema_20', 0):,.2f}\n"
        f"- **Timeframe:** {tf}\n"
        f"- **Exit:** Trailing stop (breakeven at +1.5%, trail at +2%)\n\n"
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_"
    )
    return msg


def get_current_prices():
    prices = {}
    for sym in SYMBOLS:
        try:
            prices[sym] = get_current_price(sym)
        except Exception as e:
            print(f"⚠ Could not get price for {sym}: {e}")
    return prices


def count_positions_for(positions, symbol):
    count = 0
    for p in positions:
        raw = p["raw_symbol"].replace("/", "")
        if raw == symbol or p["symbol"] == SYMBOL_NAMES.get(symbol, symbol):
            count += 1
    return count


def run():
    close_alerts = []
    open_alerts = []

    # 1. Get current prices
    print("⏳ Fetching current prices...", flush=True)
    prices = get_current_prices()
    print(f"  BTC: ${prices.get('BTCUSDT', 0):,.2f}  ETH: ${prices.get('ETHUSDT', 0):,.2f}")

    # 2. Check open positions
    positions = load_positions()
    if positions:
        print(f"📊 Checking {len(positions)} open position(s)...", flush=True)
        still_open, closed = check_trailing_stops(positions, prices)
        for pos in closed:
            close_alerts.append(format_close_message(pos))
            print(f"  ❌ Closed {pos['direction']} {pos['symbol']}: {pos['pnl_pct']:+.2f}% ({pos['exit_reason']})")
        positions = still_open
    else:
        print("📭 No open positions.", flush=True)

    # 3. Generate new signals on 15m and 1h
    new_signals = []
    for sym in SYMBOLS:
        display = SYMBOL_NAMES.get(sym, sym)
        print(f"⏳ Analyzing {display}...", flush=True)

        # Fetch data
        start_15m = int((time.time() - 30 * 86400) * 1000)
        df_15m = fetch_klines_paginated(sym, "15m", start_15m)
        df_1h = fetch_hourly_klines(sym, hours=720)  # 30 days
        df_1d = fetch_daily_klines(sym, days=120)

        # Compute indicators
        df_15m_ind = compute_all(df_15m.copy())
        df_1h_ind = compute_all(df_1h.copy())

        # Compute 1h SMA for HTF confirmation
        htf_sma_period = 50
        if len(df_1h) >= htf_sma_period:
            df_1h[f"sma_{htf_sma_period}"] = df_1h["close"].rolling(htf_sma_period).mean()

        # Check position limits
        if count_positions_for(positions, sym) >= MAX_POSITIONS_PER_ASSET:
            print(f"  {display}: max positions reached, skipping")
            continue

        # Generate signals on 15m
        sigs_15m = generate_signals_v7(sym, df_15m_ind, timeframe="15m", htf_df=df_1h)
        for s in sigs_15m:
            s["raw_symbol"] = sym
            new_signals.append(s)

        # Generate signals on 1h
        sigs_1h = generate_signals_v7(sym, df_1h_ind, timeframe="1h", htf_df=df_1d)
        for s in sigs_1h:
            s["raw_symbol"] = sym
            new_signals.append(s)

    # 4. Deduplicate — if both 15m and 1h fire same direction for same symbol, keep 1h (bigger move)
    seen = {}
    deduped = []
    for sig in new_signals:
        key = f"{sig['raw_symbol']}_{sig['type']}"
        if key not in seen:
            seen[key] = sig
            deduped.append(sig)
        else:
            # Prefer 1h over 15m
            if sig.get("timeframe") == "1h":
                deduped.remove(seen[key])
                seen[key] = sig
                deduped.append(sig)
    new_signals = deduped

    # 5. Open positions and format alerts
    for sig in new_signals:
        if count_positions_for(positions, sig["raw_symbol"]) >= MAX_POSITIONS_PER_ASSET:
            continue
        pos = open_position(sig)
        positions.append(pos)
        open_alerts.append(format_signal(sig))
        print(f"  🆕 Opened {sig['type']} {sig['symbol']} ({sig['conviction']}, {sig.get('timeframe')}) @ ${sig['price']:,.2f}")

    # 6. Save
    save_positions(positions)

    if positions:
        print(f"\n{format_status_message(positions, prices)}")

    return close_alerts, open_alerts


def main():
    close_alerts, open_alerts = run()
    all_alerts = close_alerts + open_alerts

    if not all_alerts:
        print("\nNo alerts to send.")
        return

    print(f"\n{'='*60}")
    print(f"  🚨 {len(all_alerts)} alert(s)")
    print(f"  ({len(close_alerts)} close, {len(open_alerts)} open)")
    print(f"{'='*60}\n")

    for alert in all_alerts:
        print(alert)
        print()

    combined = "\n---\n".join(all_alerts)
    with open(os.path.join(os.path.dirname(__file__), "last_alert.txt"), "w") as f:
        f.write(combined)


if __name__ == "__main__":
    main()
