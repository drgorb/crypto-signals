#!/usr/bin/env python3
"""CryptoSignals — BTC/ETH mean-reversion signal generator."""

import sys
import pandas as pd
from config import SYMBOLS, TIMEFRAMES, SYMBOL_NAMES
from data import (fetch_klines, get_current_price, fetch_funding_rate,
                  fetch_orderbook_imbalance, fetch_daily_klines, fetch_hourly_klines)
from indicators import compute_all
from signals import generate_signals
from sentiment import get_bearish_sentiment
from futures_data import compute_derivatives_score
from notify import format_summary, print_signals


def run():
    print("⏳ Fetching prediction market sentiment...")
    sentiment = get_bearish_sentiment()
    if sentiment["markets"]:
        print(f"  Found {len(sentiment['markets'])} relevant markets (bearish score: {sentiment['score']:.1%})")
    else:
        print(f"  {sentiment['reason']}")

    all_signals = []
    summaries = []

    # Pre-fetch higher-timeframe data for all symbols
    daily_data = {}
    hourly_data = {}
    data_15m = {}
    for symbol in SYMBOLS:
        display = SYMBOL_NAMES.get(symbol, symbol)
        try:
            print(f"  Fetching daily candles for {display}...")
            daily_data[symbol] = fetch_daily_klines(symbol)
        except Exception as e:
            print(f"  ⚠ Daily data error for {display}: {e}")
            daily_data[symbol] = None
        try:
            print(f"  Fetching 1h candles for {display}...")
            hourly_data[symbol] = fetch_hourly_klines(symbol)
        except Exception as e:
            print(f"  ⚠ Hourly data error for {display}: {e}")
            hourly_data[symbol] = None

    # Pre-fetch derivatives data
    derivatives_data = {}
    for symbol in SYMBOLS:
        display = SYMBOL_NAMES.get(symbol, symbol)
        try:
            print(f"  Fetching derivatives data for {display}...")
            price_now = get_current_price(symbol)
            derivatives_data[symbol] = compute_derivatives_score(symbol, price_now)
            ds = derivatives_data[symbol]
            print(f"  Derivatives score: {ds['score']:+.3f} ({ds['details']})")
        except Exception as e:
            print(f"  ⚠ Derivatives data error for {display}: {e}")
            derivatives_data[symbol] = None

    for symbol in SYMBOLS:
        display = SYMBOL_NAMES.get(symbol, symbol)
        print(f"\n⏳ Analyzing {display}...")

        # Fetch supplementary data (graceful degradation)
        funding = fetch_funding_rate(symbol)
        ob_ratio = fetch_orderbook_imbalance(symbol)
        if funding is not None:
            print(f"  Funding rate: {funding:.6f} ({funding*100:.4f}%)")
        if ob_ratio is not None:
            print(f"  Order book bid/ask ratio: {ob_ratio:.2f}")

        for tf in TIMEFRAMES:
            try:
                df = fetch_klines(symbol, tf)
                df = compute_all(df)

                # Store 15m data for cross-asset correlation
                if tf == "15m":
                    data_15m[symbol] = df

                # Build kwargs for cross-asset correlation (ETH needs BTC data)
                extra_kwargs = {}
                if symbol == "ETHUSDT":
                    extra_kwargs["btc_df_15m"] = data_15m.get("BTCUSDT")
                    extra_kwargs["btc_daily_df"] = daily_data.get("BTCUSDT")
                    extra_kwargs["btc_hourly_df"] = hourly_data.get("BTCUSDT")

                latest = df.iloc[-1]
                sigs = generate_signals(
                    symbol, df, sentiment,
                    funding_rate=funding, ob_ratio=ob_ratio,
                    daily_df=daily_data.get(symbol),
                    hourly_df=hourly_data.get(symbol),
                    derivatives_data=derivatives_data.get(symbol),
                    **extra_kwargs)
                for s in sigs:
                    s["timeframe"] = tf
                all_signals.extend(sigs)

                if not sigs:
                    rsi_val = latest.get("rsi", 0)
                    bb_pct = latest.get("bb_pct", 0.5)
                    price = latest["close"]
                    if not pd.isna(rsi_val) and not pd.isna(bb_pct):
                        summaries.append(format_summary(f"{display} ({tf})", price, rsi_val, bb_pct, sentiment))

            except Exception as e:
                print(f"  ⚠ Error on {display} {tf}: {e}")

    print_signals(all_signals, summaries)
    return all_signals


if __name__ == "__main__":
    signals = run()
    sys.exit(0 if not signals else 0)
