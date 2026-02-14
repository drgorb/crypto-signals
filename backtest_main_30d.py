#!/usr/bin/env python3
"""
Backtest using a simulated 24-hour price forecast.
Final, clean, correct implementation.
"""
import time
import random
import numpy as np
import pandas as pd

from data import fetch_klines_paginated
from indicators import rsi, atr

# --- Config ---
START_CAPITAL = 200000.0
RISK_PER_TRADE_USD = 8000.0
TEST_PERIOD_DAYS = 30
FEE_BPS = 10
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
ATR_STOP_MULT = 2.0
TIME_STOP_CANDLES = 6

def prepare_data(symbol: str, days: int) -> pd.DataFrame:
    """
    Fetches 4h data and simulates a new 24h forecast for *every 4h candle*.
    """
    print(f"Preparing data for {symbol} with 4-hourly forecast updates...")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 86400 * 1000)

    # 1. Fetch 4-Hour Data
    df_4h = fetch_klines_paginated(symbol, "4h", start_ms, now_ms)
    
    # 2. Simulate a rolling "Flawed Crystal Ball" Forecast for every 4h candle
    # The "crystal ball" peeks 6 candles (24h) ahead
    df_4h['actual_future_close'] = df_4h['close'].shift(-TIME_STOP_CANDLES)
    df_4h.dropna(inplace=True)

    def simulate_prediction(row):
        actual_change_pct = (row['actual_future_close'] / row['close']) - 1
        error_factor = random.uniform(0.9, 1.1)  # +/- 10% error
        predicted_change_pct = actual_change_pct * error_factor
        return row['close'] * (1 + predicted_change_pct)

    df_4h['pred_price_24h'] = df_4h.apply(simulate_prediction, axis=1)
    df_4h['macro_trend'] = np.where(df_4h['pred_price_24h'] > df_4h['close'], 'bullish', 'bearish')

    # 3. Add Indicators
    df_4h = rsi(df_4h, 14)
    df_4h = atr(df_4h, 14)
    df_4h.set_index('timestamp', inplace=True)
    
    return df_4h.dropna()

"""
Backtest the main "Trend Confirmation & Dip Buying" strategy with a
fixed risk of $8,000 per trade, on a NON-COMPOUNDING basis.
"""
import time
import random
import numpy as np
import pandas as pd

from data import fetch_klines_paginated
from indicators import rsi, atr

# --- Config ---
START_CAPITAL = 200000.0
RISK_PER_TRADE_USD = 8000.0
TEST_PERIOD_DAYS = 270
FEE_BPS = 10
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
ATR_STOP_MULT = 2.0
TIME_STOP_CANDLES = 6

def prepare_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetches 4h data and simulates a new 24h forecast for every 4h candle."""
    print(f"Preparing data for {symbol} with 4-hourly forecast updates...")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 86400 * 1000)

    df_4h = fetch_klines_paginated(symbol, "4h", start_ms, now_ms)
    
    df_4h['actual_future_close'] = df_4h['close'].shift(-TIME_STOP_CANDLES)
    df_4h.dropna(inplace=True)

    def simulate_prediction(row):
        actual_change_pct = (row['actual_future_close'] / row['close']) - 1
        error_factor = random.uniform(0.9, 1.1)
        predicted_change_pct = actual_change_pct * error_factor
        return row['close'] * (1 + predicted_change_pct)

    df_4h['pred_price_24h'] = df_4h.apply(simulate_prediction, axis=1)
    df_4h['macro_trend'] = np.where(df_4h['pred_price_24h'] > df_4h['close'], 'bullish', 'bearish')

    df_4h = rsi(df_4h, 14)
    df_4h = atr(df_4h, 14)
    df_4h.set_index('timestamp', inplace=True)
    
    return df_4h.dropna()


def run_backtest(df: pd.DataFrame, symbol: str) -> dict:
    print(f"Running backtest for {symbol} (Fixed ${RISK_PER_TRADE_USD} risk, non-compounding)...")
    capital = START_CAPITAL
    position = "flat"
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size_asset = 0.0
    entry_idx = 0
    trades = []
    equity_curve = [START_CAPITAL]

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # --- Manage open positions ---
        if position != "flat":
            pnl = 0
            exit_reason = None
            exit_price = 0
            
            if (i - entry_idx) >= TIME_STOP_CANDLES: exit_price, exit_reason = current_row['open'], "Time Stop"
            elif position == 'long':
                if current_row['low'] <= stop_loss: exit_price, exit_reason = stop_loss, "Stop Loss"
                elif current_row['high'] >= take_profit: exit_price, exit_reason = take_profit, "Take Profit"
            elif position == 'short':
                if current_row['high'] >= stop_loss: exit_price, exit_reason = stop_loss, "Stop Loss"
                elif current_row['low'] <= take_profit: exit_price, exit_reason = take_profit, "Take Profit"

            if exit_reason:
                pnl = (exit_price - entry_price) * position_size_asset if position == 'long' else (entry_price - exit_price) * position_size_asset
                trade_value = position_size_asset * entry_price
                # Subtract fee for the closing trade
                capital += pnl - (trade_value * (FEE_BPS / 10000))
                trades.append({"pnl_usd": pnl})
                position = "flat"

        # --- Look for new entries ---
        if position == "flat":
            trend, rsi_val = prev_row['macro_trend'], prev_row['rsi']
            
            if (trend == 'bullish' and rsi_val < RSI_OVERSOLD) or (trend == 'bearish' and rsi_val > RSI_OVERBOUGHT):
                position = 'long' if trend == 'bullish' else 'short'
                entry_price = current_row['open']
                take_profit = prev_row['pred_price_24h']
                atr_val = current_row['atr']
                
                if atr_val == 0:
                    position = "flat"; continue

                stop_loss_dist = atr_val * ATR_STOP_MULT
                position_size_asset = RISK_PER_TRADE_USD / stop_loss_dist
                
                trade_value = position_size_asset * entry_price
                capital -= trade_value * (FEE_BPS / 10000) # Fee on entry
                
                stop_loss = entry_price - stop_loss_dist if position == 'long' else entry_price + stop_loss_dist
                entry_idx = i
        
        current_equity = capital
        if position == 'long': current_equity += (current_row['close'] - entry_price) * position_size_asset
        elif position == 'short': current_equity += (entry_price - current_row['close']) * position_size_asset
        equity_curve.append(current_equity)
    
    if position != "flat":
        exit_price = df['close'].iloc[-1]
        pnl = (exit_price - entry_price) * position_size_asset if position == 'long' else (entry_price - exit_price) * position_size_asset
        capital += pnl
        trades.append({"pnl_usd": pnl})

    final_capital = equity_curve[-1]
    total_return = (final_capital - START_CAPITAL) / START_CAPITAL * 100
    wins = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    bh_return = (df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0] * 100
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min() * 100

    return {
        "symbol": symbol,
        "final_capital": f"${final_capital:,.2f}",
        "total_profit": f"${final_capital - START_CAPITAL:,.2f}",
        "total_return_pct": f"{total_return:+.2f}%",
        "buy_and_hold_pct": f"{bh_return:+.2f}%",
        "alpha_pct": f"{total_return - bh_return:+.2f}%",
        "total_trades": len(trades),
        "win_rate_pct": f"{win_rate:.1f}%",
        "max_drawdown_pct": f"{max_drawdown:.1f}%",
    }

if __name__ == "__main__":
    results = []
    for asset in ["BTCUSDT", "ETHUSDT"]:
        # Prepare data for a longer period to ensure indicators are mature
        full_df = prepare_data(asset, TEST_PERIOD_DAYS + 100) # e.g., 130 days
        
        # Slice to the last 30 days for the backtest
        last_30_days_start = full_df.index.max() - pd.Timedelta(days=30)
        df_30d = full_df[full_df.index >= last_30_days_start]

        res = run_backtest(df_30d, asset)
        results.append(res)
    
    print("\\n" + "="*80)
    print(" " * 20 + "Main Strategy (Last 30 Days Performance)")
    print("="*80)
    
    for res in results:
        print("")
        for key, val in res.items():
            print(f"  {key.replace('_', ' ').title():<25}: {val}")
            
    print("\\n" + "="*80)
