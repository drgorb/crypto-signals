#!/usr/bin/env python3
"""
Backtest a specialized "Downside Catalyst" strategy.
- Signal: Simulated TU forecast predicts >5% drop in 24h.
- Action: Enter a leveraged short position.
- Risk: Fixed $6,000 per trade (non-compounding).
- Fees: Includes both trading and borrowing fees.
"""
import time
import random
import numpy as np
import pandas as pd

from data import fetch_klines_paginated
from indicators import rsi, atr

# --- Config ---
START_CAPITAL = 200000.0  # Using a larger capital base for clarity
RISK_PER_TRADE_USD = 6000.0
TEST_PERIOD_DAYS = 270
FEE_BPS = 10  # 0.1% per side
BORROW_RATE_ANNUAL = 0.10 # 10% APR for borrowing
ATR_STOP_MULT = 2.0
TIME_STOP_CANDLES = 6
DOWNSIDE_THRESHOLD = -0.05 # -5%


def prepare_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetches 4h data and simulates a new 24h forecast for every 4h candle."""
    print(f"Preparing data for {symbol} with 4-hourly forecast updates...")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 86400 * 1000)

    df_4h = fetch_klines_paginated(symbol, "4h", start_ms, now_ms)
    
    df_4h['actual_future_close'] = df_4h['close'].shift(-TIME_STOP_CANDLES)
    df_4h['actual_change_pct'] = (df_4h['actual_future_close'] / df_4h['close']) - 1
    df_4h.dropna(inplace=True)

    def simulate_prediction(row):
        error_factor = random.uniform(0.9, 1.1)
        return row['actual_change_pct'] * error_factor

    df_4h['pred_change_pct'] = df_4h.apply(simulate_prediction, axis=1)
    df_4h['pred_price_24h'] = df_4h['close'] * (1 + df_4h['pred_change_pct'])

    df_4h = rsi(df_4h, 14)
    df_4h = atr(df_4h, 14)
    df_4h.set_index('timestamp', inplace=True)
    
    return df_4h.dropna()


def run_backtest(df: pd.DataFrame, symbol: str) -> dict:
    print(f"Running Downside Catalyst backtest for {symbol}...")
    capital = START_CAPITAL
    position = "flat"
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size_asset = 0.0
    entry_idx = 0
    trades = []
    equity_curve = [START_CAPITAL]

    BORROW_FEE_PER_CANDLE = BORROW_RATE_ANNUAL / (365 * 6) # Daily rate / 6 candles per day

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # --- Manage open short position ---
        if position == "short":
            pnl = 0
            exit_reason = None
            exit_price = 0
            
            if (i - entry_idx) >= TIME_STOP_CANDLES: exit_price, exit_reason = current_row['open'], "Time Stop"
            elif current_row['high'] >= stop_loss: exit_price, exit_reason = stop_loss, "Stop Loss"
            elif current_row['low'] <= take_profit: exit_price, exit_reason = take_profit, "Take Profit"

            if exit_reason:
                # Gross PNL
                pnl = (entry_price - exit_price) * position_size_asset
                
                # Calculate fees
                trade_value = position_size_asset * entry_price
                trading_fee = trade_value * (FEE_BPS / 10000) * 2 # Entry and Exit
                
                candles_held = i - entry_idx
                borrow_fee = trade_value * BORROW_FEE_PER_CANDLE * candles_held
                
                net_pnl = pnl - trading_fee - borrow_fee
                capital += net_pnl
                
                trades.append({"pnl_usd": net_pnl, "gross_pnl_pct": (pnl / trade_value)*100})
                position = "flat"

        # --- Look for new short entries ---
        if position == "flat":
            if prev_row['pred_change_pct'] <= DOWNSIDE_THRESHOLD:
                position = 'short'
                entry_price = current_row['open']
                take_profit = prev_row['pred_price_24h']
                atr_val = current_row['atr']
                
                if atr_val == 0: # Avoid division by zero
                    position = "flat"; continue

                stop_loss_dist = atr_val * ATR_STOP_MULT
                position_size_asset = RISK_PER_TRADE_USD / stop_loss_dist
                
                stop_loss = entry_price + stop_loss_dist
                entry_idx = i
        
        equity_curve.append(capital)
    
    # Close any final open position
    if position == "short":
        exit_price = df['close'].iloc[-1]
        pnl = (entry_price - exit_price) * position_size_asset
        trade_value = position_size_asset * entry_price
        trading_fee = trade_value * (FEE_BPS / 10000) * 2
        candles_held = len(df) - entry_idx
        borrow_fee = trade_value * BORROW_FEE_PER_CANDLE * candles_held
        net_pnl = pnl - trading_fee - borrow_fee
        capital += net_pnl
        trades.append({"pnl_usd": net_pnl, "gross_pnl_pct": (pnl / trade_value)*100})

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
        "total_trades": len(trades),
        "win_rate_pct": f"{win_rate:.1f}%",
        "max_drawdown_pct": f"{max_drawdown:.1f}%",
    }

if __name__ == "__main__":
    results = []
    for asset in ["BTCUSDT", "ETHUSDT"]:
        df = prepare_data(asset, TEST_PERIOD_DAYS + 50)
        res = run_backtest(df, asset)
        results.append(res)
    
    print("\\n" + "="*80)
    print(" " * 20 + "Downside Catalyst Strategy Backtest")
    print("="*80)
    
    for res in results:
        print("")
        for key, val in res.items():
            print(f"  {key.replace('_', ' ').title():<25}: {val}")
            
    print("\\n" + "="*80)
