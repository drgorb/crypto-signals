"""
Backtest the main "Trend Confirmation & Dip Buying" strategy with a
fixed risk of $8,000 per trade and a hard position size cap of $200,000.
NON-COMPOUNDING.
"""
# ... (imports are the same)

# --- Config ---
START_CAPITAL = 200000.0
RISK_PER_TRADE_USD = 8000.0
MAX_POSITION_USD = 200000.0
# ... (rest of config is the same)

# ... (prepare_data is the same) ...

def run_backtest(df: pd.DataFrame, symbol: str) -> dict:
    print(f"Running backtest for {symbol} (Fixed ${RISK_PER_TRADE_USD} risk, max pos ${MAX_POSITION_USD}, non-compounding)...")
    capital = START_CAPITAL
    position = "flat"
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size_asset = 0.0
    entry_idx = 0
    trades = []
    equity_curve = [START_CAPITAL]
    max_trade_size_log = 0

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # --- Manage open positions ---
        if position != "flat":
            # (exit logic is the same as before)
            pnl = 0; exit_reason = None; exit_price = 0
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
                
                if atr_val == 0 or entry_price == 0:
                    position = "flat"; continue

                stop_loss_dist = atr_val * ATR_STOP_MULT
                
                # Ideal size based on risk
                ideal_size = RISK_PER_TRADE_USD / stop_loss_dist
                
                # Max size based on capital cap
                max_size = MAX_POSITION_USD / entry_price
                
                position_size_asset = min(ideal_size, max_size)
                max_trade_size_log = max(max_trade_size_log, position_size_asset * entry_price)

                trade_value = position_size_asset * entry_price
                capital -= trade_value * (FEE_BPS / 10000) # Fee on entry
                
                stop_loss = entry_price - stop_loss_dist if position == 'long' else entry_price + stop_loss_dist
                entry_idx = i
        
        # (equity curve update is the same)
        current_equity = capital
        if position == 'long': current_equity += (current_row['close'] - entry_price) * position_size_asset
        elif position == 'short': current_equity += (entry_price - current_row['close']) * position_size_asset
        equity_curve.append(current_equity)
    
    # (final position close is the same)
    if position != "flat":
        pnl = (df['close'].iloc[-1] - entry_price) * position_size_asset if position == 'long' else (entry_price - df['close'].iloc[-1]) * position_size_asset
        capital += pnl
        trades.append({"pnl_usd": pnl})

    # --- Compile results ---
    # (compilation is the same)
    final_capital = equity_curve[-1]
    total_return = (final_capital - START_CAPITAL) / START_CAPITAL * 100
    wins = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    bh_return = (df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0] * 100
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    results = {
        "symbol": symbol,
        "final_capital": f"${final_capital:,.2f}",
        "total_profit": f"${final_capital - START_CAPITAL:,.2f}",
        "total_return_pct": f"{total_return:+.2f}%",
        "buy_and_hold_pct": f"{bh_return:+.2f}%",
        "alpha_pct": f"{total_return - bh_return:+.2f}%",
        "total_trades": len(trades),
        "win_rate_pct": f"{win_rate:.1f}%",
        "max_drawdown_pct": f"{max_drawdown:.1f}%",
        "max_trade_size_usd": f"${max_trade_size_log:,.2f}"
    }
    return results

if __name__ == "__main__":
    results = []
    for asset in ["BTCUSDT", "ETHUSDT"]:
        df = prepare_data(asset, TEST_PERIOD_DAYS + 50)
        res = run_backtest(df, asset)
        results.append(res)
    
    print("\\n" + "="*80)
    print(" " * 15 + "Main Strategy (Capped at $200k, Fixed $8k Risk, Non-Compounding)")
    print("="*80)
    
    for res in results:
        print("")
        for key, val in res.items():
            print(f"  {key.replace('_', ' ').title():<25}: {val}")
            
    print("\\n" + "="*80)
