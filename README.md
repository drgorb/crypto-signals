# CryptoSignals v7

A profitable cryptocurrency trading signal system using technical analysis and trend-following strategies.

## Overview

CryptoSignals generates automated buy/sell alerts for BTC and ETH using:
- **Technical Analysis**: Bollinger Bands, RSI, ADX, EMA
- **Trend Following**: Multi-timeframe analysis (15m + 1h)
- **Position Tracking**: Automated trailing stops
- **Risk Management**: 2% initial stop loss, dynamic trailing

## Performance

**Backtested Results (May 2025 - Feb 2026)**:
- **+12.56% return** over 9.5 months
- **37 trades** (~1 per week)
- **51.4% win rate**
- **Max drawdown**: -2.22%
- **Profitable after fees** (0.1% round-trip)

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run analysis**:
   ```bash
   python3 alert.py
   ```

3. **Backtest**:
   ```bash
   python3 backtest_v7_capital.py
   ```

## Live Trading Setup

The system can be integrated with:
- **Mattermost**: Real-time alerts
- **OpenClaw**: Automated scheduling
- **Position tracking**: JSON state management

## Strategy Evolution

### v1-v6: Learning Phase
- Started with simple Bollinger Band + RSI
- Added trailing stops, multi-timeframe analysis
- **Key lesson**: High-frequency trading gets destroyed by fees

### v7: Profitable System
- **Trend-following only**: Dropped mean-reversion signals
- **Multi-timeframe**: 15m + 1h analysis
- **No volume filter**: Volume doesn't matter for trend-following
- **Wide stops**: 2% SL gives trades room to breathe

## Key Components

### Core Files
- `signals_v7.py`: Signal generation logic
- `indicators.py`: Technical indicator calculations
- `data.py`: Binance API integration with caching
- `positions.py`: Position tracking and trailing stops
- `alert.py`: Live alert system

### Configuration
- `config_v7.py`: All strategy parameters
- Easily adjustable: RSI levels, ADX threshold, trailing stops

### Backtesting
- `backtest_v7_capital.py`: Realistic capital simulation
- Includes fees, slippage, and compounding
- Monthly P&L breakdown

## Strategy Details

### Entry Criteria
**Trend-Follow Signals**:
- ADX > 25 (strong trend)
- Price within 0.3% of 20-EMA (pullback)
- Multi-timeframe confirmation
- No volume requirement

### Exit Strategy
1. **Initial SL**: 2% from entry
2. **Breakeven**: Move SL to entry at +1.5% profit
3. **Trailing**: 1% trail distance at +2% profit
4. **Tight trail**: 0.75% trail at +3% profit

### Risk Management
- Max 1 position per asset
- Position tracking in JSON state
- 7-day maximum hold time

## Performance Analysis

The system shows consistent profitability because:
1. **Trend-following works**: Crypto trends hard
2. **Trailing stops capture big moves**: Winners run, losers are cut
3. **Low frequency**: Avoids death by fees
4. **Multi-timeframe**: Reduces false signals

## Disclaimer

- Past performance ≠ future results
- Backtest includes realistic fees and slippage
- Use appropriate position sizing
- Cryptocurrency trading carries significant risk

## License

MIT License - See LICENSE file for details