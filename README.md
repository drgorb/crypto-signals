# CryptoSignals

Mean-reversion signal generator for BTC and ETH, targeting 1% profit per trade within a ~2% band.

## How It Works

1. **Data** — Fetches 1m and 15m candles from Binance public API
2. **Indicators** — Bollinger Bands (20, 2σ), RSI (14), volume profile
3. **Sentiment** — Checks Polymarket for crypto prediction markets that might indicate directional bias
4. **Signals**:
   - **BUY**: Price ≤ lower BB + RSI < 35 + no strong bearish sentiment
   - **SELL**: Price ≥ upper BB + RSI > 65
   - TP: +1%, SL: -1%

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point — single analysis pass |
| `data.py` | Binance API data fetching |
| `indicators.py` | Bollinger Bands, RSI, volume profile |
| `signals.py` | Signal generation logic |
| `sentiment.py` | Polymarket prediction market sentiment |
| `notify.py` | Mattermost message formatting |
| `config.py` | All configuration/thresholds |

## Roadmap

- [ ] Scheduler/loop mode
- [ ] Mattermost webhook integration
- [ ] Backtesting module
- [ ] More prediction market sources
