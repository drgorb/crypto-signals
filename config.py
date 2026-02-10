"""Configuration for CryptoSignals."""

# Symbols to monitor (Binance format)
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Display names
SYMBOL_NAMES = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"}

# Timeframes
TIMEFRAMES = ["1m", "15m"]

# Indicator settings
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
RSI_PERIOD = 14

SMA_TREND_PERIOD = 50
VOLUME_SPIKE_THRESHOLD = 1.5  # current vol must be > 1.5x avg

# Signal thresholds (normal market)
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Dynamic RSI thresholds by regime
# Ranging market (narrow BB, bb_bw_pctile < 0.25)
RSI_OVERSOLD_RANGING = 35
RSI_OVERBOUGHT_RANGING = 65
# Normal market (bb_bw_pctile 0.25-0.75)
RSI_OVERSOLD_NORMAL = 30
RSI_OVERBOUGHT_NORMAL = 70

# Trade parameters (as fractions)
PROFIT_TARGET = 0.01   # 1%
STOP_LOSS = 0.01       # 1%

# ATR-based dynamic TP/SL
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.0   # SL = 1x ATR
ATR_TP_MULTIPLIER = 0.7   # TP = 0.7x ATR

# MACD settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Funding rate threshold (as decimal, 0.01% = 0.0001)
FUNDING_RATE_BEARISH = 0.0001  # above this = overleveraged longs
FUNDING_RATE_CONTRARIAN_BULLISH = -0.0005  # below this = extremely negative, standalone boost
FUNDING_RATE_SELL_BOOST = 0.0005  # above this = highly positive, boost SELL confidence

# Order book imbalance
ORDERBOOK_BID_RATIO_BULLISH = 1.2  # bid/ask ratio above this = bullish

# Binance API
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
KLINES_LIMIT = 100  # candles to fetch

# Polymarket API
POLYMARKET_API_URL = "https://gamma-api.polymarket.com"

# Sentiment
BEARISH_SENTIMENT_THRESHOLD = 0.65  # above this = strong bearish
BULLISH_SENTIMENT_THRESHOLD = 0.65  # above this = strong bullish (suppress SELL)

# Time-of-day windows (UTC hours)
HIGH_VOL_WINDOWS = [(13, 16), (0, 3)]   # US open, Asian open
LOW_VOL_WINDOWS = [(4, 8)]               # quiet hours — stricter thresholds

# --- ADX settings ---
ADX_PERIOD = 14
ADX_WEAK_THRESHOLD = 20       # ADX < 20 → ranging/weak trend
ADX_MODERATE_THRESHOLD = 40   # ADX 20-40 → moderate; > 40 → strong trend
ADX_SUPPRESS_MEAN_REVERSION = 40  # ADX above this suppresses mean-reversion trades

# --- Multi-timeframe settings ---
HOURLY_SMA_PERIOD = 50    # 1h 50-SMA for medium-term trend
DAILY_SMA_PERIOD = 20     # daily 20-SMA for macro trend
MTF_MIN_BULLISH_FOR_BUY = 2   # at least 2 of 3 timeframes must be bullish to BUY
MTF_MIN_BEARISH_FOR_SELL = 2  # at least 2 of 3 timeframes must be bearish to SELL

# --- Market structure settings ---
MARKET_STRUCTURE_LOOKBACK = 50   # candles to analyze
SWING_PIVOT_WINDOW = 3           # a swing high/low must be higher/lower than N candles on each side

# --- Cross-asset correlation ---
CORRELATION_LOOKBACK = 20        # rolling correlation period
CORRELATION_SUPPRESS_THRESHOLD = 0.7  # suppress ETH BUY if BTC bearish & corr > this

# --- Composite trend score thresholds ---
TREND_SCORE_DAILY_SMA_WEIGHT = 20
TREND_SCORE_HOURLY_SMA_WEIGHT = 15
TREND_SCORE_ADX_DI_WEIGHT = 20
TREND_SCORE_MARKET_STRUCTURE_WEIGHT = 25
TREND_SCORE_FUNDING_WEIGHT = 10
TREND_SCORE_BTC_CORR_WEIGHT = 10
TREND_SCORE_BUY_MIN = -20    # BUY requires trend score > this
TREND_SCORE_SELL_MAX = 20     # SELL requires trend score < this

# --- Momentum outlook (feature 1) ---
MOMENTUM_ROC_SHORT = 10
MOMENTUM_ROC_LONG = 20
MOMENTUM_RSI_SLOPE_WINDOW = 5
MOMENTUM_MACD_SLOPE_WINDOW = 3
TREND_SCORE_MOMENTUM_WEIGHT = 10

# --- Mean-reversion timing (feature 2) ---
REVERSION_TARGET_PCT = 0.01      # 1% reversion target
REVERSION_ADVERSE_PCT = 0.01     # 1% adverse move = failure
REVERSION_MIN_SUCCESS_RATE = 0.40  # suppress signals below 40%

# --- Prediction markets forward-looking (feature 3) ---
POLYMARKET_SEARCH_TAGS = ["crypto", "bitcoin", "ethereum", "federal-reserve", "fed", "regulation"]
POLYMARKET_SEARCH_LIMIT = 50
TREND_SCORE_PREDICTION_MARKET_WEIGHT = 15

# --- Futures/derivatives data (feature 4) ---
FUTURES_PREMIUM_BULLISH = 0.001    # > 0.1% premium = bullish
FUTURES_PREMIUM_BEARISH = -0.001   # < -0.1% premium = bearish
TREND_SCORE_DERIVATIVES_WEIGHT = 15
