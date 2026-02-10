"""Configuration for CryptoSignals v7 — quality over quantity, profitable after fees."""

# ========== SYMBOLS ==========
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SYMBOL_NAMES = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"}

# ========== TIMEFRAMES ==========
TIMEFRAMES_SIGNAL = ["15m", "1h"]  # trend-follow on both

# ========== INDICATOR SETTINGS ==========
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
RSI_PERIOD = 14
SMA_TREND_PERIOD = 50
EMA_TREND_PERIOD = 20
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14

# ========== v7 SIGNAL THRESHOLDS (strict — only cream of the crop) ==========
# Only STRONG + TREND_FOLLOW signals; no STANDARD
V7_RSI_OVERSOLD = 35         # not used (no STRONG signals)
V7_RSI_OVERBOUGHT = 65       # not used (no STRONG signals)
V7_VOLUME_THRESHOLD = 0.0    # disabled — no volume filter for trend-follow
V7_ADX_TREND_THRESHOLD = 25  # lowered for more trend-follow entries
V7_EMA_PULLBACK_PCT = 0.003  # price within 0.3% of 20-EMA

# ========== MINIMUM ATR FILTER ==========
# Round-trip cost ~0.21%, need 1% net profit → move needs 1.21%
# ATR should be at least 1.5% of price to ensure adequate volatility
V7_MIN_ATR_PCT = 0.015  # 1.5% of price

# ========== HIGHER TF CONFIRMATION ==========
V7_HTF_SMA_PERIOD = 50  # 1h 50-SMA — price must be on right side

# ========== TRAILING STOP CONFIG (wider — let trades breathe) ==========
TRAILING_15M = {
    "initial_sl_pct": 0.02,        # 2% initial stop loss
    "breakeven_trigger": 0.015,    # move SL to breakeven at +1.5%
    "trail_1_trigger": 0.02,       # start trailing at +2%
    "trail_1_distance": 0.01,      # trail 1% behind
    "trail_2_trigger": 0.03,       # tighter trail at +3%
    "trail_2_distance": 0.0075,    # trail 0.75% behind
    "max_hold_candles": 672,       # 7 days on 15m
}

# ========== POSITION SIZING ==========
V7_MAX_CONCURRENT_PER_ASSET = 1   # max 1 position per asset
V7_STRONG_LEVERAGE = 1.5          # 1.5x for STRONG signals
V7_TREND_LEVERAGE = 1.0           # 1x for trend-follow

# ========== COOLDOWNS ==========
V7_SL_COOLDOWN_CANDLES = 8  # wait 2 hours (8 × 15m) after SL

# ========== FEE MODEL ==========
# Taker scenario (Binance VIP1)
TAKER_FEE_PCT = 0.00075   # 0.075% per side
# Maker scenario
MAKER_FEE_PCT = 0.0002    # 0.02% per side
# Slippage
SLIPPAGE_PCT = 0.0003     # 0.03% per side

# ========== BINANCE API ==========
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
KLINES_LIMIT = 100
