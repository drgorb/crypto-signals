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

# ========== v7 SIGNAL THRESHOLDS ==========
V7_RSI_OVERSOLD = 35         # not used (no STRONG signals)
V7_RSI_OVERBOUGHT = 65       # not used (no STRONG signals)
V7_VOLUME_THRESHOLD = 1.5    # volume surge filter re-enabled for high vol days
V7_ADX_TREND_THRESHOLD = 25  # lowered for more trend-follow entries

# ========== VOLATILITY FILTERS ==========
V7_MIN_DAILY_VOLATILITY_PCT = 1.0  # only trade when daily range > 1%
V7_HIGH_VOL_VOLUME_MULTIPLIER = 2.0  # require 2x volume on high vol days
V7_ATR_STOP_MULTIPLIER = 1.75  # use 1.75x ATR for dynamic stops instead of fixed 2%
V7_EMA_PULLBACK_PCT = 0.003  # price within 0.3% of 20-EMA

# ========== MINIMUM ATR FILTER ==========
# Binance VIP2 round-trip cost ~0.16% (0.08%+0.08%) + slippage ~0.06% = 0.22%
# Need 1% net profit → move needs 1.22%
# ATR should be at least 1.8% of price to ensure adequate volatility
V7_MIN_ATR_PCT = 0.018  # 1.8% of price (optimized for Binance VIP2)

# ========== HIGHER TF CONFIRMATION ==========
V7_HTF_SMA_PERIOD = 50  # 1h 50-SMA — price must be on right side

# ========== TRAILING STOP CONFIG (optimized for Binance VIP2) ==========
TRAILING_15M = {
    "initial_sl_pct": 0.02,        # 2% initial stop loss
    "breakeven_trigger": 0.004,    # move SL to breakeven at +0.4% (covers VIP2 fees)
    "trail_1_trigger": 0.015,      # start trailing at +1.5%
    "trail_1_distance": 0.008,     # trail 0.8% behind
    "trail_2_trigger": 0.025,      # tighter trail at +2.5%
    "trail_2_distance": 0.005,     # trail 0.5% behind
    "max_hold_candles": 672,       # 7 days on 15m
}

# ========== POSITION SIZING ==========
V7_MAX_CONCURRENT_PER_ASSET = 1   # max 1 position per asset
V7_STRONG_LEVERAGE = 1.5          # 1.5x for STRONG signals
V7_TREND_LEVERAGE = 1.0           # 1x for trend-follow

# ========== COOLDOWNS ==========
V7_SL_COOLDOWN_CANDLES = 8  # wait 2 hours (8 × 15m) after SL

# ========== FEE MODEL ==========
# Binance VIP2 fees (without BNB discount)
TAKER_FEE_PCT = 0.0008    # 0.08% per side (VIP2)
MAKER_FEE_PCT = 0.0008    # 0.08% per side (VIP2)
# Slippage
SLIPPAGE_PCT = 0.0003     # 0.03% per side

# ========== BINANCE API ==========
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
KLINES_LIMIT = 100
