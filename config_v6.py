"""Configuration for CryptoSignals v6 — aggressive trading with trailing stops."""

# ========== SYMBOLS ==========
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SYMBOL_NAMES = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"}

# ========== TIMEFRAMES ==========
TIMEFRAMES_SIGNAL = ["5m", "15m"]

# ========== INDICATOR SETTINGS ==========
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
RSI_PERIOD = 14
SMA_TREND_PERIOD = 50
EMA_TREND_PERIOD = 20  # for trend-following pullback detection
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14

# ========== v6 STANDARD SIGNAL THRESHOLDS (loose) ==========
# BB proximity: price within this % of band width from the band
V6_BB_PROXIMITY_PCT = 0.10  # within 10% of band (not touching required)
V6_RSI_OVERSOLD = 40        # much looser than v5's 30
V6_RSI_OVERBOUGHT = 60      # much looser than v5's 70
V6_VOLUME_THRESHOLD = 1.2   # 1.2x avg (was 1.5x)

# ========== v6 STRONG SIGNAL THRESHOLDS (v5-like) ==========
V6_STRONG_RSI_OVERSOLD = 30
V6_STRONG_RSI_OVERBOUGHT = 70
V6_STRONG_VOLUME_THRESHOLD = 1.5
# Strong signals also require BB touch (price <= bb_lower or >= bb_upper)

# ========== TRAILING STOP CONFIG ==========
# 15m timeframe
TRAILING_15M = {
    "initial_sl_pct": 0.01,        # 1% initial stop loss
    "breakeven_trigger": 0.005,    # move SL to breakeven at +0.5%
    "trail_1_trigger": 0.01,       # start trailing at +1%
    "trail_1_distance": 0.005,     # trail 0.5% behind
    "trail_2_trigger": 0.02,       # tighter trail at +2%
    "trail_2_distance": 0.0075,    # trail 0.75% behind
    "max_hold_candles": 192,       # 48 hours on 15m
}

# 5m timeframe (tighter, faster)
TRAILING_5M = {
    "initial_sl_pct": 0.005,       # 0.5% initial stop loss
    "breakeven_trigger": 0.003,    # move SL to breakeven at +0.3%
    "trail_1_trigger": 0.005,      # start trailing at +0.5%
    "trail_1_distance": 0.003,     # trail 0.3% behind
    "trail_2_trigger": 0.01,       # tighter trail at +1%
    "trail_2_distance": 0.005,     # trail 0.5% behind
    "max_hold_candles": 120,       # 10 hours on 5m
}

# ========== TREND-FOLLOWING MODE ==========
V6_ADX_TREND_THRESHOLD = 30  # ADX > 30 = strong trend → use trend-following
V6_EMA_PULLBACK_PCT = 0.002  # price within 0.2% of 20-EMA = pullback

# ========== TRADE MANAGEMENT ==========
V6_MAX_CONCURRENT_PER_ASSET = 2
V6_SL_COOLDOWN_CANDLES = 4  # wait 4 candles after SL before re-entry same direction
V6_NO_COOLDOWN_AFTER_TP = True  # can re-enter immediately after trailing/TP exit

# ========== ADX / TREND (soft modifiers, not hard gates) ==========
ADX_WEAK_THRESHOLD = 20
ADX_MODERATE_THRESHOLD = 40

# ========== HIGHER TF SETTINGS (for context, not hard gates) ==========
HOURLY_SMA_PERIOD = 50
DAILY_SMA_PERIOD = 20

# ========== MARKET STRUCTURE ==========
MARKET_STRUCTURE_LOOKBACK = 50
SWING_PIVOT_WINDOW = 3
CORRELATION_LOOKBACK = 20

# ========== BINANCE API ==========
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
KLINES_LIMIT = 100

# ========== TIME-OF-DAY (informational) ==========
HIGH_VOL_WINDOWS = [(13, 16), (0, 3)]
LOW_VOL_WINDOWS = [(4, 8)]

# ========== MOMENTUM ==========
MOMENTUM_ROC_SHORT = 10
MOMENTUM_ROC_LONG = 20
MOMENTUM_RSI_SLOPE_WINDOW = 5
MOMENTUM_MACD_SLOPE_WINDOW = 3

# ========== REVERSION (informational only in v6) ==========
REVERSION_TARGET_PCT = 0.01
REVERSION_ADVERSE_PCT = 0.01
