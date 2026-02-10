"""Fetch price data from Binance public API with local CSV caching."""

import os
import requests
import pandas as pd
from config import BINANCE_BASE_URL, BINANCE_FUTURES_URL, KLINES_LIMIT

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _cache_path(symbol: str, interval: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{symbol}_{interval}.csv")


def _load_cache(symbol: str, interval: str) -> pd.DataFrame | None:
    path = _cache_path(symbol, interval)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def _save_cache(symbol: str, interval: str, df: pd.DataFrame):
    path = _cache_path(symbol, interval)
    df.to_csv(path, index=False)


def fetch_klines_paginated(symbol: str, interval: str, start_ms: int, end_ms: int = None, batch_size: int = 1000) -> pd.DataFrame:
    """Fetch klines in batches to overcome the 1000-candle API limit.
    
    Args:
        symbol: Trading pair
        interval: Candle interval (e.g. '15m')
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds (default: now)
        batch_size: Candles per request (max 1000)
    
    Returns DataFrame with all candles concatenated.
    """
    import time
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    
    # Check cache — only fetch what's missing
    cached = _load_cache(symbol, interval)
    if cached is not None and len(cached) > 0:
        cached_start = int(cached["timestamp"].iloc[0].timestamp() * 1000)
        cached_end = int(cached["timestamp"].iloc[-1].timestamp() * 1000)
        # If cache covers our range (with 1h tolerance for recent data), use it
        if cached_start <= start_ms and cached_end >= end_ms - 3600_000:
            filtered = cached[cached["timestamp"] >= pd.Timestamp(start_ms, unit="ms")]
            if len(filtered) > 0:
                return filtered.reset_index(drop=True)
        # Otherwise fetch from where cache ends
        if cached_start <= start_ms:
            start_ms = cached_end + 1
    
    url = f"{BINANCE_BASE_URL}/klines"
    all_frames = []
    cursor = start_ms
    
    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": batch_size
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            break
        
        batch_df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            batch_df[col] = batch_df[col].astype(float)
        batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"], unit="ms")
        all_frames.append(batch_df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"]].copy())
        
        # Move cursor past the last candle we got
        last_close_time = raw[-1][6]  # close_time field in ms
        cursor = last_close_time + 1
        
        if len(raw) < batch_size:
            break
    
    if not all_frames:
        return pd.DataFrame()
    
    result = pd.concat(all_frames, ignore_index=True)
    result = result.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    
    # Save to cache (merge with existing)
    cached = _load_cache(symbol, interval)
    if cached is not None:
        result = pd.concat([cached, result], ignore_index=True)
        result = result.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    _save_cache(symbol, interval, result)
    
    return result


def fetch_klines(symbol: str, interval: str, limit: int = KLINES_LIMIT) -> pd.DataFrame:
    """Fetch kline/candlestick data from Binance.
    
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    url = f"{BINANCE_BASE_URL}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"]].copy()


def fetch_hourly_klines(symbol: str, hours: int = 168) -> pd.DataFrame:
    """Fetch 1h candles for medium-term trend analysis (default: 7 days = 168 hours)."""
    import time
    start_ms = int((time.time() - hours * 3600) * 1000)
    return fetch_klines_paginated(symbol, "1h", start_ms)


def fetch_daily_klines(symbol: str, days: int = 90) -> pd.DataFrame:
    """Fetch daily candles for higher-timeframe trend filter."""
    import time
    start_ms = int((time.time() - days * 86400) * 1000)
    return fetch_klines_paginated(symbol, "1d", start_ms)


def get_current_price(symbol: str) -> float:
    """Get latest ticker price."""
    url = f"{BINANCE_BASE_URL}/ticker/price"
    resp = requests.get(url, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["price"])


def fetch_funding_rate(symbol: str) -> float | None:
    """Fetch latest funding rate from Binance Futures (cached 5 min)."""
    import time as _t, json as _j
    cache_path = os.path.join(CACHE_DIR, f"funding_{symbol}.json")
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cache_path) as f:
            c = _j.load(f)
        if _t.time() - c.get("_at", 0) < 300:
            return c.get("v")
    except Exception:
        pass
    try:
        url = f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
        resp = requests.get(url, params={"symbol": symbol, "limit": 1}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            val = float(data[-1]["fundingRate"])
            with open(cache_path, "w") as f:
                _j.dump({"_at": _t.time(), "v": val}, f)
            return val
    except Exception:
        pass
    return None


def fetch_orderbook_imbalance(symbol: str, limit: int = 20) -> float | None:
    """Fetch order book and return bid/ask volume ratio (cached 5 min)."""
    import time as _t, json as _j
    cache_path = os.path.join(CACHE_DIR, f"orderbook_{symbol}.json")
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cache_path) as f:
            c = _j.load(f)
        if _t.time() - c.get("_at", 0) < 300:
            return c.get("v")
    except Exception:
        pass
    try:
        url = f"{BINANCE_BASE_URL}/depth"
        resp = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        bid_vol = sum(float(b[1]) for b in data["bids"])
        ask_vol = sum(float(a[1]) for a in data["asks"])
        if ask_vol == 0:
            return None
        val = bid_vol / ask_vol
        with open(cache_path, "w") as f:
            _j.dump({"_at": _t.time(), "v": val}, f)
        return val
    except Exception:
        pass
    return None
