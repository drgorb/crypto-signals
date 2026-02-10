"""Futures/derivatives data from Binance Futures API (with caching)."""

import os
import time as _time
import json
import requests
from config import BINANCE_FUTURES_URL, FUTURES_PREMIUM_BULLISH, FUTURES_PREMIUM_BEARISH

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
DERIVATIVES_CACHE_TTL = 300  # 5 minutes


def _deriv_cache_path(symbol):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"derivatives_{symbol}.json")


def _load_deriv_cache(symbol):
    path = _deriv_cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            cached = json.load(f)
        if _time.time() - cached.get("_cached_at", 0) < DERIVATIVES_CACHE_TTL:
            return cached.get("data")
    except Exception:
        pass
    return None


def _save_deriv_cache(symbol, data):
    path = _deriv_cache_path(symbol)
    with open(path, "w") as f:
        json.dump({"_cached_at": _time.time(), "data": data}, f)


def fetch_premium_index(symbol: str) -> dict | None:
    """Fetch futures premium index (mark price, index price, basis).

    Returns dict with lastFundingRate, markPrice, indexPrice, or None on failure.
    """
    try:
        url = f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex"
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "mark_price": float(data["markPrice"]),
            "index_price": float(data["indexPrice"]),
            "funding_rate": float(data["lastFundingRate"]),
            "premium_pct": (float(data["markPrice"]) - float(data["indexPrice"])) / float(data["indexPrice"]),
        }
    except Exception:
        return None


def fetch_open_interest(symbol: str) -> float | None:
    """Fetch current open interest (in contracts).

    Returns OI as float, or None on failure.
    """
    try:
        url = f"{BINANCE_FUTURES_URL}/fapi/v1/openInterest"
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        return float(resp.json()["openInterest"])
    except Exception:
        return None


def fetch_recent_funding_rates(symbol: str, limit: int = 10) -> list[float]:
    """Fetch recent funding rates for trend analysis."""
    try:
        url = f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
        resp = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
        resp.raise_for_status()
        return [float(r["fundingRate"]) for r in resp.json()]
    except Exception:
        return []


def compute_derivatives_score(symbol: str, current_price: float,
                               prev_price: float | None = None,
                               prev_oi: float | None = None) -> dict:
    """Compute a derivatives sentiment score from futures data.

    Returns:
        {
            "score": float (-1.0 to 1.0),
            "premium_pct": float or None,
            "open_interest": float or None,
            "funding_rate": float or None,
            "details": str
        }
    """
    result = {"score": 0.0, "premium_pct": None, "open_interest": None,
              "funding_rate": None, "details": "no data"}

    cached = _load_deriv_cache(symbol)
    if cached is not None:
        return cached

    premium = fetch_premium_index(symbol)
    oi = fetch_open_interest(symbol)

    if premium is None and oi is None:
        return result

    components = []
    details = []

    if premium is not None:
        result["premium_pct"] = round(premium["premium_pct"], 6)
        result["funding_rate"] = round(premium["funding_rate"], 6)

        # Premium signal
        if premium["premium_pct"] > FUTURES_PREMIUM_BULLISH:
            components.append(0.5)
            details.append(f"premium {premium['premium_pct']:.4%} bullish")
        elif premium["premium_pct"] < FUTURES_PREMIUM_BEARISH:
            components.append(-0.5)
            details.append(f"premium {premium['premium_pct']:.4%} bearish")
        else:
            components.append(0.0)
            details.append(f"premium {premium['premium_pct']:.4%} neutral")

        # Funding rate as contrarian signal
        fr = premium["funding_rate"]
        if fr > 0.0005:
            components.append(-0.3)  # very positive funding = overleveraged longs
            details.append(f"funding {fr:.4%} bearish contrarian")
        elif fr < -0.0005:
            components.append(0.3)   # very negative funding = overleveraged shorts
            details.append(f"funding {fr:.4%} bullish contrarian")

    if oi is not None:
        result["open_interest"] = oi
        # OI + price direction: if we have prev values, check alignment
        if prev_oi is not None and prev_price is not None:
            oi_rising = oi > prev_oi
            price_rising = current_price > prev_price
            if oi_rising and price_rising:
                components.append(0.4)
                details.append("rising OI + rising price = bullish")
            elif oi_rising and not price_rising:
                components.append(-0.4)
                details.append("rising OI + falling price = bearish")
            elif not oi_rising and price_rising:
                details.append("falling OI + rising price = weak rally")
            else:
                details.append("falling OI + falling price = capitulation")

    if components:
        import numpy as np
        result["score"] = round(float(np.clip(np.mean(components), -1, 1)), 4)
    result["details"] = "; ".join(details) if details else "no signals"

    _save_deriv_cache(symbol, result)
    return result
