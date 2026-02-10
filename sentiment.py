"""Prediction market sentiment from Polymarket (with caching)."""

import os
import time
import requests
import json
from config import (POLYMARKET_API_URL, BEARISH_SENTIMENT_THRESHOLD, BULLISH_SENTIMENT_THRESHOLD,
                    POLYMARKET_SEARCH_TAGS, POLYMARKET_SEARCH_LIMIT)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SENTIMENT_CACHE_TTL = 300  # 5 minutes


def _sentiment_cache_path():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "polymarket_sentiment.json")


def _load_sentiment_cache():
    path = _sentiment_cache_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            cached = json.load(f)
        if time.time() - cached.get("_cached_at", 0) < SENTIMENT_CACHE_TTL:
            return cached.get("data")
    except Exception:
        pass
    return None


def _save_sentiment_cache(data):
    path = _sentiment_cache_path()
    with open(path, "w") as f:
        json.dump({"_cached_at": time.time(), "data": data}, f)


def fetch_crypto_markets() -> list[dict]:
    """Fetch crypto-related prediction markets from Polymarket.

    Searches multiple tags aggressively for forward-looking markets.
    Returns list of relevant markets with their probabilities.
    """
    all_markets = {}  # dedup by slug

    for tag in POLYMARKET_SEARCH_TAGS:
        try:
            url = f"{POLYMARKET_API_URL}/markets"
            params = {"closed": "false", "limit": POLYMARKET_SEARCH_LIMIT, "tag": tag}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            markets = resp.json()

            for m in markets:
                slug = m.get("slug", m.get("id", ""))
                if slug in all_markets:
                    continue
                title = (m.get("question") or m.get("title", "")).lower()
                # Broad keyword matching for crypto + macro relevance
                keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto",
                            "fed", "rate", "interest rate", "fomc", "regulation",
                            "sec", "etf", "stablecoin"]
                if any(kw in title for kw in keywords):
                    outcomes = m.get("outcomePrices", "[]")
                    if isinstance(outcomes, str):
                        try:
                            outcomes = json.loads(outcomes)
                        except Exception:
                            outcomes = []
                    all_markets[slug] = {
                        "title": m.get("question") or m.get("title", ""),
                        "outcomes": outcomes,
                        "volume": float(m.get("volume", 0) or 0),
                        "slug": slug,
                    }
        except Exception as e:
            print(f"  ⚠ Polymarket fetch failed for tag '{tag}': {e}")

    # Sort by volume (most liquid = most informative)
    results = sorted(all_markets.values(), key=lambda x: x["volume"], reverse=True)
    return results


def compute_prediction_market_score(markets: list[dict]) -> float:
    """Compute a forward-looking sentiment score from prediction markets.

    Returns score from -1.0 (very bearish) to +1.0 (very bullish).
    """
    if not markets:
        return 0.0

    # Categorize markets and extract directional signal
    price_bullish_kw = ["above", "ath", "all-time high", "new high", "bull", "rally",
                        "rise", "$100k", "$150k", "$200k", "$5000", "$10000"]
    price_bearish_kw = ["below", "crash", "drop", "bear", "decline", "fall"]
    # Macro: rate cuts bullish for crypto, rate hikes bearish
    macro_bullish_kw = ["cut", "lower", "ease", "dovish"]
    macro_bearish_kw = ["hike", "raise", "tighten", "hawkish"]

    scores = []
    weights = []

    for m in markets:
        title_lower = m["title"].lower()
        outcomes = m.get("outcomes", [])
        yes_prob = float(outcomes[0]) if outcomes else 0.5
        vol_weight = max(1.0, m.get("volume", 1) ** 0.3)  # log-ish volume weighting

        is_price_bull = any(kw in title_lower for kw in price_bullish_kw)
        is_price_bear = any(kw in title_lower for kw in price_bearish_kw)
        is_macro_bull = any(kw in title_lower for kw in macro_bullish_kw)
        is_macro_bear = any(kw in title_lower for kw in macro_bearish_kw)

        if is_price_bull:
            # "Will BTC be above X?" → yes_prob = bullish
            scores.append(yes_prob * 2 - 1)  # map 0..1 → -1..+1
            weights.append(vol_weight)
        elif is_price_bear:
            # "Will BTC crash?" → yes_prob = bearish
            scores.append(-(yes_prob * 2 - 1))
            weights.append(vol_weight)
        elif is_macro_bull:
            scores.append(yes_prob * 2 - 1)
            weights.append(vol_weight * 0.5)  # macro = indirect
        elif is_macro_bear:
            scores.append(-(yes_prob * 2 - 1))
            weights.append(vol_weight * 0.5)

    if not scores:
        return 0.0

    import numpy as np
    weighted_score = float(np.average(scores, weights=weights))
    return round(np.clip(weighted_score, -1.0, 1.0), 4)


def get_bearish_sentiment() -> dict:
    """Analyze prediction markets for bearish crypto sentiment.
    
    Returns:
        {
            "is_bearish": bool,
            "score": float (0-1, higher = more bearish),
            "markets": list of relevant market summaries,
            "reason": str
        }
    """
    # Check cache first
    cached_result = _load_sentiment_cache()
    if cached_result is not None:
        return cached_result

    markets = fetch_crypto_markets()
    
    if not markets:
        return {
            "is_bearish": False,
            "score": 0.0,
            "markets": [],
            "reason": "No prediction market data available"
        }
    
    # Look for bearish signals in market titles/outcomes
    bearish_keywords = ["crash", "below", "drop", "bear", "decline", "fall"]
    bullish_keywords = ["above", "bull", "rise", "rally", "ath", "high"]
    
    bearish_score = 0.0
    count = 0
    summaries = []
    
    for m in markets:
        title_lower = m["title"].lower()
        outcomes = m.get("outcomes", [])
        
        is_bearish_market = any(kw in title_lower for kw in bearish_keywords)
        is_bullish_market = any(kw in title_lower for kw in bullish_keywords)
        
        yes_prob = float(outcomes[0]) if outcomes else 0.5
        
        if is_bearish_market:
            bearish_score += yes_prob
            count += 1
        elif is_bullish_market:
            bearish_score += (1 - yes_prob)
            count += 1
        
        summaries.append(f"{m['title']} (Yes: {yes_prob:.0%})")
    
    avg_score = bearish_score / count if count > 0 else 0.0

    # Also compute bullish score (inverse)
    bullish_score = 1.0 - avg_score if count > 0 else 0.0

    # Forward-looking prediction market score (-1 to +1)
    prediction_score = compute_prediction_market_score(markets)

    result = {
        "is_bearish": avg_score > BEARISH_SENTIMENT_THRESHOLD,
        "is_bullish": bullish_score > BULLISH_SENTIMENT_THRESHOLD,
        "score": round(avg_score, 3),
        "bullish_score": round(bullish_score, 3),
        "prediction_market_score": prediction_score,
        "markets": summaries[:5],
        "reason": f"Bearish score {avg_score:.1%} from {count} markets (fwd: {prediction_score:+.2f})" if count > 0 else "No directional markets found"
    }
    _save_sentiment_cache(result)
    return result
