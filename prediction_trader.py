#!/usr/bin/env python3
"""
Prediction-based trader for TradersUnion forecasts.
Implements "Buy the dip" and "Downside Catalyst" strategies.
"""

import json
import csv
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

# Trading parameters
RISK_PER_TRADE = 8000  # $8K USD risk per trade
MAX_CAPITAL_PER_TRADE = 100000  # $100K USD max position size
CSV_PATH = "/home/mroon/crypto-trades.csv"
FORECAST_PATH = "/tmp/tu_forecast.json"

# Strategy thresholds - FOCUS ON 24H FORECAST ONLY
# Longer-term forecasts are fuzzy indicators only

BUY_24H_DIP_THRESHOLD = -1.2  # If 24h predicts >1.2% drop, short the move
SELL_24H_RISE_THRESHOLD = 1.2  # If 24h predicts >1.2% rise, buy the move

# Use longer-term as context only (not for trade decisions)
MIN_PREDICTION_CONFIDENCE = 1.0  # Minimum 1% predicted 24h move to trade


BINANCE_SYMBOL_MAP = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
}

def get_current_prices() -> Dict[str, float]:
    """Fetch live prices from Binance"""
    import urllib.request
    prices = {}
    for symbol, binance_sym in BINANCE_SYMBOL_MAP.items():
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_sym}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            prices[symbol] = float(data["price"])
        except Exception as e:
            print(f"⚠️  Failed to fetch {symbol} from Binance: {e}")
    return prices


def check_exits(live_prices: Dict[str, float]) -> List[Dict]:
    """Check open trades against live prices for SL/TP hits. Returns closed trades."""
    if not live_prices:
        return []

    trades = load_existing_trades()
    if not trades:
        return []

    closed = []
    updated_rows = []
    for trade in trades:
        symbol = trade["symbol"]
        if trade["status"] != "OPEN" or symbol not in live_prices:
            updated_rows.append(trade)
            continue

        price = live_prices[symbol]
        action = trade["action"]
        sl = float(trade["stop_loss_price"])
        tp = float(trade["take_profit_price"])
        entry = float(trade["entry_price"])

        exit_reason = None
        exit_price = price

        if action == "BUY":
            if price <= sl:
                exit_reason = "STOP_LOSS"
            elif price >= tp:
                exit_reason = "TAKE_PROFIT"
        elif action == "SELL":
            if price >= sl:
                exit_reason = "STOP_LOSS"
            elif price <= tp:
                exit_reason = "TAKE_PROFIT"

        # Also close if trade is >24h old (prediction expired)
        try:
            trade_time = datetime.fromisoformat(trade["timestamp"].replace('Z', '+00:00'))
            age_hours = (datetime.now(timezone.utc) - trade_time).total_seconds() / 3600
            if age_hours > 24 and exit_reason is None:
                exit_reason = "EXPIRED"
        except:
            pass

        if exit_reason:
            trade["status"] = exit_reason
            trade["exit_price"] = str(round(price, 2))
            trade["exit_time"] = datetime.now(timezone.utc).isoformat()
            # Calculate P&L
            shares = float(trade["shares"])
            if action == "BUY":
                pnl = (price - entry) * shares
            else:
                pnl = (entry - price) * shares
            trade["pnl_usd"] = str(round(pnl, 2))
            closed.append(trade)

        updated_rows.append(trade)

    # Rewrite CSV if any trades closed
    if closed:
        rewrite_csv(updated_rows)

    return closed


def rewrite_csv(trades: List[Dict]):
    """Rewrite entire CSV with updated trade statuses"""
    fieldnames = [
        "timestamp", "symbol", "strategy", "action", "entry_price", "target_price", "shares",
        "position_value", "stop_loss_pct", "take_profit_pct", "stop_loss_price",
        "take_profit_price", "entry_reason", "target_change_pct", "risk_usd", "status",
        "exit_price", "exit_time", "pnl_usd"
    ]
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for trade in trades:
            writer.writerow(trade)


def load_latest_forecast() -> Optional[Dict]:
    """Load the latest TradersUnion forecast from /tmp/tu_forecast.json"""
    if not os.path.exists(FORECAST_PATH):
        return None
    
    try:
        with open(FORECAST_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading forecast: {e}")
        return None


def analyze_predictions(symbol: str, forecasts: Dict) -> Tuple[str, Dict]:
    """
    Analyze forecasts and determine trading strategy.
    FOCUS ON 24H FORECAST - longer-term are fuzzy indicators only.
    
    Returns:
        (strategy_name, trade_signals) or (None, {}) if no trade
    """
    if symbol not in forecasts:
        return None, {}
    
    data = forecasts[symbol]
    current_price = data["current_price"]
    f = data["forecasts"]
    
    # PRIMARY SIGNAL: 24h forecast change
    h24_change_pct = f["24h"]["change_pct"]
    h24_target_price = f["24h"]["price"]
    
    # Skip if prediction is too small (low confidence)
    if abs(h24_change_pct) < MIN_PREDICTION_CONFIDENCE:
        return None, {}
    
    # Longer-term context (fuzzy indicators only)
    context_1mo = f.get("1mo", {}).get("change_pct", 0)
    context_12mo = f.get("12mo", {}).get("change_pct", 0)
    
    # Strategy 1: Follow 24h Bearish Prediction (SHORT)
    if h24_change_pct <= BUY_24H_DIP_THRESHOLD:
        return "24h Bearish Follow", {
            "action": "SELL",
            "entry_reason": f"24h predicts drop to ${h24_target_price:,.2f} ({h24_change_pct:.1f}%). Context: 1mo({context_1mo:.1f}%), 12mo({context_12mo:.1f}%)",
            "target_price": h24_target_price,
            "target_change": h24_change_pct,
            "stop_loss_pct": 2.0,  # 2% stop loss for shorts
            "take_profit_pct": abs(h24_change_pct) * 0.8,  # 80% of predicted drop
        }
    
    # Strategy 2: Follow 24h Bullish Prediction (BUY)
    elif h24_change_pct >= SELL_24H_RISE_THRESHOLD:
        return "24h Bullish Follow", {
            "action": "BUY",
            "entry_reason": f"24h predicts rise to ${h24_target_price:,.2f} ({h24_change_pct:.1f}%). Context: 1mo({context_1mo:.1f}%), 12mo({context_12mo:.1f}%)",
            "target_price": h24_target_price,
            "target_change": h24_change_pct,
            "stop_loss_pct": 2.0,  # 2% stop loss
            "take_profit_pct": h24_change_pct * 0.8,  # 80% of predicted rise
        }
    
    return None, {}


def calculate_position_size(price: float, stop_loss_pct: float) -> Tuple[float, float]:
    """
    Calculate position size based on risk management.
    
    Returns:
        (shares, position_value)
    """
    # Risk per trade = $8K
    # Position size = Risk / (Price * Stop Loss %)
    stop_loss_dollar_per_share = price * (stop_loss_pct / 100)
    shares = RISK_PER_TRADE / stop_loss_dollar_per_share
    position_value = shares * price
    
    # Cap at max capital
    if position_value > MAX_CAPITAL_PER_TRADE:
        shares = MAX_CAPITAL_PER_TRADE / price
        position_value = MAX_CAPITAL_PER_TRADE
    
    return shares, position_value


def create_trade_record(symbol: str, strategy: str, signals: Dict, current_price: float) -> Dict:
    """Create a trade record dictionary"""
    
    shares, position_value = calculate_position_size(current_price, signals["stop_loss_pct"])
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "strategy": strategy,
        "action": signals["action"],
        "entry_price": current_price,
        "target_price": signals.get("target_price", 0),  # 24h predicted price
        "shares": round(shares, 6),
        "position_value": round(position_value, 2),
        "stop_loss_pct": signals["stop_loss_pct"],
        "take_profit_pct": signals["take_profit_pct"],
        "stop_loss_price": round(current_price * (1 - signals["stop_loss_pct"]/100 if signals["action"] == "BUY" else 1 + signals["stop_loss_pct"]/100), 2),
        "take_profit_price": round(current_price * (1 + signals["take_profit_pct"]/100 if signals["action"] == "BUY" else 1 - signals["take_profit_pct"]/100), 2),
        "entry_reason": signals["entry_reason"],
        "target_change_pct": signals["target_change"],
        "risk_usd": min(RISK_PER_TRADE, position_value * signals["stop_loss_pct"] / 100),
        "status": "OPEN"
    }


def append_to_csv(trade: Dict):
    """Append trade to CSV file"""
    fieldnames = [
        "timestamp", "symbol", "strategy", "action", "entry_price", "target_price", "shares", 
        "position_value", "stop_loss_pct", "take_profit_pct", "stop_loss_price", 
        "take_profit_price", "entry_reason", "target_change_pct", "risk_usd", "status"
    ]
    
    file_exists = os.path.exists(CSV_PATH)
    
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(trade)


def load_existing_trades() -> List[Dict]:
    """Load existing trades to avoid duplicates"""
    if not os.path.exists(CSV_PATH):
        return []
    
    trades = []
    try:
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
    except Exception as e:
        print(f"Error loading existing trades: {e}")
    
    return trades


def is_duplicate_trade(new_trade: Dict, existing_trades: List[Dict], hours_window: int = 6) -> bool:
    """Check if this trade is a duplicate within the time window"""
    new_time = datetime.fromisoformat(new_trade["timestamp"].replace('Z', '+00:00'))
    
    for trade in existing_trades[-10:]:  # Check last 10 trades only
        try:
            trade_time = datetime.fromisoformat(trade["timestamp"].replace('Z', '+00:00'))
            time_diff = (new_time - trade_time).total_seconds() / 3600  # hours
            
            if (time_diff < hours_window and 
                trade["symbol"] == new_trade["symbol"] and
                trade["strategy"] == new_trade["strategy"] and
                trade["action"] == new_trade["action"]):
                return True
        except:
            continue
    
    return False


def main():
    """Main trading logic"""
    print(f"[{datetime.now(timezone.utc)}] Starting prediction trader...")
    
    # Step 1: Check exits on open trades
    live_prices = get_current_prices()
    closed_trades = check_exits(live_prices)
    for t in closed_trades:
        pnl = float(t.get("pnl_usd", 0))
        emoji = "💰" if pnl > 0 else "💸"
        print(f"{emoji} CLOSED {t['symbol']} {t['action']} — {t['status']} @ ${t['exit_price']} (P&L: ${pnl:+,.2f})")
    
    # Step 2: Load latest forecast for new trades
    forecast_data = load_latest_forecast()
    if not forecast_data or "data" not in forecast_data:
        print("No forecast data available")
        return
    
    forecasts = forecast_data["data"]
    existing_trades = load_existing_trades()
    
    # Don't open new trades if we already have an open trade for that symbol
    open_symbols = {t["symbol"] for t in existing_trades if t["status"] == "OPEN"}
    
    trades_created = 0
    
    # Analyze each symbol
    for symbol in ["BTC/USD", "ETH/USD"]:
        if symbol in open_symbols:
            print(f"⏳ {symbol} already has an open position — skipping")
            continue
        
        strategy, signals = analyze_predictions(symbol, forecasts)
        
        if strategy and signals:
            current_price = forecasts[symbol]["current_price"]
            trade = create_trade_record(symbol, strategy, signals, current_price)
            
            # Check for duplicates
            if not is_duplicate_trade(trade, existing_trades):
                append_to_csv(trade)
                print(f"✅ Created {strategy} trade: {signals['action']} {symbol} @ ${current_price:,.2f}")
                print(f"   Risk: ${trade['risk_usd']:,.0f}, Position: ${trade['position_value']:,.0f}")
                print(f"   SL: ${trade['stop_loss_price']}, TP: ${trade['take_profit_price']}")
                trades_created += 1
            else:
                print(f"⚠️  Skipped duplicate {strategy} trade for {symbol}")
        else:
            print(f"ℹ️  No trading signal for {symbol}")
    
    if trades_created == 0:
        print("No new trades created")
    else:
        print(f"Created {trades_created} new trades")


if __name__ == "__main__":
    main()