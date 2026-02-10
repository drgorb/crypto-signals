"""Format signal messages for Mattermost output."""

from datetime import datetime, timezone


def format_signal(signal: dict) -> str:
    """Format a single signal as a Mattermost-ready message."""
    emoji = "🟢" if signal["type"] == "BUY" else "🔴"
    strength = signal.get("strength", "")
    
    msg = f"""{emoji} **{signal['type']} {signal['symbol']}** [{strength}]
| | |
|:--|:--|
| **Price** | `${signal['price']:,.2f}` |
| **RSI** | `{signal['rsi']}` |
| **BB %** | `{signal.get('bb_pct', 'N/A')}` |
| **Take Profit** | `${signal['take_profit']:,.2f}` (+1%) |
| **Stop Loss** | `${signal['stop_loss']:,.2f}` (-1%) |
| **Vol Ratio** | `{signal.get('vol_ratio', 'N/A')}x` |
| **Sentiment** | `{signal.get('sentiment_score', 0):.1%}` bearish |

_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_"""
    return msg


def format_summary(symbol: str, price: float, rsi: float, bb_pct: float, sentiment: dict) -> str:
    """Format a no-signal summary for a symbol."""
    return (
        f"📊 **{symbol}** — No signal | "
        f"Price: `${price:,.2f}` | RSI: `{rsi:.1f}` | BB%: `{bb_pct:.3f}` | "
        f"Sentiment: `{sentiment.get('score', 0):.1%}` bearish"
    )


def print_signals(all_signals: list[dict], summaries: list[str]):
    """Print all signals and summaries to stdout."""
    print("=" * 60)
    print(f"  CryptoSignals — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)
    
    if all_signals:
        for sig in all_signals:
            print()
            print(format_signal(sig))
    else:
        print("\n  No active signals.\n")
    
    if summaries:
        print("\n" + "-" * 40)
        for s in summaries:
            print(s)
    
    print()
