"""Scrape TradersUnion BTC & ETH forecasts and format for Mattermost."""

import sys
from playwright.sync_api import sync_playwright

URLS = {
    "BTC/USD": "https://tradersunion.com/currencies/forecast/btc-usd/daily-and-weekly/",
    "ETH/USD": "https://tradersunion.com/currencies/forecast/ethusd/daily-and-weekly/",
}

HORIZONS = ["24H Prediction", "48H Prediction", "7-Day Prediction", "1-Month Prediction"]
HORIZON_LABELS = {"24H Prediction": "24h", "48H Prediction": "48h", "7-Day Prediction": "7 days", "1-Month Prediction": "1 month"}


def scrape_forecast(page, url):
    page.goto(url, timeout=30000, wait_until="networkidle")
    text = page.inner_text("body")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    current_price = None
    predictions = {}

    for i, line in enumerate(lines):
        if line == "Current price:" and i + 1 < len(lines):
            raw = lines[i + 1].replace("$", "").replace(",", "")
            try:
                current_price = float(raw)
            except ValueError:
                pass
        if line in HORIZONS and i + 2 < len(lines):
            pct_str = lines[i + 1]
            price_str = lines[i + 2].replace("$", "").replace(",", "")
            try:
                pct = float(pct_str.replace("%", "").replace("+", ""))
                price = float(price_str)
                predictions[line] = {"pct": pct, "price": price, "pct_raw": pct_str}
            except ValueError:
                pass

    return current_price, predictions


def format_message():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--headless=new"])
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        page = ctx.new_page()

        parts = ["#### 📊 TradersUnion Forecasts\n"]

        for symbol, url in URLS.items():
            current, preds = scrape_forecast(page, url)
            if current is None:
                parts.append(f"**{symbol}** — data unavailable\n")
                continue

            parts.append(f"**{symbol}** — current: ${current:,.2f}")
            for horizon in HORIZONS:
                if horizon in preds:
                    d = preds[horizon]
                    emoji = "🟢" if d["pct"] >= 0 else "🔴"
                    sign = "+" if d["pct"] >= 0 else ""
                    parts.append(f"- {HORIZON_LABELS[horizon]}: {emoji} **${d['price']:,.2f}** ({sign}{d['pct']:.1f}%)")
            parts.append("")

        browser.close()

    source_link = "[TradersUnion](https://tradersunion.com/currencies/forecast/btc-usd/daily-and-weekly/)"
    parts.append(f"*Source: {source_link}*")
    return "\n".join(parts)


if __name__ == "__main__":
    msg = format_message()
    print(msg)
