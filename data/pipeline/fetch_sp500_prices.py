from pathlib import Path

import yfinance as yf


TICKER = "^GSPC"
START_DATE = "2023-01-01"
END_DATE = "2026-06-02"  # yfinance excludes the end date.
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "sp500_daily_prices.parquet"


prices = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False,
)

if prices.empty:
    raise RuntimeError("No S&P 500 price data was returned.")

if hasattr(prices.columns, "levels"):
    prices.columns = prices.columns.get_level_values(0)

prices = prices.reset_index()
prices.columns = [str(column).lower().replace(" ", "_") for column in prices.columns]
prices["ticker"] = TICKER
prices.to_parquet(OUTPUT_PATH, index=False)

print(f"Saved {len(prices)} rows to {OUTPUT_PATH}")
