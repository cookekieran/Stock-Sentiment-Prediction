import yfinance as yf
import pandas as pd
from pathlib import Path

ALL_TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
START_DATE = "2023-01-01"
END_DATE = "2026-06-02"  # yfinance excludes the end date.

prices_df = yf.download(
    tickers=ALL_TICKERS,
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=False,
    group_by="ticker",
    progress=True,
    threads=True,
)

frames = []

for ticker in ALL_TICKERS:
    df = prices_df[ticker].copy()
    df = df.reset_index()
    df["ticker"] = ticker

    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    frames.append(df)

prices_df = pd.concat(frames)

parquet_path = Path(__file__).resolve().parent.parent / "mag7_daily_prices.parquet"

prices_df.to_parquet(parquet_path, index=False)

print("script finished")
