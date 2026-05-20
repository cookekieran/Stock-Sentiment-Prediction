import yfinance as yf
import pandas as pd
from pathlib import Path

ALL_TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
START_DATE = "2023-01-01"

prices_df = yf.download(
    tickers=ALL_TICKERS,
    start=START_DATE,
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

OUTPUT_DIR = Path("mag7_prices")
OUTPUT_DIR.mkdir(exist_ok=True)

parquet_path = OUTPUT_DIR / "mag7_daily_prices.parquet"

prices_df.to_parquet(parquet_path, index=False)

print("script finished")