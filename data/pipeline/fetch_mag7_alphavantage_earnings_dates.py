"""Fetch MAG7 quarterly earnings release dates from Alpha Vantage."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_TICKERS = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA")
API_URL = "https://www.alphavantage.co/query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--api-key-env", default="ALPHA_VANTAGE_API_KEY")
    parser.add_argument("--request-delay-seconds", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=Path("data/raw/company_fundamentals/alphavantage"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "data/processed/company_fundamentals/"
            "mag7_quarterly_earnings_dates_alphavantage.parquet"
        ),
    )
    return parser.parse_args()


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fetch_payload(ticker: str, api_key: str, retries: int) -> dict:
    query = urllib.parse.urlencode(
        {"function": "EARNINGS", "symbol": ticker, "apikey": api_key}
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(f"{API_URL}?{query}", timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "Note" in payload or "Information" in payload:
                raise RuntimeError(payload.get("Note") or payload.get("Information"))
            if "Error Message" in payload:
                raise RuntimeError(payload["Error Message"])
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(5.0 * attempt)
    raise RuntimeError(f"Failed to fetch {ticker} earnings dates: {last_error}") from last_error


def tidy_earnings(
    ticker: str,
    payload: dict,
    start_date: pd.Timestamp,
    fetched_at_utc: str,
) -> pd.DataFrame:
    reports = pd.DataFrame(payload.get("quarterlyEarnings", []))
    if reports.empty:
        raise RuntimeError(f"No quarterly earnings returned for {ticker}.")
    reports["ticker"] = ticker
    reports["fiscal_period_end"] = pd.to_datetime(reports.pop("fiscalDateEnding")).dt.normalize()
    reports["reported_date"] = pd.to_datetime(reports.pop("reportedDate")).dt.normalize()
    reports["fetched_at_utc"] = fetched_at_utc
    reports = reports[reports["fiscal_period_end"] >= start_date].copy()
    for column in reports.columns:
        if column not in {"ticker", "fiscal_period_end", "reported_date", "fetched_at_utc"}:
            reports[column] = pd.to_numeric(
                reports[column].replace({"None": None, "": None}),
                errors="coerce",
            )
    return reports.sort_values(["ticker", "fiscal_period_end"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    start_date = pd.Timestamp(args.start_date).normalize()
    frames = []
    for index, ticker in enumerate(str(value).upper() for value in args.tickers):
        if index:
            time.sleep(args.request_delay_seconds)
        payload = fetch_payload(ticker, api_key, args.retries)
        ticker_dir = args.raw_output_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        (ticker_dir / "earnings_raw.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        reports = tidy_earnings(ticker, payload, start_date, fetched_at_utc)
        reports.to_csv(ticker_dir / "earnings_quarterly.csv", index=False)
        reports.to_parquet(ticker_dir / "earnings_quarterly.parquet", index=False)
        frames.append(reports)
        print(
            f"Fetched {ticker} earnings dates: {len(reports)} quarters "
            f"({reports['reported_date'].min().date()} -> {reports['reported_date'].max().date()})"
        )
    combined = pd.concat(frames, ignore_index=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output_path, index=False)
    combined.to_csv(args.output_path.with_suffix(".csv"), index=False)
    print(f"\nWrote {len(combined):,} quarterly earnings dates to {args.output_path}")


if __name__ == "__main__":
    main()
