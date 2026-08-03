"""Fetch MAG7 quarterly income statements and balance sheets from Alpha Vantage."""

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
FUNCTIONS = {
    "income_statement": "INCOME_STATEMENT",
    "balance_sheet": "BALANCE_SHEET",
}
API_URL = "https://www.alphavantage.co/query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--api-key-env", default="ALPHA_VANTAGE_API_KEY")
    parser.add_argument("--request-delay-seconds", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=Path("data/raw/company_fundamentals/alphavantage"),
    )
    parser.add_argument(
        "--processed-output-dir",
        type=Path,
        default=Path("data/processed/company_fundamentals"),
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


def fetch_payload(
    ticker: str,
    function: str,
    api_key: str,
    retries: int,
) -> dict:
    params = urllib.parse.urlencode(
        {"function": function, "symbol": ticker, "apikey": api_key}
    )
    url = f"{API_URL}?{params}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
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
    raise RuntimeError(f"Failed to fetch {ticker} {function}: {last_error}") from last_error


def tidy_reports(
    ticker: str,
    statement_type: str,
    payload: dict,
    start_date: pd.Timestamp,
    fetched_at_utc: str,
) -> pd.DataFrame:
    reports = pd.DataFrame(payload.get("quarterlyReports", []))
    if reports.empty:
        return pd.DataFrame()
    reports["fiscal_period_end"] = pd.to_datetime(reports.pop("fiscalDateEnding")).dt.normalize()
    reports = reports[reports["fiscal_period_end"] >= start_date].copy()
    if reports.empty:
        return pd.DataFrame()
    reports.insert(0, "ticker", ticker)
    reports.insert(1, "statement_type", statement_type)
    reports["fetched_at_utc"] = fetched_at_utc
    for column in reports.columns:
        if column not in {
            "ticker",
            "statement_type",
            "fiscal_period_end",
            "reportedCurrency",
            "fetched_at_utc",
        }:
            reports[column] = pd.to_numeric(
                reports[column].replace({"None": None, "": None}),
                errors="coerce",
            )
    return reports.sort_values("fiscal_period_end").reset_index(drop=True)


def long_table(wide_statements: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "ticker",
        "statement_type",
        "fiscal_period_end",
        "reportedCurrency",
        "fetched_at_utc",
    ]
    value_columns = [column for column in wide_statements.columns if column not in id_columns]
    return wide_statements.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="line_item",
        value_name="value",
    ).sort_values(["ticker", "statement_type", "fiscal_period_end", "line_item"])


def modeling_table(statements: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "ticker",
        "statement_type",
        "fiscal_period_end",
        "reportedCurrency",
        "fetched_at_utc",
    ]
    value_columns = [column for column in statements.columns if column not in id_columns]
    long = statements.melt(
        id_vars=["ticker", "statement_type", "fiscal_period_end"],
        value_vars=value_columns,
        var_name="line_item",
        value_name="value",
    )
    long["feature"] = "fundamental_" + long["statement_type"] + "_" + long["line_item"]
    return (
        long.pivot_table(
            index=["ticker", "fiscal_period_end"],
            columns="feature",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .rename_axis(columns=None)
        .sort_values(["ticker", "fiscal_period_end"])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    start_date = pd.Timestamp(args.start_date).normalize()
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    args.raw_output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    coverage_rows = []
    request_count = 0
    for ticker in [str(value).upper() for value in args.tickers]:
        ticker_dir = args.raw_output_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        for statement_type, function in FUNCTIONS.items():
            if request_count:
                time.sleep(args.request_delay_seconds)
            payload = fetch_payload(ticker, function, api_key, args.retries)
            request_count += 1
            (ticker_dir / f"{statement_type}_raw.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            statement = tidy_reports(
                ticker,
                statement_type,
                payload,
                start_date,
                fetched_at_utc,
            )
            if statement.empty:
                raise RuntimeError(
                    f"No quarterly {statement_type} reports returned for {ticker} "
                    f"from {args.start_date} onward."
                )
            statement.to_parquet(ticker_dir / f"{statement_type}_quarterly.parquet", index=False)
            statement.to_csv(ticker_dir / f"{statement_type}_quarterly.csv", index=False)
            frames.append(statement)
            coverage_rows.append(
                {
                    "ticker": ticker,
                    "statement_type": statement_type,
                    "quarter_count": int(statement["fiscal_period_end"].nunique()),
                    "earliest_fiscal_period_end": statement["fiscal_period_end"].min(),
                    "latest_fiscal_period_end": statement["fiscal_period_end"].max(),
                    "field_count": max(len(statement.columns) - 5, 0),
                    "covers_requested_start_date": bool(
                        statement["fiscal_period_end"].min() <= start_date + pd.offsets.QuarterEnd()
                    ),
                }
            )
            print(
                f"Fetched {ticker} {statement_type}: "
                f"{statement['fiscal_period_end'].nunique()} quarters"
            )

    statements = pd.concat(frames, ignore_index=True, sort=False)
    coverage = pd.DataFrame(coverage_rows).sort_values(["ticker", "statement_type"])
    tidy = long_table(statements)
    modeling = modeling_table(statements)
    statements.to_parquet(args.raw_output_dir / "mag7_quarterly_statements_wide.parquet", index=False)
    statements.to_csv(args.raw_output_dir / "mag7_quarterly_statements_wide.csv", index=False)
    tidy.to_parquet(args.raw_output_dir / "mag7_quarterly_statements_tidy.parquet", index=False)
    tidy.to_csv(args.raw_output_dir / "mag7_quarterly_statements_tidy.csv", index=False)
    coverage.to_parquet(args.raw_output_dir / "coverage_summary.parquet", index=False)
    coverage.to_csv(args.raw_output_dir / "coverage_summary.csv", index=False)
    modeling.to_parquet(
        args.processed_output_dir / "mag7_quarterly_fundamentals_alphavantage_wide.parquet",
        index=False,
    )
    modeling.to_csv(
        args.processed_output_dir / "mag7_quarterly_fundamentals_alphavantage_wide.csv",
        index=False,
    )
    metadata = {
        "source": "Alpha Vantage",
        "requested_start_date": args.start_date,
        "fetched_at_utc": fetched_at_utc,
        "api_request_count": request_count,
        "tickers": sorted(statements["ticker"].unique().tolist()),
        "statement_types": sorted(FUNCTIONS),
        "complete_requested_coverage": bool(coverage["covers_requested_start_date"].all()),
        "modeling_rows": int(len(modeling)),
        "modeling_feature_count": max(len(modeling.columns) - 2, 0),
    }
    (args.raw_output_dir / "fetch_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("\nCoverage summary")
    print(coverage.to_string(index=False))
    print(
        f"\nWrote modeling table to "
        f"{args.processed_output_dir / 'mag7_quarterly_fundamentals_alphavantage_wide.parquet'}"
    )


if __name__ == "__main__":
    main()
