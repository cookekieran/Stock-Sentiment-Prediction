"""
Fetch broad macroeconomic series from FRED and prepare monthly features.

Outputs:
- data/raw/macro/fred_macro_raw.parquet
- data/raw/macro/fred_macro_raw.csv
- data/processed/macro_monthly.parquet
- data/processed/macro_monthly.csv

The monthly file is the one intended for joining into the DeepSeek + LTN
article dataset by article month.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES: dict[str, dict[str, str]] = {
    # Inflation
    "cpi": {
        "fred_id": "CPIAUCSL",
        "description": "Consumer Price Index for All Urban Consumers",
        "frequency": "monthly",
    },
    "core_cpi": {
        "fred_id": "CPILFESL",
        "description": "Core Consumer Price Index excluding food and energy",
        "frequency": "monthly",
    },
    "pce_price_index": {
        "fred_id": "PCEPI",
        "description": "Personal Consumption Expenditures price index",
        "frequency": "monthly",
    },
    "core_pce_price_index": {
        "fred_id": "PCEPILFE",
        "description": "Core PCE price index excluding food and energy",
        "frequency": "monthly",
    },
    # Rates and yield curve
    "fed_funds_rate": {
        "fred_id": "FEDFUNDS",
        "description": "Effective federal funds rate",
        "frequency": "monthly",
    },
    "fed_target_upper": {
        "fred_id": "DFEDTARU",
        "description": "Federal funds target range upper limit",
        "frequency": "daily",
    },
    "fed_target_lower": {
        "fred_id": "DFEDTARL",
        "description": "Federal funds target range lower limit",
        "frequency": "daily",
    },
    "treasury_2y": {
        "fred_id": "DGS2",
        "description": "2-year Treasury constant maturity rate",
        "frequency": "daily",
    },
    "treasury_10y": {
        "fred_id": "DGS10",
        "description": "10-year Treasury constant maturity rate",
        "frequency": "daily",
    },
    "treasury_10y_2y_spread": {
        "fred_id": "T10Y2Y",
        "description": "10-year Treasury minus 2-year Treasury spread",
        "frequency": "daily",
    },
    # Growth and labour
    "unemployment_rate": {
        "fred_id": "UNRATE",
        "description": "Unemployment rate",
        "frequency": "monthly",
    },
    "nonfarm_payrolls": {
        "fred_id": "PAYEMS",
        "description": "All employees, total nonfarm payrolls",
        "frequency": "monthly",
    },
    "real_gdp": {
        "fred_id": "GDPC1",
        "description": "Real gross domestic product",
        "frequency": "quarterly",
    },
    "industrial_production": {
        "fred_id": "INDPRO",
        "description": "Industrial production index",
        "frequency": "monthly",
    },
    "ism_manufacturing_pmi": {
        "fred_id": "NAPM",
        "description": "ISM manufacturing PMI",
        "frequency": "monthly",
    },
    # Market stress and liquidity
    "vix": {
        "fred_id": "VIXCLS",
        "description": "CBOE Volatility Index",
        "frequency": "daily",
    },
    "trade_weighted_dollar": {
        "fred_id": "DTWEXBGS",
        "description": "Nominal broad U.S. dollar index",
        "frequency": "daily",
    },
    "wti_oil": {
        "fred_id": "DCOILWTICO",
        "description": "WTI crude oil price",
        "frequency": "daily",
    },
    "baa_10y_credit_spread": {
        "fred_id": "BAA10Y",
        "description": "Moody's BAA corporate bond yield minus 10-year Treasury",
        "frequency": "daily",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--calculation-lookback-months",
        type=int,
        default=12,
        help="Extra months fetched before start-date so YoY features exist at start-date.",
    )
    parser.add_argument("--raw-output-dir", type=Path, default=Path("data/raw/macro"))
    parser.add_argument("--processed-output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/raw/macro/fred_series.json"))
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


def fred_request(series_id: str, start_date: str, end_date: str | None, api_key: str) -> list[dict[str, Any]]:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    if end_date:
        params["observation_end"] = end_date

    url = f"{FRED_OBSERVATIONS_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FRED HTTP {exc.code} for {series_id}: {details}") from exc

    if "error_message" in payload:
        raise RuntimeError(f"FRED error for {series_id}: {payload['error_message']}")
    return payload.get("observations", [])


def fetch_series(start_date: str, end_date: str | None, api_key: str) -> pd.DataFrame:
    frames = []

    for feature_name, meta in SERIES.items():
        try:
            observations = fred_request(meta["fred_id"], start_date, end_date, api_key)
        except Exception as exc:
            print(f"Warning: skipped {feature_name} ({meta['fred_id']}): {exc}")
            continue
        frame = pd.DataFrame(observations)
        if frame.empty:
            print(f"Warning: no observations returned for {feature_name} ({meta['fred_id']})")
            continue

        frame = frame[["date", "value"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["value"] = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
        frame["feature"] = feature_name
        frame["fred_id"] = meta["fred_id"]
        frame["description"] = meta["description"]
        frame["native_frequency"] = meta["frequency"]
        frames.append(frame)

        print(f"Fetched {feature_name} ({meta['fred_id']}): {len(frame):,} observations")

    if not frames:
        raise RuntimeError("No FRED observations were fetched.")

    return pd.concat(frames, ignore_index=True)


def make_monthly_features(raw: pd.DataFrame) -> pd.DataFrame:
    values = raw.pivot_table(index="date", columns="feature", values="value", aggfunc="last")
    monthly = values.resample("ME").last().ffill()
    monthly.index.name = "month_end"
    monthly = monthly.reset_index()
    monthly["month"] = monthly["month_end"].dt.to_period("M").astype(str)

    add_yoy(monthly, "cpi", "cpi_yoy")
    add_yoy(monthly, "core_cpi", "core_cpi_yoy")
    add_yoy(monthly, "pce_price_index", "pce_yoy")
    add_yoy(monthly, "core_pce_price_index", "core_pce_yoy")

    add_change(monthly, "fed_funds_rate", 1, "fed_funds_change_1m")
    add_change(monthly, "fed_funds_rate", 3, "fed_funds_change_3m")
    add_change(monthly, "treasury_2y", 1, "treasury_2y_change_1m")
    add_change(monthly, "treasury_10y", 1, "treasury_10y_change_1m")
    add_change(monthly, "unemployment_rate", 1, "unemployment_change_1m")
    add_change(monthly, "unemployment_rate", 3, "unemployment_change_3m")
    add_yoy(monthly, "nonfarm_payrolls", "nonfarm_payrolls_yoy")
    add_yoy(monthly, "industrial_production", "industrial_production_yoy")
    add_yoy(monthly, "real_gdp", "real_gdp_yoy")
    add_pct_change(monthly, "vix", 1, "vix_change_1m")
    add_pct_change(monthly, "wti_oil", 1, "wti_oil_change_1m")
    add_pct_change(monthly, "trade_weighted_dollar", 1, "trade_weighted_dollar_change_1m")
    add_change(monthly, "baa_10y_credit_spread", 1, "baa_10y_credit_spread_change_1m")

    monthly["high_inflation"] = (monthly["cpi_yoy"] >= 3.0).astype("Int64")
    monthly["very_high_inflation"] = (monthly["cpi_yoy"] >= 5.0).astype("Int64")
    monthly["rising_rates"] = (monthly["fed_funds_change_3m"] > 0).astype("Int64")
    monthly["falling_rates"] = (monthly["fed_funds_change_3m"] < 0).astype("Int64")
    monthly["inverted_yield_curve"] = (monthly["treasury_10y_2y_spread"] < 0).astype("Int64")
    monthly["tightening_regime"] = (
        (monthly["high_inflation"] == 1) & (monthly["rising_rates"] == 1)
    ).astype("Int64")
    monthly["easing_regime"] = (
        (monthly["falling_rates"] == 1) & (monthly["fed_funds_rate"] > 0)
    ).astype("Int64")

    monthly["macro_regime"] = "neutral"
    monthly.loc[monthly["tightening_regime"] == 1, "macro_regime"] = "tightening"
    monthly.loc[monthly["easing_regime"] == 1, "macro_regime"] = "easing"
    monthly.loc[
        (monthly["inverted_yield_curve"] == 1) & (monthly["macro_regime"] == "neutral"),
        "macro_regime",
    ] = "yield_curve_stress"

    return monthly


def add_yoy(df: pd.DataFrame, source_column: str, output_column: str) -> None:
    if source_column in df.columns:
        df[output_column] = df[source_column].pct_change(12) * 100.0


def add_change(df: pd.DataFrame, source_column: str, periods: int, output_column: str) -> None:
    if source_column in df.columns:
        df[output_column] = df[source_column].diff(periods)


def add_pct_change(df: pd.DataFrame, source_column: str, periods: int, output_column: str) -> None:
    if source_column in df.columns:
        df[output_column] = df[source_column].pct_change(periods) * 100.0


def write_outputs(
    raw: pd.DataFrame,
    monthly: pd.DataFrame,
    raw_output_dir: Path,
    processed_output_dir: Path,
    metadata_path: Path,
) -> None:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    processed_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    raw.to_parquet(raw_output_dir / "fred_macro_raw.parquet", index=False)
    raw.to_csv(raw_output_dir / "fred_macro_raw.csv", index=False)
    monthly.to_parquet(processed_output_dir / "macro_monthly.parquet", index=False)
    monthly.to_csv(processed_output_dir / "macro_monthly.csv", index=False)
    metadata_path.write_text(json.dumps(SERIES, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is missing. Add it to .env or your environment.")

    model_start_date = pd.Timestamp(args.start_date)
    fetch_start_date = (
        model_start_date - pd.DateOffset(months=args.calculation_lookback_months)
    ).strftime("%Y-%m-%d")

    raw = fetch_series(fetch_start_date, args.end_date, api_key)
    monthly = make_monthly_features(raw)
    monthly = monthly[monthly["month_end"] >= model_start_date].reset_index(drop=True)
    write_outputs(
        raw,
        monthly,
        args.raw_output_dir,
        args.processed_output_dir,
        args.metadata_path,
    )

    print("\nFRED macro data saved")
    print(f"Raw observations: {len(raw):,}")
    print(f"Monthly rows: {len(monthly):,}")
    print(f"Monthly date range: {monthly['month'].min()} -> {monthly['month'].max()}")
    print("\nMacro regime counts")
    print(monthly["macro_regime"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
