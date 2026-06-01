"""Build daily labels for news-associated material repricing events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_LABELS = {
    "no_material_event": 0,
    "risk_off_event": 1,
    "risk_on_event": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-path", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument(
        "--prices-path",
        type=Path,
        default=Path("mag7_prices/mag7_daily_prices.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-schema-path", type=Path, required=True)
    parser.add_argument("--forecast-horizon", type=int, default=5)
    parser.add_argument(
        "--material-return-threshold",
        type=float,
        default=0.05,
        help="Absolute ticker move required to label a material repricing event.",
    )
    parser.add_argument(
        "--volatility-multiple",
        type=float,
        default=0.0,
        help=(
            "Optional minimum move expressed as a multiple of realized_volatility_20d. "
            "The effective threshold is the maximum of the fixed and volatility-adjusted thresholds."
        ),
    )
    return parser.parse_args()


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(path).copy()
    prices.columns = [str(column).lower().replace(" ", "_") for column in prices.columns]
    close_column = "adj_close" if "adj_close" in prices.columns else "close"
    required = {"ticker", "date", close_column}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError(f"Price parquet is missing required columns: {missing}")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["anchor_trading_date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["event_anchor_close"] = pd.to_numeric(prices[close_column], errors="coerce")
    prices = prices.dropna(subset=["ticker", "anchor_trading_date", "event_anchor_close"])
    return prices.sort_values(["ticker", "anchor_trading_date"]).reset_index(drop=True)


def add_forward_path_labels(
    prices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("--forecast-horizon must be at least 1.")
    parts = []
    for _, ticker_df in prices.groupby("ticker", sort=False):
        ticker_df = ticker_df.sort_values("anchor_trading_date").copy()
        close = ticker_df["event_anchor_close"]
        forward_returns = pd.concat(
            [close.shift(-offset) / close - 1.0 for offset in range(1, horizon + 1)],
            axis=1,
        )
        forward_returns.columns = [f"forward_return_day_{offset}" for offset in range(1, horizon + 1)]
        ticker_df[forward_returns.columns] = forward_returns
        forward_prices = forward_returns.add(1.0).mul(close, axis=0)
        daily_path_returns = forward_prices.pct_change(axis=1, fill_method=None)
        daily_path_returns.iloc[:, 0] = forward_returns.iloc[:, 0]
        ticker_df["forward_max_return"] = forward_returns.max(axis=1, skipna=False)
        ticker_df["forward_min_return"] = forward_returns.min(axis=1, skipna=False)
        ticker_df["forward_close_return"] = close.shift(-horizon) / close - 1.0
        ticker_df["forward_max_abs_daily_path_return"] = (
            daily_path_returns.abs().max(axis=1, skipna=False)
        )
        ticker_df["_first_risk_on_day"] = np.where(
            forward_returns.ge(0.0).any(axis=1),
            forward_returns.ge(0.0).to_numpy().argmax(axis=1) + 1,
            np.nan,
        )
        ticker_df["_first_risk_off_day"] = np.where(
            forward_returns.le(0.0).any(axis=1),
            forward_returns.le(0.0).to_numpy().argmax(axis=1) + 1,
            np.nan,
        )
        parts.append(ticker_df)
    return pd.concat(parts, ignore_index=True)


def label_events(
    daily: pd.DataFrame,
    fixed_threshold: float,
    volatility_multiple: float,
) -> pd.DataFrame:
    daily = daily.copy()
    volatility = pd.to_numeric(daily.get("realized_volatility_20d"), errors="coerce").fillna(0.0)
    daily["material_event_threshold"] = np.maximum(
        fixed_threshold,
        volatility_multiple * volatility,
    )
    risk_on = daily["forward_max_return"] >= daily["material_event_threshold"]
    risk_off = daily["forward_min_return"] <= -daily["material_event_threshold"]
    daily["material_event_label"] = "no_material_event"
    daily.loc[risk_off & ~risk_on, "material_event_label"] = "risk_off_event"
    daily.loc[risk_on & ~risk_off, "material_event_label"] = "risk_on_event"

    both = risk_on & risk_off
    if both.any():
        on_days = []
        off_days = []
        for _, row in daily.loc[both].iterrows():
            on_days.append(
                next(
                    offset
                    for offset in range(1, int(row["material_event_horizon_days"]) + 1)
                    if row[f"forward_return_day_{offset}"] >= row["material_event_threshold"]
                )
            )
            off_days.append(
                next(
                    offset
                    for offset in range(1, int(row["material_event_horizon_days"]) + 1)
                    if row[f"forward_return_day_{offset}"] <= -row["material_event_threshold"]
                )
            )
        daily.loc[both, "material_event_label"] = np.where(
            np.asarray(off_days) <= np.asarray(on_days),
            "risk_off_event",
            "risk_on_event",
        )
    daily["material_event_id"] = daily["material_event_label"].map(EVENT_LABELS).astype(int)
    daily["is_material_event"] = (daily["material_event_id"] != EVENT_LABELS["no_material_event"]).astype(int)
    return daily


def main() -> None:
    args = parse_args()
    daily = pd.read_parquet(args.daily_path).copy()
    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))
    daily["ticker"] = daily["ticker"].astype(str).str.upper()
    daily["anchor_trading_date"] = pd.to_datetime(daily["anchor_trading_date"]).dt.normalize()
    prices = add_forward_path_labels(load_prices(args.prices_path), args.forecast_horizon)
    forward_columns = [
        "ticker",
        "anchor_trading_date",
        "event_anchor_close",
        "forward_max_return",
        "forward_min_return",
        "forward_close_return",
        "forward_max_abs_daily_path_return",
        *[f"forward_return_day_{offset}" for offset in range(1, args.forecast_horizon + 1)],
    ]
    daily = daily.merge(
        prices[forward_columns],
        on=["ticker", "anchor_trading_date"],
        how="left",
        validate="one_to_one",
    )
    daily["material_event_horizon_days"] = args.forecast_horizon
    daily = daily.dropna(subset=["forward_max_return", "forward_min_return"]).copy()
    daily = label_events(
        daily,
        fixed_threshold=args.material_return_threshold,
        volatility_multiple=args.volatility_multiple,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(args.output_dir / "material_event_daily.parquet", index=False)
    for split, part in daily.groupby("split", sort=False):
        part.to_parquet(args.output_dir / f"material_event_{split}.parquet", index=False)

    event_schema = {
        **schema,
        "target_column": "material_event_id",
        "target_label_column": "material_event_label",
        "target_labels": EVENT_LABELS,
        "material_event_horizon_days": args.forecast_horizon,
        "material_return_threshold": args.material_return_threshold,
        "volatility_multiple": args.volatility_multiple,
        "excluded_forward_looking_inputs": sorted(
            set(
                schema.get("excluded_forward_looking_inputs", [])
                + [
                    "event_anchor_close",
                    "forward_max_return",
                    "forward_min_return",
                    "forward_close_return",
                    "forward_max_abs_daily_path_return",
                    "material_event_id",
                    "material_event_label",
                    "is_material_event",
                    "material_event_threshold",
                    *[f"forward_return_day_{offset}" for offset in range(1, args.forecast_horizon + 1)],
                ]
            )
        ),
    }
    args.output_schema_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_schema_path.write_text(json.dumps(event_schema, indent=2), encoding="utf-8")

    print("Material event dataset created")
    print(f"Rows: {len(daily):,}")
    print(f"Forecast horizon: {args.forecast_horizon} trading days")
    print(f"Fixed material threshold: {args.material_return_threshold:.2%}")
    print(f"Volatility multiple: {args.volatility_multiple:.2f}")
    print("\nEvent targets")
    print(pd.crosstab(daily["split"], daily["material_event_label"]).to_string())
    print("\nEvent target proportions")
    print(daily["material_event_label"].value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
