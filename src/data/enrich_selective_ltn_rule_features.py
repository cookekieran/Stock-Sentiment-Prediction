"""Add leakage-safe selective predicates for LTN-style transition rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FUNDAMENTAL_SURPRISE_COLUMNS = {
    "fundamental_revenue_yoy": "rule_revenue_yoy_ticker_z",
    "fundamental_operating_margin_yoy_change": "rule_operating_margin_yoy_change_ticker_z",
}
RULE_FEATURE_COLUMNS = [
    "rule_days_since_fundamental_release",
    "rule_release_proximity_10d",
    "rule_revenue_yoy_ticker_z",
    "rule_operating_margin_yoy_change_ticker_z",
]
PERSISTENCE_WINDOWS = [3, 5, 10]
for window in PERSISTENCE_WINDOWS:
    for pressure in ["risk_on", "risk_off"]:
        RULE_FEATURE_COLUMNS.extend(
            [
                f"rule_{pressure}_shock_mean_{window}d",
                f"rule_{pressure}_shock_persistence_{window}d",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-path", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-schema-path", type=Path, required=True)
    return parser.parse_args()


def expanding_ticker_zscore(series: pd.Series) -> pd.Series:
    prior_mean = series.shift(1).expanding(min_periods=2).mean()
    prior_std = series.shift(1).expanding(min_periods=2).std()
    zscore = (series - prior_mean) / prior_std.replace(0.0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)


def release_relative_zscore(group: pd.DataFrame, source: str) -> pd.Series:
    release_dates = group["fundamental_available_from_date"]
    is_new_release = release_dates.notna() & release_dates.ne(release_dates.shift(1))
    release_values = group.loc[is_new_release, source]
    release_scores = expanding_ticker_zscore(release_values)
    daily_scores = pd.Series(np.nan, index=group.index, dtype=float)
    daily_scores.loc[release_scores.index] = release_scores
    return daily_scores.ffill().fillna(0.0)


def main() -> None:
    args = parse_args()
    daily = pd.read_parquet(args.daily_path).copy()
    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))
    daily["anchor_trading_date"] = pd.to_datetime(daily["anchor_trading_date"]).dt.normalize()
    daily["fundamental_available_from_date"] = pd.to_datetime(
        daily["fundamental_available_from_date"]
    ).dt.normalize()
    daily = daily.sort_values(["ticker", "anchor_trading_date"]).reset_index(drop=True)

    days_since_release = (
        daily["anchor_trading_date"] - daily["fundamental_available_from_date"]
    ).dt.days.clip(lower=0)
    daily["rule_days_since_fundamental_release"] = days_since_release
    daily["rule_release_proximity_10d"] = (1.0 - days_since_release / 10.0).clip(0.0, 1.0)

    for source, target in FUNDAMENTAL_SURPRISE_COLUMNS.items():
        daily[target] = 0.0
        for _, group in daily.groupby("ticker", sort=False):
            daily.loc[group.index, target] = release_relative_zscore(group, source)

    grouped = daily.groupby("ticker", group_keys=False)
    for window in PERSISTENCE_WINDOWS:
        for pressure in ["risk_on", "risk_off"]:
            source = f"driver_max_{pressure}_shock"
            mean_column = f"rule_{pressure}_shock_mean_{window}d"
            persistence_column = f"rule_{pressure}_shock_persistence_{window}d"
            daily[mean_column] = grouped[source].transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            daily[persistence_column] = grouped[source].transform(
                lambda values: values.gt(0.0).rolling(window, min_periods=1).mean()
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(args.output_dir / "latent_state_daily.parquet", index=False)
    for split, part in daily.groupby("split", sort=False):
        part.to_parquet(args.output_dir / f"latent_state_{split}.parquet", index=False)
    output_schema = {
        **schema,
        "feature_columns": list(dict.fromkeys([*schema["feature_columns"], *RULE_FEATURE_COLUMNS])),
        "selective_ltn_rule_feature_columns": RULE_FEATURE_COLUMNS,
        "selective_ltn_rule_feature_leakage_control": (
            "Release proximity uses public availability dates; ticker-relative z-scores compare each "
            "public quarterly release against prior public releases only; 3-, 5-, and 10-day shock "
            "persistence features use trailing rows including the current day."
        ),
    }
    args.output_schema_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_schema_path.write_text(json.dumps(output_schema, indent=2), encoding="utf-8")

    print("Selective LTN rule features created")
    print(f"Rows: {len(daily):,}")
    print(f"Rule features: {len(RULE_FEATURE_COLUMNS)}")
    print("\nFeature means")
    print(daily[RULE_FEATURE_COLUMNS].mean().to_string())
    print("\nRecent release rows:", int((daily["rule_release_proximity_10d"] > 0).sum()))
    print("Leakage control: prior-only ticker z-scores and trailing-only shock persistence")


if __name__ == "__main__":
    main()
