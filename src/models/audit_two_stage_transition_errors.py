"""Audit false-positive and missed regime transitions from a two-stage GRU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--daily-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forecast-horizon", type=int, required=True)
    parser.add_argument(
        "--gradual-largest-day-share",
        type=float,
        default=0.50,
        help="Treat a forward path as gradual when no single day explains this share of absolute movement.",
    )
    parser.add_argument("--sample-size", type=int, default=30)
    return parser.parse_args()


def add_forward_path_diagnostics(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    parts = []
    for _, ticker_df in daily.groupby("ticker", sort=False):
        ticker_df = ticker_df.sort_values("anchor_trading_date").copy()
        close = pd.to_numeric(ticker_df["anchor_close"], errors="coerce")
        daily_return = close.pct_change()
        future_returns = pd.concat(
            [daily_return.shift(-offset) for offset in range(1, horizon + 1)],
            axis=1,
        )
        absolute_path = future_returns.abs()
        path_abs_return = absolute_path.sum(axis=1, min_count=1)
        largest_abs_day = absolute_path.max(axis=1)
        ticker_df["future_close"] = close.shift(-horizon)
        ticker_df["future_return"] = ticker_df["future_close"] / close - 1.0
        ticker_df["largest_abs_daily_return_in_window"] = largest_abs_day
        ticker_df["path_abs_return"] = path_abs_return
        ticker_df["largest_day_share_of_path"] = largest_abs_day / path_abs_return.replace(0.0, np.nan)
        ticker_df["prior_5d_return"] = close.pct_change(5)
        ticker_df["prior_20d_return"] = close.pct_change(20)
        parts.append(ticker_df)
    return pd.concat(parts, ignore_index=True)


def categorise(row: pd.Series) -> str:
    if bool(row["is_transition"]):
        if not bool(row["predicted_transition"]):
            return "missed_transition"
        if row["pred_label"] == row["future_price_trend_label"]:
            return "correct_transition"
        return "detected_transition_wrong_destination"
    if bool(row["predicted_transition"]):
        return "false_positive_transition"
    return "correct_stable"


def mean_numeric(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else None


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path).copy()
    daily = pd.read_parquet(args.daily_path).copy()
    predictions["anchor_trading_date"] = pd.to_datetime(predictions["anchor_trading_date"]).dt.normalize()
    daily["anchor_trading_date"] = pd.to_datetime(daily["anchor_trading_date"]).dt.normalize()
    daily = add_forward_path_diagnostics(daily, args.forecast_horizon)

    diagnostic_columns = [
        "ticker",
        "anchor_trading_date",
        "anchor_close",
        "future_close",
        "future_return",
        "largest_abs_daily_return_in_window",
        "path_abs_return",
        "largest_day_share_of_path",
        "prior_5d_return",
        "prior_20d_return",
    ]
    audit = predictions.merge(
        daily[diagnostic_columns],
        on=["ticker", "anchor_trading_date"],
        how="left",
        validate="one_to_one",
    )
    audit["gradual_forward_move"] = (
        audit["largest_day_share_of_path"] < args.gradual_largest_day_share
    )
    audit["audit_category"] = audit.apply(categorise, axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(args.output_dir / "audited_predictions.parquet", index=False)
    audit.to_csv(args.output_dir / "audited_predictions.csv", index=False)

    summary_rows = []
    for (ticker, category), group in audit.groupby(["ticker", "audit_category"], sort=True):
        summary_rows.append(
            {
                "ticker": ticker,
                "audit_category": category,
                "rows": int(len(group)),
                "mean_transition_probability": mean_numeric(group, "transition_probability"),
                "mean_future_return": mean_numeric(group, "future_return"),
                "mean_largest_abs_daily_return_in_window": mean_numeric(
                    group, "largest_abs_daily_return_in_window"
                ),
                "mean_largest_day_share_of_path": mean_numeric(group, "largest_day_share_of_path"),
                "gradual_forward_move_share": float(group["gradual_forward_move"].mean()),
                "mean_driver_shock": mean_numeric(group, "driver_mean_shock_score"),
                "mean_driver_materiality": mean_numeric(group, "driver_mean_materiality"),
            }
        )
    ticker_summary = pd.DataFrame(summary_rows)
    ticker_summary.to_csv(args.output_dir / "ticker_category_summary.csv", index=False)

    overall_rows = []
    for category, group in audit.groupby("audit_category", sort=True):
        overall_rows.append(
            {
                "audit_category": category,
                "rows": int(len(group)),
                "mean_transition_probability": mean_numeric(group, "transition_probability"),
                "mean_future_return": mean_numeric(group, "future_return"),
                "mean_largest_abs_daily_return_in_window": mean_numeric(
                    group, "largest_abs_daily_return_in_window"
                ),
                "mean_largest_day_share_of_path": mean_numeric(group, "largest_day_share_of_path"),
                "gradual_forward_move_share": float(group["gradual_forward_move"].mean()),
                "mean_driver_shock": mean_numeric(group, "driver_mean_shock_score"),
                "mean_driver_materiality": mean_numeric(group, "driver_mean_materiality"),
            }
        )
        sample_columns = [
            column
            for column in [
                "ticker",
                "anchor_trading_date",
                "current_price_trend_label",
                "future_price_trend_label",
                "transition_probability",
                "destination_label",
                "pred_label",
                "future_return",
                "largest_abs_daily_return_in_window",
                "largest_day_share_of_path",
                "gradual_forward_move",
                "prior_5d_return",
                "prior_20d_return",
                "daily_market_narrative",
                "top_driver_1_label",
                "top_driver_1_summary",
                "top_driver_1_pressure",
                "top_driver_1_scope",
            ]
            if column in group.columns
        ]
        group.sort_values("transition_probability", ascending=False).head(args.sample_size)[
            sample_columns
        ].to_csv(args.output_dir / f"{category}_sample.csv", index=False)

    overall_summary = pd.DataFrame(overall_rows)
    overall_summary.to_csv(args.output_dir / "category_summary.csv", index=False)
    metrics = {
        "forecast_horizon": args.forecast_horizon,
        "gradual_largest_day_share": args.gradual_largest_day_share,
        "rows": int(len(audit)),
        "category_counts": audit["audit_category"].value_counts().to_dict(),
        "missed_transition_gradual_share": float(
            audit.loc[audit["audit_category"] == "missed_transition", "gradual_forward_move"].mean()
        ),
        "false_positive_gradual_share": float(
            audit.loc[audit["audit_category"] == "false_positive_transition", "gradual_forward_move"].mean()
        ),
    }
    (args.output_dir / "audit_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nOverall category summary")
    print(overall_summary.to_string(index=False))
    print("\nTicker/category summary")
    print(ticker_summary.to_string(index=False))


if __name__ == "__main__":
    main()
