"""Build matched raw and quality-filtered Qwen regime-transition datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRICE_COLUMNS = {
    "realized_volatility_20d",
    "rally_from_previous_low",
    "drawdown_from_previous_high",
    "recent_return",
    "drawdown_from_recent_high",
    "price_trend_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-path", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 45])
    parser.add_argument("--minimum-confidence", type=float, default=0.70)
    parser.add_argument("--maximum-uncertainty", type=float, default=0.30)
    return parser.parse_args()


def qwen_feature_columns(feature_columns: list[str]) -> list[str]:
    return [
        column
        for column in feature_columns
        if column not in PRICE_COLUMNS and not column.startswith("macro_")
    ]


def rebuild_future_targets(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    daily = daily.sort_values(["ticker", "anchor_trading_date"]).copy()
    grouped = daily.groupby("ticker", sort=False)
    daily["future_price_trend_label"] = grouped["price_trend_label"].shift(-horizon)
    daily["future_price_trend_id"] = grouped["price_trend_id"].shift(-horizon)
    return daily.dropna(subset=["price_trend_id", "future_price_trend_id"]).copy()


def write_dataset(
    daily: pd.DataFrame,
    schema: dict,
    output_dir: Path,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_dir / "latent_state_daily.parquet", index=False)
    for split, part in daily.groupby("split", sort=False):
        part.to_parquet(output_dir / f"latent_state_{split}.parquet", index=False)
    output_schema = {**schema, **metadata}
    (output_dir / "latent_state_schema.json").write_text(
        json.dumps(output_schema, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    daily = pd.read_parquet(args.daily_path).copy()
    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))
    daily["anchor_trading_date"] = pd.to_datetime(daily["anchor_trading_date"]).dt.normalize()
    feature_columns = [
        column for column in schema["feature_columns"] if column in daily.columns
    ]
    filtered_columns = qwen_feature_columns(feature_columns)
    confidence = pd.to_numeric(daily["driver_mean_confidence"], errors="coerce").fillna(0.0)
    uncertainty = pd.to_numeric(daily["driver_uncertainty"], errors="coerce").fillna(1.0)
    keep = (confidence >= args.minimum_confidence) & (
        uncertainty <= args.maximum_uncertainty
    )
    daily["qwen_quality_filter_pass"] = keep.astype(int)

    filtered = daily.copy()
    filtered.loc[~keep, filtered_columns] = 0.0

    print("Qwen quality filter")
    print(f"minimum confidence: {args.minimum_confidence:.2f}")
    print(f"maximum uncertainty: {args.maximum_uncertainty:.2f}")
    print(f"retained ticker/day packets: {int(keep.sum()):,}/{len(daily):,} ({keep.mean():.2%})")
    print(f"neutralized numeric news/Qwen columns: {len(filtered_columns)}")
    print("\nRetained packets by ticker")
    print(
        daily.groupby("ticker")["qwen_quality_filter_pass"]
        .agg(["sum", "count", "mean"])
        .to_string()
    )

    for horizon in args.horizons:
        raw_horizon = rebuild_future_targets(daily, horizon)
        filtered_horizon = rebuild_future_targets(filtered, horizon)
        common_metadata = {
            "future_regime_horizon_days": horizon,
            "qwen_filter_minimum_confidence": args.minimum_confidence,
            "qwen_filter_maximum_uncertainty": args.maximum_uncertainty,
            "qwen_filter_neutralized_feature_columns": filtered_columns,
            "qwen_filter_pass_column": "qwen_quality_filter_pass",
            "qwen_filter_pass_is_model_input": False,
        }
        write_dataset(
            raw_horizon,
            schema,
            args.output_root / f"horizon_{horizon}" / "raw",
            {**common_metadata, "qwen_quality_filter_applied": False},
        )
        write_dataset(
            filtered_horizon,
            schema,
            args.output_root / f"horizon_{horizon}" / "filtered",
            {**common_metadata, "qwen_quality_filter_applied": True},
        )
        transitions = (
            raw_horizon["price_trend_id"] != raw_horizon["future_price_trend_id"]
        )
        print(
            f"\nhorizon={horizon}: rows={len(raw_horizon):,} "
            f"transitions={int(transitions.sum()):,} stable={int((~transitions).sum()):,}"
        )


if __name__ == "__main__":
    main()
