"""Build leakage-audited, purged walk-forward datasets for the final study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRICE_FEATURES = {
    "realized_volatility_20d",
    "rally_from_previous_low",
    "drawdown_from_previous_high",
    "recent_return",
    "drawdown_from_recent_high",
    "price_trend_id",
}
FOLD_WINDOWS = {
    "fold_1": {
        "validation_start": "2024-10-01",
        "test_start": "2025-04-01",
        "test_end": "2025-09-30",
    },
    "fold_2": {
        "validation_start": "2025-04-01",
        "test_start": "2025-10-01",
        "test_end": "2026-04-30",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-path",
        type=Path,
        default=Path(
            "data/processed/qwen_quality_filter_fundamentals_transition_ablation/"
            "horizon_20/latent_state_daily.parquet"
        ),
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path(
            "data/processed/qwen_quality_filter_fundamentals_transition_ablation/"
            "horizon_20/latent_state_schema.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/final_study"),
    )
    parser.add_argument("--horizon", type=int, default=20)
    return parser.parse_args()


def permitted_features(schema: dict, daily: pd.DataFrame) -> list[str]:
    features = [column for column in schema["feature_columns"] if column in daily.columns]
    return [
        column
        for column in features
        if not column.startswith("macro_") and not column.startswith("rule_")
    ]


def add_targets(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    daily = daily.sort_values(["ticker", "anchor_trading_date"]).copy()
    grouped = daily.groupby("ticker", sort=False)
    daily["target_observation_date"] = grouped["anchor_trading_date"].shift(-horizon)
    daily["future_regime_label"] = grouped["price_trend_label"].shift(-horizon)
    daily["future_regime_id"] = grouped["price_trend_id"].shift(-horizon)
    daily["future_regime_change"] = (
        daily["future_regime_id"] != daily["price_trend_id"]
    ).astype("Int64")
    # Keep compatibility with the existing trainers while exposing precise final-study names.
    daily["future_price_trend_label"] = daily["future_regime_label"]
    daily["future_price_trend_id"] = daily["future_regime_id"]
    return daily.dropna(
        subset=["price_trend_id", "future_regime_id", "target_observation_date"]
    ).copy()


def assign_purged_split(daily: pd.DataFrame, window: dict[str, str]) -> pd.DataFrame:
    validation_start = pd.Timestamp(window["validation_start"])
    test_start = pd.Timestamp(window["test_start"])
    test_end = pd.Timestamp(window["test_end"])
    anchor = daily["anchor_trading_date"]
    target = daily["target_observation_date"]
    split = pd.Series("excluded", index=daily.index, dtype="object")
    split.loc[(anchor < validation_start) & (target < validation_start)] = "train"
    split.loc[
        (anchor >= validation_start)
        & (anchor < test_start)
        & (target < test_start)
    ] = "validation"
    split.loc[(anchor >= test_start) & (anchor <= test_end)] = "test"
    output = daily.loc[split != "excluded"].copy()
    output["split"] = split.loc[output.index]
    return output


def availability_audit(daily: pd.DataFrame, features: list[str]) -> dict[str, object]:
    fundamental_features = [column for column in features if column.startswith("fundamental_")]
    qwen_news_features = [
        column
        for column in features
        if column not in PRICE_FEATURES and column not in fundamental_features
    ]
    fundamental_leaks = int(
        (
            pd.to_datetime(daily["fundamental_available_from_date"])
            > daily["anchor_trading_date"]
        ).fillna(False).sum()
    )
    return {
        "prediction_cutoff": (
            "After the 16:00 America/New_York close on anchor_trading_date. "
            "Price features use that completed close; news published at or after "
            "the cutoff is assigned to the next trading session. Existing Alpha "
            "Vantage timestamps are timezone-naive and are explicitly assumed to "
            "represent America/New_York local time."
        ),
        "target_definition": (
            "future_regime_change equals 1 when the rolling price regime at the "
            "20th subsequent ticker observation differs from the anchor regime. "
            "It is not described as a rare structural transition."
        ),
        "feature_groups": {
            "close_available_price": sorted(PRICE_FEATURES.intersection(features)),
            "cutoff_aligned_news_and_qwen": qwen_news_features,
            "release_dated_fundamentals": fundamental_features,
        },
        "excluded_feature_groups": {
            "macro": (
                "Excluded because existing monthly macro artifacts are keyed by "
                "observation month rather than confirmed public release timestamp."
            ),
            "forward_looking": [
                "future_price_trend_id",
                "future_price_trend_label",
                "future_regime_id",
                "future_regime_label",
                "future_regime_change",
                "target_observation_date",
            ],
        },
        "fundamental_availability_leaks": fundamental_leaks,
    }


def write_fold(
    fold_name: str,
    daily: pd.DataFrame,
    features: list[str],
    source_schema: dict,
    audit: dict[str, object],
    output_root: Path,
) -> None:
    fold_dir = output_root / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(fold_dir / "daily.parquet", index=False)
    for split_name, part in daily.groupby("split", sort=False):
        part.to_parquet(fold_dir / f"{split_name}.parquet", index=False)
    schema = {
        **source_schema,
        "target_column": "future_regime_id",
        "target_label_column": "future_regime_label",
        "binary_target_column": "future_regime_change",
        "trainer_compatibility_target_column": "future_price_trend_id",
        "target_semantics": audit["target_definition"],
        "feature_columns": features,
        "walk_forward_fold": fold_name,
        "walk_forward_window": FOLD_WINDOWS[fold_name],
        "purged_boundaries": True,
        "availability_audit_path": "../availability_audit.json",
    }
    (fold_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    daily = pd.read_parquet(args.daily_path)
    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))
    daily["anchor_trading_date"] = pd.to_datetime(daily["anchor_trading_date"]).dt.normalize()
    daily = add_targets(daily, args.horizon)
    features = permitted_features(schema, daily)
    audit = availability_audit(daily, features)
    if audit["fundamental_availability_leaks"]:
        raise RuntimeError("Fundamental availability audit failed.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "availability_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    for fold_name, window in FOLD_WINDOWS.items():
        fold = assign_purged_split(daily, window)
        write_fold(fold_name, fold, features, schema, audit, args.output_root)
        print(f"{fold_name}:")
        print(fold.groupby("split").size().to_string())
        print("future regime change rates:")
        print(fold.groupby("split")["future_regime_change"].mean().to_string())


if __name__ == "__main__":
    main()
