"""
Build a daily ticker-level dataset for latent market state modelling.

This converts article-level rows into one row per ticker/trading day. News is
aggregated into gradual daily signals, then joined with macro and known market
state features. Forward-looking return fields are deliberately excluded from the
feature set; they remain useful for labels/diagnostics but should not drive the
hidden state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_LABELS = {
    "bear_drawdown": 0,
    "sideways": 1,
    "bull_rally": 2,
}

PRICE_FEATURE_COLUMNS = [
    "anchor_close",
    "volume",
    "realized_volatility_20d",
    "rally_from_previous_low",
    "drawdown_from_previous_high",
    "price_trend_id",
]

NEWS_FEATURE_COLUMNS = [
    "news_article_count",
    "news_relevant_count",
    "news_high_relevance_rate",
    "news_mean_relevance",
    "news_max_relevance",
    "news_mean_ticker_sentiment",
    "news_std_ticker_sentiment",
    "news_relevance_weighted_sentiment",
    "news_mean_overall_sentiment",
    "news_positive_share",
    "news_negative_share",
    "news_neutral_share",
]

DEEPSEEK_AGGREGATE_PREFIXES = [
    "deepseek_mean_",
    "deepseek_relevance_weighted_",
]

DRIVER_FEATURE_COLUMNS = [
    "driver_count",
    "driver_mean_direction_score",
    "driver_relevance_weighted_direction_score",
    "driver_mean_intensity",
    "driver_max_intensity",
    "driver_mean_novelty",
    "driver_max_novelty",
    "driver_mean_confidence",
    "driver_mean_materiality",
    "driver_max_materiality",
    "driver_mean_shock_score",
    "driver_abs_shock_score",
    "driver_max_risk_on_shock",
    "driver_max_risk_off_shock",
    "driver_mean_horizon_days",
    "driver_ticker_scope_share",
    "driver_sector_scope_share",
    "driver_broad_market_scope_share",
    "driver_macro_scope_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/ltn_all.parquet"),
        help="Article-level LTN dataset created by build_ltn_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for latent_state_*.parquet outputs.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path("data/processed/latent_state_schema.json"),
    )
    parser.add_argument(
        "--deepseek-features-path",
        type=Path,
        default=None,
        help=(
            "Optional article-level DeepSeek features created by "
            "build_deepseek_news_features.py. If provided, these are merged by article_uid "
            "and aggregated into daily news features."
        ),
    )
    parser.add_argument(
        "--contextual-drivers-path",
        type=Path,
        default=None,
        help=(
            "Optional article/ticker contextual drivers created by "
            "extract_contextual_drivers.py. If provided, driver signals are merged "
            "and aggregated into daily latent-state features."
        ),
    )
    parser.add_argument(
        "--require-contextual-drivers",
        action="store_true",
        help="Fail if contextual driver signals are missing from the daily dataset.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.25,
        help="Ticker relevance threshold used for high-relevance daily counts.",
    )
    return parser.parse_args()


def numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    valid = values.notna() & weights.notna()
    if not valid.any() or float(weights[valid].sum()) <= 0.0:
        return float(values.mean()) if values.notna().any() else 0.0
    return float(np.average(values[valid], weights=weights[valid]))


def sentiment_shares(labels: pd.Series) -> dict[str, float]:
    lowered = labels.fillna("").astype(str).str.lower()
    total = max(len(lowered), 1)
    positive = lowered.str.contains("bull|positive").sum()
    negative = lowered.str.contains("bear|negative").sum()
    neutral = lowered.str.contains("neutral").sum()
    return {
        "news_positive_share": float(positive / total),
        "news_negative_share": float(negative / total),
        "news_neutral_share": float(neutral / total),
    }


def aggregate_day(group: pd.DataFrame, relevance_threshold: float) -> pd.Series:
    ticker, anchor_trading_date = group.name
    relevance = numeric(group["relevance_score"])
    ticker_sentiment = numeric(group["ticker_sentiment_score"])
    overall_sentiment = (
        numeric(group["overall_sentiment_score"])
        if "overall_sentiment_score" in group.columns
        else pd.Series(np.zeros(len(group)), index=group.index)
    )
    sentiment_label_col = (
        group["ticker_sentiment_label"]
        if "ticker_sentiment_label" in group.columns
        else group.get("overall_sentiment_label", pd.Series("", index=group.index))
    )

    out = {
        "ticker": ticker,
        "anchor_trading_date": anchor_trading_date,
        "news_article_count": float(len(group)),
        "news_relevant_count": float((relevance >= relevance_threshold).sum()),
        "news_high_relevance_rate": float((relevance >= relevance_threshold).mean()),
        "news_mean_relevance": float(relevance.mean()),
        "news_max_relevance": float(relevance.max()),
        "news_mean_ticker_sentiment": float(ticker_sentiment.mean()),
        "news_std_ticker_sentiment": float(ticker_sentiment.std(ddof=0)),
        "news_relevance_weighted_sentiment": weighted_mean(ticker_sentiment, relevance),
        "news_mean_overall_sentiment": float(overall_sentiment.mean()),
    }
    out.update(sentiment_shares(sentiment_label_col))

    deepseek_columns = [column for column in group.columns if column.startswith("deepseek_")]
    for column in deepseek_columns:
        values = numeric(group[column])
        out[f"deepseek_mean_{column.removeprefix('deepseek_')}"] = float(values.mean())
        out[f"deepseek_relevance_weighted_{column.removeprefix('deepseek_')}"] = weighted_mean(
            values,
            relevance,
        )

    if "driver_shock_score" in group.columns:
        direction = numeric(group["direction_score"])
        intensity = numeric(group["driver_intensity"])
        novelty = numeric(group["driver_novelty"])
        confidence = numeric(group["driver_confidence"])
        materiality = numeric(group["driver_materiality"])
        shock = numeric(group["driver_shock_score"])
        horizon = numeric(group["horizon_days"])
        out.update(
            {
                "driver_count": float(group["driver_shock_score"].notna().sum()),
                "driver_mean_direction_score": float(direction.mean()),
                "driver_relevance_weighted_direction_score": weighted_mean(direction, relevance),
                "driver_mean_intensity": float(intensity.mean()),
                "driver_max_intensity": float(intensity.max()),
                "driver_mean_novelty": float(novelty.mean()),
                "driver_max_novelty": float(novelty.max()),
                "driver_mean_confidence": float(confidence.mean()),
                "driver_mean_materiality": float(materiality.mean()),
                "driver_max_materiality": float(materiality.max()),
                "driver_mean_shock_score": float(shock.mean()),
                "driver_abs_shock_score": float(shock.abs().mean()),
                "driver_max_risk_on_shock": float(shock.clip(lower=0.0).max()),
                "driver_max_risk_off_shock": float((-shock).clip(lower=0.0).max()),
                "driver_mean_horizon_days": float(horizon.mean()),
            }
        )
        scope = group.get("market_scope", pd.Series("unknown", index=group.index)).fillna("unknown").astype(str)
        total = max(len(scope), 1)
        out.update(
            {
                "driver_ticker_scope_share": float((scope == "ticker").sum() / total),
                "driver_sector_scope_share": float((scope == "sector").sum() / total),
                "driver_broad_market_scope_share": float((scope == "broad_market").sum() / total),
                "driver_macro_scope_share": float((scope == "macro").sum() / total),
            }
        )
        ranked = group.assign(_driver_rank=materiality * relevance.fillna(0.0))
        ranked = ranked.sort_values("_driver_rank", ascending=False).head(3)
        for idx, (_, row) in enumerate(ranked.iterrows(), start=1):
            out[f"top_driver_{idx}_label"] = row.get("driver_label", "")
            out[f"top_driver_{idx}_summary"] = row.get("driver_summary", "")
            out[f"top_driver_{idx}_pressure"] = row.get("directional_pressure", "")
            out[f"top_driver_{idx}_evidence"] = row.get("evidence", "")

    first = group.sort_values(["time_published", "article_uid"]).iloc[0]
    stable_columns = [
        "month",
        "split",
        "anchor_close",
        "volume",
        "realized_volatility_20d",
        "rally_from_previous_low",
        "drawdown_from_previous_high",
        "price_trend_label",
        "price_trend_id",
        "future_price_trend_label",
        "future_price_trend_id",
    ]
    for column in stable_columns:
        if column in group.columns:
            out[column] = first[column]

    for column in group.columns:
        if column.startswith("macro_"):
            out[column] = first[column]
    return pd.Series(out)


def build_daily_dataset(df: pd.DataFrame, relevance_threshold: float) -> pd.DataFrame:
    required = {
        "ticker",
        "anchor_trading_date",
        "time_published",
        "article_uid",
        "relevance_score",
        "ticker_sentiment_score",
        "future_price_trend_id",
        "future_price_trend_label",
        "split",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")

    work = df.copy()
    work["anchor_trading_date"] = pd.to_datetime(work["anchor_trading_date"]).dt.normalize()
    work["time_published"] = pd.to_datetime(work["time_published"], errors="coerce")
    daily = (
        work.groupby(["ticker", "anchor_trading_date"], sort=True, group_keys=False)
        .apply(lambda group: aggregate_day(group, relevance_threshold))
        .reset_index(drop=True)
    )
    daily["anchor_trading_date"] = pd.to_datetime(daily["anchor_trading_date"]).dt.normalize()
    daily = daily.sort_values(["ticker", "anchor_trading_date"]).reset_index(drop=True)
    daily["news_log_article_count"] = np.log1p(daily["news_article_count"].astype(float))
    return daily


def feature_columns(daily: pd.DataFrame) -> list[str]:
    macro_columns = sorted(
        column
        for column in daily.columns
        if column.startswith("macro_") and pd.api.types.is_numeric_dtype(daily[column])
    )
    candidates = [
        "news_log_article_count",
        *NEWS_FEATURE_COLUMNS,
        *DRIVER_FEATURE_COLUMNS,
        *[
            column
            for column in sorted(daily.columns)
            if any(column.startswith(prefix) for prefix in DEEPSEEK_AGGREGATE_PREFIXES)
        ],
        *PRICE_FEATURE_COLUMNS,
        *macro_columns,
    ]
    return list(dict.fromkeys(column for column in candidates if column in daily.columns))


def write_outputs(daily: pd.DataFrame, output_dir: Path, schema_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_dir / "latent_state_daily.parquet", index=False)
    for split, part in daily.groupby("split", sort=False):
        part.to_parquet(output_dir / f"latent_state_{split}.parquet", index=False)

    schema = {
        "target_column": "future_price_trend_id",
        "target_label_column": "future_price_trend_label",
        "target_labels": TARGET_LABELS,
        "date_column": "anchor_trading_date",
        "entity_column": "ticker",
        "feature_columns": feature_columns(daily),
        "has_contextual_driver_features": "driver_mean_shock_score" in daily.columns,
        "explanation_columns": [
            column
            for column in daily.columns
            if column.startswith("top_driver_")
        ],
        "excluded_forward_looking_inputs": [
            "next_day_return",
            "three_day_return",
            "market_sentiment_id",
            "market_sentiment_label",
            "is_market_material",
            "is_material",
        ],
    }
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def validate_contextual_drivers(daily: pd.DataFrame) -> None:
    required = {
        "driver_count",
        "driver_mean_shock_score",
        "driver_mean_materiality",
        "driver_mean_confidence",
        "top_driver_1_label",
    }
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(
            "Contextual driver features are required for the explainable pipeline, "
            f"but these daily columns are missing: {missing}. Run "
            "src/data/extract_contextual_drivers.py and pass --contextual-drivers-path."
        )
    driver_count = pd.to_numeric(daily["driver_count"], errors="coerce").fillna(0.0)
    if float(driver_count.sum()) <= 0.0:
        raise ValueError(
            "Contextual driver features are required, but no extracted drivers were "
            "merged into the daily dataset. Check the driver extraction filters and "
            "article_uid/ticker/date keys."
        )


def print_summary(daily: pd.DataFrame) -> None:
    print("Latent-state daily dataset created")
    print(f"Rows: {len(daily):,}")
    print(f"Tickers: {', '.join(sorted(daily['ticker'].unique()))}")
    print(
        "Date range: "
        f"{daily['anchor_trading_date'].min().date()} -> "
        f"{daily['anchor_trading_date'].max().date()}"
    )
    print("\nSplit sizes")
    print(daily["split"].value_counts().sort_index().to_string())
    print("\nFuture regime targets")
    print(pd.crosstab(daily["split"], daily["future_price_trend_label"]).to_string())
    print("\nFeature columns")
    for column in feature_columns(daily):
        print(f"  {column}")


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    if args.deepseek_features_path:
        deepseek = pd.read_parquet(args.deepseek_features_path)
        if "article_uid" not in deepseek.columns:
            raise ValueError(f"DeepSeek features must contain article_uid: {args.deepseek_features_path}")
        df = df.merge(deepseek, on="article_uid", how="left", validate="many_to_one")
        print(f"Merged DeepSeek article features: {args.deepseek_features_path}")
    if args.contextual_drivers_path:
        drivers = pd.read_parquet(args.contextual_drivers_path)
        required = {"article_uid", "ticker", "anchor_trading_date"}
        missing = sorted(required.difference(drivers.columns))
        if missing:
            raise ValueError(f"Contextual drivers missing required columns: {missing}")
        drivers = drivers.copy()
        drivers["anchor_trading_date"] = pd.to_datetime(drivers["anchor_trading_date"]).dt.normalize()
        df["anchor_trading_date"] = pd.to_datetime(df["anchor_trading_date"]).dt.normalize()
        df = df.merge(
            drivers,
            on=["article_uid", "ticker", "anchor_trading_date"],
            how="left",
            validate="many_to_one",
        )
        print(f"Merged contextual driver features: {args.contextual_drivers_path}")
    elif args.require_contextual_drivers:
        raise ValueError(
            "--require-contextual-drivers was set, but no --contextual-drivers-path was provided."
        )
    daily = build_daily_dataset(df, args.relevance_threshold)
    if args.require_contextual_drivers:
        validate_contextual_drivers(daily)
    write_outputs(daily, args.output_dir, args.schema_path)
    print_summary(daily)


if __name__ == "__main__":
    main()
