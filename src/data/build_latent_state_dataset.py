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
    daily = build_daily_dataset(df, args.relevance_threshold)
    write_outputs(daily, args.output_dir, args.schema_path)
    print_summary(daily)


if __name__ == "__main__":
    main()
