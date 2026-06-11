"""Build leakage-safe daily packets for transformer-context and LTN training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_CONTEXT_COLUMNS = [
    "anchor_close",
    "realized_volatility_20d",
    "rally_from_previous_low",
    "drawdown_from_previous_high",
    "recent_return",
    "drawdown_from_recent_high",
    "price_trend_label",
    "price_trend_id",
]
TARGET_COLUMNS = [
    "future_price_trend_label",
    "future_price_trend_id",
]
FUNDAMENTAL_METADATA_COLUMNS = [
    "fiscal_period_end",
    "fundamental_release_date_source",
    "fundamental_reported_date",
    "fundamental_available_from_date",
]
EXCLUDED_OFFLINE_QWEN_PREFIXES = ("driver_", "top_driver_")
EXCLUDED_OFFLINE_QWEN_COLUMNS = {
    "daily_market_narrative",
    "dominant_drivers_json",
    "net_pressure",
    "net_pressure_score",
    "qwen_quality_filter_pass",
    "raw_generation",
    "relevant_channels_json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-path", type=Path, required=True)
    parser.add_argument("--daily-context-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-articles-per-day", type=int, default=25)
    parser.add_argument("--max-article-chars", type=int, default=900)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def finite_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) and np.isfinite(numeric) else default


def article_payload(row: pd.Series, max_article_chars: int) -> dict[str, object]:
    title = clean_text(row.get("title", ""))
    summary = clean_text(row.get("summary", ""))
    article_text = clean_text(row.get("article_text", ""))
    body = summary or article_text
    return {
        "article_uid": str(row["article_uid"]),
        "time_published": (
            pd.Timestamp(row["time_published"]).isoformat()
            if pd.notna(row.get("time_published"))
            else None
        ),
        "title": title,
        "text": body[:max_article_chars],
        "source": clean_text(row.get("source", "")),
        "weak_relevance_score": finite_float(row.get("relevance_score")),
        "weak_ticker_sentiment_score": finite_float(row.get("ticker_sentiment_score")),
    }


def format_packet(ticker: str, anchor_date: pd.Timestamp, articles: list[dict[str, object]]) -> str:
    article_block = "\n".join(
        f"{index}. Title: {article['title']}\n   Text: {article['text']}"
        for index, article in enumerate(articles, start=1)
    )
    return (
        f"Ticker: {ticker}\n"
        f"Trading day: {anchor_date.date().isoformat()}\n"
        "Daily ticker news packet:\n"
        f"{article_block}"
    )


def aggregate_articles(
    articles: pd.DataFrame,
    max_articles_per_day: int,
    max_article_chars: int,
) -> pd.DataFrame:
    require_columns(
        articles,
        {"article_uid", "ticker", "anchor_trading_date", "title", "summary", "article_text"},
        "article parquet",
    )
    articles = articles.copy()
    articles["anchor_trading_date"] = pd.to_datetime(articles["anchor_trading_date"]).dt.normalize()
    articles["time_published"] = pd.to_datetime(articles.get("time_published"), errors="coerce")
    articles["_relevance_rank"] = pd.to_numeric(articles.get("relevance_score"), errors="coerce").fillna(0.0)
    articles = articles.sort_values(
        ["ticker", "anchor_trading_date", "_relevance_rank", "time_published", "article_uid"],
        ascending=[True, True, False, True, True],
    )
    rows = []
    for (ticker, anchor_date), group in articles.groupby(["ticker", "anchor_trading_date"], sort=True):
        selected = group.drop_duplicates("article_uid").head(max_articles_per_day)
        payloads = [
            article_payload(row, max_article_chars)
            for _, row in selected.iterrows()
        ]
        relevance = np.array(
            [float(article["weak_relevance_score"]) for article in payloads],
            dtype=float,
        ).clip(0.0, 1.0)
        sentiment = np.array(
            [float(article["weak_ticker_sentiment_score"]) for article in payloads],
            dtype=float,
        ).clip(-1.0, 1.0)
        relevance_total = float(relevance.sum())
        relevance_weights = (
            relevance / relevance_total
            if relevance_total > 0.0
            else np.full(len(payloads), 1.0 / max(len(payloads), 1))
        )
        directional_pressure = float(np.sum(relevance_weights * sentiment)) if len(payloads) else 0.0
        sentiment_dispersion = (
            float(np.sum(relevance_weights * np.abs(sentiment - directional_pressure)))
            if len(payloads)
            else 0.0
        )
        materiality_proxy = float(np.max(relevance * np.abs(sentiment))) if len(payloads) else 0.0
        rows.append(
            {
                "ticker": str(ticker),
                "anchor_trading_date": anchor_date,
                "packet_article_count": len(payloads),
                "packet_text": format_packet(str(ticker), anchor_date, payloads),
                "packet_articles_json": json.dumps(payloads),
                "weak_packet_mean_relevance": float(np.mean(relevance)) if len(relevance) else 0.0,
                "weak_packet_max_relevance": float(np.max(relevance)) if len(relevance) else 0.0,
                "weak_label_ticker_relevance": float(np.max(relevance)) if len(relevance) else 0.0,
                "weak_label_materiality": materiality_proxy,
                "weak_label_risk_on_pressure": max(directional_pressure, 0.0),
                "weak_label_risk_off_pressure": max(-directional_pressure, 0.0),
                "weak_label_uncertainty": float(
                    np.clip(
                        0.5 * sentiment_dispersion
                        + 0.5 * (1.0 - materiality_proxy),
                        0.0,
                        1.0,
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def daily_context_columns(daily: pd.DataFrame) -> list[str]:
    numeric_fundamentals = [
        column
        for column in daily.columns
        if column.startswith("fundamental_")
        and column not in FUNDAMENTAL_METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(daily[column])
    ]
    columns = [
        "ticker",
        "anchor_trading_date",
        "split",
        *PRICE_CONTEXT_COLUMNS,
        *TARGET_COLUMNS,
        *FUNDAMENTAL_METADATA_COLUMNS,
        *numeric_fundamentals,
    ]
    return list(dict.fromkeys(column for column in columns if column in daily.columns))


def audit_no_offline_qwen_inputs(columns: list[str]) -> None:
    leaked = sorted(
        column
        for column in columns
        if column in EXCLUDED_OFFLINE_QWEN_COLUMNS
        or column.startswith(EXCLUDED_OFFLINE_QWEN_PREFIXES)
    )
    if leaked:
        raise ValueError(f"Offline Qwen columns must not enter the end-to-end dataset: {leaked}")


def write_outputs(output_dir: Path, dataset: pd.DataFrame, schema: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_dir / "daily_packets.parquet", index=False)
    for split, part in dataset.groupby("split", sort=False):
        part.to_parquet(output_dir / f"daily_packets_{split}.parquet", index=False)
    (output_dir / "daily_packet_schema.json").write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    articles = pd.read_parquet(args.article_path)
    daily = pd.read_parquet(args.daily_context_path)
    require_columns(
        daily,
        {"ticker", "anchor_trading_date", "split", "price_trend_id", "future_price_trend_id"},
        "daily context parquet",
    )
    daily = daily.copy()
    daily["anchor_trading_date"] = pd.to_datetime(daily["anchor_trading_date"]).dt.normalize()
    context_columns = daily_context_columns(daily)
    audit_no_offline_qwen_inputs(context_columns)
    packets = aggregate_articles(
        articles,
        args.max_articles_per_day,
        args.max_article_chars,
    )
    dataset = daily[context_columns].merge(
        packets,
        on=["ticker", "anchor_trading_date"],
        how="inner",
        validate="one_to_one",
    )
    dataset = dataset.sort_values(["ticker", "anchor_trading_date"]).reset_index(drop=True)
    if "fundamental_available_from_date" in dataset:
        available = pd.to_datetime(dataset["fundamental_available_from_date"], errors="coerce")
        leaked = available.notna() & (available > dataset["anchor_trading_date"])
        if leaked.any():
            raise ValueError(f"Found {int(leaked.sum())} fundamentals rows exposed before public release.")
    schema = {
        "rows": int(len(dataset)),
        "tickers": sorted(dataset["ticker"].unique().tolist()),
        "date_min": dataset["anchor_trading_date"].min().date().isoformat(),
        "date_max": dataset["anchor_trading_date"].max().date().isoformat(),
        "split_counts": {key: int(value) for key, value in dataset["split"].value_counts().items()},
        "text_column": "packet_text",
        "article_metadata_column": "packet_articles_json",
        "automated_weak_predicate_columns": [
            "weak_label_ticker_relevance",
            "weak_label_materiality",
            "weak_label_risk_on_pressure",
            "weak_label_risk_off_pressure",
            "weak_label_uncertainty",
        ],
        "price_context_columns": [column for column in PRICE_CONTEXT_COLUMNS if column in dataset],
        "fundamental_columns": [
            column
            for column in context_columns
            if column.startswith("fundamental_")
        ],
        "target_columns": [column for column in TARGET_COLUMNS if column in dataset],
        "excluded_offline_qwen_columns": sorted(EXCLUDED_OFFLINE_QWEN_COLUMNS),
        "leakage_control": (
            "Packet text includes only articles assigned to the anchor trading date. "
            "Fundamentals are retained only from the existing leakage-safe daily dataset and "
            "audited so public availability never exceeds the anchor date. Future regime fields "
            "are downstream targets only. Automated semantic weak labels use contemporaneous "
            "article relevance and bullish/bearish sentiment only. Saved offline Qwen predicates "
            "are deliberately excluded."
        ),
    }
    write_outputs(args.output_dir, dataset, schema)
    print("End-to-end daily packet dataset created")
    print(f"Rows: {len(dataset):,}")
    print(f"Tickers: {', '.join(schema['tickers'])}")
    print(f"Date range: {schema['date_min']} -> {schema['date_max']}")
    print("Split counts")
    print(dataset["split"].value_counts().to_string())
    print(f"Mean packet articles: {dataset['packet_article_count'].mean():.2f}")
    print("Leakage audit: passed")
    print("Offline Qwen inputs: excluded")


if __name__ == "__main__":
    main()
