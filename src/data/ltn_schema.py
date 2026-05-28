"""Schema contract for the DeepSeek + LTN training dataset."""

from __future__ import annotations

import json
from pathlib import Path

CORE_COLUMNS = [
    "article_uid",
    "article_id",
    "ticker",
    "time_published",
    "signal_date",
    "anchor_trading_date",
    "month",
    "title",
    "summary",
    "article_text",
    "source",
    "url",
    "requested_entity",
    "relevance_score",
    "ticker_sentiment_score",
    "ticker_sentiment_label",
    "overall_sentiment_score",
    "overall_sentiment_label",
    "anchor_close",
    "volume",
    "next_day_return",
    "three_day_return",
    "realized_volatility_20d",
    "previous_low",
    "previous_high",
    "rally_from_previous_low",
    "drawdown_from_previous_high",
    "price_trend_label",
    "price_trend_id",
    "future_price_trend_label",
    "future_price_trend_id",
    "abs_next_day_return",
    "market_sentiment_label",
    "market_sentiment_id",
    "direction_label",
    "is_relevant",
    "is_market_material",
    "is_material",
    "split",
]

REQUIRED_COLUMNS = [
    "article_uid",
    "ticker",
    "time_published",
    "anchor_trading_date",
    "article_text",
    "relevance_score",
    "future_price_trend_label",
    "future_price_trend_id",
    "is_material",
    "split",
]

TARGET_COLUMN = "future_price_trend_id"
TARGET_LABEL_COLUMN = "future_price_trend_label"
TARGET_LABELS = {
    "bear_drawdown": 0,
    "sideways": 1,
    "bull_rally": 2,
}

STRUCTURED_FEATURE_COLUMNS = [
    "relevance_score",
    "ticker_sentiment_score",
    "overall_sentiment_score",
    "anchor_close",
    "volume",
    "next_day_return",
    "three_day_return",
    "realized_volatility_20d",
    "rally_from_previous_low",
    "drawdown_from_previous_high",
    "price_trend_id",
    "market_sentiment_id",
    "is_relevant",
    "is_market_material",
    "is_material",
    "macro_baa_10y_credit_spread",
    "macro_core_cpi_yoy",
    "macro_core_pce_yoy",
    "macro_cpi_yoy",
    "macro_pce_yoy",
    "macro_fed_funds_rate",
    "macro_fed_funds_change_1m",
    "macro_fed_funds_change_3m",
    "macro_treasury_2y",
    "macro_treasury_10y",
    "macro_treasury_10y_2y_spread",
    "macro_treasury_2y_change_1m",
    "macro_treasury_10y_change_1m",
    "macro_unemployment_rate",
    "macro_unemployment_change_3m",
    "macro_nonfarm_payrolls_yoy",
    "macro_industrial_production_yoy",
    "macro_real_gdp_yoy",
    "macro_vix",
    "macro_trade_weighted_dollar",
    "macro_wti_oil",
    "macro_high_inflation",
    "macro_very_high_inflation",
    "macro_rising_rates",
    "macro_falling_rates",
    "macro_inverted_yield_curve",
    "macro_tightening_regime",
    "macro_easing_regime",
]

PREDICATE_COLUMNS = [
    "relevance_score",
    "is_relevant",
    "is_material",
    "macro_high_inflation",
    "macro_rising_rates",
    "macro_falling_rates",
    "macro_tightening_regime",
    "macro_easing_regime",
    "macro_inverted_yield_curve",
]


def validate_dataset_columns(columns: list[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Dataset is missing required schema columns: {missing}")


def schema_dict() -> dict[str, object]:
    return {
        "target_column": TARGET_COLUMN,
        "target_label_column": TARGET_LABEL_COLUMN,
        "target_labels": TARGET_LABELS,
        "required_columns": REQUIRED_COLUMNS,
        "core_columns": CORE_COLUMNS,
        "structured_feature_columns": STRUCTURED_FEATURE_COLUMNS,
        "predicate_columns": PREDICATE_COLUMNS,
    }


def write_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema_dict(), indent=2), encoding="utf-8")
