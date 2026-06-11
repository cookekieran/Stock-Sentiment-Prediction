"""
Build the article-ticker-price dataset used by the DeepSeek + LTN pipeline.

The output rows are supervised examples with:
- article text and Alpha Vantage metadata
- ticker-level relevance/sentiment metadata
- leakage-aware market reaction labels from future returns
- chronological train/validation/test splits
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ltn_schema import CORE_COLUMNS, validate_dataset_columns, write_schema
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from ltn_schema import CORE_COLUMNS, validate_dataset_columns, write_schema


DEFAULT_TICKERS = ("NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles-dir", type=Path, default=Path("hf_dataset/articles"))
    parser.add_argument("--tickers-dir", type=Path, default=Path("hf_dataset/tickers"))
    parser.add_argument("--articles-hf-repo", default=None)
    parser.add_argument("--articles-hf-pattern", default="articles/*.parquet")
    parser.add_argument("--tickers-hf-repo", default=None)
    parser.add_argument("--tickers-hf-pattern", default="tickers/*.parquet")
    parser.add_argument(
        "--prices-path",
        type=Path,
        default=Path("mag7_prices/mag7_daily_prices.parquet"),
    )
    parser.add_argument("--prices-hf-repo", default=None)
    parser.add_argument("--prices-hf-path", default=None)
    parser.add_argument(
        "--macro-path",
        type=Path,
        default=Path("data/processed/macro_monthly.parquet"),
        help="Optional monthly macro feature parquet created by fetch_macro_fred.py.",
    )
    parser.add_argument("--macro-hf-repo", default=None)
    parser.add_argument("--macro-hf-path", default="macro_monthly.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--schema-path", type=Path, default=Path("data/processed/ltn_schema.json"))
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument(
        "--market-close-hour",
        type=int,
        default=16,
        help="Local market hour after which news is assigned to the next trading session.",
    )
    parser.add_argument(
        "--source-timezone",
        default="America/New_York",
        help=(
            "Timezone assumed for timezone-naive time_published values. The existing "
            "Alpha Vantage artifacts are treated as America/New_York local time."
        ),
    )
    parser.add_argument(
        "--market-timezone",
        default="America/New_York",
        help="Timezone used to apply the close-of-day information cutoff.",
    )
    parser.add_argument(
        "--positive-return-threshold",
        type=float,
        default=0.005,
        help="Forward return above this value is labelled bullish.",
    )
    parser.add_argument(
        "--negative-return-threshold",
        type=float,
        default=-0.005,
        help="Forward return below this value is labelled bearish.",
    )
    parser.add_argument(
        "--material-return-threshold",
        type=float,
        default=0.01,
        help="Absolute forward return required for a market-material article.",
    )
    parser.add_argument(
        "--material-relevance-threshold",
        type=float,
        default=0.25,
        help="Ticker relevance required for a semantically material article.",
    )
    parser.add_argument(
        "--trend-window-days",
        type=int,
        default=120,
        help="Trading-day lookback window for previous lows/highs used in trend labels.",
    )
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=0.20,
        help="Legacy rally/drawdown threshold retained for diagnostic features.",
    )
    parser.add_argument(
        "--regime-return-window-days",
        type=int,
        default=20,
        help="Trading-day return lookback used for directional bull/bear regime labels.",
    )
    parser.add_argument(
        "--regime-drawdown-window-days",
        type=int,
        default=60,
        help="Trading-day high-water-mark lookback used for bear regime labels.",
    )
    parser.add_argument(
        "--regime-bull-return-threshold",
        type=float,
        default=0.05,
        help="Recent return at or above this value is labelled bull_rally.",
    )
    parser.add_argument(
        "--regime-bear-return-threshold",
        type=float,
        default=-0.05,
        help="Recent return at or below this value is labelled bear_drawdown.",
    )
    parser.add_argument(
        "--regime-bear-drawdown-threshold",
        type=float,
        default=-0.10,
        help="Drawdown from the recent high at or below this value is labelled bear_drawdown.",
    )
    parser.add_argument(
        "--trend-forward-days",
        type=int,
        default=20,
        help="Trading days ahead used as the supervised future trend target.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["date", "fraction"],
        default="date",
        help="Use explicit chronological date windows or fallback fractional chronological split.",
    )
    parser.add_argument(
        "--validation-start-date",
        default="2025-04-01",
        help="First anchor_trading_date included in validation for date strategy.",
    )
    parser.add_argument(
        "--test-start-date",
        default="2025-07-01",
        help="First anchor_trading_date included in test for date strategy.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser.parse_args()


def read_parquet_dir(directory: Path, table_name: str) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {table_name}: {directory}")

    frames = []
    for file in files:
        frame = pd.read_parquet(file)
        frame["source_file"] = file.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def require_hf_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face loading requires huggingface_hub. Install it with "
            "`python -m pip install huggingface_hub`."
        ) from exc
    return HfApi, hf_hub_download


def read_hf_parquet_files(repo_id: str, pattern: str, table_name: str) -> pd.DataFrame:
    HfApi, hf_hub_download = require_hf_hub()
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = sorted(path for path in repo_files if fnmatch.fnmatch(path, pattern))
    if not parquet_files:
        raise FileNotFoundError(
            f"No Hugging Face parquet files matched {pattern!r} in {repo_id} for {table_name}."
        )

    frames = []
    for repo_path in parquet_files:
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_path)
        frame = pd.read_parquet(local_path)
        frame["source_file"] = repo_path
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def read_hf_parquet_file(repo_id: str, repo_path: str, table_name: str) -> pd.DataFrame:
    _, hf_hub_download = require_hf_hub()
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_path)
    try:
        return pd.read_parquet(local_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not load {table_name} from {repo_id}/{repo_path}") from exc


def require_columns(df: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def normalise_prices(
    prices: pd.DataFrame,
    trend_window_days: int,
    trend_threshold: float,
    regime_return_window_days: int,
    regime_drawdown_window_days: int,
    regime_bull_return_threshold: float,
    regime_bear_return_threshold: float,
    regime_bear_drawdown_threshold: float,
    trend_forward_days: int,
) -> pd.DataFrame:
    prices = prices.copy()
    prices.columns = [str(column).lower().replace(" ", "_") for column in prices.columns]

    require_columns(prices, {"date", "ticker"}, "prices")
    close_column = "adj_close" if "adj_close" in prices.columns else "close"
    if close_column not in prices.columns:
        raise ValueError("prices must contain either 'adj_close' or 'close'")

    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["trading_date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["close_for_return"] = pd.to_numeric(prices[close_column], errors="coerce")

    if "volume" in prices.columns:
        prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    else:
        prices["volume"] = np.nan

    prices = prices.dropna(subset=["ticker", "trading_date", "close_for_return"])
    prices = prices.sort_values(["ticker", "trading_date"])

    grouped = prices.groupby("ticker", group_keys=False)
    prices["one_day_return"] = grouped["close_for_return"].pct_change()
    prices["next_day_close"] = grouped["close_for_return"].shift(-1)
    prices["three_day_close"] = grouped["close_for_return"].shift(-3)
    prices["next_day_return"] = prices["next_day_close"] / prices["close_for_return"] - 1.0
    prices["three_day_return"] = prices["three_day_close"] / prices["close_for_return"] - 1.0
    prices["realized_volatility_20d"] = grouped["one_day_return"].transform(
        lambda values: values.rolling(20, min_periods=5).std()
    )
    prices["previous_low"] = grouped["close_for_return"].transform(
        lambda values: values.shift(1).rolling(trend_window_days, min_periods=20).min()
    )
    prices["previous_high"] = grouped["close_for_return"].transform(
        lambda values: values.shift(1).rolling(trend_window_days, min_periods=20).max()
    )
    prices["rally_from_previous_low"] = (
        prices["close_for_return"] / prices["previous_low"] - 1.0
    )
    prices["drawdown_from_previous_high"] = (
        prices["close_for_return"] / prices["previous_high"] - 1.0
    )
    prices["recent_return"] = grouped["close_for_return"].pct_change(regime_return_window_days)
    prices["recent_high"] = grouped["close_for_return"].transform(
        lambda values: values.shift(1).rolling(regime_drawdown_window_days, min_periods=20).max()
    )
    prices["drawdown_from_recent_high"] = prices["close_for_return"] / prices["recent_high"] - 1.0
    has_trend_history = prices["recent_return"].notna() & prices["recent_high"].notna()
    prices["price_trend_label"] = pd.Series(pd.NA, index=prices.index, dtype="object")
    prices.loc[has_trend_history, "price_trend_label"] = "sideways"
    prices.loc[
        has_trend_history & (prices["recent_return"] >= regime_bull_return_threshold),
        "price_trend_label",
    ] = "bull_rally"
    prices.loc[
        has_trend_history
        & (
            (prices["recent_return"] <= regime_bear_return_threshold)
            | (prices["drawdown_from_recent_high"] <= regime_bear_drawdown_threshold)
        ),
        "price_trend_label",
    ] = "bear_drawdown"

    trend_to_id = {"bear_drawdown": 0, "sideways": 1, "bull_rally": 2}
    prices["price_trend_id"] = prices["price_trend_label"].map(trend_to_id)
    prices["future_price_trend_label"] = grouped["price_trend_label"].shift(
        -trend_forward_days
    )
    prices["future_price_trend_id"] = prices["future_price_trend_label"].map(trend_to_id)

    return prices[
        [
            "ticker",
            "trading_date",
            "close_for_return",
            "volume",
            "next_day_return",
            "three_day_return",
            "realized_volatility_20d",
            "previous_low",
            "previous_high",
            "rally_from_previous_low",
            "drawdown_from_previous_high",
            "recent_return",
            "recent_high",
            "drawdown_from_recent_high",
            "price_trend_label",
            "price_trend_id",
            "future_price_trend_label",
            "future_price_trend_id",
        ]
    ].rename(columns={"close_for_return": "anchor_close"})


def build_article_ticker_table(articles: pd.DataFrame, tickers: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        articles,
        {"article_id", "article_uid", "time_published", "title", "summary"},
        "articles",
    )
    require_columns(
        tickers,
        {"article_id", "article_uid", "ticker", "relevance_score", "ticker_sentiment_score"},
        "tickers",
    )

    article_columns = [
        "article_id",
        "article_uid",
        "time_published",
        "title",
        "summary",
        "source",
        "url",
        "requested_entity",
        "overall_sentiment_score",
        "overall_sentiment_label",
    ]
    article_columns = [column for column in article_columns if column in articles.columns]
    duplicate_article_columns = [
        column
        for column in article_columns
        if column in tickers.columns and column not in {"article_id", "article_uid"}
    ]
    tickers = tickers.drop(columns=duplicate_article_columns)

    merged = tickers.merge(
        articles[article_columns],
        on=["article_id", "article_uid"],
        how="inner",
        validate="many_to_one",
    )

    merged["ticker"] = merged["ticker"].astype(str).str.upper()
    merged["time_published"] = pd.to_datetime(merged["time_published"], errors="coerce")
    merged["relevance_score"] = pd.to_numeric(merged["relevance_score"], errors="coerce")
    merged["ticker_sentiment_score"] = pd.to_numeric(
        merged["ticker_sentiment_score"], errors="coerce"
    )

    if "overall_sentiment_score" in merged.columns:
        merged["overall_sentiment_score"] = pd.to_numeric(
            merged["overall_sentiment_score"], errors="coerce"
        )

    merged["title"] = merged["title"].fillna("").astype(str)
    merged["summary"] = merged["summary"].fillna("").astype(str)
    merged["article_text"] = np.where(
        merged["summary"].str.len() > 0,
        merged["title"] + "\n\n" + merged["summary"],
        merged["title"],
    )

    return merged.dropna(subset=["time_published", "ticker", "article_text"])


def add_signal_dates(
    examples: pd.DataFrame,
    market_close_hour: int,
    source_timezone: str,
    market_timezone: str,
) -> pd.DataFrame:
    examples = examples.copy()
    published = pd.to_datetime(examples["time_published"], errors="coerce")
    if published.dt.tz is None:
        published = published.dt.tz_localize(
            source_timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    market_published = published.dt.tz_convert(market_timezone)
    examples["time_published_utc"] = published.dt.tz_convert("UTC")
    examples["time_published_market"] = market_published
    after_close = market_published.dt.hour >= market_close_hour
    examples["signal_date"] = market_published.dt.tz_localize(None).dt.normalize()
    examples.loc[after_close, "signal_date"] += pd.Timedelta(days=1)
    return examples


def attach_anchor_trading_dates(examples: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    anchored_frames = []
    trading_days = prices[["ticker", "trading_date"]].drop_duplicates()

    for ticker, ticker_examples in examples.groupby("ticker", sort=False):
        ticker_calendar = trading_days[trading_days["ticker"] == ticker].sort_values("trading_date")
        if ticker_calendar.empty:
            continue

        ticker_examples = ticker_examples.sort_values("signal_date")
        anchored = pd.merge_asof(
            ticker_examples,
            ticker_calendar,
            left_on="signal_date",
            right_on="trading_date",
            direction="forward",
        )
        anchored["ticker"] = ticker
        anchored_frames.append(anchored)

    if not anchored_frames:
        raise ValueError("No examples could be matched to trading dates.")

    anchored = pd.concat(anchored_frames, ignore_index=True)
    return anchored.rename(columns={"trading_date": "anchor_trading_date"})


def add_market_labels(
    examples: pd.DataFrame,
    prices: pd.DataFrame,
    positive_threshold: float,
    negative_threshold: float,
    material_return_threshold: float,
    material_relevance_threshold: float,
) -> pd.DataFrame:
    examples = examples.merge(
        prices,
        left_on=["ticker", "anchor_trading_date"],
        right_on=["ticker", "trading_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["trading_date"])

    examples = examples.dropna(subset=["next_day_return", "three_day_return"])
    examples = examples.dropna(subset=["price_trend_id", "future_price_trend_id"])

    examples["market_sentiment_label"] = "neutral"
    examples.loc[
        examples["next_day_return"] > positive_threshold, "market_sentiment_label"
    ] = "bullish"
    examples.loc[
        examples["next_day_return"] < negative_threshold, "market_sentiment_label"
    ] = "bearish"

    label_to_id = {"bearish": 0, "neutral": 1, "bullish": 2}
    examples["market_sentiment_id"] = examples["market_sentiment_label"].map(label_to_id)
    examples["direction_label"] = np.sign(examples["next_day_return"]).astype(int)

    examples["abs_next_day_return"] = examples["next_day_return"].abs()
    examples["is_market_material"] = (
        examples["abs_next_day_return"] >= material_return_threshold
    ).astype(int)
    examples["is_relevant"] = (
        examples["relevance_score"].fillna(0.0) >= material_relevance_threshold
    ).astype(int)
    examples["is_material"] = (
        (examples["is_market_material"] == 1) & (examples["is_relevant"] == 1)
    ).astype(int)

    return examples


def attach_macro_features(examples: pd.DataFrame, macro_path: Path) -> pd.DataFrame:
    if not macro_path.exists():
        print(f"Macro file not found, skipping macro join: {macro_path}")
        return examples

    macro = pd.read_parquet(macro_path)
    if "month" not in macro.columns:
        raise ValueError(f"Macro file must contain a 'month' column: {macro_path}")

    macro = macro.copy()
    macro["month"] = macro["month"].astype(str)
    macro = macro.add_prefix("macro_").rename(columns={"macro_month": "month"})

    examples = examples.copy()
    examples["month"] = examples["anchor_trading_date"].dt.to_period("M").astype(str)
    return examples.merge(macro, on="month", how="left", validate="many_to_one")


def attach_macro_frame(examples: pd.DataFrame, macro: pd.DataFrame | None) -> pd.DataFrame:
    if macro is None:
        return examples
    if "month" not in macro.columns:
        raise ValueError("Macro data must contain a 'month' column")

    macro = macro.copy()
    macro["month"] = macro["month"].astype(str)
    macro = macro.add_prefix("macro_").rename(columns={"macro_month": "month"})

    examples = examples.copy()
    examples["month"] = examples["anchor_trading_date"].dt.to_period("M").astype(str)
    return examples.merge(macro, on="month", how="left", validate="many_to_one")


def chronological_split(
    examples: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")

    examples = examples.sort_values(["anchor_trading_date", "time_published", "ticker"])
    n_rows = len(examples)
    test_start = int(n_rows * (1.0 - test_fraction))
    validation_start = int(n_rows * (1.0 - test_fraction - validation_fraction))

    train = examples.iloc[:validation_start].copy()
    validation = examples.iloc[validation_start:test_start].copy()
    test = examples.iloc[test_start:].copy()

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    return train, validation, test


def date_bounded_split(
    examples: pd.DataFrame,
    validation_start_date: str,
    test_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    examples = examples.sort_values(["anchor_trading_date", "time_published", "ticker"])
    validation_start = pd.Timestamp(validation_start_date)
    test_start = pd.Timestamp(test_start_date)
    if test_start <= validation_start:
        raise ValueError("test_start_date must be after validation_start_date.")

    train = examples[examples["anchor_trading_date"] < validation_start].copy()
    validation = examples[
        (examples["anchor_trading_date"] >= validation_start)
        & (examples["anchor_trading_date"] < test_start)
    ].copy()
    test = examples[examples["anchor_trading_date"] >= test_start].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Date-bounded split produced an empty split. "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}. "
            "Adjust --validation-start-date/--test-start-date."
        )

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    return train, validation, test


def write_outputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "ltn_train.parquet", index=False)
    validation.to_parquet(output_dir / "ltn_validation.parquet", index=False)
    test.to_parquet(output_dir / "ltn_test.parquet", index=False)
    pd.concat([train, validation, test], ignore_index=True).to_parquet(
        output_dir / "ltn_all.parquet", index=False
    )


def print_summary(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    all_examples = pd.concat([train, validation, test], ignore_index=True)
    print("LTN dataset created")
    print(f"Rows: {len(all_examples):,}")
    print(f"Tickers: {', '.join(sorted(all_examples['ticker'].unique()))}")
    print(
        "Date range: "
        f"{all_examples['anchor_trading_date'].min().date()} -> "
        f"{all_examples['anchor_trading_date'].max().date()}"
    )
    print("\nSplit sizes")
    print(all_examples["split"].value_counts().sort_index().to_string())
    print("\nMarket labels")
    print(all_examples["market_sentiment_label"].value_counts().to_string())
    print("\nMateriality")
    print(all_examples["is_material"].value_counts().sort_index().to_string())
    print("\nCurrent price trends")
    print(all_examples["price_trend_label"].value_counts().to_string())
    print("\nFuture price trend targets")
    print(all_examples["future_price_trend_label"].value_counts().to_string())


def main() -> None:
    args = parse_args()
    selected_tickers = {ticker.upper() for ticker in args.tickers}

    if args.articles_hf_repo:
        articles = read_hf_parquet_files(args.articles_hf_repo, args.articles_hf_pattern, "articles")
    else:
        articles = read_parquet_dir(args.articles_dir, "articles")

    if args.tickers_hf_repo:
        tickers = read_hf_parquet_files(args.tickers_hf_repo, args.tickers_hf_pattern, "tickers")
    else:
        tickers = read_parquet_dir(args.tickers_dir, "tickers")

    if args.prices_hf_repo:
        if not args.prices_hf_path:
            raise ValueError("--prices-hf-path is required when --prices-hf-repo is set")
        prices_raw = read_hf_parquet_file(args.prices_hf_repo, args.prices_hf_path, "prices")
    else:
        prices_raw = pd.read_parquet(args.prices_path)

    prices = normalise_prices(
        prices_raw,
        trend_window_days=args.trend_window_days,
        trend_threshold=args.trend_threshold,
        regime_return_window_days=args.regime_return_window_days,
        regime_drawdown_window_days=args.regime_drawdown_window_days,
        regime_bull_return_threshold=args.regime_bull_return_threshold,
        regime_bear_return_threshold=args.regime_bear_return_threshold,
        regime_bear_drawdown_threshold=args.regime_bear_drawdown_threshold,
        trend_forward_days=args.trend_forward_days,
    )

    examples = build_article_ticker_table(articles, tickers)
    examples = examples[examples["ticker"].isin(selected_tickers)].copy()
    prices = prices[prices["ticker"].isin(selected_tickers)].copy()

    examples = add_signal_dates(
        examples,
        args.market_close_hour,
        args.source_timezone,
        args.market_timezone,
    )
    examples = attach_anchor_trading_dates(examples, prices)
    examples = add_market_labels(
        examples=examples,
        prices=prices,
        positive_threshold=args.positive_return_threshold,
        negative_threshold=args.negative_return_threshold,
        material_return_threshold=args.material_return_threshold,
        material_relevance_threshold=args.material_relevance_threshold,
    )
    if args.macro_hf_repo:
        macro = read_hf_parquet_file(args.macro_hf_repo, args.macro_hf_path, "macro")
        examples = attach_macro_frame(examples, macro)
    else:
        examples = attach_macro_features(examples, args.macro_path)

    macro_columns = sorted(column for column in examples.columns if column.startswith("macro_"))
    candidate_columns = [
        *[column for column in CORE_COLUMNS if column in examples.columns and column != "split"],
        *[column for column in ["month", *macro_columns] if column in examples.columns],
    ]
    output_columns = list(dict.fromkeys(candidate_columns))
    examples = examples[output_columns].drop_duplicates(
        subset=["article_uid", "ticker", "anchor_trading_date"]
    )
    validate_dataset_columns([*examples.columns, "split"])

    if args.split_strategy == "date":
        train, validation, test = date_bounded_split(
            examples,
            validation_start_date=args.validation_start_date,
            test_start_date=args.test_start_date,
        )
    else:
        train, validation, test = chronological_split(
            examples,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
    write_outputs(train, validation, test, args.output_dir)
    write_schema(args.schema_path)
    print_summary(train, validation, test)


if __name__ == "__main__":
    main()
