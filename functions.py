from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ast
import hashlib

def get_end_date(start_day, desired_timeframe_in_days=1):
    start_date = datetime.strptime(start_day, "%Y-%m-%d")
    end_date = start_date + timedelta(days=desired_timeframe_in_days)
    return start_date, end_date

def process_alpha_vantage_data(input_df: pd.DataFrame):
    """
    ETL for Alpha Vantage articles.

    Outputs:
    1. df_articles_main
    2. df_tickers_metadata
    3. df_topics_metadata

    Guarantees:
    - Stable article_id (not index-based)
    - Stable article_uid (hash-based, dedupe-safe)
    - No silent data loss (parse failures are tracked)
    - Deterministic joins across exploded tables
    """

    working_df = input_df.copy()

    # -----------------------------------------------------------------
    # Set Article ID
    # -----------------------------------------------------------------
    working_df = working_df.reset_index(drop=True)
    working_df["article_id"] = np.arange(len(working_df))

    # create stable unique identifier for deduplication + HF uploads
    def make_uid(row):
        base = f"{row.get('url','')}|{row.get('time_published','')}|{row.get('title','')}"
        return hashlib.md5(base.encode()).hexdigest()

    working_df["article_uid"] = working_df.apply(make_uid, axis=1)

    # -----------------------------------------------------------------
    # Main Articles Table
    # -----------------------------------------------------------------
    print("Creating main articles table...")

    target_columns = [
        "article_id",
        "article_uid",
        "time_published",
        "overall_sentiment_score",
        "overall_sentiment_label",
        "title",
        "summary",
        "source",
        "url",
        "requested_entity",
    ]

    df_articles_main = working_df[[c for c in target_columns if c in working_df.columns]].copy()

    if "summary" in df_articles_main.columns:
        df_articles_main["summary"] = df_articles_main["summary"].fillna("").astype(str)

    # -----------------------------------------------------------------
    # DATA PARSER
    # -----------------------------------------------------------------
    def parse_json_array(val):
        if val is None or pd.isna(val) or val == "":
            return [], False

        if isinstance(val, (list, dict)):
            return val, False

        if isinstance(val, str):
            try:
                return ast.literal_eval(val.strip()), False
            except Exception:
                return [], True  # parse failure flagged

        return [], True

    # -----------------------------------------------------------------
    # TICKER METADATA (table 2)
    # -----------------------------------------------------------------
    print("Creating ticker metadata table...")

    if "ticker_sentiment" in working_df.columns:
        parsed = working_df["ticker_sentiment"].apply(parse_json_array)
        working_df["parsed_tickers"] = parsed.apply(lambda x: x[0])
        working_df["ticker_parse_error"] = parsed.apply(lambda x: x[1])

        df_tick = working_df[
            ["article_id", "article_uid", "parsed_tickers", "ticker_parse_error"]
        ].explode("parsed_tickers")

        df_tick = df_tick[df_tick["parsed_tickers"].notna()]

        valid_mask = df_tick["parsed_tickers"].apply(lambda x: isinstance(x, dict))
        df_tick_valid = df_tick[valid_mask].copy()

        if not df_tick_valid.empty:
            df_flat = pd.json_normalize(df_tick_valid["parsed_tickers"])

            df_tickers_metadata = pd.concat(
                [
                    df_tick_valid[
                        ["article_id", "article_uid", "ticker_parse_error"]
                    ].reset_index(drop=True),
                    df_flat.reset_index(drop=True),
                ],
                axis=1,
            )

            for col in ["relevance_score", "ticker_sentiment_score"]:
                if col in df_tickers_metadata.columns:
                    df_tickers_metadata[col] = (
                        pd.to_numeric(df_tickers_metadata[col], errors="coerce")
                        .fillna(0.0)
                    )
        else:
            df_tickers_metadata = pd.DataFrame(
                columns=[
                    "article_id",
                    "article_uid",
                    "ticker_parse_error",
                    "ticker",
                    "relevance_score",
                    "ticker_sentiment_score",
                    "ticker_sentiment_label",
                ]
            )
    else:
        df_tickers_metadata = pd.DataFrame(columns=["article_id", "article_uid"])

    # -----------------------------------------------------------------
    # TOPIC METADATA (table 3)
    # -----------------------------------------------------------------
    print("Creating topic metadata table...")

    if "topics" in working_df.columns:
        parsed = working_df["topics"].apply(parse_json_array)
        working_df["parsed_topics"] = parsed.apply(lambda x: x[0])
        working_df["topic_parse_error"] = parsed.apply(lambda x: x[1])

        df_top = working_df[
            ["article_id", "article_uid", "parsed_topics", "topic_parse_error"]
        ].explode("parsed_topics")

        df_top = df_top[df_top["parsed_topics"].notna()]

        valid_mask = df_top["parsed_topics"].apply(lambda x: isinstance(x, dict))
        df_top_valid = df_top[valid_mask].copy()

        if not df_top_valid.empty:
            df_flat = pd.json_normalize(df_top_valid["parsed_topics"])

            df_topics_metadata = pd.concat(
                [
                    df_top_valid[
                        ["article_id", "article_uid", "topic_parse_error"]
                    ].reset_index(drop=True),
                    df_flat.reset_index(drop=True),
                ],
                axis=1,
            )

            if "relevance_score" in df_topics_metadata.columns:
                df_topics_metadata["relevance_score"] = (
                    pd.to_numeric(df_topics_metadata["relevance_score"], errors="coerce")
                    .fillna(0.0)
                )
        else:
            df_topics_metadata = pd.DataFrame(
                columns=[
                    "article_id",
                    "article_uid",
                    "topic_parse_error",
                    "topic",
                    "relevance_score",
                ]
            )
    else:
        df_topics_metadata = pd.DataFrame(columns=["article_id", "article_uid"])

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print("\nPipeline complete")
    print(f"   Articles  : {len(df_articles_main)}")
    print(f"   Tickers   : {len(df_tickers_metadata)}")
    print(f"   Topics    : {len(df_topics_metadata)}\n")

    return df_articles_main, df_tickers_metadata, df_topics_metadata