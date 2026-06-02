"""Create a stratified packet-review template for semantic predicate labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATION_COLUMNS = [
    "label_ticker_relevance",
    "label_materiality",
    "label_risk_on_pressure",
    "label_risk_off_pressure",
    "label_uncertainty",
    "label_event_type",
    "label_scope",
    "review_notes",
    "reviewed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-packets-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--rows-per-split", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_split(group: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    if len(group) <= rows:
        return group.copy()
    group = group.copy()
    group["_relevance_band"] = pd.qcut(
        group["weak_packet_max_relevance"].rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    )
    group["_transition"] = (
        pd.to_numeric(group["price_trend_id"], errors="coerce")
        != pd.to_numeric(group["future_price_trend_id"], errors="coerce")
    ).map({True: "transition", False: "stable"})
    group["_stratum"] = (
        group["ticker"].astype(str)
        + "|"
        + group["_transition"].astype(str)
        + "|"
        + group["_relevance_band"].astype(str)
    )
    fractions = rows / len(group)
    sampled = pd.concat(
        [
            part.sample(max(1, int(round(len(part) * fractions))), random_state=seed)
            for _, part in group.groupby("_stratum", sort=False)
        ]
    ).drop_duplicates(["ticker", "anchor_trading_date"])
    if len(sampled) > rows:
        sampled = sampled.sample(rows, random_state=seed)
    elif len(sampled) < rows:
        remaining = group.drop(sampled.index)
        sampled = pd.concat(
            [sampled, remaining.sample(min(rows - len(sampled), len(remaining)), random_state=seed)]
        )
    return sampled.head(rows).copy()


def main() -> None:
    args = parse_args()
    packets = pd.read_parquet(args.daily_packets_path)
    required = {
        "ticker",
        "anchor_trading_date",
        "split",
        "packet_text",
        "packet_article_count",
        "weak_packet_max_relevance",
        "price_trend_id",
        "future_price_trend_id",
    }
    missing = sorted(required.difference(packets.columns))
    if missing:
        raise ValueError(f"Daily packet dataset is missing required columns: {missing}")
    packets["anchor_trading_date"] = pd.to_datetime(packets["anchor_trading_date"]).dt.normalize()
    sampled = pd.concat(
        [
            sample_split(group, args.rows_per_split, args.seed)
            for _, group in packets.groupby("split", sort=False)
        ],
        ignore_index=True,
    )
    sampled["packet_id"] = (
        sampled["ticker"].astype(str)
        + "_"
        + sampled["anchor_trading_date"].dt.strftime("%Y-%m-%d")
    )
    review = sampled[
        [
            "packet_id",
            "ticker",
            "anchor_trading_date",
            "split",
            "packet_article_count",
            "packet_text",
        ]
    ].copy()
    review["label_ticker_relevance"] = np.nan
    review["label_materiality"] = np.nan
    review["label_risk_on_pressure"] = np.nan
    review["label_risk_off_pressure"] = np.nan
    review["label_uncertainty"] = np.nan
    review["label_event_type"] = ""
    review["label_scope"] = ""
    review["review_notes"] = ""
    review["reviewed"] = False
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    review.sort_values(["split", "ticker", "anchor_trading_date"]).to_csv(args.output_path, index=False)
    print("Predicate annotation template created")
    print(f"Rows: {len(review):,}")
    print("Split counts")
    print(review["split"].value_counts().to_string())
    print("Labels: numeric fields use 0.0-1.0; scope and event type are reviewer categories.")
    print("Future regime labels are intentionally hidden from reviewers.")


if __name__ == "__main__":
    main()
