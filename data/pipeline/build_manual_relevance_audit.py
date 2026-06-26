"""Build a difficult, title-only MAG7 relevance labelling set.

The output deliberately mixes direct company events with ambiguous supplier,
competitor, product, market-commentary, and random headlines.  It is intended
for comparing prompt variants, not for estimating the natural prevalence of
relevant articles in the full feed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
RANDOM_SEED = 42
PER_TICKER = 60
BUCKET_SIZES = {
    "direct_company_development": 12,
    "ecosystem_or_competitor_link": 14,
    "ambiguous_company_or_commentary": 14,
    "market_or_investor_noise": 10,
    "random": 10,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "manual_labels"
    / "mag7_headline_relevance_hard_audit_420.csv"
)
HF_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--cookekieran--alphavantage-market-news"
)


def current_snapshot() -> Path:
    revision_file = HF_CACHE_ROOT / "refs" / "main"
    if not revision_file.exists():
        raise FileNotFoundError(f"Hugging Face dataset cache is unavailable: {HF_CACHE_ROOT}")
    revision = revision_file.read_text(encoding="utf-8").strip()
    snapshot = HF_CACHE_ROOT / "snapshots" / revision
    if not snapshot.exists():
        raise FileNotFoundError(f"Cached dataset snapshot is unavailable: {snapshot}")
    return snapshot


def article_frame(snapshot: Path) -> pd.DataFrame:
    paths = sorted((snapshot / "articles").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No monthly article parquet files found in {snapshot}")

    frames = [
        pd.read_parquet(path)[
            ["article_uid", "requested_entity", "time_published", "title", "source", "url"]
        ]
        for path in paths
    ]
    articles = pd.concat(frames, ignore_index=True)
    articles = articles.loc[articles["requested_entity"].isin(TICKERS)].copy()
    articles = articles.dropna(subset=["article_uid", "time_published", "title"])
    articles["time_published"] = pd.to_datetime(articles["time_published"], utc=True)
    articles["title_key"] = (
        articles["title"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    )
    return (
        articles.drop_duplicates(["requested_entity", "article_uid"])
        .drop_duplicates(["requested_entity", "title_key"])
        .sort_values(["requested_entity", "time_published", "article_uid"])
        .reset_index(drop=True)
    )


def bucket_masks(titles: pd.Series) -> dict[str, pd.Series]:
    direct = r"earnings|revenue|guidance|profit|margin|sales|demand|price hike|costs?|recall|layoffs?|job cuts|antitrust|lawsuit|court|fine|ban|regulat|probe|settlement"
    ecosystem = r"supplier|supply chain|customer|partner(?:ship)?|contract|deal|agreement|ppa|data cent(?:er|re)|cloud|chip|component|tariff|rival|competitor|exclusive"
    ambiguous = r"ceo|cfo|executive|leadership|product|launch|rollout|release|preview|rumou?r|could|may |might|expects?|sees|future|what .* means|battles on"
    noise = r"etf|stock price|shares? of|stake in|holdings?|portfolio|analyst|price target|buy rating|sell rating|should you buy|investors?"
    return {
        "direct_company_development": titles.str.contains(direct, case=False, regex=True, na=False),
        "ecosystem_or_competitor_link": titles.str.contains(ecosystem, case=False, regex=True, na=False),
        "ambiguous_company_or_commentary": titles.str.contains(ambiguous, case=False, regex=True, na=False),
        "market_or_investor_noise": titles.str.contains(noise, case=False, regex=True, na=False),
    }


def select_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    remaining = group.copy()
    masks = bucket_masks(group["title"])

    for bucket in (
        "direct_company_development",
        "ecosystem_or_competitor_link",
        "ambiguous_company_or_commentary",
        "market_or_investor_noise",
    ):
        candidates = group.loc[masks[bucket] & ~group["article_uid"].isin(
            pd.concat(selected)["article_uid"] if selected else []
        )]
        take = min(BUCKET_SIZES[bucket], len(candidates))
        if take:
            selected.append(candidates.sample(n=take, random_state=RANDOM_SEED))

    selected_frame = pd.concat(selected, ignore_index=True) if selected else group.iloc[0:0]
    remaining = group.loc[~group["article_uid"].isin(selected_frame["article_uid"])]
    random_take = min(BUCKET_SIZES["random"], len(remaining))
    if random_take:
        selected.append(remaining.sample(n=random_take, random_state=RANDOM_SEED))

    result = pd.concat(selected, ignore_index=True)
    if len(result) < PER_TICKER:
        filler = group.loc[~group["article_uid"].isin(result["article_uid"])]
        result = pd.concat(
            [result, filler.sample(n=PER_TICKER - len(result), random_state=RANDOM_SEED)],
            ignore_index=True,
        )
    return result.sample(frac=1, random_state=RANDOM_SEED).head(PER_TICKER)


def assign_split(frame: pd.DataFrame) -> pd.DataFrame:
    # Keep 30% of each ticker held out.  Never use these rows in n-shot examples.
    splits = []
    for _, group in frame.groupby("ticker", sort=False):
        shuffled = group.sample(frac=1, random_state=RANDOM_SEED).copy()
        test_count = round(len(shuffled) * 0.30)
        shuffled["split"] = "development"
        shuffled.iloc[:test_count, shuffled.columns.get_loc("split")] = "held_out_test"
        splits.append(shuffled)
    return pd.concat(splits, ignore_index=True)


def main() -> None:
    articles = article_frame(current_snapshot())
    selected = pd.concat(
        [select_for_ticker(group) for _, group in articles.groupby("requested_entity", sort=True)],
        ignore_index=True,
    ).rename(columns={"requested_entity": "ticker"})
    selected = assign_split(selected)
    selected.insert(0, "audit_id", [f"MAG7-{index:04d}" for index in range(1, len(selected) + 1)])
    selected["manual_label"] = ""
    selected["manual_reason"] = ""
    selected["label_notes"] = ""
    selected["label_basis"] = "title_only"
    selected = selected[
        [
            "audit_id",
            "ticker",
            "time_published",
            "title",
            "source",
            "url",
            "manual_label",
            "manual_reason",
            "label_notes",
            "label_basis",
            "split",
            "article_uid",
        ]
    ].sort_values(["ticker", "time_published", "audit_id"])
    assert len(selected) == len(TICKERS) * PER_TICKER
    assert selected.groupby("ticker").size().eq(PER_TICKER).all()
    assert selected["article_uid"].is_unique
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(selected)} rows to {OUTPUT_PATH}")
    print(selected.groupby(["ticker", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
