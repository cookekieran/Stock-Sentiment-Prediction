"""
Script for Magnificent 7 article-event label extraction.

Loads MAG7 news from Hugging Face, classifies each ticker-article into a
primary company event label, and maps each kept label to a leakage-safe
trading event_date.

Output: article_event_labels.parquet, one row per kept ticker-article label,
consumed directly by the modelling notebook. 
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import platform
import re
from datetime import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent

# Stale labels kept unchanged to match the archived final news processing metadata.
SCRIPT_VERSION = "mag7_daily_article_event_packet_extraction_v6"
PROMPT_VERSION = "two_shot_market_reaction_article_event_prompt_v7"
TAXONOMY_VERSION = "expanded_mag7_event_taxonomy_v5"

NEWS_DATASET = "cookekieran/alphavantage-market-news"
PRICE_DATASET = "cookekieran/mag7_prices"
PRICE_FILE = "mag7_daily_prices.parquet"

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

COMPANY_NAMES = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet / Google",
    "META": "Meta Platforms / Facebook / Instagram / WhatsApp",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
}

EVENT_TYPES = {
    "earnings_guidance",
    "demand_supply_revision",
    "regulatory_legal_outcome",
    "strategy_capital_allocation",
    "novel_product_event",
    "management_leadership_change",
    "major_customer_contract",
    "partnership_strategic_alliance",
    "product_safety_quality_issue",
    "cybersecurity_privacy_incident",
    "supply_chain_production_disruption",
    "m_a_divestiture_asset_sale",
}

EVENT_TYPES_SORTED = sorted(EVENT_TYPES)

DIRECTIONS = {"positive", "negative", "mixed", "none"}

COMPANY_RELEVANCE = {
    "direct_company",
    "supplier_customer",
    "peer_competitor",
    "not_relevant",
    "unclear",
}

CONFIDENCE = {"high", "medium", "low"}

MARKET_OPEN_NY = time(9, 30)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", default="2023-01", help="Start month, YYYY-MM.")
    parser.add_argument("--end", default="2026-05", help="End month, YYYY-MM.")
    parser.add_argument("--out-dir", default="outputs/mag7_daily_event_packets")

    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--news-revision", default="main")
    parser.add_argument("--price-revision", default="main")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-articles", type=int, default=None)

    parser.add_argument(
        "--relevance-mode",
        choices=["direct", "direct_plus_related"],
        default="direct",
        help="Whether to keep only direct company news or also supplier/customer-linked news.",
    )
    parser.add_argument(
        "--min-confidence",
        choices=["medium", "high"],
        default="medium",
        help="Minimum model confidence required for a label to be kept.",
    )

    return parser.parse_args()


def setup(out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=out / "run.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    load_dotenv(REPO_ROOT / ".env")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"HF_TOKEN was not found. Add it to {REPO_ROOT / '.env'} or the environment.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    return token


def normalise(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", normalise(value).lower())


def stable_hash(parts: list[Any]) -> str:
    raw = "||".join(canonical(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def month_range(start: str, end: str) -> list[str]:
    months = pd.period_range(start=start, end=end, freq="M")
    return [str(month) for month in months]


def choose(value: Any, allowed: set[str], default: Any) -> Any:
    value = canonical(value)
    return value if value in allowed else default


def choose_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = canonical(value)
    return value in {"true", "1", "yes", "y"}


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def input_article_id(row: pd.Series) -> str:
    if "article_uid" in row and normalise(row["article_uid"]):
        return normalise(row["article_uid"])

    return stable_hash(
        [
            row.get("ticker"),
            row.get("time_published"),
            row.get("title"),
            row.get("url"),
            row.get("summary"),
        ]
    )


def make_article_event_label_id(row: pd.Series) -> str:
    return stable_hash(
        [
            row.get("ticker"),
            row.get("article_uid"),
            row.get("time_published"),
            row.get("event_type"),
            row.get("direction"),
            row.get("event_key"),
            row.get("title"),
        ]
    )


def load_news(
    start: str,
    end: str,
    token: str,
    news_revision: str,
    max_articles: int | None,
) -> pd.DataFrame:
    frames = []

    for month in month_range(start, end):
        file_name = f"articles/{month}.parquet"
        logging.info("Downloading %s", file_name)

        path = hf_hub_download(
            repo_id=NEWS_DATASET,
            filename=file_name,
            repo_type="dataset",
            revision=news_revision,
            token=token,
        )

        frames.append(pd.read_parquet(path))

    if not frames:
        raise RuntimeError("No news files were loaded.")

    news = pd.concat(frames, ignore_index=True)

    if "requested_entity" not in news.columns:
        raise RuntimeError("News data must contain requested_entity.")

    if "time_published" not in news.columns:
        raise RuntimeError("News data must contain time_published.")

    if "title" not in news.columns:
        raise RuntimeError("News data must contain title.")

    news = news.rename(columns={"requested_entity": "ticker"})
    news["ticker"] = news["ticker"].astype(str).str.upper().str.strip()
    news = news[news["ticker"].isin(TICKERS)].copy()

    if "article_uid" not in news.columns:
        news["article_uid"] = ""

    news["time_published"] = pd.to_datetime(news["time_published"], utc=True, errors="coerce")
    news = news[news["time_published"].notna()].copy()

    news["title"] = news["title"].fillna("").astype(str)
    news["article_uid"] = news.apply(input_article_id, axis=1)

    # Important: deduplicate within ticker, not globally across tickers.
    news = news.sort_values(["ticker", "time_published", "article_uid"])
    news = news.drop_duplicates(["ticker", "article_uid"], keep="first")

    if max_articles is not None:
        news = news.head(max_articles).copy()

    return news.reset_index(drop=True)


def input_hash(news: pd.DataFrame) -> str:
    cols = ["ticker", "article_uid", "time_published", "title"]
    raw = news[cols].astype(str).sort_values(cols).to_csv(index=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prompt(row: pd.Series) -> str:
    ticker = row["ticker"]
    company_name = COMPANY_NAMES[ticker]

    title = normalise(row.get("title"))
    summary = normalise(row.get("summary"))
    source = normalise(row.get("source"))
    published = normalise(row.get("time_published"))

    return f"""
You are extracting one primary article-event label for a MAG7 stock-market event dataset.

Target ticker: {ticker}
Target company: {company_name}

Article source: {source}
Published time UTC: {published}
Title: {title}
Summary: {summary}

Task:
First decide whether this article is relevant. Then, only if relevant, assign one event_type and one direction.

Definition of relevant:
Relevant means the article describes a firm-specific event for the target company that could plausibly cause a market reaction in the target ticker.

Keep only if the article clearly describes one of these:
- earnings, revenue, profit, margin, sales, deliveries, demand, guidance, or financial outlook for the target company;
- lawsuit, investigation, fine, settlement, court ruling, regulatory approval/rejection, antitrust, privacy, labour, export-control, recall, or safety issue directly involving the target company;
- major acquisition, divestiture, asset sale, financing, capex plan, data-center buildout, factory expansion, production increase, production cut, restructuring, or layoffs by the target company;
- major product strategy shift, major product launch, major product delay, major platform launch, major robotaxi/autonomy launch, or major AI/cloud infrastructure launch by the target company;
- major customer contract, major partnership, major content-rights deal, major energy deal, or major infrastructure deal directly involving the target company;
- direct supply-chain or production risk affecting the target company's important products;
- direct competitive or demand threat specific enough to affect the target company;
- direct target-company executive/governance event likely to matter to investors.

Hard rejection rules:
- If the main company in the article is not the target company, return keep=false.
- If the title or summary is mainly about another MAG7 company, return keep=false, even if the row ticker is different.
- Do not classify another company's earnings, revenue, profit, sales, guidance, product, lawsuit, partnership, strategy, or stock movement as a target-company event.
- Do not classify another company's event as relevant merely because the target company's product, platform, cloud service, marketplace, ad business, AI model, chip, supplier, customer, or ecosystem is mentioned.
- Do not classify stock price movement, valuation, technical analysis, analyst ratings, price targets, investor sentiment, ETF flows, fund holdings, insider trades, or portfolio commentary as company events.
- Do not classify a fund, ETF, investor, shareholder, wealth manager, or holding company buying, selling, holding, praising, or criticizing shares as a target-company event.
- Do not classify third-party seller, app developer, advertiser, marketplace participant, platform user, or criminal activity as a target-company regulatory/legal event unless the article says the target company itself is sued, fined, investigated, sanctioned, forced to change conduct, or directly financially liable.
- Do not classify routine product refreshes, minor feature updates, app updates, product reviews, consumer deals, rankings, awards, rumours, teasers, leaks, or speculative launch details as market-reaction events.
- If the article is mostly broad macro, sector, AI boom, cloud, semiconductor, EV, advertising, retail, or data-center commentary without a specific target-company event, return keep=false.
- If the article only gives a weak indirect business link, return keep=false.
- If unsure, return keep=false.

Important:
- Use the article text provided, but do not invent missing facts.
- Do not keep merely because the target company is named.
- Do not keep merely because the article is about the target company's product ecosystem.
- Be conservative: false positives are worse than false negatives.
- If keep=false, use company_relevance="not_relevant", event_type=null, direction="none", event_key="", and mechanism="".

Allowed event_type values:
{", ".join(EVENT_TYPES_SORTED)}

Use event_type exactly from the allowed list.
Do not invent new event_type values.
Do not choose the closest allowed event_type unless it accurately describes the article.
If none of the allowed event_type values accurately describes the article, return keep=false with event_type=null and direction="none".

Event type guidance:
- earnings_guidance: target-company earnings, revenue, profit, margin, sales, deliveries, guidance, or financial outlook.
- demand_supply_revision: target-company demand, sales volume, order volume, deliveries, production volume, capacity, inventory, or supply availability.
- regulatory_legal_outcome: lawsuit, investigation, fine, settlement, court ruling, regulatory approval/rejection, ban, antitrust, privacy, labour, export-control, safety ruling, or legal/regulatory action directly involving the target company.
- strategy_capital_allocation: target-company capex, financing, buybacks, dividends, restructuring, cost cuts, office closures, strategic spending, data-center buildout, factory expansion, or major strategic operating shift.
- novel_product_event: major target-company product/service launch, major product strategy shift, major platform launch, major AI/cloud/autonomy launch, or major product delay.
- management_leadership_change: target-company executive, board, workforce, layoff, hiring, compensation, governance, or major internal leadership change.
- major_customer_contract: major customer contract, purchase commitment, customer adoption, or major customer deployment directly benefiting the target company.
- partnership_strategic_alliance: major partnership, alliance, collaboration, content-rights deal, energy deal, or infrastructure deal directly involving the target company.
- product_safety_quality_issue: target-company product defect, recall, safety issue, quality failure, or major service outage.
- cybersecurity_privacy_incident: data breach, cyberattack, privacy violation, security incident, or privacy enforcement issue directly involving the target company.
- supply_chain_production_disruption: target-company production, manufacturing, logistics, supplier dependency, supplier disruption, input sourcing, or production dependency problem.
- m_a_divestiture_asset_sale: target-company acquisition, merger, divestiture, takeover, asset sale, or investment disposal.

Direction definitions:
- positive: likely beneficial for the target company's fundamentals, demand, operations, reputation, or regulatory position.
- negative: likely harmful for the target company's fundamentals, demand, operations, reputation, or regulatory position.
- mixed: contains clearly material positive and negative implications.
- none: no usable directional event.

Direction rule:
Judge the event's firm-specific implication, not article tone and not realised stock price movement.
Use mixed when the article contains both a material positive channel and a material negative channel.

Example 1:
Target ticker: GOOGL
Target company: Alphabet / Google
Title: New York Times earnings boosted by digital bundling push
Summary: The New York Times reports stronger earnings from its own subscription bundle, with Google mentioned only as part of the digital advertising context.

Correct output:
{{
  "keep": false,
  "company_relevance": "not_relevant",
  "event_type": null,
  "direction": "none",
  "event_key": "",
  "mechanism": "",
  "confidence": "high",
  "rationale": "The earnings event belongs to The New York Times, not Alphabet or Google."
}}

Example 2:
Target ticker: AMZN
Target company: Amazon
Title: A Supplement Company 'Hijacked' Its Amazon Reviews to Boost Sales, According to the FTC
Summary: The FTC says a third-party supplement company manipulated product reviews on Amazon's marketplace.

Correct output:
{{
  "keep": false,
  "company_relevance": "not_relevant",
  "event_type": null,
  "direction": "none",
  "event_key": "",
  "mechanism": "",
  "confidence": "high",
  "rationale": "The legal action concerns a third-party seller using Amazon's marketplace, not a direct legal or regulatory action against Amazon."
}}

Example 3:
Target ticker: AAPL
Target company: Apple
Title: Buffett says Apple is Berkshire portfolio's best business
Summary: Warren Buffett praises Apple as a strong holding within Berkshire Hathaway's investment portfolio.

Correct output:
{{
  "keep": false,
  "company_relevance": "not_relevant",
  "event_type": null,
  "direction": "none",
  "event_key": "",
  "mechanism": "",
  "confidence": "high",
  "rationale": "Investor praise or portfolio commentary is not a firm-specific Apple event."
}}

Example 4:
Target ticker: AAPL
Target company: Apple
Title: Apple sees sales slump continuing, shares drop 2% despite beating sales expectations
Summary: Apple reports weaker sales and warns that the sales slump may continue despite beating expectations.

Correct output:
{{
  "keep": true,
  "company_relevance": "direct_company",
  "event_type": "earnings_guidance",
  "direction": "negative",
  "event_key": "sales slump continuing despite earnings beat",
  "mechanism": "continued sales weakness signals pressure on Apple demand and revenue outlook",
  "confidence": "high",
  "rationale": "The article reports Apple-specific sales and earnings information likely to affect the stock."
}}

Now classify the actual article.

Return only valid JSON with exactly these keys:
{{
  "keep": true or false,
  "company_relevance": "direct_company" or "supplier_customer" or "peer_competitor" or "not_relevant" or "unclear",
  "event_type": one allowed event_type value or null,
  "direction": "positive" or "negative" or "mixed" or "none",
  "event_key": short normalized phrase describing the selected primary event, or "",
  "mechanism": short phrase explaining why this matters for the target company, or "",
  "confidence": "high" or "medium" or "low",
  "rationale": one short sentence
}}
""".strip()

def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found.")

    return json.loads(match.group(0))


def allowed_by_relevance(company_relevance: str, relevance_mode: str) -> bool:
    if relevance_mode == "direct":
        return company_relevance == "direct_company"

    if relevance_mode == "direct_plus_related":
        return company_relevance in {"direct_company", "supplier_customer"}

    raise ValueError(f"Unknown relevance_mode: {relevance_mode}")


def allowed_by_confidence(confidence: str, min_confidence: str) -> bool:
    if min_confidence == "medium":
        return confidence in {"high", "medium"}

    if min_confidence == "high":
        return confidence == "high"

    raise ValueError(f"Unknown min_confidence: {min_confidence}")


def parse_classification(
    row: pd.Series,
    raw_response: str,
    model_id: str,
    model_revision: str,
    news_revision: str,
    start: str,
    end: str,
    loaded_input_hash: str,
    relevance_mode: str,
    min_confidence: str,
) -> dict[str, Any]:
    base = {
        "ticker": row["ticker"],
        "article_uid": row["article_uid"],
        "time_published": row["time_published"],
        "title": row.get("title", ""),
        "source": row.get("source", ""),
        "url": row.get("url", ""),
        "raw_response": raw_response,
        "parse_ok": False,
        "keep": False,
        "company_relevance": "unclear",
        "event_type": None,
        "direction": "none",
        "event_key": "",
        "mechanism": "",
        "confidence": "low",
        "rationale": "",
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "news_dataset": NEWS_DATASET,
        "news_revision": news_revision,
        "start_month": start,
        "end_month": end,
        "input_hash": loaded_input_hash,
        "relevance_mode": relevance_mode,
        "min_confidence": min_confidence,
    }

    try:
        parsed = parse_json(raw_response)
    except Exception as exc:
        base["rationale"] = f"parse_error: {exc}"
        return base

    company_relevance = choose(parsed.get("company_relevance"), COMPANY_RELEVANCE, "unclear")
    confidence = choose(parsed.get("confidence"), CONFIDENCE, "low")
    direction = choose(parsed.get("direction"), DIRECTIONS, "none")

    event_type_raw = parsed.get("event_type")
    event_type = choose(event_type_raw, EVENT_TYPES, None) if event_type_raw is not None else None

    keep = choose_bool(parsed.get("keep"))
    keep = keep and event_type is not None
    keep = keep and direction in {"positive", "negative", "mixed"}
    keep = keep and allowed_by_relevance(company_relevance, relevance_mode)
    keep = keep and allowed_by_confidence(confidence, min_confidence)

    base.update(
        {
            "parse_ok": True,
            "keep": bool(keep),
            "company_relevance": company_relevance,
            "event_type": event_type,
            "direction": direction,
            "event_key": normalise(parsed.get("event_key")),
            "mechanism": normalise(parsed.get("mechanism")),
            "confidence": confidence,
            "rationale": normalise(parsed.get("rationale")),
        }
    )

    return base


def valid_checkpoint_rows(
    checkpoint: pd.DataFrame,
    model_id: str,
    model_revision: str,
    news_revision: str,
    start: str,
    end: str,
    loaded_input_hash: str,
    relevance_mode: str,
    min_confidence: str,
) -> pd.DataFrame:
    required = {
        "ticker",
        "article_uid",
        "script_version",
        "prompt_version",
        "taxonomy_version",
        "model_id",
        "model_revision",
        "news_revision",
        "start_month",
        "end_month",
        "input_hash",
        "relevance_mode",
        "min_confidence",
    }

    if checkpoint.empty or not required.issubset(checkpoint.columns):
        return pd.DataFrame()

    mask = (
        (checkpoint["script_version"] == SCRIPT_VERSION)
        & (checkpoint["prompt_version"] == PROMPT_VERSION)
        & (checkpoint["taxonomy_version"] == TAXONOMY_VERSION)
        & (checkpoint["model_id"] == model_id)
        & (checkpoint["model_revision"] == model_revision)
        & (checkpoint["news_revision"] == news_revision)
        & (checkpoint["start_month"] == start)
        & (checkpoint["end_month"] == end)
        & (checkpoint["input_hash"] == loaded_input_hash)
        & (checkpoint["relevance_mode"] == relevance_mode)
        & (checkpoint["min_confidence"] == min_confidence)
    )

    return checkpoint.loc[mask].copy()


def classify(
    news: pd.DataFrame,
    out: Path,
    token: str,
    model_id: str,
    model_revision: str,
    news_revision: str,
    start: str,
    end: str,
    loaded_input_hash: str,
    batch_size: int,
    checkpoint_every: int,
    relevance_mode: str,
    min_confidence: str,
) -> pd.DataFrame:
    checkpoint_path = out / "classification_checkpoint.parquet"

    if checkpoint_path.exists():
        checkpoint = pd.read_parquet(checkpoint_path)
        done = valid_checkpoint_rows(
            checkpoint,
            model_id=model_id,
            model_revision=model_revision,
            news_revision=news_revision,
            start=start,
            end=end,
            loaded_input_hash=loaded_input_hash,
            relevance_mode=relevance_mode,
            min_confidence=min_confidence,
        )
    else:
        done = pd.DataFrame()

    done_keys = set()
    if not done.empty:
        done_keys = set(zip(done["ticker"].astype(str), done["article_uid"].astype(str)))

    news_keys = list(zip(news["ticker"].astype(str), news["article_uid"].astype(str)))
    todo = news[[key not in done_keys for key in news_keys]].copy()

    logging.info("Loaded %d checkpoint rows", len(done))
    logging.info("Classifying %d remaining ticker-articles", len(todo))

    if todo.empty:
        return done.drop_duplicates(["ticker", "article_uid"], keep="last").reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    revision=model_revision,
    token=token,
    trust_remote_code=True)

    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = [] if done.empty else done.to_dict("records")

    for start_idx in range(0, len(todo), batch_size):
        batch = todo.iloc[start_idx : start_idx + batch_size].copy()

        messages = [
            [{"role": "user", "content": prompt(row)}]
            for _, row in batch.iterrows()
        ]

        texts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = encoded["input_ids"].shape[1]
        generated = output[:, prompt_length:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for (_, row), raw_response in zip(batch.iterrows(), decoded):
            records.append(
                parse_classification(
                    row=row,
                    raw_response=raw_response,
                    model_id=model_id,
                    model_revision=model_revision,
                    news_revision=news_revision,
                    start=start,
                    end=end,
                    loaded_input_hash=loaded_input_hash,
                    relevance_mode=relevance_mode,
                    min_confidence=min_confidence,
                )
            )

        if len(records) % checkpoint_every < batch_size:
            checkpoint_df = pd.DataFrame(records)
            checkpoint_df = checkpoint_df.drop_duplicates(["ticker", "article_uid"], keep="last")
            checkpoint_df.to_parquet(checkpoint_path, index=False)
            logging.info("Checkpointed %d rows", len(checkpoint_df))

    classified = pd.DataFrame(records)
    classified = classified.drop_duplicates(["ticker", "article_uid"], keep="last")
    classified.to_parquet(checkpoint_path, index=False)

    return classified.reset_index(drop=True)


def make_article_event_labels(classified: pd.DataFrame) -> pd.DataFrame:
    labels = classified[classified["keep"]].copy()

    if labels.empty:
        return pd.DataFrame(
            columns=[
                "article_event_label_id",
                "ticker",
                "article_uid",
                "time_published",
                "event_type",
                "direction",
                "company_relevance",
                "event_key",
                "mechanism",
                "confidence",
                "rationale",
                "representative_title",
                "article_count",
                "source_article_uids",
                "raw_response",
                "parse_ok",
            ]
        )

    labels["representative_title"] = labels["title"].fillna("").astype(str)
    labels["article_count"] = 1
    labels["source_article_uids"] = labels["article_uid"].astype(str)
    labels["article_event_label_id"] = labels.apply(make_article_event_label_id, axis=1)

    cols = [
        "article_event_label_id",
        "ticker",
        "article_uid",
        "time_published",
        "event_type",
        "direction",
        "company_relevance",
        "event_key",
        "mechanism",
        "confidence",
        "rationale",
        "representative_title",
        "article_count",
        "source_article_uids",
        "raw_response",
        "parse_ok",
    ]

    labels = labels[cols].copy()
    labels = labels.sort_values(["ticker", "time_published", "article_uid"])
    labels = labels.drop_duplicates(["ticker", "article_uid"], keep="first")

    return labels.reset_index(drop=True)


def load_price_calendar(token: str, price_revision: str) -> dict[str, list[pd.Timestamp]]:
    path = hf_hub_download(
        repo_id=PRICE_DATASET,
        filename=PRICE_FILE,
        repo_type="dataset",
        revision=price_revision,
        token=token,
    )

    prices = pd.read_parquet(path)

    if "date" not in prices.columns:
        raise RuntimeError("Price data must contain date.")

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date
    prices = prices[prices["date"].notna()].copy()

    calendars: dict[str, list[pd.Timestamp]] = {}

    if "ticker" in prices.columns:
        for ticker, group in prices.groupby("ticker"):
            calendars[str(ticker).upper()] = sorted(pd.to_datetime(group["date"]).unique())
    else:
        shared = sorted(pd.to_datetime(prices["date"]).unique())
        for ticker in TICKERS:
            calendars[ticker] = shared

    return calendars


def leakage_safe_event_date(
    published_utc: pd.Timestamp,
    ticker: str,
    calendars: dict[str, list[pd.Timestamp]],
) -> tuple[Any, bool]:
    if pd.isna(published_utc):
        return None, False

    if ticker not in calendars or not calendars[ticker]:
        return None, False

    published_ny = published_utc.tz_convert("America/New_York")
    published_day = pd.Timestamp(published_ny.date())

    if published_ny.time() < MARKET_OPEN_NY:
        candidate = published_day
    else:
        candidate = published_day + pd.Timedelta(days=1)

    for trading_day in calendars[ticker]:
        trading_day = pd.Timestamp(trading_day).normalize()
        if trading_day >= candidate:
            return trading_day.date(), True

    return None, False


def add_event_dates(
    labels: pd.DataFrame,
    token: str,
    price_revision: str,
) -> pd.DataFrame:
    labels = labels.copy()

    if labels.empty:
        labels["event_date"] = pd.Series(dtype="object")
        labels["event_date_mapped"] = pd.Series(dtype="bool")
        return labels

    calendars = load_price_calendar(token, price_revision)

    mapped = labels.apply(
        lambda row: leakage_safe_event_date(
            published_utc=row["time_published"],
            ticker=row["ticker"],
            calendars=calendars,
        ),
        axis=1,
    )

    labels["event_date"] = [item[0] for item in mapped]
    labels["event_date_mapped"] = [item[1] for item in mapped]

    return labels


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).to_dict().items()}


def write_samples(
    out: Path,
    classified: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    classified.head(200).to_csv(out / "classification_sample.csv", index=False)
    labels.head(200).to_csv(out / "article_event_labels_sample.csv", index=False)


def write_manifest(
    out: Path,
    run_args: argparse.Namespace,
    news: pd.DataFrame,
    classified: pd.DataFrame,
    labels: pd.DataFrame,
    loaded_input_hash: str,
) -> None:
    parse_ok = int(classified["parse_ok"].sum()) if "parse_ok" in classified.columns else 0
    parse_failures = int(len(classified) - parse_ok)
    kept = int(classified["keep"].sum()) if "keep" in classified.columns else 0

    mapped = int(labels["event_date_mapped"].sum()) if "event_date_mapped" in labels.columns else 0
    unmapped = int(len(labels) - mapped)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "news_dataset": NEWS_DATASET,
        "news_revision": run_args.news_revision,
        "price_dataset": PRICE_DATASET,
        "price_revision": run_args.price_revision,
        "model_id": run_args.model_id,
        "model_revision": run_args.model_revision,
        "start_month": run_args.start,
        "end_month": run_args.end,
        "input_hash": loaded_input_hash,
        "relevance_mode": run_args.relevance_mode,
        "min_confidence": run_args.min_confidence,
        "counts": {
            "input_ticker_articles": int(len(news)),
            "classified_ticker_articles": int(len(classified)),
            "parse_ok": parse_ok,
            "parse_failures": parse_failures,
            "parse_failure_pct": round(100 * parse_failures / max(len(classified), 1), 4),
            "kept_article_event_labels": int(len(labels)),
            "kept_pct_of_classified": round(100 * kept / max(len(classified), 1), 4),
            "event_date_mapped": mapped,
            "event_date_unmapped": unmapped,
            "event_date_mapping_failure_pct": round(100 * unmapped / max(len(labels), 1), 4),
        },
        "distributions": {
            "input_articles_by_ticker": value_counts_dict(news["ticker"]),
            "kept_labels_by_ticker": value_counts_dict(labels["ticker"]) if not labels.empty else {},
            "kept_labels_by_event_type": value_counts_dict(labels["event_type"]) if not labels.empty else {},
            "kept_labels_by_direction": value_counts_dict(labels["direction"]) if not labels.empty else {},
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface_hub"),
            "pandas": package_version("pandas"),
            "pyarrow": package_version("pyarrow"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "outputs": {
            "classification_checkpoint": "classification_checkpoint.parquet",
            "article_event_labels": "article_event_labels.parquet",
            "article_event_labels_audit": "article_event_labels_audit.csv",
            "manifest": "manifest.json",
        },
        "notes": [
            "No 7-day fuzzy event deduplication is applied.",
            "Exact duplicate articles are removed by ticker + article_uid.",
            "Each kept ticker-article becomes one primary article-event label.",
            "If an article mentions multiple possible events, the prompt asks the model to select the single most material direct event.",
        ],
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    run_args = args()
    out = Path(run_args.out_dir)

    token = setup(out)

    logging.info("Loading news")
    news = load_news(
        start=run_args.start,
        end=run_args.end,
        token=token,
        news_revision=run_args.news_revision,
        max_articles=run_args.max_articles,
    )

    loaded_input_hash = input_hash(news)

    logging.info("Input ticker-articles: %d", len(news))
    logging.info("Input hash: %s", loaded_input_hash)

    classified = classify(
        news=news,
        out=out,
        token=token,
        model_id=run_args.model_id,
        model_revision=run_args.model_revision,
        news_revision=run_args.news_revision,
        start=run_args.start,
        end=run_args.end,
        loaded_input_hash=loaded_input_hash,
        batch_size=run_args.batch_size,
        checkpoint_every=run_args.checkpoint_every,
        relevance_mode=run_args.relevance_mode,
        min_confidence=run_args.min_confidence,
    )

    logging.info("Making article event labels")
    labels = make_article_event_labels(classified)

    logging.info("Mapping article event labels to leakage-safe trading dates")
    labels = add_event_dates(
        labels=labels,
        token=token,
        price_revision=run_args.price_revision,
    )

    classified.to_parquet(out / "classification_checkpoint.parquet", index=False)

    labels.to_parquet(out / "article_event_labels.parquet", index=False)
    labels.to_csv(out / "article_event_labels_audit.csv", index=False)

    write_samples(out, classified, labels)
    write_manifest(out, run_args, news, classified, labels, loaded_input_hash)

    logging.info("Done")
    logging.info("Article event labels: %d", len(labels))
    logging.info("Output directory: %s", out)


if __name__ == "__main__":
    main()
