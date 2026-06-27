"""Run the MAG7 4-shot headline relevance filter over the full news archive.

This is intended for the University GPU container. It checkpoints after every
batch so the run can be safely resumed if the job stops halfway through.

Default behaviour:
  - downloads monthly article parquet files from the Hugging Face dataset used
    by the rest of this project;
  - filters article/ticker pairs for the MAG7 tickers;
  - runs the fixed 4-shot relevance prompt;
  - appends each completed batch to a JSONL checkpoint;
  - writes final parquet files for all predictions and relevant predictions.

Example:
    python scripts/run_mag7_relevance_filter.py --start 2023-01 --end 2026-05

Smoke test:
    python scripts/run_mag7_relevance_filter.py --start 2023-01 --end 2023-01 --max-rows 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
NEWS_REPO = "cookekieran/alphavantage-market-news"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
N_SHOTS = 4

# Fictional MAG7-style 4-shot examples. None are from the manual audit CSV.
FOUR_SHOT_EXAMPLES = [
    (
        "Apple secures exclusive United States broadcast rights for a major global sports series under a five-year agreement.",
        "relevant",
    ),
    (
        "Apple refreshes an existing headset with a faster chip and minor design changes.",
        "not relevant",
    ),
    (
        "A software company expands its collaboration with Google Cloud to deploy a cybersecurity platform for enterprise customers.",
        "relevant",
    ),
    (
        "Tesla earnings preview: investors are watching margins, deliveries, and self-driving progress.",
        "not relevant",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01", help="First month to process, formatted YYYY-MM.")
    parser.add_argument("--end", default="2026-05", help="Last month to process, formatted YYYY-MM.")
    parser.add_argument("--input-path", default=None, help="Optional local CSV/parquet input instead of Hugging Face monthly files.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None, help="Use only for smoke tests.")
    parser.add_argument("--checkpoint-name", default="relevance_predictions_checkpoint.jsonl")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution. Not recommended for the full run.")
    return parser.parse_args()


def setup(out_dir: Path, allow_cpu: bool) -> str | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler()],
    )

    # Support both names used across this project/container.
    for env_file in [Path(".env"), Path("env.txt")]:
        if env_file.exists():
            load_dotenv(env_file)

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA is not available. Use a GPU container or pass --allow-cpu for a tiny smoke test.")

    return os.getenv("HF_TOKEN")


def make_prompt(ticker: str, published_at: pd.Timestamp, title: str) -> str:
    published_at = pd.Timestamp(published_at)
    demos = "\n\nFICTIONAL LABELLED EXAMPLES\n" + "\n\n".join(
        f"Headline: {headline}\nClassification: {label}"
        for headline, label in FOUR_SHOT_EXAMPLES
    )

    return f"""You perform a first-stage company-news relevance filter for ticker {ticker}.

Classify as relevant when the headline reports an actual, concrete company development, including:

- earnings, revenue, margin, demand, delivery, or guidance results;
- a new named commercial deal, rights agreement, contract, financing decision, supply disruption, recall, or regulatory/legal outcome involving the target;
- a newly announced or expanded cloud/product partnership where the target's named service is part of the commercial or technical arrangement;
- a direct competitive development that explicitly changes demand for, substitutes for, restricts, or displaces the target's product;
- an announced strategic or capital-allocation decision by the target company.

Classify as not relevant when the headline is mainly:

- a routine hardware refresh, minor feature update, award, product review, teaser, rumour, or future-product preview;
- an earnings preview or stock-market article saying investors are watching a future result, rather than reporting the actual result;
- an analyst target, investor opinion, executive opinion, valuation discussion, fund holding, or ETF story;
- a third-party competitor, supplier, utility, or infrastructure story that does not state a direct new commercial, legal, or operational action by or with the target;
- broad macro, index, tariff, or sector commentary.

Do not predict returns or use information outside the supplied date and headline.{demos}

TARGET HEADLINE
Ticker: {ticker}
Publication date: {published_at.date().isoformat()}
Headline: {title}
Classification:""".strip()


def read_jsonl_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning("Skipping malformed checkpoint line %d in %s", line_number, path)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates("row_key", keep="last")


def append_jsonl_checkpoint(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def row_key(article_uid: str, ticker: str, published_at: str, title: str) -> str:
    raw = f"{article_uid}|{ticker}|{published_at}|{title}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def normalise_news_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()

    if "ticker" not in frame.columns:
        if "requested_entity" in frame.columns:
            frame["ticker"] = frame["requested_entity"]
        elif "symbol" in frame.columns:
            frame["ticker"] = frame["symbol"]
        else:
            raise ValueError("Input needs a ticker, requested_entity, or symbol column.")

    if "article_uid" not in frame.columns:
        if "article_id" in frame.columns:
            frame["article_uid"] = frame["article_id"]
        elif "url" in frame.columns:
            frame["article_uid"] = frame["url"]
        else:
            frame["article_uid"] = frame.index.astype(str)

    if "time_published" not in frame.columns:
        for candidate in ["published_at", "publication_date", "datetime", "date"]:
            if candidate in frame.columns:
                frame["time_published"] = frame[candidate]
                break
        else:
            raise ValueError("Input needs a time_published or equivalent date column.")

    if "title" not in frame.columns:
        for candidate in ["headline", "summary"]:
            if candidate in frame.columns:
                frame["title"] = frame[candidate]
                break
        else:
            raise ValueError("Input needs a title or headline column.")

    news = frame[["article_uid", "ticker", "time_published", "title"]].copy()
    news["ticker"] = news["ticker"].astype(str).str.upper()
    news = news.loc[news["ticker"].isin(TICKERS)]
    news = news.dropna(subset=["time_published", "title"])
    news["time_published"] = pd.to_datetime(news["time_published"], utc=True, errors="coerce")
    news = news.dropna(subset=["time_published"])
    news["title"] = news["title"].astype(str).str.strip()
    news = news.loc[news["title"].ne("")]
    news["row_key"] = [
        row_key(str(article_uid), str(ticker), str(time_published), str(title))
        for article_uid, ticker, time_published, title in news[["article_uid", "ticker", "time_published", "title"]].itertuples(index=False)
    ]
    news = news.drop_duplicates("row_key")
    return news.sort_values(["ticker", "time_published", "article_uid"]).reset_index(drop=True)


def load_local_input(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(f"Unsupported input format: {input_path}. Use CSV or parquet.")


def load_hf_news(start: str, end: str, token: str | None) -> pd.DataFrame:
    frames = []
    for month in pd.period_range(start, end, freq="M").astype(str):
        logging.info("Loading Hugging Face news month %s", month)
        path = hf_hub_download(
            NEWS_REPO,
            repo_type="dataset",
            filename=f"articles/{month}.parquet",
            token=token,
        )
        frames.append(pd.read_parquet(path))

    if not frames:
        raise ValueError("No news months loaded.")

    return pd.concat(frames, ignore_index=True)


def load_news(args: argparse.Namespace, token: str | None) -> pd.DataFrame:
    if args.input_path:
        raw = load_local_input(Path(args.input_path))
    else:
        raw = load_hf_news(args.start, args.end, token)

    news = normalise_news_frame(raw)
    if args.max_rows:
        news = news.head(args.max_rows)

    logging.info("Loaded %d MAG7 article/ticker rows", len(news))
    return news


def load_model(model_id: str) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"), padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Intentionally move the whole model to one device. If the GPU does not have
    # enough memory, fail clearly instead of silently offloading layers to CPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=os.getenv("HF_TOKEN"),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    logging.info("Loaded %s on %s", model_id, device)
    return tokenizer, model, device


def parse_label(raw_response: str) -> str:
    labels = re.findall(r"\bnot relevant\b|\brelevant\b", raw_response.casefold())
    return labels[-1] if labels else ""


def classify_batch(
    batch: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> list[dict]:
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": make_prompt(row.ticker, row.time_published, row.title)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in batch.itertuples(index=False)
    ]

    encoded = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    suffixes = generated[:, encoded["input_ids"].shape[1] :]
    rows = []

    for row, suffix in zip(batch.itertuples(index=False), suffixes):
        raw_response = tokenizer.decode(suffix, skip_special_tokens=True).strip()
        predicted_label = parse_label(raw_response)
        rows.append(
            {
                "row_key": row.row_key,
                "article_uid": row.article_uid,
                "ticker": row.ticker,
                "time_published": row.time_published.isoformat(),
                "title": row.title,
                "n_shots": N_SHOTS,
                "predicted_label": predicted_label,
                "is_relevant": predicted_label == "relevant",
                "raw_response": raw_response,
            }
        )

    return rows


def run_filter(args: argparse.Namespace, news: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    checkpoint_path = out_dir / args.checkpoint_name
    done = read_jsonl_checkpoint(checkpoint_path)

    seen = set(done["row_key"]) if not done.empty else set()
    todo = news.loc[~news["row_key"].isin(seen)].reset_index(drop=True)

    logging.info(
        "Rows total=%d, already checkpointed=%d, remaining=%d",
        len(news),
        len(seen),
        len(todo),
    )

    if todo.empty:
        return done

    tokenizer, model, device = load_model(args.model_id)

    for start in range(0, len(todo), args.batch_size):
        batch = todo.iloc[start : start + args.batch_size]
        records = classify_batch(batch, tokenizer, model, device)
        append_jsonl_checkpoint(checkpoint_path, records)

        completed = min(start + len(batch), len(todo))
        relevant = sum(row["is_relevant"] for row in records)
        logging.info(
            "Checkpointed batch rows %d-%d of %d remaining; batch relevant=%d",
            start + 1,
            completed,
            len(todo),
            relevant,
        )

    return read_jsonl_checkpoint(checkpoint_path)


def write_outputs(predictions: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    predictions = predictions.drop_duplicates("row_key", keep="last").sort_values(["ticker", "time_published", "article_uid"])
    relevant = predictions.loc[predictions["is_relevant"]].copy()
    unparsed = predictions.loc[predictions["predicted_label"].eq("")].copy()

    predictions.to_parquet(out_dir / "relevance_predictions.parquet", index=False)
    relevant.to_parquet(out_dir / "relevant_article_ticker_rows.parquet", index=False)
    if not unparsed.empty:
        unparsed.to_parquet(out_dir / "unparsed_model_outputs.parquet", index=False)

    manifest = {
        "model_id": args.model_id,
        "n_shots": N_SHOTS,
        "start": args.start,
        "end": args.end,
        "input_path": args.input_path,
        "total_rows": int(len(predictions)),
        "relevant_rows": int(len(relevant)),
        "not_relevant_rows": int(predictions["predicted_label"].eq("not relevant").sum()),
        "unparsed_rows": int(predictions["predicted_label"].eq("").sum()),
        "checkpoint_file": args.checkpoint_name,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Wrote %d predictions and %d relevant rows to %s", len(predictions), len(relevant), out_dir)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir or "outputs/mag7_relevance_filter_4shot")
    token = setup(out_dir, args.allow_cpu)
    news = load_news(args, token)
    predictions = run_filter(args, news, out_dir)
    write_outputs(predictions, out_dir, args)


if __name__ == "__main__":
    main()
