"""
Extract open-ended contextual market drivers from news with DeepSeek.

Unlike a fixed schema such as "geopolitical risk" or "inflation pressure", this
script lets DeepSeek name the currently relevant driver in natural language and
then records general update signals: risk direction, intensity, novelty,
confidence, time horizon, and scope. These signals can be aggregated into the
latent-state dataset, while the driver text remains available for explanations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DIRECTION_TO_SCORE = {
    "strong_risk_off": -1.0,
    "risk_off": -0.65,
    "mixed": 0.0,
    "neutral": 0.0,
    "risk_on": 0.65,
    "strong_risk_on": 1.0,
}

HORIZON_TO_DAYS = {
    "intraday": 1.0,
    "days": 3.0,
    "weeks": 20.0,
    "months": 60.0,
    "unknown": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/ltn_all.parquet"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/contextual_driver_features.parquet"),
    )
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=None,
        help="Only extract drivers for rows with relevance_score at least this value.",
    )
    parser.add_argument(
        "--top-n-per-ticker-day",
        type=int,
        default=None,
        help="Keep only the top N most relevant articles for each ticker/trading day.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "validation", "test"],
        help="Restrict extraction to one or more dataset splits. Repeat the flag for multiple splits.",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Write partial output after this many new rows.",
    )
    return parser.parse_args()


def clamp01(value, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return float(min(1.0, max(0.0, numeric)))


def normalise_choice(value, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in allowed else default


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def build_prompt(row: pd.Series) -> str:
    title = str(row.get("title", "") or "")
    summary = str(row.get("summary", "") or "")
    ticker = str(row.get("ticker", "") or "")
    date = str(row.get("anchor_trading_date", "") or "")[:10]
    return f"""You are analysing financial news for market-regime forecasting.

Ticker: {ticker}
Date: {date}
Title: {title}
Summary: {summary}

Identify the most important contextual market driver in this news item for the
given ticker. Do not use a fixed category list: name the driver in the terms that
best fit this specific period and event. If the article is not market-moving for
the ticker, say so with low intensity and low confidence.

Return ONLY valid JSON with this exact shape:
{{
  "driver_label": "short open-ended label",
  "driver_summary": "one sentence explaining the driver",
  "directional_pressure": "strong_risk_off|risk_off|mixed|neutral|risk_on|strong_risk_on",
  "intensity": 0.0,
  "novelty": 0.0,
  "confidence": 0.0,
  "time_horizon": "intraday|days|weeks|months|unknown",
  "market_scope": "ticker|sector|broad_market|macro|unknown",
  "evidence": "short quote/paraphrase from the article"
}}"""


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    columns = [
        "article_uid",
        "ticker",
        "anchor_trading_date",
        "title",
        "summary",
        "article_text",
        "relevance_score",
        "split",
    ]
    full = pd.read_parquet(args.input_path)
    df = full[[column for column in columns if column in full.columns]].copy()
    required = {"article_uid", "ticker", "anchor_trading_date", "article_text"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if "title" not in df.columns:
        df["title"] = df["article_text"].fillna("").astype(str).str.split("\n").str[0]
    if "summary" not in df.columns:
        df["summary"] = df["article_text"]
    if "relevance_score" not in df.columns:
        df["relevance_score"] = 0.0
    df["anchor_trading_date"] = pd.to_datetime(df["anchor_trading_date"]).dt.normalize()
    if args.split:
        if "split" not in df.columns:
            raise ValueError("--split was provided, but the input has no split column.")
        df = df[df["split"].isin(args.split)].copy()
    if args.start_date:
        df = df[df["anchor_trading_date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        df = df[df["anchor_trading_date"] <= pd.Timestamp(args.end_date)].copy()
    if args.min_relevance is not None:
        df = df[pd.to_numeric(df["relevance_score"], errors="coerce").fillna(0.0) >= args.min_relevance].copy()
    df = df.drop_duplicates(subset=["article_uid", "ticker", "anchor_trading_date"])
    if args.top_n_per_ticker_day:
        if args.top_n_per_ticker_day < 1:
            raise ValueError("--top-n-per-ticker-day must be at least 1.")
        df = (
            df.assign(_rank_relevance=pd.to_numeric(df["relevance_score"], errors="coerce").fillna(0.0))
            .sort_values(["anchor_trading_date", "ticker", "_rank_relevance"], ascending=[True, True, False])
            .groupby(["ticker", "anchor_trading_date"], group_keys=False)
            .head(args.top_n_per_ticker_day)
            .drop(columns=["_rank_relevance"])
        )
    df = df.sort_values(["anchor_trading_date", "ticker", "article_uid"]).reset_index(drop=True)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    return df


def load_existing(path: Path, force: bool) -> pd.DataFrame | None:
    if force or not path.exists():
        return None
    return pd.read_parquet(path)


def save_output(path: Path, frames: list[pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.concat(frames, ignore_index=True)
    output = output.drop_duplicates(subset=["article_uid", "ticker", "anchor_trading_date"], keep="last")
    output.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(
        json.dumps({"rows": int(len(output)), "columns": list(output.columns)}, indent=2),
        encoding="utf-8",
    )


def normalise_driver(raw: dict, source: pd.Series) -> dict:
    direction = normalise_choice(
        raw.get("directional_pressure"),
        set(DIRECTION_TO_SCORE),
        "neutral",
    )
    horizon = normalise_choice(raw.get("time_horizon"), set(HORIZON_TO_DAYS), "unknown")
    scope = normalise_choice(
        raw.get("market_scope"),
        {"ticker", "sector", "broad_market", "macro", "unknown"},
        "unknown",
    )
    intensity = clamp01(raw.get("intensity"))
    novelty = clamp01(raw.get("novelty"))
    confidence = clamp01(raw.get("confidence"))
    direction_score = DIRECTION_TO_SCORE[direction]
    materiality = intensity * confidence
    shock_score = direction_score * materiality
    return {
        "article_uid": str(source["article_uid"]),
        "ticker": str(source["ticker"]),
        "anchor_trading_date": pd.Timestamp(source["anchor_trading_date"]),
        "driver_label": str(raw.get("driver_label", "no clear driver"))[:200],
        "driver_summary": str(raw.get("driver_summary", ""))[:600],
        "directional_pressure": direction,
        "direction_score": float(direction_score),
        "driver_intensity": float(intensity),
        "driver_novelty": float(novelty),
        "driver_confidence": float(confidence),
        "driver_materiality": float(materiality),
        "driver_shock_score": float(shock_score),
        "time_horizon": horizon,
        "horizon_days": float(HORIZON_TO_DAYS[horizon]),
        "market_scope": scope,
        "evidence": str(raw.get("evidence", ""))[:600],
        "source_relevance_score": clamp01(source.get("relevance_score"), default=0.0),
    }


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(args)
    if rows.empty:
        raise ValueError("No rows selected for contextual driver extraction. Loosen the filters and try again.")
    existing = load_existing(args.output_path, args.force)
    frames: list[pd.DataFrame] = []
    if existing is not None and not existing.empty:
        existing = existing.copy()
        existing["anchor_trading_date"] = pd.to_datetime(existing["anchor_trading_date"]).dt.normalize()
        done_keys = set(
            zip(
                existing["article_uid"].astype(str),
                existing["ticker"].astype(str),
                existing["anchor_trading_date"].astype(str),
            )
        )
        row_keys = list(
            zip(
                rows["article_uid"].astype(str),
                rows["ticker"].astype(str),
                rows["anchor_trading_date"].astype(str),
            )
        )
        rows = rows[[key not in done_keys for key in row_keys]].copy()
        frames.append(existing)
        print(f"Resuming from {len(done_keys):,} existing contextual drivers.")

    if rows.empty:
        print(f"No new rows to process. Output already exists: {args.output_path}")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    generated_rows = []
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        batch = rows.iloc[start:end]
        prompts = [build_prompt(row) for _, row in batch.iterrows()]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for local_idx, (_, source) in enumerate(batch.iterrows()):
            generated = tokenizer.decode(
                output_ids[local_idx, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            parsed = extract_json(generated)
            normalised = normalise_driver(parsed, source)
            normalised["raw_generation"] = generated[:1200]
            generated_rows.append(normalised)

        if generated_rows and len(generated_rows) % args.checkpoint_every == 0:
            frames.append(pd.DataFrame(generated_rows))
            generated_rows = []
            save_output(args.output_path, frames)
            print(f"Processed {end:,}/{len(rows):,} new rows")

    if generated_rows:
        frames.append(pd.DataFrame(generated_rows))
    save_output(args.output_path, frames)
    print(f"Wrote contextual drivers to {args.output_path}")


if __name__ == "__main__":
    main()
