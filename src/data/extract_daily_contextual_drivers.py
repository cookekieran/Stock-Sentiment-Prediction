"""
Extract ticker/day contextual market driver predicates with DeepSeek.

This is the dissertation-facing DeepSeek interpreter. It groups all available
news for a ticker and trading day into one packet, asks DeepSeek to interpret the
daily context, and writes one structured predicate row per ticker/day. The output
is intended to feed the daily latent-state GRU and LTN-style logic losses.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from contextual_macro_channels import map_macro_channels


DIRECTION_TO_SCORE = {
    "risk_off": -1.0,
    "mixed": 0.0,
    "neutral": 0.0,
    "risk_on": 1.0,
}

HORIZON_TO_DAYS = {
    "days": 3.0,
    "weeks": 20.0,
    "months": 60.0,
    "unknown": 0.0,
}

SCOPES = ["ticker", "sector", "broad_market", "macro", "irrelevant"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/ltn_all.parquet"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/daily_contextual_driver_predicates.parquet"),
    )
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--max-articles-per-day", type=int, default=25)
    parser.add_argument("--max-article-chars", type=int, default=700)
    parser.add_argument("--ticker", action="append", help="Restrict to one or more tickers. Repeat for multiple.")
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "validation", "test"],
        help="Restrict extraction to one or more dataset splits. Repeat for multiple.",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
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


def normalise_scopes(value) -> list[str]:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    scopes = [scope for scope in text.split("|") if scope in SCOPES]
    return list(dict.fromkeys(scopes)) or ["irrelevant"]


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def load_daily_packets(args: argparse.Namespace) -> pd.DataFrame:
    columns = [
        "article_uid",
        "ticker",
        "anchor_trading_date",
        "time_published",
        "title",
        "summary",
        "article_text",
        "split",
        "price_trend_label",
        "future_price_trend_label",
    ]
    full = pd.read_parquet(args.input_path)
    df = full[[column for column in columns if column in full.columns]].copy()
    required = {"article_uid", "ticker", "anchor_trading_date", "article_text"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df["anchor_trading_date"] = pd.to_datetime(df["anchor_trading_date"]).dt.normalize()
    df["time_published"] = pd.to_datetime(df.get("time_published"), errors="coerce")
    if "title" not in df.columns:
        df["title"] = df["article_text"].fillna("").astype(str).str.split("\n").str[0]
    if "summary" not in df.columns:
        df["summary"] = df["article_text"]

    if args.ticker:
        df = df[df["ticker"].isin(args.ticker)].copy()
    if args.split:
        if "split" not in df.columns:
            raise ValueError("--split was provided, but the input has no split column.")
        df = df[df["split"].isin(args.split)].copy()
    if args.start_date:
        df = df[df["anchor_trading_date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        df = df[df["anchor_trading_date"] <= pd.Timestamp(args.end_date)].copy()
    if df.empty:
        raise ValueError("No article rows selected for daily DeepSeek driver extraction.")

    packets = []
    grouped = df.sort_values(["ticker", "anchor_trading_date", "time_published", "article_uid"]).groupby(
        ["ticker", "anchor_trading_date"],
        sort=True,
    )
    for (ticker, date), group in grouped:
        group = group.head(args.max_articles_per_day)
        packets.append(
            {
                "ticker": ticker,
                "anchor_trading_date": date,
                "article_count": int(len(group)),
                "articles": group.to_dict("records"),
                "current_regime": str(group.get("price_trend_label", pd.Series(["unknown"])).iloc[0]),
                "future_regime": str(group.get("future_price_trend_label", pd.Series(["unknown"])).iloc[0]),
            }
        )
    packets_df = pd.DataFrame(packets)
    if args.max_days:
        packets_df = packets_df.head(args.max_days).copy()
    return packets_df.reset_index(drop=True)


def format_article(row: dict, idx: int, max_chars: int) -> str:
    title = str(row.get("title", "") or "").strip()
    summary = str(row.get("summary", "") or "").strip()
    text = str(row.get("article_text", "") or "").strip()
    body = summary or text
    body = body[:max_chars]
    return f"{idx}. Title: {title}\n   Text: {body}"


def build_prompt(packet: pd.Series, max_article_chars: int) -> str:
    ticker = str(packet["ticker"])
    date = pd.Timestamp(packet["anchor_trading_date"]).date().isoformat()
    articles = packet["articles"]
    article_block = "\n".join(
        format_article(row, idx, max_article_chars)
        for idx, row in enumerate(articles, start=1)
    )
    return f"""You are DeepSeek acting as a financial market-state interpreter.

Task:
Read all news items for this ticker and trading day as one daily news packet.
Do not classify each article independently. Infer the dominant daily market
narrative, the contextual market drivers, and how strongly this daily context
should update a persistent hidden market regime state.

Ticker: {ticker}
Trading day: {date}
Current ticker regime: {packet.get("current_regime", "unknown")}
News items:
{article_block}

Return ONLY valid JSON with this exact shape:
{{
  "daily_market_narrative": "short daily narrative",
  "dominant_drivers": [
    {{
      "driver_label": "open-ended label",
      "driver_summary": "short explanation",
      "scope": "ticker|sector|broad_market|macro|irrelevant",
      "directional_pressure": "risk_on|risk_off|mixed|neutral",
      "intensity": 0.0,
      "novelty": 0.0,
      "confidence": 0.0,
      "time_horizon": "days|weeks|months|unknown",
      "relevant_channels": [
        {{
          "channel": "open-ended channel name",
          "importance": 0.0,
          "available_variable": "optional mapped variable or empty string"
        }}
      ]
    }}
  ],
  "net_pressure": "risk_on|risk_off|mixed|neutral",
  "update_strength": 0.0,
  "uncertainty": 0.0
}}"""


def load_existing(path: Path, force: bool) -> pd.DataFrame | None:
    if force or not path.exists():
        return None
    return pd.read_parquet(path)


def save_output(path: Path, frames: list[pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.concat(frames, ignore_index=True)
    output = output.drop_duplicates(subset=["ticker", "anchor_trading_date"], keep="last")
    output.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(
        json.dumps({"rows": int(len(output)), "columns": list(output.columns)}, indent=2),
        encoding="utf-8",
    )


def summarise_channels(drivers: list[dict]) -> tuple[list[dict], dict[str, float]]:
    channels = []
    price_variable_importance = 0.0
    for driver in drivers:
        for raw_channel in driver.get("relevant_channels", []) or []:
            if not isinstance(raw_channel, dict):
                continue
            importance = clamp01(raw_channel.get("importance"))
            variable = str(raw_channel.get("available_variable", "") or "").strip()
            channels.append(
                {
                    "channel": str(raw_channel.get("channel", "") or "")[:160],
                    "importance": importance,
                    "available_variable": variable[:120],
                }
            )
            if variable in {
                "anchor_close",
                "volume",
                "realized_volatility_20d",
                "rally_from_previous_low",
                "drawdown_from_previous_high",
                "price_trend_id",
            }:
                price_variable_importance = max(price_variable_importance, importance)
    if channels:
        mean_importance = float(np.mean([channel["importance"] for channel in channels]))
    else:
        mean_importance = 0.0
    macro_features, macro_matches = map_macro_channels(drivers)
    return channels, {
        "driver_channel_count": float(len(channels)),
        "driver_mean_channel_importance": mean_importance,
        "driver_price_variable_channel_importance": float(price_variable_importance),
        "driver_macro_channel_matches_json": json.dumps(macro_matches),
        **macro_features,
    }


def normalise_daily_predicates(raw: dict, packet: pd.Series) -> dict:
    drivers = raw.get("dominant_drivers", [])
    if not isinstance(drivers, list):
        drivers = []
    drivers = [driver for driver in drivers if isinstance(driver, dict)]
    update_strength = clamp01(raw.get("update_strength"))
    uncertainty = clamp01(raw.get("uncertainty"))
    net_pressure = normalise_choice(raw.get("net_pressure"), set(DIRECTION_TO_SCORE), "neutral")
    net_pressure_score = DIRECTION_TO_SCORE[net_pressure]

    normalised_drivers = []
    for driver in drivers:
        scopes = normalise_scopes(driver.get("scope"))
        pressure = normalise_choice(driver.get("directional_pressure"), set(DIRECTION_TO_SCORE), "neutral")
        intensity = clamp01(driver.get("intensity"))
        novelty = clamp01(driver.get("novelty"))
        confidence = clamp01(driver.get("confidence"))
        materiality = intensity * confidence
        shock = DIRECTION_TO_SCORE[pressure] * materiality * update_strength
        normalised_drivers.append(
            {
                "driver_label": str(driver.get("driver_label", "no clear driver"))[:200],
                "driver_summary": str(driver.get("driver_summary", ""))[:600],
                "scope": "|".join(scopes),
                "scopes": scopes,
                "directional_pressure": pressure,
                "direction_score": float(DIRECTION_TO_SCORE[pressure]),
                "intensity": float(intensity),
                "novelty": float(novelty),
                "confidence": float(confidence),
                "materiality": float(materiality),
                "shock": float(shock),
                "time_horizon": normalise_choice(driver.get("time_horizon"), set(HORIZON_TO_DAYS), "unknown"),
                "relevant_channels": driver.get("relevant_channels", []) or [],
            }
        )

    channels, channel_features = summarise_channels(normalised_drivers)
    count = max(len(normalised_drivers), 1)
    direction = np.array([driver["direction_score"] for driver in normalised_drivers], dtype=float)
    intensity = np.array([driver["intensity"] for driver in normalised_drivers], dtype=float)
    novelty = np.array([driver["novelty"] for driver in normalised_drivers], dtype=float)
    confidence = np.array([driver["confidence"] for driver in normalised_drivers], dtype=float)
    materiality = np.array([driver["materiality"] for driver in normalised_drivers], dtype=float)
    shock = np.array([driver["shock"] for driver in normalised_drivers], dtype=float)
    horizon = np.array([HORIZON_TO_DAYS[driver["time_horizon"]] for driver in normalised_drivers], dtype=float)
    scopes = [scope for driver in normalised_drivers for scope in driver["scopes"]]

    out = {
        "ticker": str(packet["ticker"]),
        "anchor_trading_date": pd.Timestamp(packet["anchor_trading_date"]),
        "daily_market_narrative": str(raw.get("daily_market_narrative", ""))[:1000],
        "net_pressure": net_pressure,
        "net_pressure_score": float(net_pressure_score),
        "driver_update_strength": float(update_strength),
        "driver_uncertainty": float(uncertainty),
        "driver_count": float(len(normalised_drivers)),
        "driver_mean_direction_score": float(direction.mean()) if len(direction) else 0.0,
        "driver_relevance_weighted_direction_score": float(net_pressure_score * update_strength),
        "driver_mean_intensity": float(intensity.mean()) if len(intensity) else 0.0,
        "driver_max_intensity": float(intensity.max()) if len(intensity) else 0.0,
        "driver_mean_novelty": float(novelty.mean()) if len(novelty) else 0.0,
        "driver_max_novelty": float(novelty.max()) if len(novelty) else 0.0,
        "driver_mean_confidence": float(confidence.mean()) if len(confidence) else 0.0,
        "driver_mean_materiality": float(materiality.mean()) if len(materiality) else 0.0,
        "driver_max_materiality": float(materiality.max()) if len(materiality) else 0.0,
        "driver_mean_shock_score": float(shock.mean()) if len(shock) else 0.0,
        "driver_abs_shock_score": float(np.abs(shock).mean()) if len(shock) else 0.0,
        "driver_max_risk_on_shock": float(np.clip(shock, 0.0, None).max()) if len(shock) else 0.0,
        "driver_max_risk_off_shock": float(np.clip(-shock, 0.0, None).max()) if len(shock) else 0.0,
        "driver_mean_horizon_days": float(horizon.mean()) if len(horizon) else 0.0,
        "relevant_channels_json": json.dumps(channels),
        "dominant_drivers_json": json.dumps(normalised_drivers),
    }
    for scope in SCOPES:
        out[f"driver_{scope}_scope_share"] = float(scopes.count(scope) / count)
    out.update(channel_features)

    ranked = sorted(normalised_drivers, key=lambda driver: driver["materiality"], reverse=True)[:3]
    for idx, driver in enumerate(ranked, start=1):
        out[f"top_driver_{idx}_label"] = driver["driver_label"]
        out[f"top_driver_{idx}_summary"] = driver["driver_summary"]
        out[f"top_driver_{idx}_pressure"] = driver["directional_pressure"]
        out[f"top_driver_{idx}_scope"] = driver["scope"]
    return out


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    packets = load_daily_packets(args)
    existing = load_existing(args.output_path, args.force)
    frames: list[pd.DataFrame] = []
    if existing is not None and not existing.empty:
        existing = existing.copy()
        existing["anchor_trading_date"] = pd.to_datetime(existing["anchor_trading_date"]).dt.normalize()
        done_keys = set(zip(existing["ticker"].astype(str), existing["anchor_trading_date"].astype(str)))
        packet_keys = list(zip(packets["ticker"].astype(str), packets["anchor_trading_date"].astype(str)))
        packets = packets[[key not in done_keys for key in packet_keys]].copy()
        frames.append(existing)
        print(f"Resuming from {len(done_keys):,} existing daily driver rows.")

    if packets.empty:
        print(f"No new ticker/day packets to process. Output already exists: {args.output_path}")
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
    for start in range(0, len(packets), args.batch_size):
        end = min(start + args.batch_size, len(packets))
        batch = packets.iloc[start:end]
        prompts = [build_prompt(row, args.max_article_chars) for _, row in batch.iterrows()]
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
        for local_idx, (_, packet) in enumerate(batch.iterrows()):
            generated = tokenizer.decode(
                output_ids[local_idx, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            parsed = extract_json(generated)
            row = normalise_daily_predicates(parsed, packet)
            row["raw_generation"] = generated[:2000]
            generated_rows.append(row)

        if generated_rows and len(generated_rows) % args.checkpoint_every == 0:
            frames.append(pd.DataFrame(generated_rows))
            generated_rows = []
            save_output(args.output_path, frames)
            print(f"Processed {end:,}/{len(packets):,} new ticker/day packets")

    if generated_rows:
        frames.append(pd.DataFrame(generated_rows))
    save_output(args.output_path, frames)
    print(f"Wrote daily contextual driver predicates to {args.output_path}")


if __name__ == "__main__":
    main()
