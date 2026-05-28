"""
Create DeepSeek-derived article features for the latent-state model.

This script runs a frozen DeepSeek model over each unique article text and saves
compact numerical features. The daily latent-state dataset can then aggregate
these features by ticker/trading day, so training does not repeatedly run the
large language model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/ltn_all.parquet"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/deepseek_article_features.parquet"),
    )
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=32,
        help="Number of deterministic random projection dimensions to save.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Optional smoke-test limit on unique articles.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output instead of resuming from existing article features.",
    )
    return parser.parse_args()


def load_articles(path: Path, max_articles: int | None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["article_uid", "article_text"])
    df = df.dropna(subset=["article_uid", "article_text"])
    df = df.drop_duplicates(subset=["article_uid"]).sort_values("article_uid").reset_index(drop=True)
    if max_articles:
        df = df.head(max_articles).copy()
    return df


def projection_matrix(hidden_size: int, projection_dim: int) -> np.ndarray:
    rng = np.random.default_rng(20260528)
    matrix = rng.normal(0.0, 1.0 / np.sqrt(projection_dim), size=(hidden_size, projection_dim))
    return matrix.astype(np.float32)


def load_existing(path: Path, force: bool) -> pd.DataFrame | None:
    if force or not path.exists():
        return None
    return pd.read_parquet(path)


def save_checkpoint(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    metadata_path = path.with_suffix(".json")
    metadata = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    articles = load_articles(args.input_path, args.max_articles)
    existing = load_existing(args.output_path, args.force)
    if existing is not None and not existing.empty:
        done = set(existing["article_uid"].astype(str))
        articles = articles[~articles["article_uid"].astype(str).isin(done)].copy()
        outputs = [existing]
        print(f"Resuming from {len(done):,} existing article features.")
    else:
        outputs = []

    if articles.empty:
        print(f"No new articles to process. Output already exists: {args.output_path}")
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

    hidden_size = int(model.config.hidden_size)
    projector = torch.as_tensor(projection_matrix(hidden_size, args.projection_dim), device=device)

    rows = []
    texts = articles["article_text"].fillna("").astype(str).tolist()
    ids = articles["article_uid"].astype(str).tolist()
    for start in range(0, len(texts), args.batch_size):
        end = min(start + args.batch_size, len(texts))
        batch_texts = texts[start:end]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)
            hidden = output.hidden_states[-1]
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=device), lengths].float()
            projected = pooled @ projector

        pooled_np = pooled.detach().cpu().numpy()
        projected_np = projected.detach().cpu().numpy()
        for local_idx, article_uid in enumerate(ids[start:end]):
            row = {
                "article_uid": article_uid,
                "deepseek_embedding_norm": float(np.linalg.norm(pooled_np[local_idx])),
                "deepseek_embedding_mean": float(pooled_np[local_idx].mean()),
                "deepseek_embedding_std": float(pooled_np[local_idx].std()),
            }
            for dim in range(args.projection_dim):
                row[f"deepseek_rp_{dim:02d}"] = float(projected_np[local_idx, dim])
            rows.append(row)

        if len(rows) >= 1000:
            outputs.append(pd.DataFrame(rows))
            rows = []
            save_checkpoint(args.output_path, pd.concat(outputs, ignore_index=True))
            print(f"Processed {min(end, len(texts)):,}/{len(texts):,} new articles")

    if rows:
        outputs.append(pd.DataFrame(rows))
    result = pd.concat(outputs, ignore_index=True)
    result = result.drop_duplicates(subset=["article_uid"], keep="last")
    save_checkpoint(args.output_path, result)
    print(f"Wrote {len(result):,} article features to {args.output_path}")


if __name__ == "__main__":
    main()
