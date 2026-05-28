"""
Train a frozen-DeepSeek baseline for future price trend classification.

The baseline predicts `future_price_trend_id` from:
- DeepSeek article embedding
- structured ticker/price/macro features

This is the comparison point for the LTN-constrained model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from common import (
    available_feature_columns,
    class_weights,
    classification_report,
    fit_feature_stats,
    labels,
    load_split,
    transform_features,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/ltn_train.parquet"))
    parser.add_argument(
        "--validation-path", type=Path, default=Path("data/processed/ltn_validation.parquet")
    )
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/ltn_test.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/deepseek_baseline"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


class TrendDataset:
    def __init__(self, df, features: np.ndarray, y: np.ndarray):
        self.texts = df["article_text"].fillna("").astype(str).tolist()
        self.features = features
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.texts[idx], self.features[idx], int(self.y[idx])


class DeepSeekTrendClassifier:
    def __init__(self, model_id: str, structured_dim: int, dropout: float, device: str | None):
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = int(self.encoder.config.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + structured_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 3),
        ).to(self.device)

    def batch_logits(self, texts, features, max_length: int):
        torch = self.torch
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        features_t = torch.as_tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            outputs = self.encoder(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=self.device), lengths]
            pooled = pooled.float()

        combined = torch.cat([pooled, features_t], dim=1)
        return self.head(combined)


def iterate_batches(dataset: TrendDataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        texts, features, y = zip(*(dataset[idx] for idx in range(start, end)))
        yield texts, np.stack(features), np.array(y, dtype=np.int64)


def evaluate(model, dataset: TrendDataset, max_length: int, batch_size: int) -> dict:
    torch = model.torch
    model.head.eval()
    preds = []
    truth = []
    with torch.no_grad():
        for texts, features, y in iterate_batches(dataset, batch_size):
            logits = model.batch_logits(texts, features, max_length)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            truth.append(y)
    return classification_report(np.concatenate(truth), np.concatenate(preds))


def main() -> None:
    args = parse_args()
    train_df = load_split(args.train_path, args.max_train_rows)
    validation_df = load_split(args.validation_path, args.max_validation_rows)
    test_df = load_split(args.test_path, args.max_test_rows)

    feature_columns = available_feature_columns(train_df)
    feature_stats = fit_feature_stats(train_df, feature_columns)
    train_set = TrendDataset(train_df, transform_features(train_df, feature_columns, feature_stats), labels(train_df))
    validation_set = TrendDataset(
        validation_df, transform_features(validation_df, feature_columns, feature_stats), labels(validation_df)
    )
    test_set = TrendDataset(test_df, transform_features(test_df, feature_columns, feature_stats), labels(test_df))

    model = DeepSeekTrendClassifier(args.model_id, len(feature_columns), args.dropout, args.device)
    torch = model.torch
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate)
    weights = torch.as_tensor(class_weights(train_set.y), dtype=torch.float32, device=model.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.head.train()
        losses = []
        for texts, features, y in iterate_batches(train_set, args.batch_size):
            optimizer.zero_grad()
            logits = model.batch_logits(texts, features, args.max_length)
            target = torch.as_tensor(y, dtype=torch.long, device=model.device)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_report = evaluate(model, validation_set, args.max_length, args.batch_size)
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"val_accuracy={validation_report['accuracy']:.4f} "
            f"val_macro_f1={validation_report['macro_f1']:.4f}"
        )
        if validation_report["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_report["macro_f1"]
            torch.save(model.head.state_dict(), args.output_dir / "best_head.pt")
            write_json(
                args.output_dir / "training_metadata.json",
                {
                    "model_id": args.model_id,
                    "feature_columns": feature_columns,
                    "feature_stats": feature_stats,
                    "best_validation": validation_report,
                },
            )

    model.head.load_state_dict(torch.load(args.output_dir / "best_head.pt", map_location=model.device))
    test_report = evaluate(model, test_set, args.max_length, args.batch_size)
    write_json(args.output_dir / "test_metrics.json", test_report)
    print(f"test_accuracy={test_report['accuracy']:.4f} test_macro_f1={test_report['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
