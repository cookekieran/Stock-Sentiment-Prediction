"""
Train a temporal latent-state market regime model.

The model consumes one daily row per ticker and updates a GRU hidden state across
time. It predicts the future market regime from the latest hidden state, while
optional losses discourage abrupt regime probability changes and encode soft
macro/news transition logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from common import classification_report, write_json


LABEL_NAMES = ["bear_drawdown", "sideways", "bull_rally"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-path", type=Path, default=Path("data/processed/latent_state_daily.parquet"))
    parser.add_argument("--schema-path", type=Path, default=Path("data/processed/latent_state_schema.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/latent_state_gru"))
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--logic-weight", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def read_schema(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Schema not found: {path}. Run src/data/build_latent_state_dataset.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Daily latent-state dataset not found: {path}. "
            "Run src/data/build_latent_state_dataset.py first."
        )
    df = pd.read_parquet(path)
    df["anchor_trading_date"] = pd.to_datetime(df["anchor_trading_date"])
    return df.sort_values(["ticker", "anchor_trading_date"]).reset_index(drop=True)


def fit_feature_stats(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    stats = {}
    for column in feature_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        mean = float(values.mean()) if values.notna().any() else 0.0
        std = float(values.std()) if values.notna().any() else 1.0
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        stats[column] = {"mean": mean, "std": std}
    return stats


def transform_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    arrays = []
    for column in feature_columns:
        values = pd.to_numeric(df[column], errors="coerce").fillna(stats[column]["mean"])
        arrays.append(((values - stats[column]["mean"]) / stats[column]["std"]).to_numpy(np.float32))
    return np.stack(arrays, axis=1).astype(np.float32)


def raw_features(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    arrays = []
    for column in feature_columns:
        arrays.append(pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy(np.float32))
    return np.stack(arrays, axis=1).astype(np.float32)


class DailySequenceDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        stats: dict[str, dict[str, float]],
        sequence_length: int,
    ):
        self.df = df.sort_values(["ticker", "anchor_trading_date"]).reset_index(drop=True)
        self.feature_columns = feature_columns
        self.x = transform_features(self.df, feature_columns, stats)
        self.raw = raw_features(self.df, feature_columns)
        self.y = pd.to_numeric(self.df["future_price_trend_id"], errors="raise").to_numpy(np.int64)
        self.indices: list[tuple[int, int]] = []

        for _, group in self.df.groupby("ticker", sort=False):
            positions = group.index.to_numpy()
            if len(positions) < sequence_length:
                continue
            for end_offset in range(sequence_length, len(positions) + 1):
                window = positions[end_offset - sequence_length : end_offset]
                self.indices.append((int(window[0]), int(window[-1]) + 1))

        if not self.indices:
            raise ValueError(
                f"No sequences created. sequence_length={sequence_length} is too large for this split."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start, end = self.indices[idx]
        return self.x[start:end], self.raw[start:end], self.y[end - 1]


def sequence_labels(dataset: DailySequenceDataset) -> np.ndarray:
    return np.array([dataset.y[end - 1] for _, end in dataset.indices], dtype=np.int64)


def make_loader(dataset: DailySequenceDataset, batch_size: int, balanced: bool):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    labels = sequence_labels(dataset)
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


class LatentStateGRU:
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str | None,
    ):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.model = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        ).to(self.device)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        ).to(self.device)

    def logits(self, x):
        hidden_sequence, _ = self.model(x)
        sequence_logits = self.head(hidden_sequence)
        final_logits = sequence_logits[:, -1, :]
        return final_logits, sequence_logits

    def parameters(self):
        yield from self.model.parameters()
        yield from self.head.parameters()


def feature_index(feature_columns: list[str], name: str) -> int | None:
    try:
        return feature_columns.index(name)
    except ValueError:
        return None


def get_raw(raw, feature_columns: list[str], name: str):
    import torch

    idx = feature_index(feature_columns, name)
    if idx is None:
        return torch.zeros(raw.shape[:2], dtype=raw.dtype, device=raw.device)
    return raw[:, :, idx]


def fuzzy_implication_loss(antecedent, consequent):
    truth = 1.0 - antecedent + antecedent * consequent
    return (1.0 - truth.clamp(0.0, 1.0)).mean()


def logic_loss(sequence_logits, raw, feature_columns: list[str]):
    probs = sequence_logits.softmax(dim=-1)
    bear = probs[:, :, 0]
    sideways = probs[:, :, 1]
    bull = probs[:, :, 2]

    sentiment = get_raw(raw, feature_columns, "news_relevance_weighted_sentiment").clamp(-1.0, 1.0)
    positive_news = sentiment.clamp(min=0.0)
    negative_news = (-sentiment).clamp(min=0.0)
    relevant_news = get_raw(raw, feature_columns, "news_high_relevance_rate").clamp(0.0, 1.0)
    high_inflation = get_raw(raw, feature_columns, "macro_high_inflation").clamp(0.0, 1.0)
    rising_rates = get_raw(raw, feature_columns, "macro_rising_rates").clamp(0.0, 1.0)
    tightening = get_raw(raw, feature_columns, "macro_tightening_regime").clamp(0.0, 1.0)
    easing = get_raw(raw, feature_columns, "macro_easing_regime").clamp(0.0, 1.0)
    falling_rates = get_raw(raw, feature_columns, "macro_falling_rates").clamp(0.0, 1.0)
    inverted_curve = get_raw(raw, feature_columns, "macro_inverted_yield_curve").clamp(0.0, 1.0)

    losses = [
        fuzzy_implication_loss(relevant_news * negative_news * tightening, bear),
        fuzzy_implication_loss(relevant_news * positive_news * easing, bull),
        fuzzy_implication_loss(high_inflation * rising_rates, 1.0 - bull),
        fuzzy_implication_loss(inverted_curve * negative_news, bear + sideways),
        fuzzy_implication_loss(falling_rates * easing * positive_news, 1.0 - bear),
    ]
    return sum(losses) / len(losses)


def smoothness_loss(sequence_logits, raw, feature_columns: list[str]):
    if sequence_logits.shape[1] < 2:
        return sequence_logits.new_tensor(0.0)
    probs = sequence_logits.softmax(dim=-1)
    diffs = (probs[:, 1:, :] - probs[:, :-1, :]).pow(2).sum(dim=-1)

    sentiment = get_raw(raw, feature_columns, "news_relevance_weighted_sentiment").clamp(-1.0, 1.0)
    relevance = get_raw(raw, feature_columns, "news_high_relevance_rate").clamp(0.0, 1.0)
    volume = get_raw(raw, feature_columns, "news_log_article_count")
    shock = (
        (sentiment[:, 1:] - sentiment[:, :-1]).abs()
        + 0.5 * (relevance[:, 1:] - relevance[:, :-1]).abs()
        + 0.05 * volume[:, 1:].clamp(min=0.0)
    )
    calm_weight = (-shock).exp().detach()
    return (diffs * calm_weight).mean()


def evaluate(model: LatentStateGRU, dataset: DailySequenceDataset, batch_size: int) -> dict:
    import torch
    from torch.utils.data import DataLoader

    model.model.eval()
    model.head.eval()
    preds = []
    truth = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, _, y in loader:
            x = x.to(model.device, dtype=torch.float32)
            logits, _ = model.logits(x)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            truth.append(y.numpy())
    return classification_report(np.concatenate(truth), np.concatenate(preds))


def main() -> None:
    args = parse_args()
    schema = read_schema(args.schema_path)
    daily = load_daily(args.daily_path)
    feature_columns = [column for column in schema["feature_columns"] if column in daily.columns]

    train_df = daily[daily["split"] == "train"].copy()
    validation_df = daily[daily["split"] == "validation"].copy()
    test_df = daily[daily["split"] == "test"].copy()
    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            f"Expected non-empty train/validation/test splits; got "
            f"{len(train_df)}, {len(validation_df)}, {len(test_df)}."
        )

    stats = fit_feature_stats(train_df, feature_columns)
    train_set = DailySequenceDataset(train_df, feature_columns, stats, args.sequence_length)
    validation_set = DailySequenceDataset(validation_df, feature_columns, stats, args.sequence_length)
    test_set = DailySequenceDataset(test_df, feature_columns, stats, args.sequence_length)

    model = LatentStateGRU(
        input_dim=len(feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device,
    )
    torch = model.torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = make_loader(train_set, args.batch_size, balanced=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.model.train()
        model.head.train()
        losses = []
        task_losses = []
        smooth_losses = []
        logic_losses = []

        for x, raw, y in train_loader:
            x = x.to(model.device, dtype=torch.float32)
            raw = raw.to(model.device, dtype=torch.float32)
            y = y.to(model.device, dtype=torch.long)

            optimizer.zero_grad()
            final_logits, sequence_logits = model.logits(x)
            task = criterion(final_logits, y)
            smooth = smoothness_loss(sequence_logits, raw, feature_columns)
            logic = logic_loss(sequence_logits, raw, feature_columns)
            loss = task + args.smoothness_weight * smooth + args.logic_weight * logic
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            task_losses.append(float(task.detach().cpu()))
            smooth_losses.append(float(smooth.detach().cpu()))
            logic_losses.append(float(logic.detach().cpu()))

        validation_report = evaluate(model, validation_set, args.batch_size)
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"task={np.mean(task_losses):.4f} smooth={np.mean(smooth_losses):.4f} "
            f"logic={np.mean(logic_losses):.4f} "
            f"val_accuracy={validation_report['accuracy']:.4f} "
            f"val_macro_f1={validation_report['macro_f1']:.4f}"
        )
        if validation_report["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_report["macro_f1"]
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
                "validation": validation_report,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    model.model.load_state_dict(best_state["model"])
    model.head.load_state_dict(best_state["head"])
    torch.save(
        {"model": best_state["model"], "head": best_state["head"]},
        args.output_dir / "best_latent_state_gru.pt",
    )

    test_report = evaluate(model, test_set, args.batch_size)
    write_json(
        args.output_dir / "training_metadata.json",
        {
            "feature_columns": feature_columns,
            "feature_stats": stats,
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "smoothness_weight": args.smoothness_weight,
            "logic_weight": args.logic_weight,
            "best_validation": best_state["validation"],
            "model_description": (
                "Daily ticker-level GRU latent state updated by aggregated news, "
                "macro features, and known market state."
            ),
        },
    )
    write_json(args.output_dir / "test_metrics.json", test_report)
    print(f"test_accuracy={test_report['accuracy']:.4f} test_macro_f1={test_report['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
