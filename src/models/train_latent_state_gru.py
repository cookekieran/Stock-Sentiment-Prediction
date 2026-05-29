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
    parser.add_argument(
        "--persistence-prior-strength",
        type=float,
        default=0.0,
        help=(
            "Add a logit prior toward the current observed price_trend_id. "
            "This makes the hidden state persistent by default and lets the GRU learn deviations."
        ),
    )
    parser.add_argument(
        "--sampler",
        choices=["natural", "balanced"],
        default="natural",
        help="Use chronological/natural class frequency or balanced replacement sampling.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "inverse", "sqrt_inverse"],
        default="sqrt_inverse",
        help="Class weighting for cross-entropy on final regime labels.",
    )
    parser.add_argument(
        "--diagnostics-every",
        type=int,
        default=1,
        help="Print validation prediction distribution every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save validation/test prediction probabilities for later explanations.",
    )
    parser.add_argument(
        "--require-contextual-drivers",
        action="store_true",
        help="Fail unless contextual driver features are present and non-empty.",
    )
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


def evaluate_label_counts(dataset: DailySequenceDataset) -> dict[str, int]:
    labels = sequence_labels(dataset)
    return {LABEL_NAMES[idx]: int((labels == idx).sum()) for idx in range(len(LABEL_NAMES))}


def make_loader(dataset: DailySequenceDataset, batch_size: int, sampler: str):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if sampler == "natural":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


def class_weight_tensor(labels: np.ndarray, mode: str, device):
    import torch

    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    counts[counts == 0.0] = 1.0
    if mode == "inverse":
        weights = counts.sum() / (len(counts) * counts)
    elif mode == "sqrt_inverse":
        weights = np.sqrt(counts.sum() / (len(counts) * counts))
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


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

    if feature_index(feature_columns, "driver_mean_shock_score") is not None:
        shock = get_raw(raw, feature_columns, "driver_mean_shock_score").clamp(-1.0, 1.0)
        positive_news = get_raw(raw, feature_columns, "driver_max_risk_on_shock").clamp(0.0, 1.0)
        negative_news = get_raw(raw, feature_columns, "driver_max_risk_off_shock").clamp(0.0, 1.0)
        materiality = get_raw(raw, feature_columns, "driver_mean_materiality").clamp(0.0, 1.0)
        novelty = get_raw(raw, feature_columns, "driver_mean_novelty").clamp(0.0, 1.0)
        confidence = get_raw(raw, feature_columns, "driver_mean_confidence").clamp(0.0, 1.0)
        update_strength = get_raw(raw, feature_columns, "driver_update_strength").clamp(0.0, 1.0)
        uncertainty = get_raw(raw, feature_columns, "driver_uncertainty").clamp(0.0, 1.0)
        irrelevant_scope = get_raw(raw, feature_columns, "driver_irrelevant_scope_share").clamp(0.0, 1.0)
        macro_channel = get_raw(raw, feature_columns, "driver_macro_variable_channel_importance").clamp(0.0, 1.0)
        relevant_news = (materiality * confidence).clamp(0.0, 1.0)
        conflicting_news = (positive_news * negative_news).sqrt().clamp(0.0, 1.0)
        if probs.shape[1] > 1:
            prob_shift = (probs[:, 1:, :] - probs[:, :-1, :]).abs().sum(dim=-1).clamp(0.0, 1.0)
            calm_or_irrelevant = ((1.0 - update_strength[:, 1:]) * (1.0 - novelty[:, 1:]) + irrelevant_scope[:, 1:]).clamp(0.0, 1.0)
            persistent_risk_off = (negative_news[:, 1:] * negative_news[:, :-1] * confidence[:, 1:]).clamp(0.0, 1.0)
            persistent_risk_on = (positive_news[:, 1:] * positive_news[:, :-1] * confidence[:, 1:]).clamp(0.0, 1.0)
        else:
            prob_shift = sequence_logits.new_zeros(sequence_logits.shape[:2])
            calm_or_irrelevant = sequence_logits.new_zeros(sequence_logits.shape[:2])
            persistent_risk_off = sequence_logits.new_zeros(sequence_logits.shape[:2])
            persistent_risk_on = sequence_logits.new_zeros(sequence_logits.shape[:2])
    else:
        sentiment = get_raw(raw, feature_columns, "news_relevance_weighted_sentiment").clamp(-1.0, 1.0)
        shock = sentiment
        positive_news = sentiment.clamp(min=0.0)
        negative_news = (-sentiment).clamp(min=0.0)
        relevant_news = get_raw(raw, feature_columns, "news_high_relevance_rate").clamp(0.0, 1.0)
        novelty = relevant_news
        uncertainty = sequence_logits.new_zeros(sequence_logits.shape[:2])
        macro_channel = sequence_logits.new_ones(sequence_logits.shape[:2])
        conflicting_news = sequence_logits.new_zeros(sequence_logits.shape[:2])
        prob_shift = sequence_logits.new_zeros(sequence_logits.shape[:2])
        calm_or_irrelevant = sequence_logits.new_zeros(sequence_logits.shape[:2])
        persistent_risk_off = sequence_logits.new_zeros(sequence_logits.shape[:2])
        persistent_risk_on = sequence_logits.new_zeros(sequence_logits.shape[:2])
    high_inflation = get_raw(raw, feature_columns, "macro_high_inflation").clamp(0.0, 1.0)
    rising_rates = get_raw(raw, feature_columns, "macro_rising_rates").clamp(0.0, 1.0)
    tightening = get_raw(raw, feature_columns, "macro_tightening_regime").clamp(0.0, 1.0)
    easing = get_raw(raw, feature_columns, "macro_easing_regime").clamp(0.0, 1.0)
    falling_rates = get_raw(raw, feature_columns, "macro_falling_rates").clamp(0.0, 1.0)
    inverted_curve = get_raw(raw, feature_columns, "macro_inverted_yield_curve").clamp(0.0, 1.0)

    losses = [
        fuzzy_implication_loss(relevant_news * negative_news * (0.5 + 0.5 * novelty), 1.0 - bull),
        fuzzy_implication_loss(relevant_news * negative_news * macro_channel * tightening, bear + sideways),
        fuzzy_implication_loss(relevant_news * positive_news * macro_channel * easing, bull),
        fuzzy_implication_loss(macro_channel * high_inflation * rising_rates, 1.0 - bull),
        fuzzy_implication_loss(macro_channel * inverted_curve * negative_news, bear + sideways),
        fuzzy_implication_loss(macro_channel * falling_rates * easing * positive_news, 1.0 - bear),
        fuzzy_implication_loss((conflicting_news + uncertainty).clamp(0.0, 1.0), sideways),
    ]
    if probs.shape[1] > 1:
        losses.extend(
            [
                fuzzy_implication_loss(calm_or_irrelevant, 1.0 - prob_shift),
                fuzzy_implication_loss(persistent_risk_off, 1.0 - bull[:, 1:]),
                fuzzy_implication_loss(persistent_risk_on, 1.0 - bear[:, 1:]),
            ]
        )
    return sum(losses) / len(losses)


def smoothness_loss(sequence_logits, raw, feature_columns: list[str]):
    if sequence_logits.shape[1] < 2:
        return sequence_logits.new_tensor(0.0)
    probs = sequence_logits.softmax(dim=-1)
    diffs = (probs[:, 1:, :] - probs[:, :-1, :]).pow(2).sum(dim=-1)

    if feature_index(feature_columns, "driver_mean_shock_score") is not None:
        sentiment = get_raw(raw, feature_columns, "driver_mean_shock_score").clamp(-1.0, 1.0)
        relevance = get_raw(raw, feature_columns, "driver_mean_materiality").clamp(0.0, 1.0)
    else:
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


def add_persistence_prior(logits, raw, feature_columns: list[str], strength: float):
    if strength <= 0.0:
        return logits
    idx = feature_index(feature_columns, "price_trend_id")
    if idx is None:
        return logits

    import torch

    trend = raw[:, :, idx].round().long().clamp(0, len(LABEL_NAMES) - 1)
    prior = torch.nn.functional.one_hot(trend, num_classes=len(LABEL_NAMES)).to(
        dtype=logits.dtype,
        device=logits.device,
    )
    if logits.ndim == 2:
        prior = prior[:, -1, :]
    return logits + strength * prior


def evaluate(
    model: LatentStateGRU,
    dataset: DailySequenceDataset,
    batch_size: int,
    persistence_prior_strength: float = 0.0,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    model.model.eval()
    model.head.eval()
    preds = []
    truth = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, raw, y in loader:
            x = x.to(model.device, dtype=torch.float32)
            raw = raw.to(model.device, dtype=torch.float32)
            logits, _ = model.logits(x)
            logits = add_persistence_prior(
                logits,
                raw,
                dataset.feature_columns,
                persistence_prior_strength,
            )
            preds.append(logits.argmax(dim=1).cpu().numpy())
            truth.append(y.numpy())
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(preds)
    report = classification_report(y_true, y_pred)
    report["prediction_counts"] = {
        LABEL_NAMES[idx]: int((y_pred == idx).sum()) for idx in range(len(LABEL_NAMES))
    }
    report["truth_counts"] = {
        LABEL_NAMES[idx]: int((y_true == idx).sum()) for idx in range(len(LABEL_NAMES))
    }
    return report


def collect_predictions(
    model: LatentStateGRU,
    dataset: DailySequenceDataset,
    batch_size: int,
    persistence_prior_strength: float = 0.0,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    model.model.eval()
    model.head.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    cursor = 0
    with torch.no_grad():
        for x, raw, y in loader:
            x = x.to(model.device, dtype=torch.float32)
            raw = raw.to(model.device, dtype=torch.float32)
            logits, sequence_logits = model.logits(x)
            logits = add_persistence_prior(
                logits,
                raw,
                dataset.feature_columns,
                persistence_prior_strength,
            )
            sequence_logits = add_persistence_prior(
                sequence_logits,
                raw,
                dataset.feature_columns,
                persistence_prior_strength,
            )
            probs = logits.softmax(dim=1).cpu().numpy()
            sequence_probs = sequence_logits.softmax(dim=2).cpu().numpy()
            preds = probs.argmax(axis=1)
            for batch_idx in range(len(y)):
                start, end = dataset.indices[cursor]
                final = dataset.df.iloc[end - 1]
                first = dataset.df.iloc[start]
                row = {
                    "ticker": final["ticker"],
                    "anchor_trading_date": final["anchor_trading_date"],
                    "sequence_start_date": first["anchor_trading_date"],
                    "true_label_id": int(y[batch_idx]),
                    "true_label": LABEL_NAMES[int(y[batch_idx])],
                    "pred_label_id": int(preds[batch_idx]),
                    "pred_label": LABEL_NAMES[int(preds[batch_idx])],
                    "prob_bear_drawdown": float(probs[batch_idx, 0]),
                    "prob_sideways": float(probs[batch_idx, 1]),
                    "prob_bull_rally": float(probs[batch_idx, 2]),
                    "start_prob_bear_drawdown": float(sequence_probs[batch_idx, 0, 0]),
                    "start_prob_sideways": float(sequence_probs[batch_idx, 0, 1]),
                    "start_prob_bull_rally": float(sequence_probs[batch_idx, 0, 2]),
                    "current_price_trend_label": final.get("price_trend_label", ""),
                    "future_price_trend_label": final.get("future_price_trend_label", ""),
                }
                for column in dataset.df.columns:
                    if (
                        column.startswith("top_driver_")
                        or column.startswith("driver_")
                        or column in {"daily_market_narrative", "net_pressure", "relevant_channels_json"}
                    ):
                        value = final.get(column)
                        if pd.api.types.is_scalar(value):
                            row[column] = value
                rows.append(row)
                cursor += 1
    return pd.DataFrame(rows)


def majority_baseline_report(dataset: DailySequenceDataset, majority_class: int) -> dict:
    y_true = sequence_labels(dataset)
    y_pred = np.full_like(y_true, fill_value=majority_class)
    report = classification_report(y_true, y_pred)
    report["prediction_counts"] = {
        LABEL_NAMES[idx]: int((y_pred == idx).sum()) for idx in range(len(LABEL_NAMES))
    }
    report["truth_counts"] = {
        LABEL_NAMES[idx]: int((y_true == idx).sum()) for idx in range(len(LABEL_NAMES))
    }
    return report


def current_regime_baseline_report(dataset: DailySequenceDataset) -> dict | None:
    idx = feature_index(dataset.feature_columns, "price_trend_id")
    if idx is None:
        return None
    y_true = sequence_labels(dataset)
    y_pred = []
    for _, end in dataset.indices:
        y_pred.append(int(round(float(dataset.raw[end - 1, idx]))))
    y_pred = np.asarray(y_pred, dtype=np.int64).clip(0, len(LABEL_NAMES) - 1)
    report = classification_report(y_true, y_pred)
    report["prediction_counts"] = {
        LABEL_NAMES[idx]: int((y_pred == idx).sum()) for idx in range(len(LABEL_NAMES))
    }
    report["truth_counts"] = {
        LABEL_NAMES[idx]: int((y_true == idx).sum()) for idx in range(len(LABEL_NAMES))
    }
    return report


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{label}={counts.get(label, 0)}" for label in LABEL_NAMES)


def print_report_summary(name: str, report: dict) -> None:
    print(f"{name}_accuracy={report['accuracy']:.4f} {name}_macro_f1={report['macro_f1']:.4f}")
    if "truth_counts" in report and "prediction_counts" in report:
        print(f"  {name} truth: {format_counts(report['truth_counts'])}")
        print(f"  {name} pred : {format_counts(report['prediction_counts'])}")


def validate_contextual_driver_inputs(daily: pd.DataFrame, feature_columns: list[str]) -> None:
    required_features = {
        "driver_mean_shock_score",
        "driver_mean_materiality",
        "driver_mean_confidence",
        "driver_max_risk_on_shock",
        "driver_max_risk_off_shock",
    }
    missing_features = sorted(required_features.difference(feature_columns))
    missing_text = "top_driver_1_label" not in daily.columns
    if missing_features or missing_text:
        details = []
        if missing_features:
            details.append(f"missing driver feature columns: {missing_features}")
        if missing_text:
            details.append("missing top_driver_1_label explanation text")
        raise ValueError(
            "Contextual driver inputs are required for this explainable run, but "
            + "; ".join(details)
            + ". Rebuild the daily dataset with --contextual-drivers-path and "
            "--require-contextual-drivers."
        )
    driver_count = pd.to_numeric(daily.get("driver_count"), errors="coerce").fillna(0.0)
    if float(driver_count.sum()) <= 0.0:
        raise ValueError("Contextual driver columns exist, but driver_count is zero for every row.")


def main() -> None:
    args = parse_args()
    schema = read_schema(args.schema_path)
    daily = load_daily(args.daily_path)
    feature_columns = [column for column in schema["feature_columns"] if column in daily.columns]
    if args.require_contextual_drivers:
        validate_contextual_driver_inputs(daily, feature_columns)

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
    weights = class_weight_tensor(sequence_labels(train_set), args.class_weighting, model.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    train_loader = make_loader(train_set, args.batch_size, sampler=args.sampler)

    print(f"train sequences: {len(train_set)}")
    print(f"validation sequences: {len(validation_set)}")
    print(f"test sequences: {len(test_set)}")
    print(f"train label counts: {format_counts(evaluate_label_counts(train_set))}")
    print(f"validation label counts: {format_counts(evaluate_label_counts(validation_set))}")
    print(f"test label counts: {format_counts(evaluate_label_counts(test_set))}")
    print(
        f"sampler={args.sampler} class_weighting={args.class_weighting} "
        f"persistence_prior_strength={args.persistence_prior_strength}"
    )

    train_majority = int(np.bincount(sequence_labels(train_set), minlength=3).argmax())
    print_report_summary("validation_majority_baseline", majority_baseline_report(validation_set, train_majority))
    test_current_baseline = current_regime_baseline_report(test_set)
    if test_current_baseline is not None:
        print_report_summary("test_current_regime_baseline", test_current_baseline)

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
            final_logits = add_persistence_prior(
                final_logits,
                raw,
                feature_columns,
                args.persistence_prior_strength,
            )
            sequence_logits = add_persistence_prior(
                sequence_logits,
                raw,
                feature_columns,
                args.persistence_prior_strength,
            )
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

        validation_report = evaluate(
            model,
            validation_set,
            args.batch_size,
            persistence_prior_strength=args.persistence_prior_strength,
        )
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"task={np.mean(task_losses):.4f} smooth={np.mean(smooth_losses):.4f} "
            f"logic={np.mean(logic_losses):.4f} "
            f"val_accuracy={validation_report['accuracy']:.4f} "
            f"val_macro_f1={validation_report['macro_f1']:.4f}"
        )
        if args.diagnostics_every and epoch % args.diagnostics_every == 0:
            print(f"  val truth: {format_counts(validation_report['truth_counts'])}")
            print(f"  val pred : {format_counts(validation_report['prediction_counts'])}")
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

    test_report = evaluate(
        model,
        test_set,
        args.batch_size,
        persistence_prior_strength=args.persistence_prior_strength,
    )
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
            "persistence_prior_strength": args.persistence_prior_strength,
            "sampler": args.sampler,
            "class_weighting": args.class_weighting,
            "best_validation": best_state["validation"],
            "model_description": (
                "Daily ticker-level GRU latent state updated by aggregated news, "
                "macro features, and known market state."
            ),
        },
    )
    write_json(args.output_dir / "test_metrics.json", test_report)
    print_report_summary("test", test_report)
    if args.save_predictions:
        validation_predictions = collect_predictions(
            model,
            validation_set,
            args.batch_size,
            persistence_prior_strength=args.persistence_prior_strength,
        )
        test_predictions = collect_predictions(
            model,
            test_set,
            args.batch_size,
            persistence_prior_strength=args.persistence_prior_strength,
        )
        validation_predictions.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
        test_predictions.to_parquet(args.output_dir / "test_predictions.parquet", index=False)
        print(f"Saved predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
