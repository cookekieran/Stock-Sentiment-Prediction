"""
Train a two-stage GRU that detects regime transitions before choosing a destination.

Stage 1 predicts whether the current ticker regime will change at the configured
forecast horizon. Stage 2 is trained only on genuine transitions and predicts
the destination regime. At inference, stable predictions preserve the current
regime and destination logits are masked so a proposed transition cannot point
back to the current regime.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from common import classification_report, write_json
from train_latent_state_gru import (
    DailySequenceDataset,
    LABEL_NAMES,
    evaluate_label_counts,
    fit_feature_stats,
    load_daily,
    raw_features,
    read_schema,
    sequence_labels,
    transform_features,
    validate_contextual_driver_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-path", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/two_stage_transition_gru"))
    parser.add_argument(
        "--feature-set",
        choices=["full", "price_only", "price_macro", "price_qwen", "price_fundamentals", "price_qwen_fundamentals"],
        default="price_qwen",
    )
    parser.add_argument(
        "--detector-feature-set",
        choices=["full", "price_only", "price_macro", "price_qwen", "price_fundamentals", "price_qwen_fundamentals"],
        default=None,
        help="Override --feature-set for the transition detector.",
    )
    parser.add_argument(
        "--destination-feature-set",
        choices=["full", "price_only", "price_macro", "price_qwen", "price_fundamentals", "price_qwen_fundamentals"],
        default=None,
        help="Override --feature-set for the destination classifier.",
    )
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--destination-loss-weight", type=float, default=1.0)
    parser.add_argument("--logic-weight", type=float, default=0.0)
    parser.add_argument(
        "--logic-rule-set",
        choices=[
            "none",
            "news",
            "fundamentals",
            "all",
            "selective_release",
            "selective_surprise",
            "selective_persistence",
            "selective_all",
        ],
        default="none",
        help="Apply auditable LTN-style fuzzy constraints to transition and destination probabilities.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["final_macro_f1", "transition_pr_auc"],
        default="final_macro_f1",
        help="Validation metric used to select the saved epoch.",
    )
    parser.add_argument(
        "--transition-pos-weight",
        choices=["auto", "none"],
        default="auto",
        help="Balance transition rows in the binary loss using the training split.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument(
        "--minimum-stable-accuracy",
        type=float,
        default=0.70,
        help="Minimum validation stable accuracy required when selecting the transition threshold.",
    )
    parser.add_argument("--require-contextual-drivers", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def current_regime_ids(dataset: DailySequenceDataset) -> np.ndarray:
    idx = dataset.feature_columns.index("price_trend_id")
    return np.array(
        [int(round(float(dataset.raw[end - 1, idx]))) for _, end in dataset.indices],
        dtype=np.int64,
    ).clip(0, len(LABEL_NAMES) - 1)


def transition_labels(dataset: DailySequenceDataset) -> np.ndarray:
    return (sequence_labels(dataset) != current_regime_ids(dataset)).astype(np.int64)


def transition_feature_columns(feature_columns: list[str], feature_set: str) -> list[str]:
    price_columns = {
        "realized_volatility_20d",
        "rally_from_previous_low",
        "drawdown_from_previous_high",
        "recent_return",
        "drawdown_from_recent_high",
        "price_trend_id",
    }
    fundamentals = {column for column in feature_columns if column.startswith("fundamental_")}
    macro = {column for column in feature_columns if column.startswith("macro_")}
    rule_features = {column for column in feature_columns if column.startswith("rule_")}
    qwen = set(feature_columns).difference(price_columns, fundamentals, macro, rule_features)
    requested = {
        "full": set(feature_columns),
        "price_only": price_columns,
        "price_macro": price_columns | macro,
        "price_qwen": price_columns | qwen,
        "price_fundamentals": price_columns | fundamentals,
        "price_qwen_fundamentals": price_columns | qwen | fundamentals,
    }[feature_set]
    selected = [column for column in feature_columns if column in requested]
    if not selected:
        raise ValueError(f"No features selected for feature_set={feature_set}.")
    return selected


def fuzzy_implication_loss(antecedent, consequent):
    truth = 1.0 - antecedent + antecedent * consequent
    return (1.0 - truth.clamp(0.0, 1.0)).mean()


def final_raw_feature(raw, feature_columns: list[str], name: str):
    import torch

    try:
        index = feature_columns.index(name)
    except ValueError:
        return torch.zeros(raw.shape[0], dtype=raw.dtype, device=raw.device)
    return raw[:, -1, index]


def transition_logic_losses(
    transition_logits,
    destination_logits,
    raw,
    feature_columns: list[str],
    rule_set: str,
) -> dict[str, object]:
    """Return named fuzzy-rule violations for the selected LTN-style rule set."""
    import torch

    if rule_set == "none":
        return {}
    transition = transition_logits.sigmoid()
    destination = destination_logits.softmax(dim=-1)
    bear = destination[:, 0]
    sideways = destination[:, 1]
    bull = destination[:, 2]

    confidence = final_raw_feature(raw, feature_columns, "driver_mean_confidence").clamp(0.0, 1.0)
    uncertainty = final_raw_feature(raw, feature_columns, "driver_uncertainty").clamp(0.0, 1.0)
    materiality = final_raw_feature(raw, feature_columns, "driver_mean_materiality").clamp(0.0, 1.0)
    novelty = final_raw_feature(raw, feature_columns, "driver_mean_novelty").clamp(0.0, 1.0)
    intensity = final_raw_feature(raw, feature_columns, "driver_mean_intensity").clamp(0.0, 1.0)
    update_strength = final_raw_feature(raw, feature_columns, "driver_update_strength").clamp(0.0, 1.0)
    ticker_scope = final_raw_feature(raw, feature_columns, "driver_ticker_scope_share").clamp(0.0, 1.0)
    irrelevant_scope = final_raw_feature(raw, feature_columns, "driver_irrelevant_scope_share").clamp(0.0, 1.0)
    risk_on = final_raw_feature(raw, feature_columns, "driver_max_risk_on_shock").clamp(0.0, 1.0)
    risk_off = final_raw_feature(raw, feature_columns, "driver_max_risk_off_shock").clamp(0.0, 1.0)
    release_proximity = final_raw_feature(raw, feature_columns, "rule_release_proximity_10d").clamp(0.0, 1.0)
    revenue_surprise = final_raw_feature(raw, feature_columns, "rule_revenue_yoy_ticker_z")
    margin_surprise = final_raw_feature(
        raw,
        feature_columns,
        "rule_operating_margin_yoy_change_ticker_z",
    )
    persistent_risk_on = final_raw_feature(
        raw,
        feature_columns,
        "rule_risk_on_shock_persistence_3d",
    ).clamp(0.0, 1.0)
    persistent_risk_off = final_raw_feature(
        raw,
        feature_columns,
        "rule_risk_off_shock_persistence_3d",
    ).clamp(0.0, 1.0)
    mean_risk_on_3d = final_raw_feature(raw, feature_columns, "rule_risk_on_shock_mean_3d").clamp(0.0, 1.0)
    mean_risk_off_3d = final_raw_feature(raw, feature_columns, "rule_risk_off_shock_mean_3d").clamp(0.0, 1.0)

    material_news = (
        confidence
        * (1.0 - uncertainty)
        * materiality
        * novelty
        * intensity
        * (0.5 + 0.5 * ticker_scope)
    ).clamp(0.0, 1.0)
    weak_or_irrelevant_news = (
        (1.0 - update_strength)
        * (1.0 - materiality)
        + irrelevant_scope
    ).clamp(0.0, 1.0)
    conflicting_news = (risk_on * risk_off).sqrt().clamp(0.0, 1.0)

    losses: dict[str, object] = {}
    if rule_set in {"news", "all"}:
        losses["material_news_implies_transition"] = fuzzy_implication_loss(
            material_news,
            transition,
        )
        losses["weak_or_irrelevant_news_implies_persistence"] = fuzzy_implication_loss(
            weak_or_irrelevant_news,
            1.0 - transition,
        )
        losses["conflicting_news_implies_sideways_destination"] = fuzzy_implication_loss(
            conflicting_news * transition,
            sideways,
        )

    if rule_set in {"fundamentals", "all"}:
        revenue_yoy = final_raw_feature(raw, feature_columns, "fundamental_revenue_yoy")
        operating_margin_yoy = final_raw_feature(
            raw,
            feature_columns,
            "fundamental_operating_margin_yoy_change",
        )
        deteriorating_fundamentals = (
            torch.sigmoid(-5.0 * revenue_yoy)
            * torch.sigmoid(-15.0 * operating_margin_yoy)
        ).clamp(0.0, 1.0)
        improving_fundamentals = (
            torch.sigmoid(5.0 * revenue_yoy)
            * torch.sigmoid(15.0 * operating_margin_yoy)
        ).clamp(0.0, 1.0)
        aligned_risk_off = (deteriorating_fundamentals * risk_off * confidence).clamp(0.0, 1.0)
        aligned_risk_on = (improving_fundamentals * risk_on * confidence).clamp(0.0, 1.0)
        losses["deteriorating_fundamentals_and_risk_off_imply_transition"] = (
            fuzzy_implication_loss(aligned_risk_off, transition)
        )
        losses["deteriorating_fundamentals_and_risk_off_imply_bear_destination"] = (
            fuzzy_implication_loss(aligned_risk_off * transition, bear)
        )
        losses["improving_fundamentals_and_risk_on_imply_transition"] = (
            fuzzy_implication_loss(aligned_risk_on, transition)
        )
        losses["improving_fundamentals_and_risk_on_imply_bull_destination"] = (
            fuzzy_implication_loss(aligned_risk_on * transition, bull)
        )
    if rule_set in {"selective_release", "selective_all"}:
        release_material_news = (release_proximity * material_news).clamp(0.0, 1.0)
        losses["recent_release_and_material_news_imply_transition"] = (
            fuzzy_implication_loss(release_material_news, transition)
        )
    if rule_set in {"selective_surprise", "selective_all"}:
        downside_surprise = (
            torch.sigmoid(-2.0 * revenue_surprise)
            * torch.sigmoid(-2.0 * margin_surprise)
            * release_proximity
            * risk_off
            * confidence
        ).clamp(0.0, 1.0)
        upside_surprise = (
            torch.sigmoid(2.0 * revenue_surprise)
            * torch.sigmoid(2.0 * margin_surprise)
            * release_proximity
            * risk_on
            * confidence
        ).clamp(0.0, 1.0)
        losses["ticker_relative_downside_surprise_implies_transition"] = (
            fuzzy_implication_loss(downside_surprise, transition)
        )
        losses["ticker_relative_downside_surprise_implies_bear_destination"] = (
            fuzzy_implication_loss(downside_surprise * transition, bear)
        )
        losses["ticker_relative_upside_surprise_implies_transition"] = (
            fuzzy_implication_loss(upside_surprise, transition)
        )
        losses["ticker_relative_upside_surprise_implies_bull_destination"] = (
            fuzzy_implication_loss(upside_surprise * transition, bull)
        )
    if rule_set in {"selective_persistence", "selective_all"}:
        persistent_off = (
            persistent_risk_off * mean_risk_off_3d * confidence * (1.0 - uncertainty)
        ).clamp(0.0, 1.0)
        persistent_on = (
            persistent_risk_on * mean_risk_on_3d * confidence * (1.0 - uncertainty)
        ).clamp(0.0, 1.0)
        losses["persistent_risk_off_shocks_imply_transition"] = fuzzy_implication_loss(
            persistent_off,
            transition,
        )
        losses["persistent_risk_off_shocks_imply_bear_destination"] = (
            fuzzy_implication_loss(persistent_off * transition, bear)
        )
        losses["persistent_risk_on_shocks_imply_transition"] = fuzzy_implication_loss(
            persistent_on,
            transition,
        )
        losses["persistent_risk_on_shocks_imply_bull_destination"] = (
            fuzzy_implication_loss(persistent_on * transition, bull)
        )
    return losses


def mean_logic_loss(losses: dict[str, object], reference):
    if not losses:
        return reference.new_tensor(0.0)
    return sum(losses.values()) / len(losses)


class DualInputSequenceDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        detector_feature_columns: list[str],
        destination_feature_columns: list[str],
        raw_feature_columns: list[str],
        detector_stats: dict[str, dict[str, float]],
        destination_stats: dict[str, dict[str, float]],
        sequence_length: int,
    ):
        self.df = df.sort_values(["ticker", "anchor_trading_date"]).reset_index(drop=True)
        self.detector_feature_columns = detector_feature_columns
        self.destination_feature_columns = destination_feature_columns
        self.feature_columns = raw_feature_columns
        self.detector_x = transform_features(self.df, detector_feature_columns, detector_stats)
        self.destination_x = transform_features(self.df, destination_feature_columns, destination_stats)
        self.raw = raw_features(self.df, raw_feature_columns)
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
        return (
            self.detector_x[start:end],
            self.destination_x[start:end],
            self.raw[start:end],
            self.y[end - 1],
        )


class TwoStageTransitionGRU:
    def __init__(
        self,
        detector_input_dim: int,
        destination_input_dim: int,
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
        self.detector_encoder = nn.GRU(
            input_size=detector_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        ).to(self.device)
        self.destination_encoder = nn.GRU(
            input_size=destination_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        ).to(self.device)
        self.transition_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        ).to(self.device)
        self.destination_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(LABEL_NAMES)),
        ).to(self.device)

    def logits(self, detector_x, destination_x):
        detector_hidden_sequence, _ = self.detector_encoder(detector_x)
        destination_hidden_sequence, _ = self.destination_encoder(destination_x)
        transition_logits = self.transition_head(detector_hidden_sequence[:, -1, :]).squeeze(-1)
        destination_logits = self.destination_head(destination_hidden_sequence[:, -1, :])
        return transition_logits, destination_logits

    def parameters(self):
        yield from self.detector_encoder.parameters()
        yield from self.destination_encoder.parameters()
        yield from self.transition_head.parameters()
        yield from self.destination_head.parameters()

    def train(self):
        self.detector_encoder.train()
        self.destination_encoder.train()
        self.transition_head.train()
        self.destination_head.train()

    def eval(self):
        self.detector_encoder.eval()
        self.destination_encoder.eval()
        self.transition_head.eval()
        self.destination_head.eval()


def mask_current_destination(destination_logits, current):
    masked = destination_logits.clone()
    masked.scatter_(1, current[:, None], float("-inf"))
    return masked


def binary_report(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_pred = (probabilities >= threshold).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "actual_transition_rows": int(y_true.sum()),
        "predicted_transition_rows": int(y_pred.sum()),
    }


def tune_threshold(
    arrays: dict[str, np.ndarray],
    minimum_stable_accuracy: float,
) -> tuple[float, dict]:
    candidates = np.linspace(0.05, 0.95, 91)
    y_true = (arrays["true_id"] != arrays["current_id"]).astype(np.int64)
    reports = []
    for threshold in candidates:
        predicted_transition = arrays["transition_probability"] >= threshold
        final_ids = np.where(predicted_transition, arrays["destination_id"], arrays["current_id"])
        transition_rows = y_true.astype(bool)
        stable_rows = ~transition_rows
        detector = binary_report(y_true, arrays["transition_probability"], threshold)
        regime = classification_report(arrays["true_id"], final_ids)
        detector["final_macro_f1"] = regime["macro_f1"]
        detector["final_accuracy"] = regime["accuracy"]
        detector["stable_accuracy"] = (
            float((final_ids[stable_rows] == arrays["true_id"][stable_rows]).mean())
            if stable_rows.any()
            else 1.0
        )
        reports.append(detector)
    eligible = [report for report in reports if report["stable_accuracy"] >= minimum_stable_accuracy]
    if not eligible:
        eligible = reports
    best = max(
        eligible,
        key=lambda report: (report["final_macro_f1"], report["f1"], report["precision"]),
    )
    return float(best["threshold"]), best


def collect_arrays(model: TwoStageTransitionGRU, dataset: DualInputSequenceDataset, batch_size: int) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    transition_probabilities = []
    destination_probabilities = []
    destinations = []
    currents = []
    truth = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for detector_x, destination_x, raw, y in loader:
            detector_x = detector_x.to(model.device, dtype=torch.float32)
            destination_x = destination_x.to(model.device, dtype=torch.float32)
            y = y.to(model.device, dtype=torch.long)
            current = raw[:, -1, dataset.feature_columns.index("price_trend_id")].round().long()
            current = current.clamp(0, len(LABEL_NAMES) - 1).to(model.device)
            transition_logits, destination_logits = model.logits(detector_x, destination_x)
            masked_destination_logits = mask_current_destination(destination_logits, current)
            transition_probabilities.append(transition_logits.sigmoid().cpu().numpy())
            destination_probabilities.append(masked_destination_logits.softmax(dim=1).cpu().numpy())
            destinations.append(masked_destination_logits.argmax(dim=1).cpu().numpy())
            currents.append(current.cpu().numpy())
            truth.append(y.cpu().numpy())
    return {
        "transition_probability": np.concatenate(transition_probabilities),
        "destination_probabilities": np.concatenate(destination_probabilities),
        "destination_id": np.concatenate(destinations),
        "current_id": np.concatenate(currents),
        "true_id": np.concatenate(truth),
    }


def evaluate(model: TwoStageTransitionGRU, dataset: DualInputSequenceDataset, batch_size: int, threshold: float) -> dict:
    arrays = collect_arrays(model, dataset, batch_size)
    true_transition = (arrays["true_id"] != arrays["current_id"]).astype(np.int64)
    predicted_transition = arrays["transition_probability"] >= threshold
    final_ids = np.where(predicted_transition, arrays["destination_id"], arrays["current_id"])
    transition_rows = true_transition.astype(bool)
    stable_rows = ~transition_rows
    destination_accuracy = (
        float((arrays["destination_id"][transition_rows] == arrays["true_id"][transition_rows]).mean())
        if transition_rows.any()
        else None
    )
    return {
        "transition_detector": binary_report(true_transition, arrays["transition_probability"], threshold),
        "destination_accuracy_on_true_transitions": destination_accuracy,
        "final_regime": {
            **classification_report(arrays["true_id"], final_ids),
            "transition_accuracy": float((final_ids[transition_rows] == arrays["true_id"][transition_rows]).mean())
            if transition_rows.any()
            else None,
            "stable_accuracy": float((final_ids[stable_rows] == arrays["true_id"][stable_rows]).mean())
            if stable_rows.any()
            else None,
        },
    }


def collect_predictions(
    model: TwoStageTransitionGRU,
    dataset: DualInputSequenceDataset,
    batch_size: int,
    threshold: float,
) -> pd.DataFrame:
    arrays = collect_arrays(model, dataset, batch_size)
    true_transition = arrays["true_id"] != arrays["current_id"]
    predicted_transition = arrays["transition_probability"] >= threshold
    final_ids = np.where(predicted_transition, arrays["destination_id"], arrays["current_id"])
    rows = []
    for cursor, (start, end) in enumerate(dataset.indices):
        final = dataset.df.iloc[end - 1]
        row = {
            "ticker": final["ticker"],
            "anchor_trading_date": final["anchor_trading_date"],
            "sequence_start_date": dataset.df.iloc[start]["anchor_trading_date"],
            "current_price_trend_label": LABEL_NAMES[arrays["current_id"][cursor]],
            "future_price_trend_label": LABEL_NAMES[arrays["true_id"][cursor]],
            "is_transition": bool(true_transition[cursor]),
            "transition_probability": float(arrays["transition_probability"][cursor]),
            "predicted_transition": bool(predicted_transition[cursor]),
            "destination_label": LABEL_NAMES[arrays["destination_id"][cursor]],
            "destination_prob_bear_drawdown": float(arrays["destination_probabilities"][cursor, 0]),
            "destination_prob_sideways": float(arrays["destination_probabilities"][cursor, 1]),
            "destination_prob_bull_rally": float(arrays["destination_probabilities"][cursor, 2]),
            "pred_label_id": int(final_ids[cursor]),
            "pred_label": LABEL_NAMES[final_ids[cursor]],
            "true_label_id": int(arrays["true_id"][cursor]),
            "true_label": LABEL_NAMES[arrays["true_id"][cursor]],
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
    return pd.DataFrame(rows)


def format_regime_counts(dataset: DualInputSequenceDataset) -> str:
    counts = evaluate_label_counts(dataset)
    return ", ".join(f"{label}={counts[label]}" for label in LABEL_NAMES)


def print_report(name: str, report: dict) -> None:
    detector = report["transition_detector"]
    regime = report["final_regime"]
    print(
        f"{name}_transition precision={detector['precision']:.4f} recall={detector['recall']:.4f} "
        f"f1={detector['f1']:.4f} pr_auc={detector['pr_auc']:.4f} roc_auc={detector['roc_auc']:.4f} "
        f"predicted={detector['predicted_transition_rows']} actual={detector['actual_transition_rows']}"
    )
    print(
        f"{name}_destination_accuracy_on_true_transitions="
        f"{report['destination_accuracy_on_true_transitions']:.4f}"
    )
    print(
        f"{name}_final accuracy={regime['accuracy']:.4f} macro_f1={regime['macro_f1']:.4f} "
        f"transition_accuracy={regime['transition_accuracy']:.4f} stable_accuracy={regime['stable_accuracy']:.4f}"
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    schema = read_schema(args.schema_path)
    daily = load_daily(args.daily_path)
    available_features = [column for column in schema["feature_columns"] if column in daily.columns]
    if args.require_contextual_drivers:
        validate_contextual_driver_inputs(daily, available_features)
    detector_feature_set = args.detector_feature_set or args.feature_set
    destination_feature_set = args.destination_feature_set or args.feature_set
    detector_features = transition_feature_columns(available_features, detector_feature_set)
    destination_features = transition_feature_columns(available_features, destination_feature_set)
    if "price_trend_id" not in available_features:
        raise ValueError("price_trend_id must be present to define transition labels.")

    train_df = daily[daily["split"] == "train"].copy()
    validation_df = daily[daily["split"] == "validation"].copy()
    test_df = daily[daily["split"] == "test"].copy()
    detector_stats = fit_feature_stats(train_df, detector_features)
    destination_stats = fit_feature_stats(train_df, destination_features)
    train_set = DualInputSequenceDataset(
        train_df,
        detector_features,
        destination_features,
        available_features,
        detector_stats,
        destination_stats,
        args.sequence_length,
    )
    validation_set = DualInputSequenceDataset(
        validation_df,
        detector_features,
        destination_features,
        available_features,
        detector_stats,
        destination_stats,
        args.sequence_length,
    )
    test_set = DualInputSequenceDataset(
        test_df,
        detector_features,
        destination_features,
        available_features,
        detector_stats,
        destination_stats,
        args.sequence_length,
    )

    train_transitions = transition_labels(train_set)
    validation_transitions = transition_labels(validation_set)
    test_transitions = transition_labels(test_set)
    print(f"train sequences: {len(train_set)} transitions={int(train_transitions.sum())} stable={int((train_transitions == 0).sum())}")
    print(f"validation sequences: {len(validation_set)} transitions={int(validation_transitions.sum())} stable={int((validation_transitions == 0).sum())}")
    print(f"test sequences: {len(test_set)} transitions={int(test_transitions.sum())} stable={int((test_transitions == 0).sum())}")
    print(f"train destination labels: {format_regime_counts(train_set)}")
    print(
        f"detector_feature_set={detector_feature_set} detector_feature_count={len(detector_features)} "
        f"destination_feature_set={destination_feature_set} "
        f"destination_feature_count={len(destination_features)}"
    )

    model = TwoStageTransitionGRU(
        detector_input_dim=len(detector_features),
        destination_input_dim=len(destination_features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device,
    )
    transition_weight = 1.0
    if args.transition_pos_weight == "auto":
        positives = max(int(train_transitions.sum()), 1)
        negatives = int((train_transitions == 0).sum())
        transition_weight = negatives / positives
    transition_criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(transition_weight, dtype=torch.float32, device=model.device)
    )
    destination_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_selection_score = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        transition_losses = []
        destination_losses = []
        logic_losses = []
        for detector_x, destination_x, raw, y in loader:
            detector_x = detector_x.to(model.device, dtype=torch.float32)
            destination_x = destination_x.to(model.device, dtype=torch.float32)
            y = y.to(model.device, dtype=torch.long)
            current = raw[:, -1, available_features.index("price_trend_id")].round().long()
            current = current.clamp(0, len(LABEL_NAMES) - 1).to(model.device)
            is_transition = (current != y).to(dtype=torch.float32)
            optimizer.zero_grad()
            transition_logits, destination_logits = model.logits(detector_x, destination_x)
            transition_loss = transition_criterion(transition_logits, is_transition)
            transition_mask = is_transition.bool()
            if transition_mask.any():
                destination_loss = destination_criterion(destination_logits[transition_mask], y[transition_mask])
            else:
                destination_loss = destination_logits.sum() * 0.0
            rule_losses = transition_logic_losses(
                transition_logits,
                destination_logits,
                raw.to(model.device, dtype=torch.float32),
                available_features,
                args.logic_rule_set,
            )
            logic_loss = mean_logic_loss(rule_losses, transition_logits)
            loss = (
                transition_loss
                + args.destination_loss_weight * destination_loss
                + args.logic_weight * logic_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            transition_losses.append(float(transition_loss.detach().cpu()))
            destination_losses.append(float(destination_loss.detach().cpu()))
            logic_losses.append(float(logic_loss.detach().cpu()))

        validation_arrays = collect_arrays(model, validation_set, args.batch_size)
        threshold, detector_report = tune_threshold(
            validation_arrays,
            args.minimum_stable_accuracy,
        )
        validation_report = evaluate(model, validation_set, args.batch_size, threshold)
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} transition_loss={np.mean(transition_losses):.4f} "
            f"destination_loss={np.mean(destination_losses):.4f} logic_loss={np.mean(logic_losses):.4f} "
            f"threshold={threshold:.2f} "
            f"val_transition_f1={detector_report['f1']:.4f} val_pr_auc={detector_report['pr_auc']:.4f} "
            f"val_final_accuracy={validation_report['final_regime']['accuracy']:.4f} "
            f"val_final_macro_f1={validation_report['final_regime']['macro_f1']:.4f}"
        )
        selection_score = {
            "final_macro_f1": validation_report["final_regime"]["macro_f1"],
            "transition_pr_auc": detector_report["pr_auc"],
        }[args.selection_metric]
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            epochs_without_improvement = 0
            best_state = {
            "detector_encoder": {
                k: v.detach().cpu().clone() for k, v in model.detector_encoder.state_dict().items()
            },
            "destination_encoder": {
                k: v.detach().cpu().clone() for k, v in model.destination_encoder.state_dict().items()
            },
                "transition_head": {k: v.detach().cpu().clone() for k, v in model.transition_head.state_dict().items()},
                "destination_head": {k: v.detach().cpu().clone() for k, v in model.destination_head.state_dict().items()},
                "threshold": threshold,
                "validation": validation_report,
                "epoch": epoch,
            }
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_state['epoch']}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    model.detector_encoder.load_state_dict(best_state["detector_encoder"])
    model.destination_encoder.load_state_dict(best_state["destination_encoder"])
    model.transition_head.load_state_dict(best_state["transition_head"])
    model.destination_head.load_state_dict(best_state["destination_head"])
    torch.save(best_state, args.output_dir / "best_two_stage_transition_gru.pt")

    test_report = evaluate(model, test_set, args.batch_size, best_state["threshold"])
    print(f"selected_validation_threshold={best_state['threshold']:.2f}")
    print_report("validation", best_state["validation"])
    print_report("test", test_report)
    write_json(args.output_dir / "test_metrics.json", test_report)
    write_json(
        args.output_dir / "training_metadata.json",
        {
            "detector_feature_columns": detector_features,
            "destination_feature_columns": destination_features,
            "raw_feature_columns": available_features,
            "detector_feature_stats": detector_stats,
            "destination_feature_stats": destination_stats,
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "detector_feature_set": detector_feature_set,
            "destination_feature_set": destination_feature_set,
            "minimum_stable_accuracy": args.minimum_stable_accuracy,
            "seed": args.seed,
            "destination_loss_weight": args.destination_loss_weight,
            "logic_weight": args.logic_weight,
            "logic_rule_set": args.logic_rule_set,
            "logic_rule_names": list(
                transition_logic_losses(
                    torch.zeros(1, device=model.device),
                    torch.zeros((1, len(LABEL_NAMES)), device=model.device),
                    torch.zeros((1, 1, len(available_features)), device=model.device),
                    available_features,
                    args.logic_rule_set,
                )
            ),
            "selection_metric": args.selection_metric,
            "transition_pos_weight": args.transition_pos_weight,
            "resolved_transition_pos_weight": transition_weight,
            "best_epoch": best_state["epoch"],
            "selected_validation_threshold": best_state["threshold"],
            "best_validation": best_state["validation"],
            "model_description": (
                "Two-stage daily GRU: detect whether the current regime changes, "
                "then predict the destination regime only when a transition is proposed."
            ),
        },
    )
    if args.save_predictions:
        collect_predictions(model, validation_set, args.batch_size, best_state["threshold"]).to_parquet(
            args.output_dir / "validation_predictions.parquet", index=False
        )
        collect_predictions(model, test_set, args.batch_size, best_state["threshold"]).to_parquet(
            args.output_dir / "test_predictions.parquet", index=False
        )
        print(f"Saved predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
