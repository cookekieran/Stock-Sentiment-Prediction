"""Train a GRU to detect news-associated material ticker repricing events."""

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

from train_latent_state_gru import (
    LatentStateGRU,
    fit_feature_stats,
    load_daily,
    transform_features,
    validate_contextual_driver_inputs,
)


EVENT_LABELS = ["no_material_event", "risk_off_event", "risk_on_event"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-path", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-set",
        choices=[
            "full",
            "price_only",
            "price_macro",
            "price_qwen",
            "qwen_only",
            "fundamentals_only",
            "price_fundamentals",
            "price_qwen_fundamentals",
        ],
        default="price_qwen",
    )
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--class-weighting", choices=["none", "inverse", "sqrt_inverse"], default="sqrt_inverse")
    parser.add_argument("--minimum-no-event-accuracy", type=float, default=0.70)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--require-contextual-drivers", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class MaterialEventSequenceDataset:
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
        self.y = pd.to_numeric(self.df["material_event_id"], errors="raise").to_numpy(np.int64)
        self.indices: list[tuple[int, int]] = []
        for _, group in self.df.groupby("ticker", sort=False):
            positions = group.index.to_numpy()
            for end_offset in range(sequence_length, len(positions) + 1):
                window = positions[end_offset - sequence_length : end_offset]
                self.indices.append((int(window[0]), int(window[-1]) + 1))
        if not self.indices:
            raise ValueError(f"No sequences created for sequence_length={sequence_length}.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start, end = self.indices[idx]
        return self.x[start:end], self.y[end - 1]


def sequence_labels(dataset: MaterialEventSequenceDataset) -> np.ndarray:
    return np.asarray([dataset.y[end - 1] for _, end in dataset.indices], dtype=np.int64)


def event_feature_columns(feature_columns: list[str], feature_set: str) -> list[str]:
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
    qwen = set(feature_columns).difference(price_columns, fundamentals, macro)
    requested = {
        "full": set(feature_columns),
        "price_only": price_columns,
        "price_macro": price_columns | macro,
        "price_qwen": price_columns | qwen,
        "qwen_only": qwen,
        "fundamentals_only": fundamentals,
        "price_fundamentals": price_columns | fundamentals,
        "price_qwen_fundamentals": price_columns | qwen | fundamentals,
    }[feature_set]
    selected = [column for column in feature_columns if column in requested]
    if not selected:
        raise ValueError(f"No features selected for feature_set={feature_set}.")
    return selected


def class_weights(labels: np.ndarray, mode: str, device):
    import torch

    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=len(EVENT_LABELS)).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (len(EVENT_LABELS) * counts)
    if mode == "sqrt_inverse":
        weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(EVENT_LABELS)),
        zero_division=0,
    )
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(len(EVENT_LABELS)), average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(EVENT_LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(EVENT_LABELS))).tolist(),
    }


def collect_arrays(model: LatentStateGRU, dataset: MaterialEventSequenceDataset, batch_size: int) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    model.model.eval()
    model.head.eval()
    probabilities = []
    truth = []
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits, _ = model.logits(x.to(model.device, dtype=torch.float32))
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
            truth.append(y.numpy())
    return {"probabilities": np.concatenate(probabilities), "truth": np.concatenate(truth)}


def predictions_for_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    event_probability = probabilities[:, 1:].sum(axis=1)
    direction = probabilities[:, 1:].argmax(axis=1) + 1
    return np.where(event_probability >= threshold, direction, 0)


def evaluate_arrays(arrays: dict[str, np.ndarray], threshold: float) -> dict:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    probabilities = arrays["probabilities"]
    truth = arrays["truth"]
    pred = predictions_for_threshold(probabilities, threshold)
    true_event = truth != 0
    predicted_event = pred != 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_event,
        predicted_event,
        average="binary",
        zero_division=0,
    )
    no_event = ~true_event
    return {
        "threshold": float(threshold),
        "event_detector": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(average_precision_score(true_event, probabilities[:, 1:].sum(axis=1))),
            "roc_auc": float(roc_auc_score(true_event, probabilities[:, 1:].sum(axis=1))),
            "actual_event_rows": int(true_event.sum()),
            "predicted_event_rows": int(predicted_event.sum()),
            "no_event_accuracy": float((pred[no_event] == 0).mean()) if no_event.any() else 1.0,
        },
        "direction_accuracy_on_true_events": float((pred[true_event] == truth[true_event]).mean()) if true_event.any() else None,
        "event_direction": classification_metrics(truth, pred),
    }


def tune_threshold(arrays: dict[str, np.ndarray], minimum_no_event_accuracy: float) -> tuple[float, dict]:
    reports = [evaluate_arrays(arrays, threshold) for threshold in np.linspace(0.05, 0.95, 91)]
    eligible = [
        report
        for report in reports
        if report["event_detector"]["no_event_accuracy"] >= minimum_no_event_accuracy
    ]
    if not eligible:
        eligible = reports
    best = max(
        eligible,
        key=lambda report: (
            report["event_direction"]["macro_f1"],
            report["event_detector"]["f1"],
            report["event_detector"]["precision"],
        ),
    )
    return best["threshold"], best


def collect_predictions(
    model: LatentStateGRU,
    dataset: MaterialEventSequenceDataset,
    batch_size: int,
    threshold: float,
) -> pd.DataFrame:
    arrays = collect_arrays(model, dataset, batch_size)
    probabilities = arrays["probabilities"]
    pred = predictions_for_threshold(probabilities, threshold)
    rows = []
    for cursor, (start, end) in enumerate(dataset.indices):
        final = dataset.df.iloc[end - 1]
        row = {
            "ticker": final["ticker"],
            "anchor_trading_date": final["anchor_trading_date"],
            "sequence_start_date": dataset.df.iloc[start]["anchor_trading_date"],
            "true_label_id": int(arrays["truth"][cursor]),
            "true_label": EVENT_LABELS[arrays["truth"][cursor]],
            "pred_label_id": int(pred[cursor]),
            "pred_label": EVENT_LABELS[pred[cursor]],
            "prob_no_material_event": float(probabilities[cursor, 0]),
            "prob_risk_off_event": float(probabilities[cursor, 1]),
            "prob_risk_on_event": float(probabilities[cursor, 2]),
            "prob_material_event": float(probabilities[cursor, 1:].sum()),
            "forward_max_return": final["forward_max_return"],
            "forward_min_return": final["forward_min_return"],
            "forward_close_return": final["forward_close_return"],
            "material_event_threshold": final["material_event_threshold"],
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


def print_report(name: str, report: dict) -> None:
    detector = report["event_detector"]
    direction = report["event_direction"]
    print(
        f"{name}_event precision={detector['precision']:.4f} recall={detector['recall']:.4f} "
        f"f1={detector['f1']:.4f} pr_auc={detector['pr_auc']:.4f} roc_auc={detector['roc_auc']:.4f} "
        f"predicted={detector['predicted_event_rows']} actual={detector['actual_event_rows']} "
        f"no_event_accuracy={detector['no_event_accuracy']:.4f}"
    )
    print(
        f"{name}_direction accuracy={direction['accuracy']:.4f} macro_f1={direction['macro_f1']:.4f} "
        f"direction_accuracy_on_true_events={report['direction_accuracy_on_true_events']:.4f}"
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
    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))
    daily = load_daily(args.daily_path)
    available_features = [column for column in schema["feature_columns"] if column in daily.columns]
    if args.require_contextual_drivers:
        validate_contextual_driver_inputs(daily, available_features)
    input_features = event_feature_columns(available_features, args.feature_set)
    train_df = daily[daily["split"] == "train"].copy()
    validation_df = daily[daily["split"] == "validation"].copy()
    test_df = daily[daily["split"] == "test"].copy()
    stats = fit_feature_stats(train_df, input_features)
    train_set = MaterialEventSequenceDataset(train_df, input_features, stats, args.sequence_length)
    validation_set = MaterialEventSequenceDataset(validation_df, input_features, stats, args.sequence_length)
    test_set = MaterialEventSequenceDataset(test_df, input_features, stats, args.sequence_length)
    train_labels = sequence_labels(train_set)
    print(f"train sequences: {len(train_set)} labels={np.bincount(train_labels, minlength=3).tolist()}")
    print(f"validation sequences: {len(validation_set)} labels={np.bincount(sequence_labels(validation_set), minlength=3).tolist()}")
    print(f"test sequences: {len(test_set)} labels={np.bincount(sequence_labels(test_set), minlength=3).tolist()}")
    print(f"feature_set={args.feature_set} feature_count={len(input_features)}")

    model = LatentStateGRU(
        input_dim=len(input_features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_labels, args.class_weighting, model.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    best_state = None
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.model.train()
        model.head.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad()
            logits, _ = model.logits(x.to(model.device, dtype=torch.float32))
            loss = criterion(logits, y.to(model.device, dtype=torch.long))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_arrays = collect_arrays(model, validation_set, args.batch_size)
        threshold, report = tune_threshold(validation_arrays, args.minimum_no_event_accuracy)
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} threshold={threshold:.2f} "
            f"val_macro_f1={report['event_direction']['macro_f1']:.4f} "
            f"val_event_f1={report['event_detector']['f1']:.4f} "
            f"val_pr_auc={report['event_detector']['pr_auc']:.4f}"
        )
        if report["event_direction"]["macro_f1"] > best_macro_f1:
            best_macro_f1 = report["event_direction"]["macro_f1"]
            epochs_without_improvement = 0
            best_state = {
                "model": {key: value.detach().cpu().clone() for key, value in model.model.state_dict().items()},
                "head": {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()},
                "threshold": threshold,
                "validation": report,
                "epoch": epoch,
            }
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_state['epoch']}.")
                break
    model.model.load_state_dict(best_state["model"])
    model.head.load_state_dict(best_state["head"])
    test_report = evaluate_arrays(collect_arrays(model, test_set, args.batch_size), best_state["threshold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_dir / "best_material_event_gru.pt")
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "feature_columns": input_features,
                "feature_stats": stats,
                "feature_set": args.feature_set,
                "sequence_length": args.sequence_length,
                "best_epoch": best_state["epoch"],
                "selected_validation_threshold": best_state["threshold"],
                "best_validation": best_state["validation"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"selected_validation_threshold={best_state['threshold']:.2f}")
    print_report("validation", best_state["validation"])
    print_report("test", test_report)
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
