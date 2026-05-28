"""Shared utilities for DeepSeek baseline and DeepSeek+LTN experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SRC_DIR / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.append(str(DATA_DIR))

from ltn_schema import STRUCTURED_FEATURE_COLUMNS, TARGET_COLUMN, validate_dataset_columns


def load_split(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_dataset_columns(list(df.columns))
    if max_rows:
        df = df.head(max_rows).copy()
    return df


def available_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in STRUCTURED_FEATURE_COLUMNS if column in df.columns]


def fit_feature_stats(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
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
        values = (values - stats[column]["mean"]) / stats[column]["std"]
        arrays.append(values.to_numpy(dtype=np.float32))
    if not arrays:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.stack(arrays, axis=1)


def labels(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df[TARGET_COLUMN], errors="raise").to_numpy(dtype=np.int64)


def class_weights(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    labels_list = ["bear_drawdown", "sideways", "bull_rally"]
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        confusion[int(truth), int(pred)] += 1

    per_class = {}
    f1s = []
    for idx, name in enumerate(labels_list):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[idx, :].sum()),
        }

    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
