"""Audit Qwen-GRU transition overrides against persistence and future regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import classification_report


LABEL_NAMES = ["bear_drawdown", "sideways", "bull_rally"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
PROBABILITY_COLUMNS = [
    "prob_bear_drawdown",
    "prob_sideways",
    "prob_bull_rally",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.21)
    parser.add_argument("--sample-size", type=int, default=25)
    return parser.parse_args()


def categorise(row: pd.Series) -> str:
    transition = bool(row["is_transition"])
    override = bool(row["accept_override"])
    correct = bool(row["hybrid_correct"])
    if override and transition and correct:
        return "true_positive_override"
    if override and not correct:
        return "false_positive_override"
    if transition and not override:
        return "missed_transition"
    if not transition and not override and correct:
        return "correct_persistence"
    return "other"


def mean_numeric(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else None


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.predictions_path).copy()
    probabilities = df[PROBABILITY_COLUMNS].to_numpy()
    current_ids = df["current_price_trend_label"].map(LABEL_TO_ID).to_numpy()
    truth_ids = pd.to_numeric(df["true_label_id"], errors="raise").to_numpy(np.int64)
    neural_ids = probabilities.argmax(axis=1)
    current_probabilities = probabilities[np.arange(len(df)), current_ids]
    alternatives = probabilities.copy()
    alternatives[np.arange(len(df)), current_ids] = -np.inf
    alternative_ids = alternatives.argmax(axis=1)
    alternative_probabilities = alternatives.max(axis=1)
    margins = alternative_probabilities - current_probabilities
    accept_override = margins >= args.threshold
    hybrid_ids = np.where(accept_override, alternative_ids, current_ids)

    df["raw_gru_label"] = [LABEL_NAMES[idx] for idx in neural_ids]
    df["hybrid_label"] = [LABEL_NAMES[idx] for idx in hybrid_ids]
    df["best_alternative_label"] = [LABEL_NAMES[idx] for idx in alternative_ids]
    df["current_regime_probability"] = current_probabilities
    df["best_alternative_probability"] = alternative_probabilities
    df["override_margin"] = margins
    df["accept_override"] = accept_override
    df["is_transition"] = current_ids != truth_ids
    df["hybrid_correct"] = hybrid_ids == truth_ids
    df["audit_category"] = df.apply(categorise, axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_dir / "audited_predictions.parquet", index=False)
    df.to_csv(args.output_dir / "audited_predictions.csv", index=False)

    summary_rows = []
    for category, group in df.groupby("audit_category", sort=True):
        summary_rows.append(
            {
                "audit_category": category,
                "rows": int(len(group)),
                "mean_override_margin": mean_numeric(group, "override_margin"),
                "mean_driver_update_strength": mean_numeric(group, "driver_update_strength"),
                "mean_driver_uncertainty": mean_numeric(group, "driver_uncertainty"),
                "mean_driver_materiality": mean_numeric(group, "driver_mean_materiality"),
                "mean_driver_shock": mean_numeric(group, "driver_mean_shock_score"),
            }
        )
        columns = [
            column
            for column in [
                "ticker",
                "anchor_trading_date",
                "current_price_trend_label",
                "future_price_trend_label",
                "raw_gru_label",
                "hybrid_label",
                "override_margin",
                "daily_market_narrative",
                "net_pressure",
                "driver_update_strength",
                "driver_uncertainty",
                "driver_mean_shock_score",
                "top_driver_1_label",
                "top_driver_1_summary",
                "top_driver_1_pressure",
                "top_driver_1_scope",
            ]
            if column in group.columns
        ]
        sample = group.sort_values("override_margin", ascending=False).head(args.sample_size)
        sample[columns].to_csv(args.output_dir / f"{category}_sample.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("audit_category")
    summary.to_csv(args.output_dir / "category_summary.csv", index=False)

    hybrid_report = classification_report(truth_ids, hybrid_ids)
    transition = current_ids != truth_ids
    stable = ~transition
    metrics = {
        "threshold": args.threshold,
        "rows": int(len(df)),
        "accepted_overrides": int(accept_override.sum()),
        "accuracy": hybrid_report["accuracy"],
        "macro_f1": hybrid_report["macro_f1"],
        "transition_rows": int(transition.sum()),
        "transition_accuracy": float((hybrid_ids[transition] == truth_ids[transition]).mean()),
        "stable_accuracy": float((hybrid_ids[stable] == truth_ids[stable]).mean()),
        "category_counts": df["audit_category"].value_counts().to_dict(),
    }
    (args.output_dir / "audit_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print("\nCategory summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
