"""Run the matched, multi-seed final walk-forward study and summarize results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


TRAINED_VARIANTS = {
    "price_only": {"feature_set": "price_only", "logic_rule_set": "none", "logic_weight": 0.0},
    "price_fundamentals": {
        "feature_set": "price_fundamentals",
        "logic_rule_set": "none",
        "logic_weight": 0.0,
    },
    "price_qwen_fundamentals": {
        "feature_set": "price_qwen_fundamentals",
        "logic_rule_set": "none",
        "logic_weight": 0.0,
    },
    "price_qwen_fundamentals_ltn": {
        "feature_set": "price_qwen_fundamentals",
        "logic_rule_set": "all",
        "logic_weight": 0.1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/final_study"))
    parser.add_argument("--output-root", type=Path, default=Path("models/final_study"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    return parser.parse_args()


def persistence_row(fold: str, seed: int, daily: pd.DataFrame) -> dict[str, object]:
    test = daily[daily["split"] == "test"].copy()
    truth_change = test["future_regime_change"].astype(int).to_numpy()
    probabilities = np.zeros(len(test), dtype=float)
    predicted_regime = test["price_trend_id"].astype(int).to_numpy()
    true_regime = test["future_regime_id"].astype(int).to_numpy()
    return {
        "fold": fold,
        "seed": seed,
        "variant": "persistence",
        "transition_pr_auc": float(average_precision_score(truth_change, probabilities)),
        "transition_f1": 0.0,
        "final_accuracy": float((predicted_regime == true_regime).mean()),
        "final_macro_f1": float(f1_score(true_regime, predicted_regime, average="macro", labels=[0, 1, 2])),
    }


def trained_row(fold: str, seed: int, variant: str, metrics_path: Path) -> dict[str, object]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "fold": fold,
        "seed": seed,
        "variant": variant,
        "transition_pr_auc": metrics["transition_detector"]["pr_auc"],
        "transition_f1": metrics["transition_detector"]["f1"],
        "final_accuracy": metrics["final_regime"]["accuracy"],
        "final_macro_f1": metrics["final_regime"]["macro_f1"],
    }


def main() -> None:
    args = parse_args()
    rows = []
    for fold_dir in sorted(args.data_root.glob("fold_*")):
        fold = fold_dir.name
        daily_path = fold_dir / "daily.parquet"
        schema_path = fold_dir / "schema.json"
        daily = pd.read_parquet(daily_path)
        for seed in args.seeds:
            rows.append(persistence_row(fold, seed, daily))
            for variant, config in TRAINED_VARIANTS.items():
                output_dir = args.output_root / fold / variant / f"seed_{seed}"
                metrics_path = output_dir / "test_metrics.json"
                if not metrics_path.exists():
                    command = [
                        sys.executable,
                        "src/models/train_two_stage_transition_gru.py",
                        "--daily-path",
                        str(daily_path),
                        "--schema-path",
                        str(schema_path),
                        "--output-dir",
                        str(output_dir),
                        "--feature-set",
                        config["feature_set"],
                        "--logic-rule-set",
                        config["logic_rule_set"],
                        "--logic-weight",
                        str(config["logic_weight"]),
                        "--selection-metric",
                        "transition_pr_auc",
                        "--sequence-length",
                        "10",
                        "--epochs",
                        str(args.epochs),
                        "--early-stopping-patience",
                        "4",
                        "--seed",
                        str(seed),
                        "--batch-size",
                        str(args.batch_size),
                        "--hidden-size",
                        str(args.hidden_size),
                        "--dropout",
                        "0.35",
                        "--learning-rate",
                        "0.0003",
                        "--minimum-stable-accuracy",
                        "0.70",
                        "--save-predictions",
                    ]
                    subprocess.run(command, check=True)
                rows.append(trained_row(fold, seed, variant, metrics_path))
    results = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_root / "all_results.csv", index=False)
    summary = (
        results.groupby("variant")
        .agg(
            transition_pr_auc_mean=("transition_pr_auc", "mean"),
            transition_pr_auc_std=("transition_pr_auc", "std"),
            transition_f1_mean=("transition_f1", "mean"),
            final_accuracy_mean=("final_accuracy", "mean"),
            final_macro_f1_mean=("final_macro_f1", "mean"),
        )
        .sort_values("transition_pr_auc_mean", ascending=False)
    )
    summary.to_csv(args.output_root / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
