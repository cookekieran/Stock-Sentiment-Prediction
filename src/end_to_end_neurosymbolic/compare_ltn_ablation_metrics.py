"""Summarize end-to-end LTN ablation metrics from model output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def load_row(path: Path) -> dict[str, object]:
    metrics_path = path / "test_metrics.json"
    metadata_path = path / "training_metadata.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    detector = metrics["transition_detector"]
    final = metrics["final_regime"]
    return {
        "run": path.name,
        "logic_rule_set": metadata.get("logic_rule_set"),
        "logic_weight": metadata.get("loss_weights", {}).get("logic"),
        "approved_rule_count": metadata.get("approved_rule_count"),
        "best_epoch": metadata.get("best_epoch"),
        "selected_threshold": detector.get("threshold"),
        "transition_pr_auc": detector.get("pr_auc"),
        "transition_precision": detector.get("precision"),
        "transition_recall": detector.get("recall"),
        "transition_f1": detector.get("f1"),
        "final_accuracy": final.get("accuracy"),
        "macro_f1": final.get("macro_f1"),
        "transition_accuracy": final.get("transition_accuracy"),
        "stable_accuracy": final.get("stable_accuracy"),
        "predicted_transition_rows": detector.get("predicted_transition_rows"),
        "actual_transition_rows": detector.get("actual_transition_rows"),
    }


def main() -> None:
    args = parse_args()
    rows = [load_row(path) for path in sorted(args.output_root.iterdir()) if path.is_dir()]
    summary = pd.DataFrame(rows).sort_values(["logic_weight", "run"], na_position="first")
    summary_path = args.summary_path or args.output_root / "ltn_ablation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
