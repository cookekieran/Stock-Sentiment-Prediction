"""Compare article-level and latent-state experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("models/deepseek_baseline/test_metrics.json"),
    )
    parser.add_argument(
        "--ltn-metrics",
        type=Path,
        default=Path("models/deepseek_ltn/test_metrics.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/comparison/model_comparison.json"),
    )
    parser.add_argument(
        "--latent-state-metrics",
        type=Path,
        default=Path("models/latent_state_gru/test_metrics.json"),
    )
    return parser.parse_args()


def read_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    baseline = read_metrics(args.baseline_metrics)
    ltn = read_metrics(args.ltn_metrics)
    latent = read_metrics(args.latent_state_metrics) if args.latent_state_metrics.exists() else None

    comparison = {
        "deepseek_baseline": {
            "accuracy": baseline["accuracy"],
            "macro_f1": baseline["macro_f1"],
        },
        "deepseek_ltn": {
            "accuracy": ltn["accuracy"],
            "macro_f1": ltn["macro_f1"],
        },
        "deepseek_ltn_delta": {
            "accuracy": ltn["accuracy"] - baseline["accuracy"],
            "macro_f1": ltn["macro_f1"] - baseline["macro_f1"],
        },
        "baseline_per_class": baseline.get("per_class", {}),
        "ltn_per_class": ltn.get("per_class", {}),
    }
    if latent:
        comparison["latent_state_gru"] = {
            "accuracy": latent["accuracy"],
            "macro_f1": latent["macro_f1"],
        }
        comparison["latent_state_delta_vs_deepseek_ltn"] = {
            "accuracy": latent["accuracy"] - ltn["accuracy"],
            "macro_f1": latent["macro_f1"] - ltn["macro_f1"],
        }
        comparison["latent_state_per_class"] = latent.get("per_class", {})

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("Experiment comparison")
    print(f"baseline accuracy: {baseline['accuracy']:.4f}")
    print(f"ltn accuracy     : {ltn['accuracy']:.4f}")
    print(f"delta accuracy   : {comparison['deepseek_ltn_delta']['accuracy']:.4f}")
    print(f"baseline macro F1: {baseline['macro_f1']:.4f}")
    print(f"ltn macro F1     : {ltn['macro_f1']:.4f}")
    print(f"delta macro F1   : {comparison['deepseek_ltn_delta']['macro_f1']:.4f}")
    if latent:
        print(f"latent accuracy  : {latent['accuracy']:.4f}")
        print(f"latent macro F1  : {latent['macro_f1']:.4f}")
    else:
        print(f"latent metrics not found: {args.latent_state_metrics}")


if __name__ == "__main__":
    main()
