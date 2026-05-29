"""
Explain a latent-state regime prediction using saved predictions and drivers.

The explanation combines:
- DeepSeek contextual drivers saved in the daily dataset/prediction rows
- GRU probability movement over the recent window
- LTN-style rule activation scores from contextual driver features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROB_COLUMNS = ["prob_bear_drawdown", "prob_sideways", "prob_bull_rally"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("models/deepseek_gru_with_rules/test_predictions.parquet"),
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date", required=True, help="Anchor trading date, e.g. 2026-04-15")
    parser.add_argument("--lookback-days", type=int, default=8)
    parser.add_argument(
        "--allow-missing-drivers",
        action="store_true",
        help="Print the explanation even if no contextual driver text was saved.",
    )
    return parser.parse_args()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def rule_activations(row: pd.Series) -> dict[str, float]:
    risk_off = clamp(row.get("driver_max_risk_off_shock", 0.0))
    risk_on = clamp(row.get("driver_max_risk_on_shock", 0.0))
    materiality = clamp(row.get("driver_mean_materiality", 0.0))
    novelty = clamp(row.get("driver_mean_novelty", 0.0))
    confidence = clamp(row.get("driver_mean_confidence", 0.0))
    abs_shock = clamp(row.get("driver_abs_shock_score", 0.0))
    conflict = (risk_off * risk_on) ** 0.5

    return {
        "persistent risk-off evidence should reduce bull confidence": risk_off * materiality * confidence,
        "risk-on evidence should reduce bear confidence": risk_on * materiality * confidence,
        "novel high-materiality news may justify a larger state update": novelty * materiality,
        "conflicting drivers should raise uncertainty/sideways probability": conflict,
        "low-impact news should preserve the previous state": 1.0 - abs_shock,
    }


def describe_probability_move(window: pd.DataFrame) -> str:
    first = window.iloc[0]
    last = window.iloc[-1]
    moves = {
        "bear_drawdown": last["prob_bear_drawdown"] - first["prob_bear_drawdown"],
        "sideways": last["prob_sideways"] - first["prob_sideways"],
        "bull_rally": last["prob_bull_rally"] - first["prob_bull_rally"],
    }
    ordered = sorted(moves.items(), key=lambda item: abs(item[1]), reverse=True)
    parts = [f"{label} {delta:+.3f}" for label, delta in ordered]
    return ", ".join(parts)


def driver_lines(row: pd.Series) -> list[str]:
    lines = []
    for idx in range(1, 4):
        label = row.get(f"top_driver_{idx}_label", "")
        if not isinstance(label, str) or not label.strip():
            continue
        pressure = row.get(f"top_driver_{idx}_pressure", "")
        summary = row.get(f"top_driver_{idx}_summary", "")
        evidence = row.get(f"top_driver_{idx}_evidence", "")
        line = f"{idx}. {label} [{pressure}]"
        if isinstance(summary, str) and summary.strip():
            line += f": {summary}"
        if isinstance(evidence, str) and evidence.strip():
            line += f" Evidence: {evidence}"
        lines.append(line)
    return lines


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    predictions["anchor_trading_date"] = pd.to_datetime(predictions["anchor_trading_date"]).dt.normalize()
    target_date = pd.Timestamp(args.date).normalize()
    ticker_rows = predictions[predictions["ticker"].astype(str).str.upper() == args.ticker.upper()]
    ticker_rows = ticker_rows.sort_values("anchor_trading_date")
    if ticker_rows.empty:
        raise ValueError(f"No predictions found for ticker {args.ticker}")

    eligible = ticker_rows[ticker_rows["anchor_trading_date"] <= target_date].tail(args.lookback_days)
    if eligible.empty:
        raise ValueError(f"No predictions found for {args.ticker} on/before {target_date.date()}")

    final = eligible.iloc[-1]
    print(f"Prediction explanation for {args.ticker.upper()} on {final['anchor_trading_date'].date()}")
    print()
    print("Model output")
    print(f"  predicted: {final['pred_label']} | actual: {final.get('true_label', '')}")
    print(
        "  probabilities: "
        f"bear={final['prob_bear_drawdown']:.3f}, "
        f"sideways={final['prob_sideways']:.3f}, "
        f"bull={final['prob_bull_rally']:.3f}"
    )
    print(f"  current regime: {final.get('current_price_trend_label', '')}")
    print(f"  recent probability movement: {describe_probability_move(eligible)}")
    print()

    drivers = driver_lines(final)
    if not drivers and not args.allow_missing_drivers:
        raise ValueError(
            "No contextual driver text was saved for this prediction row. "
            "For the explainable dissertation pipeline, rebuild the daily dataset "
            "with --contextual-drivers-path and train with --save-predictions."
        )
    print("DeepSeek contextual drivers")
    if drivers:
        for line in drivers:
            print(f"  {line}")
    else:
        print("  No contextual driver text was saved for this row.")
    print()

    print("LTN-style rule activation")
    activations = rule_activations(final)
    for name, score in sorted(activations.items(), key=lambda item: item[1], reverse=True):
        print(f"  {score:.3f}  {name}")
    print()
    print("Interpretation")
    top_rule, top_score = max(activations.items(), key=lambda item: item[1])
    print(
        "  The prediction is explained as a persistent regime estimate adjusted by "
        "recent DeepSeek contextual drivers. The strongest symbolic signal here is "
        f"'{top_rule}' with activation {top_score:.3f}."
    )


if __name__ == "__main__":
    main()
