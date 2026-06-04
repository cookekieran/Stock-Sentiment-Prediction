"""Mine candidate fuzzy LTN rules for human economic review.

The script discovers simple candidate implications from train rows and scores
them on validation rows. It does not use the test split, so approved rules can
still be evaluated honestly later.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-packets-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-antecedents", type=int, default=3)
    parser.add_argument("--min-train-support", type=float, default=0.03)
    parser.add_argument("--min-validation-support", type=float, default=0.03)
    parser.add_argument("--min-validation-lift", type=float, default=1.05)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def sigmoid(values, scale: float = 1.0):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-scale * np.clip(values, -20.0, 20.0)))


def clamp01(values):
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def train_percentile_transform(train: pd.Series, values: pd.Series) -> np.ndarray:
    ordered = np.sort(pd.to_numeric(train, errors="coerce").dropna().to_numpy(float))
    if len(ordered) == 0:
        return np.zeros(len(values), dtype=float)
    raw = pd.to_numeric(values, errors="coerce").fillna(np.nanmedian(ordered)).to_numpy(float)
    return np.searchsorted(ordered, raw, side="right") / len(ordered)


def build_atoms(df: pd.DataFrame, train_reference: pd.DataFrame) -> dict[str, np.ndarray]:
    current = numeric(df, "price_trend_id").round().astype(int)
    revenue_yoy = numeric(df, "fundamental_revenue_yoy")
    margin_change = numeric(df, "fundamental_operating_margin_yoy_change")
    recent_return = numeric(df, "recent_return")
    recent_drawdown = numeric(df, "drawdown_from_recent_high")
    volatility = numeric(df, "realized_volatility_20d")
    quarter_age = numeric(df, "fundamental_quarter_age_days", default=999.0)
    has_statement = numeric(df, "fundamental_has_public_statement")

    volatility_percentile = train_percentile_transform(
        numeric(train_reference, "realized_volatility_20d"),
        volatility,
    )
    return {
        "high_ticker_relevance": clamp01(numeric(df, "weak_label_ticker_relevance")),
        "material_news": clamp01(numeric(df, "weak_label_materiality")),
        "risk_on_news": clamp01(numeric(df, "weak_label_risk_on_pressure")),
        "risk_off_news": clamp01(numeric(df, "weak_label_risk_off_pressure")),
        "high_uncertainty": clamp01(numeric(df, "weak_label_uncertainty")),
        "recent_public_earnings_release": clamp01(np.exp(-quarter_age / 10.0) * has_statement),
        "negative_revenue_growth": sigmoid(-revenue_yoy, scale=5.0),
        "positive_revenue_growth": sigmoid(revenue_yoy, scale=5.0),
        "margin_deterioration": sigmoid(-margin_change, scale=10.0),
        "margin_improvement": sigmoid(margin_change, scale=10.0),
        "volatility_spike": clamp01(volatility_percentile),
        "drawdown_stress": clamp01(-recent_drawdown * 5.0),
        "positive_price_momentum": clamp01(recent_return * 10.0),
        "negative_price_momentum": clamp01(-recent_return * 10.0),
        "current_bear_regime": (current == 0).astype(float).to_numpy(),
        "current_sideways_regime": (current == 1).astype(float).to_numpy(),
        "current_bull_regime": (current == 2).astype(float).to_numpy(),
    }


def build_consequents(df: pd.DataFrame) -> dict[str, np.ndarray]:
    current = numeric(df, "price_trend_id").round().astype(int).to_numpy()
    future = numeric(df, "future_price_trend_id").round().astype(int).to_numpy()
    transition = (current != future).astype(float)
    return {
        "transition": transition,
        "persistence": 1.0 - transition,
        "bear_destination": (future == 0).astype(float),
        "sideways_destination": (future == 1).astype(float),
        "bull_destination": (future == 2).astype(float),
        "bear_transition": ((transition == 1.0) & (future == 0)).astype(float),
        "bull_transition": ((transition == 1.0) & (future == 2)).astype(float),
        "sideways_transition": ((transition == 1.0) & (future == 1)).astype(float),
    }


def antecedent_truth(atoms: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    values = np.ones(len(next(iter(atoms.values()))), dtype=float)
    for name in names:
        values *= atoms[name]
    return values


def score_rule(antecedent: np.ndarray, consequent: np.ndarray) -> dict[str, float]:
    support = float(antecedent.mean())
    active_rate = float((antecedent >= 0.5).mean())
    weighted_hits = float((antecedent * consequent).sum())
    confidence = weighted_hits / float(antecedent.sum() + EPS)
    base_rate = float(consequent.mean())
    lift = confidence / float(base_rate + EPS)
    satisfaction = float((1.0 - antecedent + antecedent * consequent).mean())
    return {
        "support": support,
        "active_rate": active_rate,
        "confidence": confidence,
        "base_rate": base_rate,
        "lift": lift,
        "satisfaction": satisfaction,
        "violation": 1.0 - satisfaction,
    }


def bootstrap_confidence(
    antecedent: np.ndarray,
    consequent: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if repeats <= 0:
        return np.nan, np.nan
    values = []
    n = len(antecedent)
    for _ in range(repeats):
        indices = rng.integers(0, n, size=n)
        score = score_rule(antecedent[indices], consequent[indices])
        values.append(score["confidence"])
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def economic_hint(antecedent_names: tuple[str, ...], consequent_name: str) -> str:
    antecedents = set(antecedent_names)
    if "risk_off_news" in antecedents and "drawdown_stress" in antecedents:
        return "plausible risk-off/drawdown stress rule"
    if "recent_public_earnings_release" in antecedents and (
        "negative_revenue_growth" in antecedents or "margin_deterioration" in antecedents
    ):
        return "plausible negative earnings/fundamentals rule"
    if "recent_public_earnings_release" in antecedents and (
        "positive_revenue_growth" in antecedents or "margin_improvement" in antecedents
    ):
        return "plausible positive earnings/fundamentals rule"
    if "risk_on_news" in antecedents and consequent_name in {"bull_destination", "bull_transition"}:
        return "plausible risk-on destination rule"
    if "risk_off_news" in antecedents and consequent_name in {"bear_destination", "bear_transition"}:
        return "plausible risk-off destination rule"
    if "high_uncertainty" in antecedents and consequent_name == "sideways_destination":
        return "plausible uncertainty/sideways rule"
    if "material_news" not in antecedents and consequent_name == "transition":
        return "review carefully: transition rule without material-news gate"
    return "review economic plausibility"


def formula_text(antecedents: tuple[str, ...], consequent: str) -> str:
    lhs = " AND ".join(antecedents)
    return f"{lhs} => {consequent}"


def mine_rules(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    train_atoms = build_atoms(train, train)
    validation_atoms = build_atoms(validation, train)
    train_consequents = build_consequents(train)
    validation_consequents = build_consequents(validation)
    rows = []
    atom_names = sorted(train_atoms)
    for size in range(1, args.max_antecedents + 1):
        for antecedent_names in itertools.combinations(atom_names, size):
            train_antecedent = antecedent_truth(train_atoms, antecedent_names)
            validation_antecedent = antecedent_truth(validation_atoms, antecedent_names)
            for consequent_name in sorted(train_consequents):
                train_score = score_rule(train_antecedent, train_consequents[consequent_name])
                if train_score["support"] < args.min_train_support:
                    continue
                validation_score = score_rule(validation_antecedent, validation_consequents[consequent_name])
                if validation_score["support"] < args.min_validation_support:
                    continue
                if validation_score["lift"] < args.min_validation_lift:
                    continue
                boot_mean, boot_std = bootstrap_confidence(
                    validation_antecedent,
                    validation_consequents[consequent_name],
                    args.bootstrap_repeats,
                    rng,
                )
                rows.append(
                    {
                        "approved": "",
                        "rule_text": formula_text(antecedent_names, consequent_name),
                        "antecedents": " & ".join(antecedent_names),
                        "consequent": consequent_name,
                        "antecedent_count": size,
                        "train_support": train_score["support"],
                        "train_active_rate": train_score["active_rate"],
                        "train_confidence": train_score["confidence"],
                        "train_base_rate": train_score["base_rate"],
                        "train_lift": train_score["lift"],
                        "train_satisfaction": train_score["satisfaction"],
                        "validation_support": validation_score["support"],
                        "validation_active_rate": validation_score["active_rate"],
                        "validation_confidence": validation_score["confidence"],
                        "validation_base_rate": validation_score["base_rate"],
                        "validation_lift": validation_score["lift"],
                        "validation_satisfaction": validation_score["satisfaction"],
                        "validation_confidence_bootstrap_mean": boot_mean,
                        "validation_confidence_bootstrap_std": boot_std,
                        "economic_hint": economic_hint(antecedent_names, consequent_name),
                        "review_notes": "",
                    }
                )
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["ranking_score"] = (
        result["validation_lift"]
        * result["validation_confidence"]
        * np.sqrt(result["validation_support"])
        / (1.0 + result["validation_confidence_bootstrap_std"].fillna(0.0))
    )
    return result.sort_values(
        ["ranking_score", "validation_lift", "validation_confidence"],
        ascending=False,
    ).head(args.top_k)


def main() -> None:
    args = parse_args()
    packets = pd.read_parquet(args.daily_packets_path)
    required = {
        "split",
        "price_trend_id",
        "future_price_trend_id",
        "weak_label_ticker_relevance",
        "weak_label_materiality",
        "weak_label_risk_on_pressure",
        "weak_label_risk_off_pressure",
        "weak_label_uncertainty",
    }
    missing = sorted(required.difference(packets.columns))
    if missing:
        raise ValueError(f"Daily packet dataset is missing required columns: {missing}")
    train = packets[packets["split"] == "train"].copy()
    validation = packets[packets["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Both train and validation splits are required for rule mining.")
    candidates = mine_rules(train, validation, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "candidate_ltn_rules.csv"
    candidates.to_csv(output_path, index=False)
    metadata = {
        "rows": int(len(candidates)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "max_antecedents": args.max_antecedents,
        "min_train_support": args.min_train_support,
        "min_validation_support": args.min_validation_support,
        "min_validation_lift": args.min_validation_lift,
        "bootstrap_repeats": args.bootstrap_repeats,
        "note": "Rules are mined on train and scored on validation only. Do not mine or select rules using test rows.",
    }
    (args.output_dir / "candidate_ltn_rules_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("Candidate LTN rules mined")
    print(f"Rows: {len(candidates):,}")
    print(f"Output: {output_path}")
    if not candidates.empty:
        print("\nTop 10 candidates")
        print(
            candidates[
                [
                    "rule_text",
                    "validation_support",
                    "validation_confidence",
                    "validation_lift",
                    "economic_hint",
                ]
            ].head(10).to_string(index=False)
        )


if __name__ == "__main__":
    main()
