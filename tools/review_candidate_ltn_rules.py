from __future__ import annotations

import csv
import argparse
from pathlib import Path


DEFAULT_SOURCE = Path(r"C:\Users\kiera\Downloads\candidate_ltn_rules.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/end_to_end_neurosymbolic/rule_review")


APPROVE_EXACT = {
    "negative_price_momentum => bear_destination": "Strong bearish continuation rule: price momentum is directly aligned with bear destination.",
    "current_bear_regime AND negative_price_momentum => bear_destination": "Plausible persistence/continuation rule: bear regime plus negative momentum supports bear destination.",
    "negative_price_momentum AND volatility_spike => bear_destination": "Plausible stress rule: negative momentum with volatility spike supports bear destination.",
    "current_bear_regime AND negative_price_momentum AND volatility_spike => bear_destination": "Plausible high-conviction bearish continuation rule.",
    "drawdown_stress AND negative_price_momentum => bear_destination": "Plausible drawdown-continuation rule.",
    "negative_price_momentum AND negative_revenue_growth => bear_destination": "Good combined market/fundamental bearish rule.",
    "negative_price_momentum AND negative_revenue_growth AND volatility_spike => bear_destination": "Good high-stress bearish rule, although support should be watched.",
    "current_bull_regime AND negative_revenue_growth AND volatility_spike => bear_transition": "Economically plausible bull-to-bear warning rule; lower support but high lift.",
    "positive_price_momentum AND positive_revenue_growth => bull_destination": "Good combined market/fundamental bullish rule.",
    "positive_revenue_growth AND risk_on_news => bull_destination": "Plausible fundamentals plus positive news rule.",
    "positive_price_momentum AND risk_on_news => bull_destination": "Plausible market confirmation plus positive news rule.",
    "current_bull_regime AND risk_on_news => persistence": "Plausible bull-regime persistence rule when news is supportive.",
}


MAYBE_CONTAINS = [
    (
        "current_bear_regime AND positive_revenue_growth",
        "Possible recovery rule, but could be mean reversion. Keep only if paired with recent earnings/risk-on evidence.",
    ),
    (
        "current_bear_regime AND margin_improvement",
        "Possible recovery rule, but broad. Prefer a version gated by earnings release or risk-on news.",
    ),
    (
        "high_uncertainty AND negative_price_momentum",
        "Possible bearish risk rule, but uncertainty alone can be ambiguous.",
    ),
    (
        "risk_on_news => bull_destination",
        "Economically plausible but weak lift; useful as a soft rule, not a decisive one.",
    ),
    (
        "recent_public_earnings_release",
        "Too broad by itself. Useful only when combined with surprise/momentum/news direction.",
    ),
]


REJECT_PATTERNS = [
    ("current_bear_regime => bull_transition", "Reject as broad mean-reversion artifact: no catalyst or mechanism."),
    (
        "drawdown_stress => bull_transition",
        "Reject as broad mean-reversion artifact: drawdown alone does not imply bullish transition.",
    ),
    (
        "current_bear_regime AND drawdown_stress",
        "Reject or require extra catalyst: bear drawdown alone mostly encodes regime state.",
    ),
    ("high_ticker_relevance =>", "Reject: relevance says the article is on-topic, not directional."),
    (
        "current_bear_regime AND high_ticker_relevance",
        "Reject: ticker relevance is not an economic transition driver.",
    ),
]


INFORMATIVE_ANTECEDENTS = {
    "positive_revenue_growth",
    "negative_revenue_growth",
    "margin_improvement",
    "margin_deterioration",
    "positive_price_momentum",
    "negative_price_momentum",
    "volatility_spike",
    "risk_on_news",
    "risk_off_news",
    "high_uncertainty",
    "recent_public_earnings_release",
}


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def classify(row: dict[str, str]) -> tuple[str, str]:
    text = row["rule_text"]
    consequent = row["consequent"]
    support = as_float(row, "validation_support")
    lift = as_float(row, "validation_lift")
    bootstrap_std = as_float(row, "validation_confidence_bootstrap_std")

    if text in APPROVE_EXACT:
        return "KEEP", APPROVE_EXACT[text]

    for pattern, note in REJECT_PATTERNS:
        if pattern in text:
            return "REJECT", note

    antecedents = {part.strip() for part in row.get("antecedents", "").split("&") if part.strip()}
    if consequent.endswith("transition") and not (antecedents & INFORMATIVE_ANTECEDENTS):
        return "REJECT", "Reject: transition rule has no economic catalyst beyond current regime/state."

    for pattern, note in MAYBE_CONTAINS:
        if pattern in text:
            return "MAYBE", note

    if "high_ticker_relevance" in text:
        return "MAYBE", "Ticker relevance is useful as a data-quality gate, but it is not an economic driver."

    if consequent == "bear_destination" and "negative_price_momentum" in text and "positive_revenue_growth" in text:
        return "MAYBE", "Mixed signal: bearish price action but positive revenue growth. Test separately rather than keeping as a core rule."

    if consequent == "bear_destination" and "negative_price_momentum" in text and lift >= 1.5 and support >= 0.05:
        return "KEEP", "Plausible bearish destination rule with adequate validation support/lift."

    if (
        consequent == "bull_destination"
        and any(token in text for token in ["positive_price_momentum", "positive_revenue_growth", "risk_on_news"])
        and lift >= 1.08
        and support >= 0.05
    ):
        return "MAYBE", "Plausible bullish destination rule, but validate it against stability/false positives."

    if consequent == "persistence" and "current_bull_regime" in text and any(
        token in text for token in ["risk_on_news", "positive_price_momentum"]
    ):
        return "MAYBE", "Plausible persistence confirmation rule, not a transition detector."

    if lift < 1.1:
        return "REJECT", "Reject: validation lift is too weak to justify adding rule complexity."

    if bootstrap_std > 0.06:
        return "MAYBE", "Statistically noisy bootstrap estimate; inspect before using."

    return "REJECT", "Not selected: either weak economic mechanism, too broad, or redundant with better rules."


def format_rule(row: dict[str, str]) -> str:
    return (
        f"- `{row['rule_text']}` | "
        f"support={as_float(row, 'validation_support'):.3f}, "
        f"conf={as_float(row, 'validation_confidence'):.3f}, "
        f"lift={as_float(row, 'validation_lift'):.3f} | "
        f"{row['codex_review_note']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review mined candidate LTN rules and mark KEEP/MAYBE/REJECT.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reviewed_csv = args.output_dir / "candidate_ltn_rules_reviewed.csv"
    summary_md = args.output_dir / "recommended_ltn_rules.md"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        decision, note = classify(row)
        row["recommended_decision"] = decision
        row["codex_review_note"] = note

    fieldnames = list(rows[0].keys())
    for extra in ["recommended_decision", "codex_review_note"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    with reviewed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    keep = [row for row in rows if row["recommended_decision"] == "KEEP"]
    maybe = [row for row in rows if row["recommended_decision"] == "MAYBE"]
    reject = [row for row in rows if row["recommended_decision"] == "REJECT"]

    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write("# Recommended Candidate LTN Rules\n\n")
        handle.write("## Keep first\n")
        for row in keep[:20]:
            handle.write(format_rule(row) + "\n")
        handle.write("\n## Maybe / ablation candidates\n")
        for row in maybe[:30]:
            handle.write(format_rule(row) + "\n")
        handle.write("\n## Main rejection pattern\n")
        handle.write(
            "Reject broad current-regime-only or drawdown-only transition rules unless they are gated by "
            "price momentum, fundamentals, earnings timing, or directional news.\n"
        )

    print(f"Wrote {reviewed_csv}")
    print(f"Wrote {summary_md}")
    print(f"KEEP={len(keep)} MAYBE={len(maybe)} REJECT={len(reject)}")
    print("Top KEEP rules:")
    for row in keep[:12]:
        print(
            f"- {row['rule_text']} | "
            f"lift={as_float(row, 'validation_lift'):.3f} "
            f"support={as_float(row, 'validation_support'):.3f}"
        )


if __name__ == "__main__":
    main()
