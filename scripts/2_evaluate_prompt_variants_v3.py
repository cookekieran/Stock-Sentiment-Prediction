"""
Evaluates 0-shot through 6-shot variants of the MAG7 market-reaction
relevance prompt against a manually labelled audit set. 

A headline counts as relevant only if it plausibly describes a firm-specific
event that could move the target MAG7 ticker. 
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

COMPANY_NAMES = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet / Google",
    "META": "Meta Platforms / Facebook / Instagram / WhatsApp",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
}

PROMPT_VARIANTS = [
    "zero_shot",
    "one_shot",
    "two_shot",
    "three_shot",
    "four_shot",
    "five_shot",
    "six_shot",
]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--audit-csv", default="mag7_headline_relevance_hard_audit_420_relabelled_clean.csv")
    parser.add_argument("--out-dir", default="outputs/prompt_eval_market_reaction_relevance")

    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        default=PROMPT_VARIANTS,
        choices=PROMPT_VARIANTS,
    )

    return parser.parse_args()


def setup(out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=out / "prompt_eval.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    load_dotenv(REPO_ROOT / ".env")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"HF_TOKEN was not found. Add it to {REPO_ROOT / '.env'} or the environment.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this in the GPU container.")

    return token


def normalise(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", normalise(value).lower())


def load_audit(path: Path, max_rows: int | None) -> pd.DataFrame:
    audit = pd.read_csv(path)

    required = {"audit_id", "ticker", "title", "manual_label", "split"}
    missing = required - set(audit.columns)
    if missing:
        raise RuntimeError(f"Audit CSV is missing required columns: {sorted(missing)}")

    audit["ticker"] = audit["ticker"].astype(str).str.upper().str.strip()
    audit = audit[audit["ticker"].isin(TICKERS)].copy()

    audit["manual_label_clean"] = audit["manual_label"].map(canonical)
    audit["manual_keep"] = audit["manual_label_clean"].eq("relevant")

    bad_labels = sorted(set(audit["manual_label_clean"]) - {"relevant", "not relevant"})
    if bad_labels:
        raise RuntimeError(f"Unexpected manual labels: {bad_labels}")

    if max_rows is not None:
        audit = audit.head(max_rows).copy()

    return audit.reset_index(drop=True)


def output_schema() -> str:
    return """
Return only valid JSON with exactly these keys:
{
  "keep": true or false,
  "rejection_reason": "not_rejected" or "stock_or_analyst_commentary" or "investor_or_fund_activity" or "minor_or_routine_product_update" or "award_or_ranking" or "another_company_event" or "weak_indirect_business_link" or "third_party_platform_activity" or "former_employee_or_alumnus" or "broad_market_or_sector_news" or "consumer_advice_or_deal" or "unclear_or_not_material",
  "confidence": "high" or "medium" or "low",
  "rationale": one short sentence
}
""".strip()


def market_reaction_rules() -> str:
    return """
Task:
Decide whether the headline is Relevant or Not Relevant for a MAG7 stock-market event dataset.

Definition of Relevant:
Relevant means the headline describes a firm-specific event for the target company that could plausibly cause a market reaction in the target ticker.

Keep only if the headline itself clearly describes one of these:
- earnings, revenue, profit, margin, sales, deliveries, demand, guidance, or financial outlook for the target company;
- lawsuit, investigation, fine, settlement, court ruling, regulatory approval/rejection, antitrust, privacy, labour, export-control, recall, or safety issue directly involving the target company;
- major acquisition, divestiture, asset sale, financing, capex plan, data-center buildout, factory expansion, production increase, production cut, restructuring, or layoffs by the target company;
- major product strategy shift, major product launch, major product delay, major platform launch, major robotaxi/autonomy launch, or major AI/cloud infrastructure launch by the target company;
- major customer contract, major partnership, major content-rights deal, major energy deal, or major infrastructure deal directly involving the target company;
- direct supply-chain or production risk affecting the target company's important products;
- direct competitive or demand threat that is specific enough to affect the target company, such as a major customer choosing a competitor's chips instead of Nvidia's;
- direct target-company executive/governance event likely to matter to investors.

Reject if the headline is mainly:
- stock price movement, stock performance, valuation, technical analysis, analyst rating, price target, investment advice, or pre-earnings speculation;
- a fund, ETF, investor, insider, pension fund, or wealth manager buying, selling, holding, or commenting on shares;
- a routine product refresh, minor feature update, app update, product review, consumer deal, product comparison, rumour, teaser, leak, or speculative launch detail;
- customer satisfaction award, brand ranking, product award, listicle, or consumer advice;
- another company's earnings, lawsuit, product, partnership, strategy, sales, or stock movement, including when the headline is mainly about another MAG7 company (reject even if the row's target ticker is different from the company the headline is actually about);
- a third-party seller, app developer, advertiser, platform user, or criminal using the target company's platform or products, unless the headline says the target company itself is sued, fined, investigated, sanctioned, or directly financially liable;
- broad macro, sector, AI boom, cloud, semiconductor, EV, advertising, retail, or data-center news without a specific market-relevant event for the target company;
- a weak indirect business link where the target company is only mentioned as a customer, supplier, platform, cloud provider, ad business, AI model, chip, benchmark, comparison, or possible beneficiary;
- a former employee, former executive, alumnus, or ex-board member acting outside the target company.

Important:
- Use headline-only evidence.
- Do not assume missing details.
- Do not keep merely because the target company is named.
- Do not keep merely because the headline is about the target company's product ecosystem.
- If a direct company action looks routine or minor, return keep=false.
- If unsure, return keep=false.
- Be conservative: false positives are worse than false negatives.
""".strip()


def all_examples() -> list[str]:
    return [
        """
Example 1:
Target ticker: AAPL
Target company: Apple
Headline: Apple sees sales slump continuing, shares drop 2% despite beating sales expectations

Correct output:
{
  "keep": true,
  "rejection_reason": "not_rejected",
  "confidence": "high",
  "rationale": "The headline reports Apple sales weakness and earnings-related information likely to affect the stock."
}
""".strip(),
        """
Example 2:
Target ticker: AAPL
Target company: Apple
Headline: Apple launches new Vision Pro with M5 chip and Dual Knit Band, starting at $3,499

Correct output:
{
  "keep": false,
  "rejection_reason": "minor_or_routine_product_update",
  "confidence": "high",
  "rationale": "The headline describes a product refresh/detail rather than a clearly market-moving Apple event."
}
""".strip(),
        """
Example 3:
Target ticker: AMZN
Target company: Amazon
Headline: EPAM Signs Strategic Collaboration Agreement with AWS to Help Organizations Become Cloud-Native

Correct output:
{
  "keep": false,
  "rejection_reason": "weak_indirect_business_link",
  "confidence": "high",
  "rationale": "The headline is mainly about EPAM using or partnering with AWS and does not clearly indicate a market-moving Amazon event."
}
""".strip(),
        """
Example 4:
Target ticker: NVDA
Target company: Nvidia
Headline: Oracle To Deploy 50,000 AMD AI Chips In A Sign Of Growing Competition For Nvidia

Correct output:
{
  "keep": true,
  "rejection_reason": "not_rejected",
  "confidence": "high",
  "rationale": "The headline describes a specific competitive demand threat to Nvidia from a major customer choosing AMD chips."
}
""".strip(),
        """
Example 5:
Target ticker: TSLA
Target company: Tesla
Headline: Tesla recalls more than 11,000 Cybertrucks for issues with trunk bed, windshield wiper

Correct output:
{
  "keep": true,
  "rejection_reason": "not_rejected",
  "confidence": "high",
  "rationale": "A Tesla vehicle recall is a direct product safety/regulatory event likely to matter to investors."
}
""".strip(),
        """
Example 6:
Target ticker: AAPL
Target company: Apple
Headline: APPLE INC : Jefferies reiterates its Sell rating - MarketScreener

Correct output:
{
  "keep": false,
  "rejection_reason": "stock_or_analyst_commentary",
  "confidence": "high",
  "rationale": "An analyst rating is market commentary rather than a firm-specific Apple event."
}
""".strip(),
    ]


def shot_count(variant: str) -> int:
    return {
        "zero_shot": 0,
        "one_shot": 1,
        "two_shot": 2,
        "three_shot": 3,
        "four_shot": 4,
        "five_shot": 5,
        "six_shot": 6,
    }[variant]


def few_shot_examples(variant: str) -> str:
    n = shot_count(variant)
    if n == 0:
        return ""
    return "\n\n".join(all_examples()[:n])


def build_prompt(row: pd.Series, variant: str) -> str:
    ticker = row["ticker"]
    company_name = COMPANY_NAMES[ticker]

    title = normalise(row.get("title"))
    source = normalise(row.get("source"))
    published = normalise(row.get("time_published"))

    examples = few_shot_examples(variant)
    examples_block = f"\n\nExamples:\n\n{examples}\n" if examples else ""

    return f"""
You are a conservative financial-news relevance judge.

Target ticker: {ticker}
Target company: {company_name}

Source: {source}
Published time UTC: {published}
Headline: {title}

{market_reaction_rules()}
{examples_block}

Now classify the actual headline.

{output_schema()}
""".strip()


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found.")

    return json.loads(match.group(0))


def parse_model_output(raw: str) -> dict[str, Any]:
    base = {
        "parse_ok": False,
        "pred_keep": False,
        "rejection_reason": "unclear_or_not_material",
        "confidence": "low",
        "rationale": "",
    }

    allowed_reasons = {
        "not_rejected",
        "stock_or_analyst_commentary",
        "investor_or_fund_activity",
        "minor_or_routine_product_update",
        "award_or_ranking",
        "another_company_event",
        "weak_indirect_business_link",
        "third_party_platform_activity",
        "former_employee_or_alumnus",
        "broad_market_or_sector_news",
        "consumer_advice_or_deal",
        "unclear_or_not_material",
    }

    try:
        parsed = parse_json(raw)
    except Exception as exc:
        base["rationale"] = f"parse_error: {exc}"
        return base

    rejection_reason = canonical(parsed.get("rejection_reason"))
    if rejection_reason not in allowed_reasons:
        rejection_reason = "unclear_or_not_material"

    confidence = canonical(parsed.get("confidence"))
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    pred_keep = bool(parsed.get("keep"))
    if pred_keep:
        rejection_reason = "not_rejected"

    base.update(
        {
            "parse_ok": True,
            "pred_keep": pred_keep,
            "rejection_reason": rejection_reason,
            "confidence": confidence,
            "rationale": normalise(parsed.get("rationale")),
        }
    )

    return base


def run_variant(
    audit: pd.DataFrame,
    variant: str,
    out: Path,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    batch_size: int,
) -> pd.DataFrame:
    output_path = out / f"predictions_{variant}.csv"

    if output_path.exists():
        logging.info("Loading cached predictions for %s", variant)
        return pd.read_csv(output_path)

    logging.info("Running prompt variant: %s", variant)

    records = []

    for start_idx in range(0, len(audit), batch_size):
        batch = audit.iloc[start_idx : start_idx + batch_size].copy()

        messages = [
            [{"role": "user", "content": build_prompt(row, variant)}]
            for _, row in batch.iterrows()
        ]

        texts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = encoded["input_ids"].shape[1]
        generated = output[:, prompt_length:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for (_, row), raw in zip(batch.iterrows(), decoded):
            parsed = parse_model_output(raw)

            record = {
                "prompt_variant": variant,
                "n_shot": shot_count(variant),
                "audit_id": row["audit_id"],
                "ticker": row["ticker"],
                "split": row["split"],
                "time_published": row.get("time_published", ""),
                "title": row["title"],
                "source": row.get("source", ""),
                "url": row.get("url", ""),
                "article_uid": row.get("article_uid", ""),
                "manual_label": row["manual_label"],
                "manual_keep": bool(row["manual_keep"]),
                "raw_response": raw,
            }
            record.update(parsed)
            records.append(record)

        logging.info("%s: processed %d / %d", variant, min(start_idx + batch_size, len(audit)), len(audit))

    predictions = pd.DataFrame(records)
    predictions.to_csv(output_path, index=False)

    return predictions


def binary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["manual_keep"].astype(bool)
    y_pred = frame["pred_keep"].astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(frame), 1)

    return {
        "n": int(len(frame)),
        "manual_relevant": int(y_true.sum()),
        "pred_relevant": int(y_pred.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "parse_failure_count": int((~frame["parse_ok"].astype(bool)).sum()),
    }


def bootstrap_binary_metrics_ci(frame: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    """
    Bootstrap 95% CIs for precision/recall/F1/accuracy by resampling rows
    with replacement.

    Each audit row is an independent headline judgment (unlike the
    return-prediction pipeline, where multiple articles can share one
    realised-return outcome), so plain row-level resampling is appropriate
    here.
    """
    y_true = frame["manual_keep"].astype(bool).to_numpy()
    y_pred = frame["pred_keep"].astype(bool).to_numpy()
    n = len(frame)

    rng = np.random.default_rng(seed)
    boot_precision = np.empty(n_boot)
    boot_recall = np.empty(n_boot)
    boot_f1 = np.empty(n_boot)
    boot_accuracy = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]

        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        tn = int((~yt & ~yp).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        boot_precision[b] = precision
        boot_recall[b] = recall
        boot_f1[b] = 2 * precision * recall / max(precision + recall, 1e-12)
        boot_accuracy[b] = (tp + tn) / n

    def ci(arr: np.ndarray) -> tuple[float, float]:
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    precision_lo, precision_hi = ci(boot_precision)
    recall_lo, recall_hi = ci(boot_recall)
    f1_lo, f1_hi = ci(boot_f1)
    accuracy_lo, accuracy_hi = ci(boot_accuracy)

    return {
        "precision_ci_low": round(precision_lo, 4),
        "precision_ci_high": round(precision_hi, 4),
        "recall_ci_low": round(recall_lo, 4),
        "recall_ci_high": round(recall_hi, 4),
        "f1_ci_low": round(f1_lo, 4),
        "f1_ci_high": round(f1_hi, 4),
        "accuracy_ci_low": round(accuracy_lo, 4),
        "accuracy_ci_high": round(accuracy_hi, 4),
    }


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for variant, variant_frame in predictions.groupby("prompt_variant"):
        for split_name in ["all", "development", "held_out_test"]:
            if split_name == "all":
                split_frame = variant_frame
            else:
                split_frame = variant_frame[variant_frame["split"] == split_name]

            if split_frame.empty:
                continue

            row = {
                "prompt_variant": variant,
                "n_shot": shot_count(variant),
                "split": split_name,
            }
            row.update(binary_metrics(split_frame))
            row.update(bootstrap_binary_metrics_ci(split_frame))
            rows.append(row)

    return pd.DataFrame(rows)


def write_error_files(predictions: pd.DataFrame, out: Path) -> None:
    cols = [
        "audit_id",
        "ticker",
        "split",
        "manual_label",
        "pred_keep",
        "rejection_reason",
        "confidence",
        "title",
        "rationale",
        "url",
    ]

    for variant, frame in predictions.groupby("prompt_variant"):
        false_positives = frame[(~frame["manual_keep"].astype(bool)) & (frame["pred_keep"].astype(bool))]
        false_negatives = frame[(frame["manual_keep"].astype(bool)) & (~frame["pred_keep"].astype(bool))]

        false_positives[cols].to_csv(out / f"false_positives_{variant}.csv", index=False)
        false_negatives[cols].to_csv(out / f"false_negatives_{variant}.csv", index=False)


def choose_best_prompt(metrics: pd.DataFrame) -> dict[str, Any]:
    development = metrics[metrics["split"] == "development"].copy()

    if development.empty:
        development = metrics[metrics["split"] == "all"].copy()

    development = development.sort_values(
        ["f1", "precision", "recall", "accuracy"],
        ascending=[False, False, False, False],
    )

    return development.iloc[0].to_dict()


def main() -> None:
    run_args = args()
    out = Path(run_args.out_dir)

    token = setup(out)

    audit = load_audit(Path(run_args.audit_csv), run_args.max_rows)

    logging.info("Loaded audit rows: %d", len(audit))
    logging.info("Manual labels:\n%s", audit["manual_label_clean"].value_counts().to_string())
    logging.info("Splits:\n%s", audit["split"].value_counts().to_string())

    tokenizer = AutoTokenizer.from_pretrained(
        run_args.model_id,
        revision=run_args.model_revision,
        token=token,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        run_args.model_id,
        revision=run_args.model_revision,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_predictions = []

    for variant in run_args.prompt_variants:
        predictions = run_variant(
            audit=audit,
            variant=variant,
            out=out,
            tokenizer=tokenizer,
            model=model,
            batch_size=run_args.batch_size,
        )
        all_predictions.append(predictions)

    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_csv(out / "all_prompt_predictions.csv", index=False)

    metrics = evaluate_predictions(combined)
    metrics = metrics.sort_values(["split", "f1", "precision"], ascending=[True, False, False])
    metrics.to_csv(out / "prompt_metrics.csv", index=False)

    write_error_files(combined, out)

    best = choose_best_prompt(metrics)

    summary = {
        "best_prompt_by_development_f1": best,
        "prompt_variants": run_args.prompt_variants,
        "audit_csv": str(run_args.audit_csv),
        "model_id": run_args.model_id,
        "model_revision": run_args.model_revision,
        "selection_rule": "Highest development F1, then precision, then recall, then accuracy.",
        "outputs": {
            "prompt_metrics": "prompt_metrics.csv",
            "all_prompt_predictions": "all_prompt_predictions.csv",
            "false_positive_files": "false_positives_<prompt_variant>.csv",
            "false_negative_files": "false_negatives_<prompt_variant>.csv",
        },
    }

    (out / "best_prompt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPrompt metrics:")
    print(metrics.to_string(index=False))

    print("\nBest prompt selected on development split:")
    print(json.dumps(best, indent=2))

    development_metrics = metrics[metrics["split"] == "development"].copy()
    ranked_split = development_metrics if not development_metrics.empty else metrics[metrics["split"] == "all"].copy()
    top3 = ranked_split.sort_values("f1", ascending=False).head(3)

    print("\nTop 3 variants by F1, with bootstrap 95% CIs:")
    print(
        top3[
            ["prompt_variant", "n", "f1", "f1_ci_low", "f1_ci_high", "precision", "recall"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()