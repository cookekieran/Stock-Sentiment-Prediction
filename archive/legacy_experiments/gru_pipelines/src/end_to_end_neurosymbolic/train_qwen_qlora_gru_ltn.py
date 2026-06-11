"""Train an end-to-end Qwen QLoRA, GRU, and quantified fuzzy-LTN model."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PREDICATE_NAMES = [
    "ticker_relevance",
    "materiality",
    "risk_on_pressure",
    "risk_off_pressure",
    "uncertainty",
]
WEAK_LABEL_COLUMNS = [f"weak_label_{name}" for name in PREDICATE_NAMES]
PRICE_COLUMNS = [
    "realized_volatility_20d",
    "rally_from_previous_low",
    "drawdown_from_previous_high",
    "recent_return",
    "drawdown_from_recent_high",
    "price_trend_id",
]
FUNDAMENTAL_COLUMNS = [
    "fundamental_gross_margin",
    "fundamental_operating_margin",
    "fundamental_net_margin",
    "fundamental_ebitda_margin",
    "fundamental_current_ratio",
    "fundamental_liabilities_to_assets",
    "fundamental_debt_to_equity",
    "fundamental_cash_to_assets",
    "fundamental_revenue_yoy",
    "fundamental_gross_profit_yoy",
    "fundamental_operating_income_yoy",
    "fundamental_net_income_yoy",
    "fundamental_ebitda_yoy",
    "fundamental_assets_yoy",
    "fundamental_cash_yoy",
    "fundamental_inventory_yoy",
    "fundamental_receivables_yoy",
    "fundamental_debt_yoy",
    "fundamental_gross_margin_yoy_change",
    "fundamental_operating_margin_yoy_change",
    "fundamental_net_margin_yoy_change",
    "fundamental_ebitda_margin_yoy_change",
    "fundamental_quarter_age_days",
    "fundamental_has_public_statement",
]
TABULAR_COLUMNS = [*PRICE_COLUMNS, *FUNDAMENTAL_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-packets-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--tabular-hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--predicate-loss-weight", type=float, default=0.5)
    parser.add_argument("--logic-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--logic-rule-set",
        choices=["original", "mined_keep"],
        default="original",
        help="Use the original hand-written formulas or mined rules marked KEEP in --approved-rules-path.",
    )
    parser.add_argument(
        "--approved-rules-path",
        type=Path,
        default=None,
        help="Reviewed candidate_ltn_rules CSV containing recommended_decision=KEEP rows.",
    )
    parser.add_argument("--destination-loss-weight", type=float, default=1.0)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-validation-sequences", type=int, default=None)
    parser.add_argument("--max-test-sequences", type=int, default=None)
    parser.add_argument("--minimum-stable-accuracy", type=float, default=0.70)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Daily packet dataset is missing required columns: {missing}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_tabular_stats(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats = {}
    for column in TABULAR_COLUMNS:
        values = pd.to_numeric(train[column], errors="coerce")
        mean = float(values.mean()) if values.notna().any() else 0.0
        std = float(values.std()) if values.notna().any() else 1.0
        stats[column] = {"mean": mean, "std": std if np.isfinite(std) and std > 1e-8 else 1.0}
    return stats


def transform_tabular(df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> np.ndarray:
    arrays = []
    for column in TABULAR_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce").fillna(stats[column]["mean"])
        arrays.append(((values - stats[column]["mean"]) / stats[column]["std"]).to_numpy(np.float32))
    return np.stack(arrays, axis=1)


class PacketSequenceDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        stats: dict[str, dict[str, float]],
        sequence_length: int,
        max_sequences: int | None,
        seed: int,
    ):
        import torch

        self.df = df.sort_values(["ticker", "anchor_trading_date"]).reset_index(drop=True)
        self.packet_texts = self.df["packet_text"].fillna("").astype(str).tolist()
        self.raw_tabular = torch.tensor(
            self.df[TABULAR_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32),
            dtype=torch.float32,
        )
        self.tabular = torch.tensor(transform_tabular(self.df, stats), dtype=torch.float32)
        self.predicates = torch.tensor(
            self.df[WEAK_LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32),
            dtype=torch.float32,
        )
        self.current = torch.tensor(
            pd.to_numeric(self.df["price_trend_id"], errors="raise").to_numpy(np.int64),
            dtype=torch.long,
        )
        self.future = torch.tensor(
            pd.to_numeric(self.df["future_price_trend_id"], errors="raise").to_numpy(np.int64),
            dtype=torch.long,
        )
        self.indices: list[tuple[int, int]] = []
        for _, group in self.df.groupby("ticker", sort=False):
            positions = group.index.to_numpy()
            for end_offset in range(sequence_length, len(positions) + 1):
                window = positions[end_offset - sequence_length : end_offset]
                self.indices.append((int(window[0]), int(window[-1]) + 1))
        if max_sequences is not None and len(self.indices) > max_sequences:
            self.indices = self.stratified_sample_indices(max_sequences, seed)
        if not self.indices:
            raise ValueError("No sequences created. Reduce --sequence-length.")

    def __len__(self) -> int:
        return len(self.indices)

    def stratified_sample_indices(self, max_sequences: int, seed: int) -> list[tuple[int, int]]:
        rng = np.random.default_rng(seed)
        strata: dict[tuple[str, bool, int], list[tuple[int, int]]] = {}
        for start, end in self.indices:
            current = int(self.current[end - 1])
            future = int(self.future[end - 1])
            ticker = str(self.df.iloc[end - 1]["ticker"])
            strata.setdefault((ticker, current != future, future), []).append((start, end))
        sampled = []
        for values in strata.values():
            count = max(1, int(round(max_sequences * len(values) / len(self.indices))))
            selected = rng.choice(len(values), size=min(count, len(values)), replace=False)
            sampled.extend(values[index] for index in selected)
        if len(sampled) > max_sequences:
            selected = rng.choice(len(sampled), size=max_sequences, replace=False)
            sampled = [sampled[index] for index in selected]
        elif len(sampled) < max_sequences:
            remaining = list(set(self.indices).difference(sampled))
            selected = rng.choice(len(remaining), size=min(max_sequences - len(sampled), len(remaining)), replace=False)
            sampled.extend(remaining[index] for index in selected)
        return sorted(sampled)

    def __getitem__(self, idx: int) -> dict[str, object]:
        start, end = self.indices[idx]
        return {
            "packet_texts": self.packet_texts[start:end],
            "raw_tabular": self.raw_tabular[start:end],
            "tabular": self.tabular[start:end],
            "weak_predicates": self.predicates[start:end],
            "current": self.current[end - 1],
            "future": self.future[end - 1],
            "ticker": str(self.df.iloc[end - 1]["ticker"]),
            "anchor_trading_date": pd.Timestamp(self.df.iloc[end - 1]["anchor_trading_date"]).date().isoformat(),
        }


def collate_sequences(batch: list[dict[str, object]]) -> dict[str, object]:
    import torch

    return {
        "packet_texts": [text for item in batch for text in item["packet_texts"]],
        "raw_tabular": torch.stack([item["raw_tabular"] for item in batch]),
        "tabular": torch.stack([item["tabular"] for item in batch]),
        "weak_predicates": torch.stack([item["weak_predicates"] for item in batch]),
        "current": torch.stack([item["current"] for item in batch]),
        "future": torch.stack([item["future"] for item in batch]),
        "ticker": [item["ticker"] for item in batch],
        "anchor_trading_date": [item["anchor_trading_date"] for item in batch],
    }


class RealLogic:
    """Product-t-norm Real Logic operators with quantified satisfaction."""

    @staticmethod
    def not_(truth):
        return 1.0 - truth

    @staticmethod
    def and_(*truths):
        result = truths[0]
        for truth in truths[1:]:
            result = result * truth
        return result

    @staticmethod
    def implies(antecedent, consequent):
        return 1.0 - antecedent + antecedent * consequent

    @staticmethod
    def forall(truth, p: float = 2.0):
        return 1.0 - ((1.0 - truth.clamp(0.0, 1.0)).pow(p).mean() + 1e-8).pow(1.0 / p)


@dataclass
class ModelOutputs:
    predicates: object
    transition_logits: object
    destination_logits: object
    reaction: object


def load_qwen_qlora(args: argparse.Namespace):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if quantization_config is not None:
        qwen = prepare_model_for_kbit_training(qwen)
        qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    qwen.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return tokenizer, get_peft_model(qwen, lora_config)


class EndToEndNeuroSymbolicModel:
    def __init__(self, args: argparse.Namespace):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = torch.device(args.device)
        self.tokenizer, self.qwen = load_qwen_qlora(args)
        qwen_hidden = int(self.qwen.config.hidden_size)
        self.predicate_head = nn.Sequential(
            nn.Linear(qwen_hidden, args.hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, len(PREDICATE_NAMES)),
        ).to(self.device)
        self.tabular_projection = nn.Sequential(
            nn.Linear(len(TABULAR_COLUMNS), args.tabular_hidden_size),
            nn.GELU(),
            nn.LayerNorm(args.tabular_hidden_size),
        ).to(self.device)
        self.gru = nn.GRU(
            len(PREDICATE_NAMES) + args.tabular_hidden_size,
            args.hidden_size,
            batch_first=True,
        ).to(self.device)
        self.transition_head = nn.Linear(args.hidden_size, 1).to(self.device)
        self.destination_head = nn.Linear(args.hidden_size, 3).to(self.device)
        self.reaction_head = nn.Linear(args.hidden_size, 1).to(self.device)

    def parameters(self):
        yield from self.qwen.parameters()
        yield from self.predicate_head.parameters()
        yield from self.tabular_projection.parameters()
        yield from self.gru.parameters()
        yield from self.transition_head.parameters()
        yield from self.destination_head.parameters()
        yield from self.reaction_head.parameters()

    def train(self):
        for module in self.modules():
            module.train()

    def eval(self):
        for module in self.modules():
            module.eval()

    def modules(self):
        return [
            self.qwen,
            self.predicate_head,
            self.tabular_projection,
            self.gru,
            self.transition_head,
            self.destination_head,
            self.reaction_head,
        ]

    def encode_packets(self, packet_texts: list[str], max_length: int):
        torch = self.torch
        tokens = self.tokenizer(
            packet_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        outputs = self.qwen(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.sigmoid(self.predicate_head(pooled.float()))

    def forward(self, batch: dict[str, object], max_length: int) -> ModelOutputs:
        batch_size, sequence_length, _ = batch["tabular"].shape
        predicates = self.encode_packets(batch["packet_texts"], max_length)
        predicates = predicates.view(batch_size, sequence_length, -1)
        tabular = self.tabular_projection(batch["tabular"].to(self.device))
        hidden_sequence, _ = self.gru(self.torch.cat([predicates, tabular], dim=-1))
        hidden = hidden_sequence[:, -1]
        return ModelOutputs(
            predicates=predicates,
            transition_logits=self.transition_head(hidden).squeeze(-1),
            destination_logits=self.destination_head(hidden),
            reaction=self.reaction_head(hidden).squeeze(-1),
        )


def grounded_ltn_formulas(outputs: ModelOutputs, batch: dict[str, object]) -> dict[str, object]:
    import torch

    logic = RealLogic()
    predicates = outputs.predicates[:, -1]
    relevance, materiality, risk_on, risk_off, uncertainty = predicates.unbind(dim=-1)
    transition = outputs.transition_logits.sigmoid()
    destination = outputs.destination_logits.softmax(dim=-1)
    bear, sideways, bull = destination.unbind(dim=-1)
    current = batch["current"].to(transition.device)
    recent_revenue = batch["tabular"][:, -1, TABULAR_COLUMNS.index("fundamental_revenue_yoy")].to(transition.device)
    recent_margin = batch["tabular"][:, -1, TABULAR_COLUMNS.index("fundamental_operating_margin_yoy_change")].to(transition.device)
    negative_fundamentals = logic.and_(torch.sigmoid(-recent_revenue), torch.sigmoid(-recent_margin))
    positive_fundamentals = logic.and_(torch.sigmoid(recent_revenue), torch.sigmoid(recent_margin))
    conflicting_news = logic.and_(risk_on, risk_off)
    weak_news = logic.and_(logic.not_(relevance), logic.not_(materiality))
    is_bear = (current == 0).float()
    is_bull = (current == 2).float()
    return {
        "irrelevant_news_implies_low_materiality": logic.forall(
            logic.implies(logic.not_(relevance), logic.not_(materiality))
        ),
        "conflicting_news_implies_uncertainty": logic.forall(
            logic.implies(conflicting_news, uncertainty)
        ),
        "weak_news_implies_persistence": logic.forall(
            logic.implies(weak_news, logic.not_(transition))
        ),
        "risk_off_and_negative_fundamentals_imply_transition": logic.forall(
            logic.implies(logic.and_(risk_off, negative_fundamentals), transition)
        ),
        "risk_off_transition_implies_bear_destination": logic.forall(
            logic.implies(logic.and_(risk_off, transition), bear)
        ),
        "risk_on_and_positive_fundamentals_imply_transition": logic.forall(
            logic.implies(logic.and_(risk_on, positive_fundamentals), transition)
        ),
        "risk_on_transition_implies_bull_destination": logic.forall(
            logic.implies(logic.and_(risk_on, transition), bull)
        ),
        "bear_with_risk_on_implies_non_bear_destination": logic.forall(
            logic.implies(logic.and_(is_bear, risk_on, transition), logic.not_(bear))
        ),
        "bull_with_risk_off_implies_non_bull_destination": logic.forall(
            logic.implies(logic.and_(is_bull, risk_off, transition), logic.not_(bull))
        ),
        "conflicting_transition_news_implies_sideways_destination": logic.forall(
            logic.implies(logic.and_(conflicting_news, transition), sideways)
        ),
    }


def load_keep_rule_specs(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        raise ValueError("--logic-rule-set mined_keep requires --approved-rules-path.")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keep = [
        row
        for row in rows
        if row.get("recommended_decision", "").strip().upper() == "KEEP"
        or row.get("approved", "").strip().upper() in {"KEEP", "Y", "YES", "TRUE", "1"}
    ]
    if not keep:
        raise ValueError(f"No KEEP rules found in {path}.")
    return keep


def final_raw_feature(batch: dict[str, object], column: str):
    return batch["raw_tabular"][:, -1, TABULAR_COLUMNS.index(column)]


def grounded_mined_keep_formulas(
    outputs: ModelOutputs,
    batch: dict[str, object],
    rule_specs: list[dict[str, str]],
) -> dict[str, object]:
    import torch

    logic = RealLogic()
    device = outputs.transition_logits.device
    predicates = outputs.predicates[:, -1]
    relevance, materiality, risk_on, risk_off, uncertainty = predicates.unbind(dim=-1)
    transition = outputs.transition_logits.sigmoid()
    destination = outputs.destination_logits.softmax(dim=-1)
    bear, sideways, bull = destination.unbind(dim=-1)
    current = batch["current"].to(device)

    recent_return = final_raw_feature(batch, "recent_return").to(device)
    recent_drawdown = final_raw_feature(batch, "drawdown_from_recent_high").to(device)
    volatility = final_raw_feature(batch, "realized_volatility_20d").to(device)
    revenue_yoy = final_raw_feature(batch, "fundamental_revenue_yoy").to(device)
    margin_change = final_raw_feature(batch, "fundamental_operating_margin_yoy_change").to(device)
    quarter_age = final_raw_feature(batch, "fundamental_quarter_age_days").to(device)
    has_statement = final_raw_feature(batch, "fundamental_has_public_statement").to(device).clamp(0.0, 1.0)

    # These atoms mirror mine_candidate_ltn_rules.py, but keep them differentiable
    # where they depend on model predicates or continuous financial features.
    volatility_centered = (volatility - volatility.mean()) / volatility.std(unbiased=False).clamp(min=1e-6)
    atoms = {
        "high_ticker_relevance": relevance,
        "material_news": materiality,
        "risk_on_news": risk_on,
        "risk_off_news": risk_off,
        "high_uncertainty": uncertainty,
        "recent_public_earnings_release": (torch.exp(-quarter_age.clamp(min=0.0) / 10.0) * has_statement).clamp(0.0, 1.0),
        "negative_revenue_growth": torch.sigmoid(-revenue_yoy * 5.0),
        "positive_revenue_growth": torch.sigmoid(revenue_yoy * 5.0),
        "margin_deterioration": torch.sigmoid(-margin_change * 10.0),
        "margin_improvement": torch.sigmoid(margin_change * 10.0),
        "volatility_spike": torch.sigmoid(volatility_centered),
        "drawdown_stress": (-recent_drawdown * 5.0).clamp(0.0, 1.0),
        "positive_price_momentum": (recent_return * 10.0).clamp(0.0, 1.0),
        "negative_price_momentum": (-recent_return * 10.0).clamp(0.0, 1.0),
        "current_bear_regime": (current == 0).float(),
        "current_sideways_regime": (current == 1).float(),
        "current_bull_regime": (current == 2).float(),
    }
    consequents = {
        "transition": transition,
        "persistence": logic.not_(transition),
        "bear_destination": bear,
        "sideways_destination": sideways,
        "bull_destination": bull,
        "bear_transition": logic.and_(transition, bear),
        "sideways_transition": logic.and_(transition, sideways),
        "bull_transition": logic.and_(transition, bull),
    }

    satisfactions = {}
    for index, row in enumerate(rule_specs, start=1):
        antecedent_names = [part.strip() for part in row["antecedents"].split("&") if part.strip()]
        consequent_name = row["consequent"].strip()
        unknown_atoms = sorted(set(antecedent_names).difference(atoms))
        if unknown_atoms:
            raise ValueError(f"Rule {row['rule_text']} uses unknown atoms: {unknown_atoms}")
        if consequent_name not in consequents:
            raise ValueError(f"Rule {row['rule_text']} uses unknown consequent: {consequent_name}")

        antecedent = atoms[antecedent_names[0]]
        for name in antecedent_names[1:]:
            antecedent = logic.and_(antecedent, atoms[name])
        safe_name = row["rule_text"].replace(" ", "_").replace("=>", "implies").replace("&", "and")
        satisfactions[f"mined_keep_{index:02d}_{safe_name}"] = logic.forall(
            logic.implies(antecedent, consequents[consequent_name])
        )
    return satisfactions


def compute_losses(
    outputs: ModelOutputs,
    batch: dict[str, object],
    args: argparse.Namespace,
    rule_specs: list[dict[str, str]] | None,
) -> tuple[object, dict[str, float], dict[str, object]]:
    import torch
    import torch.nn.functional as functional

    device = outputs.transition_logits.device
    current = batch["current"].to(device)
    future = batch["future"].to(device)
    weak_predicates = batch["weak_predicates"].to(device)
    transition_target = (current != future).float()
    transition_loss = functional.binary_cross_entropy_with_logits(
        outputs.transition_logits,
        transition_target,
    )
    transition_mask = transition_target.bool()
    destination_loss = (
        functional.cross_entropy(outputs.destination_logits[transition_mask], future[transition_mask])
        if transition_mask.any()
        else outputs.destination_logits.sum() * 0.0
    )
    predicate_loss = functional.binary_cross_entropy(outputs.predicates, weak_predicates)
    reaction_target = (future.float() - current.float()) / 2.0
    reaction_loss = functional.mse_loss(torch.tanh(outputs.reaction), reaction_target)
    if args.logic_rule_set == "mined_keep":
        satisfactions = grounded_mined_keep_formulas(outputs, batch, rule_specs or [])
    else:
        satisfactions = grounded_ltn_formulas(outputs, batch)
    logic_loss = 1.0 - torch.stack(list(satisfactions.values())).mean()
    loss = (
        transition_loss
        + args.destination_loss_weight * destination_loss
        + args.predicate_loss_weight * predicate_loss
        + args.reaction_loss_weight * reaction_loss
        + args.logic_loss_weight * logic_loss
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "transition_loss": float(transition_loss.detach().cpu()),
        "destination_loss": float(destination_loss.detach().cpu()),
        "predicate_loss": float(predicate_loss.detach().cpu()),
        "reaction_loss": float(reaction_loss.detach().cpu()),
        "logic_loss": float(logic_loss.detach().cpu()),
    }
    return loss, metrics, satisfactions


def summarize_epoch(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in metric_rows[0]
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
    }


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    predicted = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None,
        "actual_transition_rows": int(y_true.sum()),
        "predicted_transition_rows": int(predicted.sum()),
    }


def regime_metrics(
    true_ids: np.ndarray,
    current_ids: np.ndarray,
    destination_ids: np.ndarray,
    transition_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    actual_transition = true_ids != current_ids
    predicted_transition = transition_probabilities >= threshold
    final_ids = np.where(predicted_transition, destination_ids, current_ids)
    return {
        **classification_metrics(true_ids, final_ids),
        "transition_accuracy": (
            float((final_ids[actual_transition] == true_ids[actual_transition]).mean())
            if actual_transition.any()
            else None
        ),
        "stable_accuracy": (
            float((final_ids[~actual_transition] == true_ids[~actual_transition]).mean())
            if (~actual_transition).any()
            else None
        ),
    }


def tune_threshold(arrays: dict[str, np.ndarray], minimum_stable_accuracy: float) -> float:
    reports = []
    for threshold in np.linspace(0.05, 0.95, 91):
        regime = regime_metrics(
            arrays["future_id"],
            arrays["current_id"],
            arrays["destination_id"],
            arrays["transition_probability"],
            float(threshold),
        )
        detector = binary_metrics(
            arrays["future_id"] != arrays["current_id"],
            arrays["transition_probability"],
            float(threshold),
        )
        reports.append({**detector, **regime})
    eligible = [
        report
        for report in reports
        if report["stable_accuracy"] is not None
        and report["stable_accuracy"] >= minimum_stable_accuracy
    ]
    if not eligible:
        eligible = reports
    return float(max(eligible, key=lambda report: (report["macro_f1"], report["f1"], report["precision"]))["threshold"])


def evaluate(
    model,
    loader,
    args: argparse.Namespace,
    rule_specs: list[dict[str, str]] | None,
    threshold: float | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    import torch

    model.eval()
    metrics = []
    rows = []
    formula_totals: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            batch["tabular"] = batch["tabular"].to(model.device)
            outputs = model.forward(batch, args.max_length)
            _, batch_metrics, satisfactions = compute_losses(outputs, batch, args, rule_specs)
            metrics.append(batch_metrics)
            probabilities = outputs.transition_logits.sigmoid().cpu().numpy()
            destination_logits = outputs.destination_logits.clone()
            current = batch["current"].cpu().numpy()
            future = batch["future"].cpu().numpy()
            destination_logits.scatter_(
                1,
                torch.tensor(current, device=destination_logits.device)[:, None],
                float("-inf"),
            )
            destination_probabilities = destination_logits.softmax(dim=-1).cpu().numpy()
            destinations = destination_probabilities.argmax(axis=1)
            predicted_predicates = outputs.predicates[:, -1].cpu().numpy()
            for index in range(len(current)):
                row = {
                    "ticker": batch["ticker"][index],
                    "anchor_trading_date": batch["anchor_trading_date"][index],
                    "current_id": int(current[index]),
                    "future_id": int(future[index]),
                    "is_transition": bool(current[index] != future[index]),
                    "transition_probability": float(probabilities[index]),
                    "destination_id": int(destinations[index]),
                    "destination_prob_bear": float(destination_probabilities[index, 0]),
                    "destination_prob_sideways": float(destination_probabilities[index, 1]),
                    "destination_prob_bull": float(destination_probabilities[index, 2]),
                    "reaction_score": float(torch.tanh(outputs.reaction[index]).cpu()),
                }
                for predicate_index, predicate_name in enumerate(PREDICATE_NAMES):
                    row[f"predicate_{predicate_name}"] = float(predicted_predicates[index, predicate_index])
                rows.append(row)
            for name, satisfaction in satisfactions.items():
                formula_totals.setdefault(name, []).append(float(satisfaction.cpu()))
    predictions = pd.DataFrame(rows)
    arrays = {
        "future_id": predictions["future_id"].to_numpy(np.int64),
        "current_id": predictions["current_id"].to_numpy(np.int64),
        "destination_id": predictions["destination_id"].to_numpy(np.int64),
        "transition_probability": predictions["transition_probability"].to_numpy(float),
    }
    if threshold is None:
        threshold = tune_threshold(arrays, args.minimum_stable_accuracy)
    predictions["predicted_transition"] = predictions["transition_probability"] >= threshold
    predictions["predicted_id"] = np.where(
        predictions["predicted_transition"],
        predictions["destination_id"],
        predictions["current_id"],
    )
    summary = summarize_epoch(metrics)
    summary["transition_detector"] = binary_metrics(
        arrays["future_id"] != arrays["current_id"],
        arrays["transition_probability"],
        threshold,
    )
    summary["final_regime"] = regime_metrics(
        arrays["future_id"],
        arrays["current_id"],
        arrays["destination_id"],
        arrays["transition_probability"],
        threshold,
    )
    summary["formula_satisfaction"] = {
        name: float(np.mean(values))
        for name, values in formula_totals.items()
    }
    return summary, predictions


def save_adapter_and_heads(model, output_dir: Path, metadata: dict[str, object]) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    model.qwen.save_pretrained(output_dir / "qwen_qlora_adapter")
    torch.save(
        {
            "predicate_head": model.predicate_head.state_dict(),
            "tabular_projection": model.tabular_projection.state_dict(),
            "gru": model.gru.state_dict(),
            "transition_head": model.transition_head.state_dict(),
            "destination_head": model.destination_head.state_dict(),
            "reaction_head": model.reaction_head.state_dict(),
        },
        output_dir / "neurosymbolic_heads.pt",
    )
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def capture_model_state(model) -> dict[str, object]:
    from peft import get_peft_model_state_dict

    def cpu_state(module):
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}

    return {
        "qwen_qlora_adapter": {
            name: value.detach().cpu().clone()
            for name, value in get_peft_model_state_dict(model.qwen).items()
        },
        "predicate_head": cpu_state(model.predicate_head),
        "tabular_projection": cpu_state(model.tabular_projection),
        "gru": cpu_state(model.gru),
        "transition_head": cpu_state(model.transition_head),
        "destination_head": cpu_state(model.destination_head),
        "reaction_head": cpu_state(model.reaction_head),
    }


def trainable_parameter_summary(model) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "trainable": int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad)),
        "total_exposed": int(sum(parameter.numel() for parameter in parameters)),
    }


def restore_model_state(model, state: dict[str, object]) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model.qwen, state["qwen_qlora_adapter"])
    model.predicate_head.load_state_dict(state["predicate_head"])
    model.tabular_projection.load_state_dict(state["tabular_projection"])
    model.gru.load_state_dict(state["gru"])
    model.transition_head.load_state_dict(state["transition_head"])
    model.destination_head.load_state_dict(state["destination_head"])
    model.reaction_head.load_state_dict(state["reaction_head"])


def validate_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    packets = pd.read_parquet(path)
    require_columns(
        packets,
        {
            "ticker",
            "anchor_trading_date",
            "split",
            "packet_text",
            "price_trend_id",
            "future_price_trend_id",
            *TABULAR_COLUMNS,
            *WEAK_LABEL_COLUMNS,
        },
    )
    train = packets[packets["split"] == "train"].copy()
    if train.empty:
        raise ValueError("No training rows found.")
    return packets, fit_tabular_stats(train)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    packets, stats = validate_dataset(args.daily_packets_path)
    rule_specs = load_keep_rule_specs(args.approved_rules_path) if args.logic_rule_set == "mined_keep" else None
    print(f"packet rows: {len(packets):,}")
    print(f"tabular features: {len(TABULAR_COLUMNS)}")
    print(f"semantic predicates: {', '.join(PREDICATE_NAMES)}")
    if rule_specs is not None:
        print(f"LTN rule set: mined_keep ({len(rule_specs)} rules)")
    if args.dry_run:
        print("Dry run passed: dataset schema is valid. Qwen was not loaded.")
        return
    import torch
    from torch.utils.data import DataLoader

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for live Qwen QLoRA training.")
    train = PacketSequenceDataset(
        packets[packets["split"] == "train"],
        stats,
        args.sequence_length,
        args.max_train_sequences,
        args.seed,
    )
    validation = PacketSequenceDataset(
        packets[packets["split"] == "validation"],
        stats,
        args.sequence_length,
        args.max_validation_sequences,
        args.seed,
    )
    test = PacketSequenceDataset(
        packets[packets["split"] == "test"],
        stats,
        args.sequence_length,
        args.max_test_sequences,
        args.seed,
    )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )
    test_loader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )
    model = EndToEndNeuroSymbolicModel(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
    parameter_summary = trainable_parameter_summary(model)
    print(
        f"trainable parameters: {parameter_summary['trainable']:,} "
        f"of {parameter_summary['total_exposed']:,} exposed parameters",
        flush=True,
    )
    best_validation_pr_auc = -1.0
    best_epoch = None
    best_state = None
    best_validation_metrics = None
    best_threshold = None
    start_epoch = 1
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        restore_model_state(model, checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_validation_pr_auc = checkpoint["best_validation_pr_auc"]
        best_epoch = checkpoint["best_epoch"]
        best_state = checkpoint["best_state"]
        best_validation_metrics = checkpoint["best_validation_metrics"]
        best_threshold = checkpoint["best_threshold"]
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming after epoch {checkpoint['epoch']} from {args.resume_checkpoint}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        rows = []
        for step, batch in enumerate(train_loader, start=1):
            batch["tabular"] = batch["tabular"].to(model.device)
            outputs = model.forward(batch, args.max_length)
            loss, metrics, _ = compute_losses(outputs, batch, args, rule_specs)
            (loss / args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            rows.append(metrics)
            if args.log_every and (step % args.log_every == 0 or step == len(train_loader)):
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={metrics['loss']:.4f} logic_loss={metrics['logic_loss']:.4f}",
                    flush=True,
                )
        train_metrics = summarize_epoch(rows)
        validation_metrics, validation_predictions = evaluate(model, validation_loader, args, rule_specs)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_logic_loss={train_metrics['logic_loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_pr_auc={validation_metrics['transition_detector']['pr_auc']:.4f}"
        )
        if validation_metrics["transition_detector"]["pr_auc"] > best_validation_pr_auc:
            best_validation_pr_auc = validation_metrics["transition_detector"]["pr_auc"]
            best_epoch = epoch
            best_state = capture_model_state(model)
            best_validation_metrics = validation_metrics
            best_threshold = validation_metrics["transition_detector"]["threshold"]
            save_adapter_and_heads(
                model,
                args.output_dir,
                {
                    "status": "best_validation_checkpoint",
                    "best_epoch": best_epoch,
                    "best_validation": best_validation_metrics,
                },
            )
            validation_predictions.to_parquet(
                args.output_dir / "validation_predictions.parquet",
                index=False,
            )
            write_json(args.output_dir / "best_validation_metrics.json", best_validation_metrics)
            print(f"Saved improved checkpoint after epoch {epoch}.", flush=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state": capture_model_state(model),
                "optimizer_state": optimizer.state_dict(),
                "best_validation_pr_auc": best_validation_pr_auc,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "best_validation_metrics": best_validation_metrics,
                "best_threshold": best_threshold,
            },
            args.output_dir / "latest_training_checkpoint.pt",
        )
        print(f"Saved resumable checkpoint after epoch {epoch}.", flush=True)
    if best_state is None:
        raise RuntimeError("Training did not produce a best model checkpoint.")
    restore_model_state(model, best_state)
    test_metrics, test_predictions = evaluate(model, test_loader, args, rule_specs, best_threshold)
    metadata = {
        "model_id": args.model_id,
        "best_epoch": best_epoch,
        "best_validation": best_validation_metrics,
        "test": test_metrics,
        "tabular_columns": TABULAR_COLUMNS,
        "tabular_stats": stats,
        "predicate_names": PREDICATE_NAMES,
        "parameter_summary": parameter_summary,
        "loss_weights": {
            "predicate": args.predicate_loss_weight,
            "logic": args.logic_loss_weight,
            "destination": args.destination_loss_weight,
            "reaction": args.reaction_loss_weight,
        },
        "logic_rule_set": args.logic_rule_set,
        "approved_rules_path": str(args.approved_rules_path) if args.approved_rules_path else None,
        "approved_rule_count": len(rule_specs or []),
        "architecture": (
            "Live Qwen QLoRA packet encoder with differentiable predicate heads, "
            "ticker-sequence GRU, transition and destination heads, reaction auxiliary "
            "head, and quantified Real Logic fuzzy knowledge-base loss."
        ),
    }
    save_adapter_and_heads(model, args.output_dir, metadata)
    test_predictions.to_parquet(args.output_dir / "test_predictions.parquet", index=False)
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    print(f"Best validation transition PR-AUC: {best_validation_pr_auc:.4f}")
    print(f"Selected validation threshold: {best_threshold:.2f}")
    print(f"Test transition PR-AUC: {test_metrics['transition_detector']['pr_auc']:.4f}")
    print(
        f"Test final accuracy: {test_metrics['final_regime']['accuracy']:.4f} "
        f"macro-F1: {test_metrics['final_regime']['macro_f1']:.4f}"
    )
    print(f"Saved QLoRA adapter and neuro-symbolic heads to {args.output_dir}")


if __name__ == "__main__":
    main()
