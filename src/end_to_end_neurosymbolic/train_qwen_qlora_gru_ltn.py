"""Train an end-to-end Qwen QLoRA, GRU, and quantified fuzzy-LTN model."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--destination-loss-weight", type=float, default=1.0)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-validation-sequences", type=int, default=None)
    parser.add_argument("--max-test-sequences", type=int, default=None)
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
    ):
        import torch

        self.df = df.sort_values(["ticker", "anchor_trading_date"]).reset_index(drop=True)
        self.packet_texts = self.df["packet_text"].fillna("").astype(str).tolist()
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
        if max_sequences is not None:
            self.indices = self.indices[:max_sequences]
        if not self.indices:
            raise ValueError("No sequences created. Reduce --sequence-length.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, object]:
        start, end = self.indices[idx]
        return {
            "packet_texts": self.packet_texts[start:end],
            "tabular": self.tabular[start:end],
            "weak_predicates": self.predicates[start:end],
            "current": self.current[end - 1],
            "future": self.future[end - 1],
        }


def collate_sequences(batch: list[dict[str, object]]) -> dict[str, object]:
    import torch

    return {
        "packet_texts": [text for item in batch for text in item["packet_texts"]],
        "tabular": torch.stack([item["tabular"] for item in batch]),
        "weak_predicates": torch.stack([item["weak_predicates"] for item in batch]),
        "current": torch.stack([item["current"] for item in batch]),
        "future": torch.stack([item["future"] for item in batch]),
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
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if quantization_config is not None:
        qwen = prepare_model_for_kbit_training(qwen)
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


def compute_losses(
    outputs: ModelOutputs,
    batch: dict[str, object],
    args: argparse.Namespace,
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


def evaluate(model, loader, args: argparse.Namespace) -> dict[str, object]:
    import torch
    from sklearn.metrics import average_precision_score

    model.eval()
    metrics = []
    probabilities = []
    targets = []
    formula_totals: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            batch["tabular"] = batch["tabular"].to(model.device)
            outputs = model.forward(batch, args.max_length)
            _, batch_metrics, satisfactions = compute_losses(outputs, batch, args)
            metrics.append(batch_metrics)
            probabilities.extend(outputs.transition_logits.sigmoid().cpu().tolist())
            current = batch["current"].cpu().numpy()
            future = batch["future"].cpu().numpy()
            targets.extend((current != future).astype(np.int64).tolist())
            for name, satisfaction in satisfactions.items():
                formula_totals.setdefault(name, []).append(float(satisfaction.cpu()))
    summary = summarize_epoch(metrics)
    summary["transition_pr_auc"] = float(average_precision_score(targets, probabilities))
    summary["formula_satisfaction"] = {
        name: float(np.mean(values))
        for name, values in formula_totals.items()
    }
    return summary


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
    print(f"packet rows: {len(packets):,}")
    print(f"tabular features: {len(TABULAR_COLUMNS)}")
    print(f"semantic predicates: {', '.join(PREDICATE_NAMES)}")
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
    )
    validation = PacketSequenceDataset(
        packets[packets["split"] == "validation"],
        stats,
        args.sequence_length,
        args.max_validation_sequences,
    )
    test = PacketSequenceDataset(
        packets[packets["split"] == "test"],
        stats,
        args.sequence_length,
        args.max_test_sequences,
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
    best_validation_pr_auc = -1.0
    best_epoch = None
    best_state = None
    best_validation_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        rows = []
        for step, batch in enumerate(train_loader, start=1):
            batch["tabular"] = batch["tabular"].to(model.device)
            outputs = model.forward(batch, args.max_length)
            loss, metrics, _ = compute_losses(outputs, batch, args)
            (loss / args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            rows.append(metrics)
        train_metrics = summarize_epoch(rows)
        validation_metrics = evaluate(model, validation_loader, args)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_logic_loss={train_metrics['logic_loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_pr_auc={validation_metrics['transition_pr_auc']:.4f}"
        )
        if validation_metrics["transition_pr_auc"] > best_validation_pr_auc:
            best_validation_pr_auc = validation_metrics["transition_pr_auc"]
            best_epoch = epoch
            best_state = capture_model_state(model)
            best_validation_metrics = validation_metrics
    if best_state is None:
        raise RuntimeError("Training did not produce a best model checkpoint.")
    restore_model_state(model, best_state)
    test_metrics = evaluate(model, test_loader, args)
    metadata = {
        "model_id": args.model_id,
        "best_epoch": best_epoch,
        "best_validation": best_validation_metrics,
        "test": test_metrics,
        "tabular_columns": TABULAR_COLUMNS,
        "tabular_stats": stats,
        "predicate_names": PREDICATE_NAMES,
        "loss_weights": {
            "predicate": args.predicate_loss_weight,
            "logic": args.logic_loss_weight,
            "destination": args.destination_loss_weight,
            "reaction": args.reaction_loss_weight,
        },
        "architecture": (
            "Live Qwen QLoRA packet encoder with differentiable predicate heads, "
            "ticker-sequence GRU, transition and destination heads, reaction auxiliary "
            "head, and quantified Real Logic fuzzy knowledge-base loss."
        ),
    }
    save_adapter_and_heads(model, args.output_dir, metadata)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Best validation transition PR-AUC: {best_validation_pr_auc:.4f}")
    print(f"Test transition PR-AUC: {test_metrics['transition_pr_auc']:.4f}")
    print(f"Saved QLoRA adapter and neuro-symbolic heads to {args.output_dir}")


if __name__ == "__main__":
    main()
