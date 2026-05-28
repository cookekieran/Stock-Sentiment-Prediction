"""
Train a DeepSeek trend classifier with LTN-style fuzzy logic constraints.

This keeps the same frozen DeepSeek encoder and classifier head as the baseline,
then adds LTNtorch formulas over macro/materiality predicates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from common import (
    available_feature_columns,
    class_weights,
    classification_report,
    fit_feature_stats,
    labels,
    load_split,
    transform_features,
    write_json,
)
from train_deepseek_baseline import DeepSeekTrendClassifier, iterate_batches


PREDICATE_COLUMNS = [
    "relevance_score",
    "is_material",
    "macro_high_inflation",
    "macro_rising_rates",
    "macro_falling_rates",
    "macro_tightening_regime",
    "macro_easing_regime",
    "macro_inverted_yield_curve",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/ltn_train.parquet"))
    parser.add_argument(
        "--validation-path", type=Path, default=Path("data/processed/ltn_validation.parquet")
    )
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/ltn_test.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/deepseek_ltn"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--logic-weight", type=float, default=0.2)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


class LtnTrendDataset:
    def __init__(self, df, features: np.ndarray, y: np.ndarray):
        self.texts = df["article_text"].fillna("").astype(str).tolist()
        self.features = features
        self.y = y
        predicates = {}
        for column in PREDICATE_COLUMNS:
            if column in df.columns:
                predicates[column] = df[column].fillna(0.0)
            else:
                predicates[column] = 0.0
        self.predicates = np.stack(
            [
                np.asarray(
                    predicates[column] if np.ndim(predicates[column]) else np.repeat(predicates[column], len(df)),
                    dtype=np.float32,
                )
                for column in PREDICATE_COLUMNS
            ],
            axis=1,
        )
        self.predicates = np.nan_to_num(self.predicates, nan=0.0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.texts[idx], self.features[idx], self.predicates[idx], int(self.y[idx])


def iterate_ltn_batches(dataset: LtnTrendDataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        texts, features, predicates, y = zip(*(dataset[idx] for idx in range(start, end)))
        yield texts, np.stack(features), np.stack(predicates), np.array(y, dtype=np.int64)


class LTNConstraintSystem:
    """LTN predicates and formulas for macro-aware trend constraints."""

    def __init__(self):
        import ltn

        self.ltn = ltn
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier="f")
        self.SatAgg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError())

        self.Bear = ltn.Predicate(func=lambda x: x[:, 0])
        self.Sideways = ltn.Predicate(func=lambda x: x[:, 1])
        self.Bull = ltn.Predicate(func=lambda x: x[:, 2])
        self.HighRelevance = ltn.Predicate(func=lambda x: x[:, 3].clamp(0.0, 1.0))
        self.LowRelevance = ltn.Predicate(func=lambda x: (1.0 - x[:, 3]).clamp(0.0, 1.0))
        self.Material = ltn.Predicate(func=lambda x: x[:, 4].clamp(0.0, 1.0))
        self.HighInflation = ltn.Predicate(func=lambda x: x[:, 5].clamp(0.0, 1.0))
        self.RisingRates = ltn.Predicate(func=lambda x: x[:, 6].clamp(0.0, 1.0))
        self.FallingRates = ltn.Predicate(func=lambda x: x[:, 7].clamp(0.0, 1.0))
        self.Tightening = ltn.Predicate(func=lambda x: x[:, 8].clamp(0.0, 1.0))
        self.Easing = ltn.Predicate(func=lambda x: x[:, 9].clamp(0.0, 1.0))
        self.InvertedCurve = ltn.Predicate(func=lambda x: x[:, 10].clamp(0.0, 1.0))

    def loss(self, logits, predicates):
        import torch

        probs = logits.softmax(dim=1)
        relevance = predicates[:, 0].clamp(0.0, 1.0).unsqueeze(1)
        high_relevance = ((relevance - 0.25) / 0.75).clamp(0.0, 1.0)

        # x columns:
        # 0 bear prob, 1 sideways prob, 2 bull prob,
        # 3 high relevance, 4 material, 5 high inflation, 6 rising rates,
        # 7 falling rates, 8 tightening, 9 easing, 10 inverted curve.
        individuals = torch.cat(
            [probs, high_relevance, predicates[:, 1:8].clamp(0.0, 1.0)],
            dim=1,
        )

        x = self.ltn.Variable("x", individuals)

        formulas = [
            self.Forall(
                x,
                self.Implies(
                    self.And(self.And(self.Material(x), self.HighRelevance(x)), self.Easing(x)),
                    self.Bull(x),
                ),
            ),
            self.Forall(
                x,
                self.Implies(
                    self.And(self.And(self.Material(x), self.HighRelevance(x)), self.Tightening(x)),
                    self.Bear(x),
                ),
            ),
            self.Forall(
                x,
                self.Implies(
                    self.And(self.HighInflation(x), self.RisingRates(x)),
                    self.Not(self.Bull(x)),
                ),
            ),
            self.Forall(
                x,
                self.Implies(
                    self.And(self.InvertedCurve(x), self.Material(x)),
                    self.Or(self.Bear(x), self.Sideways(x)),
                ),
            ),
            self.Forall(x, self.Implies(self.LowRelevance(x), self.Sideways(x))),
            self.Forall(
                x,
                self.Implies(
                    self.And(self.FallingRates(x), self.Easing(x)),
                    self.Not(self.Bear(x)),
                ),
            ),
        ]
        return 1.0 - self.SatAgg(*formulas)


def evaluate(model, dataset: LtnTrendDataset, max_length: int, batch_size: int) -> dict:
    torch = model.torch
    model.head.eval()
    preds = []
    truth = []
    with torch.no_grad():
        for texts, features, _, y in iterate_ltn_batches(dataset, batch_size):
            logits = model.batch_logits(texts, features, max_length)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            truth.append(y)
    return classification_report(np.concatenate(truth), np.concatenate(preds))


def main() -> None:
    args = parse_args()
    train_df = load_split(args.train_path, args.max_train_rows)
    validation_df = load_split(args.validation_path, args.max_validation_rows)
    test_df = load_split(args.test_path, args.max_test_rows)

    feature_columns = available_feature_columns(train_df)
    feature_stats = fit_feature_stats(train_df, feature_columns)
    train_set = LtnTrendDataset(train_df, transform_features(train_df, feature_columns, feature_stats), labels(train_df))
    validation_set = LtnTrendDataset(
        validation_df, transform_features(validation_df, feature_columns, feature_stats), labels(validation_df)
    )
    test_set = LtnTrendDataset(test_df, transform_features(test_df, feature_columns, feature_stats), labels(test_df))

    model = DeepSeekTrendClassifier(args.model_id, len(feature_columns), args.dropout, args.device)
    torch = model.torch
    ltn_constraints = LTNConstraintSystem()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate)
    weights = torch.as_tensor(class_weights(train_set.y), dtype=torch.float32, device=model.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.head.train()
        losses = []
        task_losses = []
        logic_losses = []
        for texts, features, predicates, y in iterate_ltn_batches(train_set, args.batch_size):
            optimizer.zero_grad()
            logits = model.batch_logits(texts, features, args.max_length)
            target = torch.as_tensor(y, dtype=torch.long, device=model.device)
            predicate_t = torch.as_tensor(predicates, dtype=torch.float32, device=model.device)
            task_loss = criterion(logits, target)
            constraint_loss = ltn_constraints.loss(logits, predicate_t)
            loss = task_loss + args.logic_weight * constraint_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            task_losses.append(float(task_loss.detach().cpu()))
            logic_losses.append(float(constraint_loss.detach().cpu()))

        validation_report = evaluate(model, validation_set, args.max_length, args.batch_size)
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"task={np.mean(task_losses):.4f} logic={np.mean(logic_losses):.4f} "
            f"val_accuracy={validation_report['accuracy']:.4f} "
            f"val_macro_f1={validation_report['macro_f1']:.4f}"
        )
        if validation_report["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_report["macro_f1"]
            torch.save(model.head.state_dict(), args.output_dir / "best_head.pt")
            write_json(
                args.output_dir / "training_metadata.json",
                {
                    "model_id": args.model_id,
                    "feature_columns": feature_columns,
                    "feature_stats": feature_stats,
                    "logic_weight": args.logic_weight,
                    "predicate_columns": PREDICATE_COLUMNS,
                    "best_validation": validation_report,
                    "rules": [
                        "material & high_relevance & easing -> bull_rally",
                        "material & high_relevance & tightening -> bear_drawdown",
                        "high_inflation & rising_rates -> not bull_rally",
                        "inverted_yield_curve & material -> bear_drawdown or sideways",
                        "low_relevance -> sideways",
                        "falling_rates & easing -> not bear_drawdown",
                    ],
                },
            )

    model.head.load_state_dict(torch.load(args.output_dir / "best_head.pt", map_location=model.device))
    test_report = evaluate(model, test_set, args.max_length, args.batch_size)
    write_json(args.output_dir / "test_metrics.json", test_report)
    print(f"test_accuracy={test_report['accuracy']:.4f} test_macro_f1={test_report['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
