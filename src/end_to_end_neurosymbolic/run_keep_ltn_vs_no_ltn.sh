#!/usr/bin/env bash
set -euo pipefail

DAILY_PACKETS_PATH="${DAILY_PACKETS_PATH:-data/processed/end_to_end_neurosymbolic/horizon_20/daily_packets.parquet}"
MINED_RULES_PATH="${MINED_RULES_PATH:-outputs/end_to_end_neurosymbolic/rule_mining_horizon_20/candidate_ltn_rules.csv}"
RULE_REVIEW_DIR="${RULE_REVIEW_DIR:-outputs/end_to_end_neurosymbolic/rule_review_horizon_20}"
APPROVED_RULES_PATH="${APPROVED_RULES_PATH:-${RULE_REVIEW_DIR}/candidate_ltn_rules_reviewed.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-models/end_to_end_neurosymbolic/keep_rule_ltn_ablation}"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-25}"
LTN_WEIGHT="${LTN_WEIGHT:-0.2}"

MAX_TRAIN_ARGS=()
if [[ -n "${MAX_TRAIN_SEQUENCES:-}" ]]; then
  MAX_TRAIN_ARGS+=(--max-train-sequences "${MAX_TRAIN_SEQUENCES}")
fi
if [[ -n "${MAX_VALIDATION_SEQUENCES:-}" ]]; then
  MAX_TRAIN_ARGS+=(--max-validation-sequences "${MAX_VALIDATION_SEQUENCES}")
fi
if [[ -n "${MAX_TEST_SEQUENCES:-}" ]]; then
  MAX_TRAIN_ARGS+=(--max-test-sequences "${MAX_TEST_SEQUENCES}")
fi

python tools/review_candidate_ltn_rules.py \
  --source "${MINED_RULES_PATH}" \
  --output-dir "${RULE_REVIEW_DIR}"

COMMON_ARGS=(
  --daily-packets-path "${DAILY_PACKETS_PATH}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --log-every "${LOG_EVERY}"
  --seed "${SEED}"
  --logic-rule-set mined_keep
  --approved-rules-path "${APPROVED_RULES_PATH}"
  "${MAX_TRAIN_ARGS[@]}"
)

python src/end_to_end_neurosymbolic/train_qwen_qlora_gru_ltn.py \
  "${COMMON_ARGS[@]}" \
  --logic-loss-weight 0 \
  --output-dir "${OUTPUT_ROOT}/no_ltn_seed${SEED}"

python src/end_to_end_neurosymbolic/train_qwen_qlora_gru_ltn.py \
  "${COMMON_ARGS[@]}" \
  --logic-loss-weight "${LTN_WEIGHT}" \
  --output-dir "${OUTPUT_ROOT}/keep_ltn_w${LTN_WEIGHT}_seed${SEED}"

python src/end_to_end_neurosymbolic/compare_ltn_ablation_metrics.py \
  --output-root "${OUTPUT_ROOT}"
