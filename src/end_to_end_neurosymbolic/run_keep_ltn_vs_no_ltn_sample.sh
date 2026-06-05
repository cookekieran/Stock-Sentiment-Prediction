#!/usr/bin/env bash
set -euo pipefail

# Short version of the mined KEEP-rule ablation. This compares the same model
# with and without LTN gradient pressure on a capped stratified sample.
export EPOCHS="${EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export MAX_TRAIN_SEQUENCES="${MAX_TRAIN_SEQUENCES:-200}"
export MAX_VALIDATION_SEQUENCES="${MAX_VALIDATION_SEQUENCES:-200}"
export MAX_TEST_SEQUENCES="${MAX_TEST_SEQUENCES:-200}"
export LOG_EVERY="${LOG_EVERY:-10}"
export LTN_WEIGHT="${LTN_WEIGHT:-0.2}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-models/end_to_end_neurosymbolic/keep_rule_ltn_ablation_sample}"

bash src/end_to_end_neurosymbolic/run_keep_ltn_vs_no_ltn.sh
