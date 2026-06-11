# End-to-End Neuro-Symbolic Pipeline

Status: optional GPU extension. The final local study did not run this model
because the available runtime had no CUDA device. It must pass the documented
tiny GPU smoke test before being included in any reported comparison.

This folder contains the separate research path for a jointly trained model:

```text
daily ticker news packet
        |
        v
Qwen2.5-7B-Instruct + QLoRA adapters
        |
        v
trainable semantic predicate heads
        |
        v
GRU persistent ticker state
        |
        v
transition detector + destination classifier
        |
        v
LTN knowledge-base satisfaction loss
```

The existing offline-Qwen pipeline remains available under `src/data` and
`src/models`. Do not use previously generated Qwen predicates as inputs to this
end-to-end model: the live Qwen encoder and trainable predicate heads replace
that fixed boundary.

## 1. Build Daily Text Packets

The first artifact is one row per ticker/trading day containing:

- the raw daily news packet used by the live Qwen encoder;
- article metadata and automated weak semantic labels for predicate supervision;
- contemporaneous price context;
- leakage-safe public quarterly fundamentals;
- current and future regime labels;
- chronological train, validation, and test splits.

```powershell
conda run -n stocks_env python src\end_to_end_neurosymbolic\build_daily_packet_dataset.py `
  --article-path data\processed\horizon_20\ltn\ltn_all.parquet `
  --daily-context-path data\processed\qwen_quality_filter_fundamentals_transition_ablation\horizon_20\latent_state_daily.parquet `
  --output-dir data\processed\end_to_end_neurosymbolic\horizon_20
```

The builder deliberately excludes saved Qwen narratives, predicates, and raw
generations. Future regime labels remain targets only. Fundamentals are audited
to ensure that their public availability date never exceeds the anchor date.

Automated predicate targets are calculated from contemporaneous article
relevance and bullish/bearish sentiment:

```text
weak_label_ticker_relevance
weak_label_materiality
weak_label_risk_on_pressure
weak_label_risk_off_pressure
weak_label_uncertainty
```

These labels supervise the live Qwen predicate heads without using future price
movements. Future bullish, sideways, and bearish regime labels supervise the
downstream GRU transition task separately.

## 2. Optional Predicate Review

Manual review is optional. It is useful only if the dissertation needs an
independent semantic-quality evaluation of whether Qwen learned sensible
relevance and materiality predicates. It is not required to train the automated
pipeline. The template samples packets across tickers, chronological splits,
transition status, and weak relevance bands while hiding future regime labels
from the reviewer.

```powershell
conda run -n stocks_env python src\end_to_end_neurosymbolic\build_predicate_annotation_template.py `
  --daily-packets-path data\processed\end_to_end_neurosymbolic\horizon_20\daily_packets.parquet `
  --output-path outputs\end_to_end_neurosymbolic\predicate_annotation_template.csv `
  --rows-per-split 150
```

If used, reviewers should score numeric predicate fields from `0.0` to `1.0`,
then add event-type and scope categories.

## Next Implementation Steps

## 3. Train Live Qwen + QLoRA + GRU + LTN

`train_qwen_qlora_gru_ltn.py` implements the first fully integrated training
graph:

```text
raw packet text
        |
        v
live Qwen2.5-7B-Instruct + QLoRA adapters
        |
        v
five differentiable semantic predicate heads
        |
        v
daily predicate vector + price context + public fundamentals
        |
        v
ticker-sequence GRU
        |
        v
transition, destination, and market-reaction heads
        |
        v
supervised losses + quantified fuzzy-LTN knowledge-base loss
```

The quantified LTN formulas currently encode:

- irrelevant news implies low materiality;
- conflicting risk-on and risk-off news implies uncertainty;
- weak news implies regime persistence;
- aligned fundamental deterioration and risk-off news imply a transition;
- a risk-off transition implies a bear destination;
- aligned fundamental improvement and risk-on news imply a transition;
- a risk-on transition implies a bull destination;
- risk-on evidence can move an existing bear state away from bear;
- risk-off evidence can move an existing bull state away from bull;
- conflicting transition evidence implies a sideways destination.

Install the GPU dependencies inside the Jupyter container:

```bash
python -m pip install -r src/end_to_end_neurosymbolic/requirements-gpu.in
```

Run a tiny GPU smoke test first:

```bash
python src/end_to_end_neurosymbolic/train_qwen_qlora_gru_ltn.py \
  --daily-packets-path data/processed/end_to_end_neurosymbolic/horizon_20/daily_packets.parquet \
  --output-dir models/end_to_end_neurosymbolic/smoke_qwen_qlora_gru_ltn \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-train-sequences 2 \
  --max-validation-sequences 2 \
  --max-test-sequences 2
```

After that succeeds, run the first baseline:

```bash
python src/end_to_end_neurosymbolic/train_qwen_qlora_gru_ltn.py \
  --daily-packets-path data/processed/end_to_end_neurosymbolic/horizon_20/daily_packets.parquet \
  --output-dir models/end_to_end_neurosymbolic/qwen_qlora_gru_ltn_seed42 \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --log-every 25 \
  --seed 42
```

The trainer saves:

- the Qwen QLoRA adapter;
- the predicate, GRU, transition, destination, and reaction heads;
- validation PR-AUC;
- test PR-AUC from the restored best-validation checkpoint;
- validation-only threshold tuning and thresholded test metrics;
- validation and test prediction parquets with learned semantic predicates;
- quantified formula-satisfaction values;
- training metadata and feature normalization statistics.
- a resumable checkpoint after every epoch.

If a run stops after a completed epoch, resume it with:

```bash
python src/end_to_end_neurosymbolic/train_qwen_qlora_gru_ltn.py \
  --daily-packets-path data/processed/end_to_end_neurosymbolic/horizon_20/daily_packets.parquet \
  --output-dir models/end_to_end_neurosymbolic/qwen_qlora_gru_ltn_seed42 \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --log-every 25 \
  --seed 42 \
  --resume-checkpoint models/end_to_end_neurosymbolic/qwen_qlora_gru_ltn_seed42/latest_training_checkpoint.pt
```

## Next Implementation Steps

1. Run the tiny Qwen QLoRA smoke test in the GPU container.
2. Compare ablations with the LTN and predicate-supervision losses disabled.
3. Extend the quantified knowledge base with earnings-event predicates and
   trailing semantic persistence after the initial integrated run is stable.

## 4. Mine Candidate LTN Rules

Use `mine_candidate_ltn_rules.py` to discover candidate fuzzy implications from
the train split and score them on validation only. The output CSV includes an
`approved` column for human economic review before any rule is added back to the
LTN knowledge base.

```bash
python src/end_to_end_neurosymbolic/mine_candidate_ltn_rules.py \
  --daily-packets-path data/processed/end_to_end_neurosymbolic/horizon_20/daily_packets.parquet \
  --output-dir outputs/end_to_end_neurosymbolic/rule_mining_horizon_20 \
  --max-antecedents 3 \
  --min-train-support 0.03 \
  --min-validation-support 0.03 \
  --min-validation-lift 1.05 \
  --bootstrap-repeats 500 \
  --top-k 300
```

Review `candidate_ltn_rules.csv` in the morning. Approve only rules that are
economically plausible and were not selected using the test split.

## 5. Compare Approved Rules Against No LTN

Use the sampled runner for a quick diagnostic before committing GPU time to a
full ablation:

```bash
bash src/end_to_end_neurosymbolic/run_keep_ltn_vs_no_ltn_sample.sh
```

This runs the same Qwen QLoRA and GRU architecture twice on identical
stratified samples:

- `no_ltn`: approved rules are audited, but their loss weight is zero;
- `keep_ltn`: only rules marked `KEEP` contribute to the differentiable logic
  loss.

The runner writes `ltn_ablation_summary.csv` under
`models/end_to_end_neurosymbolic/keep_rule_ltn_ablation_sample`. Compare
transition PR-AUC, transition F1, final macro-F1, and stable accuracy before
deciding whether the approved rules justify a full-data run.
