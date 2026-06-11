# Transformer-Context Neuro-Symbolic Pipeline

Status: active architecture direction.

The market-state representation must be created by the transformer's context
window. The final architecture must not append a GRU or another recurrent state
module after the language model.

```text
ordered ticker-history context window
  = daily news packets
  + contemporaneous price context
  + publicly available fundamentals
        |
        v
causal transformer with trainable adapters
        |
        +--> contextual market-state representation
        +--> semantic predicate heads
        |
        v
transition, destination, and reaction heads
        |
        v
supervised losses + fuzzy-LTN knowledge-base loss
```

## State Definition

For an anchor ticker/date, construct one chronological context window containing
the recent history available at that date. Each day must have explicit date and
section delimiters. The transformer uses causal self-attention across the full
window, so the anchor-date representation is conditioned on the preceding news,
market context, and fundamentals.

Use the final anchor-state token, or a clearly documented pooling operation over
the anchor-day tokens, as the market-state representation. Prediction and
predicate heads consume that representation directly. Do not feed transformer
outputs into a separate recurrent network.

A transformer context window is finite rather than permanently persistent.
Sequence length, truncation policy, and the number of historical trading days
are therefore experimental variables that must be reported and ablated.

## Retained Components

- `build_daily_packet_dataset.py`: builds leakage-safe daily text and tabular
  packets. Extend it to emit chronological multi-day contexts.
- `build_predicate_annotation_template.py`: supports independent review of
  learned semantic predicates.
- `mine_candidate_ltn_rules.py`: discovers candidate rules using train and
  validation data only.
- `requirements-gpu.in`: GPU dependencies for the live transformer path.

## Required Replacement Work

1. Build chronological context windows per ticker without crossing information
   cutoffs or dataset splits.
2. Add explicit daily boundary tokens and serialize only information available
   by each trading-day cutoff.
3. Train a live transformer with adapter tuning and prediction/predicate heads
   attached directly to its contextual representation.
4. Apply LTN losses to learned predicates and predictions.
5. Compare context-window lengths and pooling choices against persistence and
   non-contextual transformer baselines.

All superseded GRU trainers, runners, audits, documentation, and model outputs
are preserved under `archive/legacy_experiments/gru_pipelines/`.
