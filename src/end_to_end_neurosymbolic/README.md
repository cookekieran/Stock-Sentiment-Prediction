# End-to-End Neuro-Symbolic Pipeline

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
- article metadata and weak semantic labels for later predicate supervision;
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

## 2. Create A Predicate-Review Template

The Qwen predicate heads need a manually reviewed subset. The template samples
packets across tickers, chronological splits, transition status, and weak
relevance bands while hiding future regime labels from the reviewer.

```powershell
conda run -n stocks_env python src\end_to_end_neurosymbolic\build_predicate_annotation_template.py `
  --daily-packets-path data\processed\end_to_end_neurosymbolic\horizon_20\daily_packets.parquet `
  --output-path outputs\end_to_end_neurosymbolic\predicate_annotation_template.csv `
  --rows-per-split 150
```

Reviewers should score numeric predicate fields from `0.0` to `1.0`, then add
event-type and scope categories. This reviewed subset supports both predicate
supervision and held-out predicate-quality evaluation.

## Next Implementation Steps

1. Review and label the stratified subset of packets for ticker relevance,
   materiality, directional pressure, uncertainty, and event type.
2. Add a Qwen + QLoRA packet encoder with trainable semantic predicate heads.
3. Feed daily predicate vectors, price context, and fundamentals into a GRU.
4. Ground quantified LTN formulas over packet-day examples and optimize logical
   satisfaction jointly with supervised predicate and transition losses.
