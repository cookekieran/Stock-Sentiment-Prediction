# Final Walk-Forward Results

## Outcome

The final matched study used two purged walk-forward folds and seeds `7`, `21`,
and `42`. The primary metric was future-regime-change PR-AUC.

| Model | PR-AUC mean | PR-AUC std | Change F1 mean | Final macro-F1 mean |
|---|---:|---:|---:|---:|
| Price + leakage-safe fundamentals | **0.610** | 0.061 | 0.118 | 0.418 |
| Price + fundamentals + Qwen | 0.569 | 0.039 | **0.244** | 0.413 |
| Price + fundamentals + Qwen + LTN | 0.569 | 0.040 | 0.240 | 0.416 |
| Price only | 0.555 | 0.099 | 0.100 | 0.404 |
| Persistence | 0.512 | 0.000 | 0.000 | **0.463** |

## Interpretation

Price plus publicly available fundamentals produced the strongest average
ranking of future regime changes. Qwen predicates increased thresholded change
F1, but reduced average PR-AUC and final macro-F1 relative to the
price-plus-fundamentals control.

Adding LTN constraints produced essentially no change in mean PR-AUC:

```text
Qwen without LTN: 0.569126
Qwen with LTN:    0.569115
```

The evidence therefore does not support the claim that the tested
neuro-symbolic constraints improve predictive ranking. The constraints remain
useful as an auditable representation of economic assumptions, but their
predictive contribution is negligible in this study.

Persistence achieved the strongest final macro-F1 while detecting no changes.
This highlights the tension between preserving stable regimes and detecting
future regime changes. PR-AUC remains the primary metric because it evaluates
the change detector without relying on a validation-selected threshold.

## Limitations

- The target is a change in a rolling price-regime definition, not a rare
  structural market transition.
- The dataset contains only seven highly correlated large technology companies.
- Adjacent targets overlap; purged folds reduce boundary leakage but do not make
  daily observations statistically independent.
- Qwen predicates were generated before the final study and may contain noisy or
  asymmetric semantic signals.
- The timezone-naive Alpha Vantage news timestamps are assumed to represent
  America/New_York local time; the source artifacts do not independently encode
  timezone provenance.
- Existing macro artifacts were excluded because their public release timestamps
  were not sufficiently auditable.

## End-to-End Qwen Extension

The local runtime reported `torch.cuda.is_available() == False`. The live
Qwen2.5-7B QLoRA model requires CUDA, so its GPU smoke test could not be run on
this machine. In accordance with the protocol, it remains an optional extension
and is not part of the final result.

Detailed generated results are written to:

```text
models/final_study/all_results.csv
models/final_study/summary.csv
```
