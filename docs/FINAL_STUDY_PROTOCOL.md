# Transformer-Context Final Study Protocol

Status: replacement study design. No GRU results belong in the final study.

## Research Question

Does a transformer that constructs a contextual market-state representation
from an ordered history of ticker-day packets improve out-of-time
future-regime-change forecasting, and do differentiable logic constraints add
value to that representation?

## Architecture

For each ticker and anchor date, construct a chronological context window from
only information available by the prediction cutoff:

- daily news packets;
- contemporaneous price context;
- publicly available fundamentals;
- explicit date and daily-boundary tokens.

A causal transformer attends over this complete history. Transition,
destination, reaction, and semantic-predicate heads consume the contextual
anchor-date representation directly. The model must not use a GRU, LSTM, or
separate recurrent state module.

## Information Availability

The prediction is made after the 16:00 America/New_York close. Price features use
the completed close. News at or after the cutoff is assigned to the next trading
session. Fundamentals are joined only after
`fundamental_available_from_date`. Future regime labels are targets only.

Context construction must not cross chronological split boundaries or include
information after the anchor-date cutoff.

## Target And Evaluation

The primary target is `future_regime_change`: whether the rolling price regime
at the configured future ticker observation differs from the anchor regime.

Use purged walk-forward folds and repeated seeds. Select checkpoints,
hyperparameters, and decision thresholds using validation data only.

Primary metric: transition PR-AUC.

Secondary metrics: transition F1, final regime macro-F1, final accuracy, and
semantic-predicate quality where reviewed labels are available.

## Required Comparisons

1. Persistence.
2. Non-contextual transformer using only the anchor-day packet.
3. Transformer context window without Qwen predicate supervision or LTN loss.
4. Transformer context window with predicate supervision.
5. Transformer context window with predicate supervision and LTN loss.

Report ablations for context-window length, truncation policy, and anchor-state
pooling. Match all other training settings within each comparison.
