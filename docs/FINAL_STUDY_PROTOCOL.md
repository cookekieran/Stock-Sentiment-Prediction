# Final Study Protocol

## Research Question

Do Qwen-derived semantic predicates and differentiable logic constraints improve
out-of-time MAG7 future-regime-change forecasting beyond price and leakage-safe
fundamentals?

## Information Availability

The prediction is made after the 16:00 America/New_York close. Price features use
the completed close. News is converted into the market timezone and news at or
after the cutoff is assigned to the next trading session. Existing timezone-naive
Alpha Vantage timestamps are explicitly assumed to represent America/New_York
local time. Fundamentals are joined only after `fundamental_available_from_date`.

Existing macro features are excluded because their monthly observation keys do
not prove when each value was publicly available.

## Target

`future_regime_change` is one when the rolling price regime at the 20th
subsequent ticker observation differs from the anchor regime. Earlier project
documents called this a transition. The final study uses the more precise name
because the event is common and may reflect rolling-window mechanics.

## Evaluation

Two purged walk-forward folds prevent a target observation from crossing into
the next split. Results are averaged over seeds `7`, `21`, and `42`.

Primary metric: transition PR-AUC.

Secondary metrics: transition F1, final regime macro-F1, and final accuracy.

## Matched Models

1. Persistence.
2. Price only.
3. Price plus leakage-safe fundamentals.
4. Price plus fundamentals plus Qwen predicates.
5. Price plus fundamentals plus Qwen predicates plus LTN constraints.

The test folds are used only for final reporting by the runner. Model checkpoint
selection and transition-threshold selection use each fold's validation period.
