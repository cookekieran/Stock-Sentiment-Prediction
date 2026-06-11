# Explainable MAG7 Market-Regime Prediction: Project Decision History

## Purpose of This Document

This document records the development of the dissertation project, with particular
attention to:

- the decisions that were made;
- the evidence that motivated each decision;
- approaches that did not work;
- why the project moved to the next approach;
- the current architecture and strongest results;
- the work that should be completed next.

The project is an explainable neuro-symbolic market-regime prediction system for:

```text
AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
```

The target regimes are:

```text
0 = bear_drawdown
1 = sideways
2 = bull_rally
```

The intended prediction target is the future ticker-level market regime, rather
than sentiment about an individual article.

---

## 1. Original Article-Level Approach

### Initial Idea

The earliest design treated individual financial-news articles as separate model
inputs. An LLM would interpret each article, and the resulting representation
would be used to predict a market regime.

### Why It Was Rejected

This framing was not economically realistic. A single article should not normally
cause an entire ticker-level regime to change immediately. Market regimes are
persistent states that evolve from accumulated evidence, price action, company
information, and broader economic conditions.

An article-level design also creates several problems:

- multiple articles on the same day can produce contradictory predictions;
- days with more articles receive more state updates;
- individual low-quality articles can have excessive influence;
- the model does not naturally represent regime persistence.

### Decision

Replace article-level prediction with a daily ticker-level latent-state system.

The revised conceptual architecture became:

```text
daily ticker news packet
+ ticker price context
+ macroeconomic context
+ current ticker regime
        ↓
LLM extracts contextual daily drivers
        ↓
logic rules reason over relevant predicates
        ↓
GRU updates a persistent ticker-level hidden state once per day
        ↓
classifier predicts a future regime
```

---

## 2. DeepSeek-R1 Versus Qwen

### DeepSeek-R1 Experiment

The first structured-predicate extraction model tested was:

```text
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

### What Went Wrong

The reasoning-style model frequently produced long reasoning passages instead of
reliable raw JSON. Explicit instructions to return only JSON were not sufficient.

This made it unsuitable for a pipeline that required predictable schema-constrained
daily predicates.

### Decision

Replace DeepSeek-R1 with:

```text
Qwen/Qwen2.5-7B-Instruct
```

Qwen produced structured contextual predicate output more reliably.

### Dissertation Interpretation

The decision was not that DeepSeek was incapable of reasoning. It was that the
reasoning-style distilled model was poorly suited to dependable structured
extraction in this pipeline.

A suitable dissertation statement is:

> Reasoning-style DeepSeek-R1 distilled models were tested but found unreliable
> for schema-constrained predicate extraction. An instruction-tuned Qwen model
> was therefore used to operationalise the contextual reasoning layer into
> structured daily predicates.

---

## 3. Daily Qwen Predicate Extraction

### Implemented Approach

The extraction script groups all news for a ticker and anchor trading date into
one packet. Qwen processes the packet once and returns:

- a daily market narrative;
- open-ended driver labels and summaries;
- driver scope;
- directional pressure;
- intensity;
- novelty;
- confidence;
- time horizon;
- relevant semantic channels;
- net pressure;
- update strength;
- uncertainty.

The important design choice was to keep driver labels open-ended. Examples include:

```text
AI capex optimism
Fed pivot uncertainty
semiconductor export restrictions
Iran conflict escalation
Tesla delivery weakness
```

### Completed Extraction

The full MAG7 extraction produced:

```text
5,115 ticker/day packets
```

The main generated files included:

```text
data/processed/daily_contextual_driver_predicates_mag7_qwen.parquet
data/processed/daily_contextual_driver_predicates_mag7_qwen_sanitized.parquet
```

### Apparent Invalid JSON Problem

A diagnostic reported hundreds of invalid `raw_generation` rows. A later check
reported:

```text
Invalid rows: 397
```

Every supposedly invalid generation was exactly 2,000 characters long.

The cause was:

```python
row["raw_generation"] = generated[:2000]
```

The saved raw diagnostic preview was truncated to 2,000 characters. The parsed
structured fields had already been extracted successfully.

Verification showed:

```text
rows: 5115
rows with extracted drivers: 5115
rows without extracted drivers: 0
blank narratives: 0
```

### Decision

Do not regenerate the full Qwen dataset. The apparent JSON failure concerned only
the stored raw preview, not the usable parsed columns.

### Remaining Qwen Quality Concerns

Manual inspection still revealed genuine semantic noise:

- unrelated-company news was sometimes classified as ticker-level;
- irrelevant stories sometimes received directional pressure;
- Qwen sometimes invented internal variable names;
- some driver scopes were imperfect;
- neutral days were not always explicitly represented as neutral.

These issues later motivated confidence filtering and more selective rule design.

---

## 4. Clarifying the Neuro-Symbolic Architecture

### Initial Confusion

There was uncertainty over whether the system counted as neuro-symbolic AI and
whether an LTN should be directly attached to Qwen.

### Current Interpretation

The current system is a practical neuro-symbolic pipeline:

```text
Qwen extracts predicates
        ↓
GRU learns sequential representations
        ↓
differentiable fuzzy rules regularise predictions
```

The current implementation is not a fully end-to-end LTN connected directly to
Qwen:

- Qwen is not updated during GRU training;
- logic losses do not backpropagate into Qwen;
- the GRU and fuzzy rule losses are differentiable;
- predicates provide an interpretable interface between language and prediction.

### Decision

Continue with the practical modular architecture rather than attempting a
computationally expensive end-to-end Qwen-plus-LTN system during the main
dissertation experiments.

The safest description is:

> An LTN-style fuzzy-regularized neuro-symbolic regime-transition model.

The rules primarily provide auditable reasoning constraints and regularization.
They do not make the internal reasoning of Qwen itself fully explainable.

---

## 5. Initial Latent-State GRU And Persistence Failure

### Original GRU Objective

The first GRU predicted one of three future regimes directly. It used a persistence
prior because current regimes are usually informative about future regimes.

### Initial Label Distribution

The first labels were heavily dominated by `bull_rally`, particularly in the
training and validation periods.

Example initial split counts:

```text
train:
bear_drawdown = 339
sideways      = 849
bull_rally    = 2086

validation:
bear_drawdown = 110
sideways      = 25
bull_rally    = 286
```

### Initial Results

Persistence baseline:

```text
accuracy = 0.7620
macro-F1 = 0.6420
```

GRU without rules:

```text
accuracy = 0.7539
macro-F1 = 0.6338
transition accuracy = 0.0000
stable accuracy = 0.9894
```

GRU with initial rules:

```text
accuracy = 0.7539
macro-F1 = 0.6346
transition accuracy = 0.0031
stable accuracy = 0.9884
```

### What This Revealed

The model was effectively copying the current regime. It performed well on easy
stable days and almost completely failed when regimes changed.

This was a critical evaluation insight: overall accuracy alone made the model
appear useful even when it learned almost nothing about regime transitions.

### Attempted Repair: Reduce Persistence And Add Class Weighting

Reducing persistence strength and applying inverse class weighting made the model
unstable. In one rules experiment, it collapsed toward bear predictions and
performance deteriorated sharply.

### Decision

Stop treating persistence strength and class balancing as the main solution.
Evaluate transition and stable cases separately, and reconsider the target
definition and model framing.

---

## 6. Mapping Qwen Drivers To Macroeconomic Variables

### Problem

Qwen generated useful semantic channel descriptions such as:

```text
inflation pressure
Fed policy
interest rates
yield curve inversion
oil prices
market volatility
```

However, the code expected specific names such as:

```text
macro_cpi_yoy
macro_fed_funds_rate
macro_treasury_10y_2y_spread
macro_wti_oil
macro_vix
macro_unemployment_rate
```

The naming mismatch meant the macro logic gate rarely activated.

### Implemented Repair

A deterministic keyword mapping translated open-ended Qwen text into controlled
macro channel flags.

Example mappings:

```text
inflation, CPI, consumer prices -> macro_cpi_yoy
Fed, interest rates, tightening -> macro_fed_funds_rate
yield curve, Treasury spread -> macro_treasury_10y_2y_spread
VIX, volatility, risk aversion -> macro_vix
oil, crude, energy prices -> macro_wti_oil
unemployment, payrolls, job market -> macro_unemployment_rate
```

### Results

The mapping successfully activated macro channels, but the macro-mapped GRU still
failed to detect transitions effectively:

```text
accuracy = 0.7126
macro-F1 = 0.5999
transition accuracy = 0.0062
stable accuracy = 0.9333
```

### Decision

Keep the mapping because it improves semantic correctness and explainability, but
do not treat it as a solution to predictive transition detection.

---

## 7. Revising The Regime Labels

### Problem

Plots showed that the original labels classified long periods as bullish and did
not align clearly with meaningful price drops or rallies.

### Revised Label Proposal

A more balanced definition used recent return and drawdown:

```python
if return_20d <= -0.05 or drawdown_60d <= -0.10:
    return "bear_drawdown"
if return_20d >= 0.05:
    return "bull_rally"
return "sideways"
```

### Resulting Balance

```text
bull_rally       35.3%
bear_drawdown    32.4%
sideways         32.3%
```

This was substantially more balanced overall, although ticker-specific differences
remained. TSLA, for example, had a much larger bear proportion.

### Decision

Use the revised labels because they better represent economically meaningful
price behavior and reduce the original bull-class dominance.

---

## 8. Feature Ablations And The Macro Collapse

### Reason For Ablations

The model continued to produce implausible class distributions. Explicit feature
sets were introduced to determine whether price, Qwen predicates, or macro data
were causing the problem.

Feature sets included:

```text
price_only
price_macro
price_qwen
price_fundamentals
price_qwen_fundamentals
full
```

### Price-Only Result

```text
accuracy = 0.4495
macro-F1 = 0.4365
transition accuracy = 0.2295
stable accuracy = 0.6882
```

The price-only model produced all three classes and behaved relatively sensibly.

### Price-Plus-Macro Result

```text
accuracy = 0.3773
macro-F1 = 0.1826
predicted bear = 1357
predicted sideways = 0
predicted bull = 0
```

The model predicted bear for every test row.

### Interpretation

Raw macroeconomic level variables were likely nonstationary and encoded time-period
differences rather than useful transition signals. The model learned that the
validation/test economic environment looked different from training and treated
that shift as universally bearish.

### Decision

- Do not use raw macroeconomic levels indiscriminately.
- Prefer MoM, YoY, spread, and change-based macro features.
- Keep macro inputs separate in ablations.
- Do not assume more data automatically improves the model.

### Price-Plus-Qwen Result

```text
accuracy = 0.4031
macro-F1 = 0.3694
transition accuracy = 0.4136
stable accuracy = 0.3917
```

Qwen improved sensitivity to transitions but caused many false positives and
substantially reduced stable accuracy.

This established a recurring pattern:

> Qwen increases transition responsiveness, but its noisy signals make the model
> override stable regimes too frequently.

---

## 9. Hybrid Persistence Override

### Motivation

The price-plus-Qwen GRU detected more transitions than persistence but performed
poorly on stable days. A hybrid system used persistence as the default and allowed
the model to override it only when the alternative regime probability exceeded
the current-regime probability by a threshold.

### Example Results

```text
Persistence:
accuracy = 0.4797
macro-F1 = 0.4725
transition accuracy = 0.0000
stable accuracy = 1.0000

Raw price + Qwen GRU:
accuracy = 0.4031
macro-F1 = 0.3694
transition accuracy = 0.4136
stable accuracy = 0.3917

Hybrid threshold 0.21:
accuracy = 0.4886
macro-F1 = 0.4729
transition accuracy = 0.0977
stable accuracy = 0.9124
```

### Interpretation

The hybrid slightly beat persistence overall while retaining some transition
detection. However, it still detected relatively few true transitions.

### Decision

Keep the idea of a conservative transition gate, but change the model framing so
transition detection is learned explicitly rather than derived indirectly from
three-class probabilities.

---

## 10. Forecast-Horizon Experiments

### Motivation

A 20-day regime forecast may be too difficult for daily news. Experiments compared:

```text
5-day, 10-day, and 20-day horizons
```

### Price-Plus-Qwen Results

#### 5-Day

```text
Persistence accuracy = 0.6714
Persistence macro-F1 = 0.6672

Price + Qwen accuracy = 0.5932
Price + Qwen macro-F1 = 0.5602
transition accuracy = 0.3009
stable accuracy = 0.7362
```

#### 10-Day

```text
Persistence accuracy = 0.5626
Persistence macro-F1 = 0.5572

Price + Qwen accuracy = 0.4488
Price + Qwen macro-F1 = 0.4448
transition accuracy = 0.3089
stable accuracy = 0.5575
```

#### 20-Day

```text
Persistence accuracy = 0.4797
Persistence macro-F1 = 0.4725

Price + Qwen accuracy = 0.4031
Price + Qwen macro-F1 = 0.3694
transition accuracy = 0.4136
stable accuracy = 0.3917
```

### Interpretation

Shorter horizons were easier overall, but the model still did not beat persistence.
At longer horizons, it detected more transitions but made substantially more
stable-day errors.

### Decision

The central challenge was not simply choosing a better horizon. The classification
objective itself needed to focus explicitly on regime changes.

---

## 11. Moving To A Two-Stage Transition Model

### New Framing

The three-class model was dominated by easy stable rows. The task was reformulated:

```text
Stage 1: Will the regime change within the forecast horizon?

Stage 2: If a transition is likely, which regime comes next?
```

This produced:

```text
src/models/train_two_stage_transition_gru.py
```

### Why This Was Better

The two-stage structure directly optimizes the difficult question instead of
allowing stable rows to dominate the loss. It also supports clearer evaluation:

- transition precision;
- transition recall;
- transition F1;
- transition PR-AUC;
- destination accuracy;
- stable accuracy;
- final regime accuracy.

### Initial Problem

Early two-stage models often predicted nearly every row as a transition. Threshold
tuning was required to balance transition recall against stable accuracy.

Example tuned results:

```text
5-day:
transition F1 = 0.4381
stable accuracy = 0.7341

10-day:
transition F1 = 0.4155
stable accuracy = 0.7699

20-day:
transition F1 = 0.3626
stable accuracy = 0.7803
```

### Decision

Keep the two-stage architecture as the primary model. It exposes the actual
research problem much more honestly than the original three-class GRU.

---

## 12. Oracle Diagnostics

### Purpose

Oracle experiments were used to identify whether performance was primarily limited
by transition detection or by predicting the destination after a transition.

### Results

#### 5-Day

```text
Actual two-stage accuracy = 0.5661
Oracle transition-detector accuracy = 0.8585
Oracle destination-classifier accuracy = 0.8137
Destination accuracy when transition detected = 0.5150
```

#### 10-Day

```text
Actual two-stage accuracy = 0.5064
Oracle transition-detector accuracy = 0.7866
Oracle destination-classifier accuracy = 0.7112
Destination accuracy when transition detected = 0.4928
```

#### 20-Day

```text
Actual two-stage accuracy = 0.4458
Oracle transition-detector accuracy = 0.7760
Oracle destination-classifier accuracy = 0.6183
Destination accuracy when transition detected = 0.5160
```

### Interpretation

There is substantial potential performance if the model can identify which rows
are transitions. Destination classification is imperfect but not the primary
bottleneck.

### Decision

Focus subsequent work on improving transition PR-AUC and transition detection,
rather than optimizing overall three-class accuracy.

---

## 13. Filtering Qwen Predicate Data

### Hypothesis

Qwen predicates might be useful only when Qwen is confident and uncertainty is
low. Including every predicate row may make the news channel too noisy.

### Implemented Filter

The filtered-Qwen dataset retained predicates satisfying approximately:

```text
mean confidence >= 0.70
uncertainty <= 0.30
```

Low-quality Qwen rows were neutralized rather than removed, preserving the daily
time series.

### Result

Approximately:

```text
58.50% of ticker/day packets were retained
```

### Interpretation

Filtering reduced noise and produced a more defensible news representation.
However, filtered Qwen still did not independently solve transition detection.

### Decision

Use filtered Qwen for combined experiments. Do not regenerate Qwen output yet.
First test deterministic filtering, scope weighting, and combinations with better
company-specific information.

---

## 14. Adding Company Fundamentals

### Motivation

Daily news may not be meaningful without company-specific financial context. For
example, an earnings-related article may be more useful when accompanied by actual
revenue, margin, cash-flow, or balance-sheet changes.

### Data Source Decision

Yahoo Finance was considered but did not reliably provide sufficiently complete
quarterly history back to 2023.

Alpha Vantage was selected for historical quarterly:

- income statements;
- balance sheets;
- earnings release dates.

### Leakage Risk

Quarterly financial information must not be visible before its public release.
The join therefore uses the actual reported/public availability date rather than
the fiscal quarter end.

### Decision

Create leakage-safe daily fundamentals features that become available only after
the statement's public release date.

### Fundamentals Ablation Results At 20 Days

| Variant | Transition F1 | PR-AUC | Transition Accuracy | Stable Accuracy |
|---|---:|---:|---:|---:|
| Price only | 0.0353 | 0.5694 | 0.0092 | 0.9724 |
| Filtered Qwen | 0.4186 | 0.5838 | 0.1905 | 0.7561 |
| Price + fundamentals | 0.5532 | **0.6145** | 0.2488 | 0.6423 |
| Price + filtered Qwen + fundamentals | **0.6220** | 0.6042 | **0.3333** | 0.4878 |

### Interpretation

This was one of the most important findings:

- fundamentals clearly improve transition prediction;
- fundamentals-only provides the best current transition ranking;
- filtered Qwen plus fundamentals increases recall and transition F1;
- Qwen still introduces more false positives and reduces stability.

### Decision

Use company fundamentals as a core input. Treat Qwen as a potentially useful
contextual signal that improves sensitivity, rather than the strongest standalone
predictor.

---

## 15. Generic LTN-Style Rules

### Initial Rules

Differentiable fuzzy implication losses were added:

```text
loss = transition_loss + destination_loss + logic_weight * logic_loss
```

Generic rules included:

1. Material, novel, confident, low-uncertainty news implies transition.
2. Weak or irrelevant news implies persistence.
3. Conflicting risk-on and risk-off news implies sideways destination.
4. Declining revenue and operating margin with risk-off news imply transition
   and bear destination.
5. Improving revenue and operating margin with risk-on news imply transition
   and bull destination.

### Results

The generic rules produced almost no PR-AUC improvement:

```text
best generic rules PR-AUC ≈ 0.6043
no-rules combined model PR-AUC ≈ 0.6042
```

Strong weights often harmed ranking.

### Interpretation

The rules were economically plausible but too broad. They activated on too many
ordinary situations and did not distinguish true regime changes from normal
market variation.

### Decision

Keep the rules for explainability and regularization, but develop more selective
antecedents.

---

## 16. Selective Leakage-Safe LTN Rules

### New Hypotheses

Rules should focus on conditions more likely to be genuinely discriminative:

- proximity to an earnings release;
- unusually large changes relative to the ticker's own history;
- persistent Qwen shocks across multiple days;
- eventually, ticker-specific thresholds.

### Added Rule Features

```text
rule_days_since_fundamental_release
rule_release_proximity_10d
rule_revenue_yoy_ticker_z
rule_operating_margin_yoy_change_ticker_z
rule_risk_on_shock_mean_3d
rule_risk_off_shock_mean_3d
rule_risk_on_shock_persistence_3d
rule_risk_off_shock_persistence_3d
```

### Leakage Controls

- release proximity uses public availability dates;
- ticker-relative z-scores compare each quarterly release against prior public
  releases only;
- repeated daily copies of quarterly data are not treated as new observations;
- Qwen persistence uses only trailing/current days;
- `rule_*` features are visible to fuzzy rules but excluded from GRU inputs.

Timing audit:

```text
release_date_after_anchor = 0
negative_days_since_release = 0
```

### Selective Rule Families

#### `selective_release`

```text
recent public quarterly release
AND material contextual news
    => transition
```

#### `selective_surprise`

```text
large negative ticker-relative revenue and margin surprise
AND recent release
AND matching Qwen risk-off pressure
    => transition and bear destination
```

The positive equivalent implies a bull destination.

#### `selective_persistence`

```text
persistent trailing risk-off Qwen shocks
    => transition and bear destination
```

The risk-on equivalent implies a bull destination.

#### `selective_all`

Combines all selective rule families.

### Results

| Experiment | Transition F1 | PR-AUC | Final Accuracy | Stable Accuracy |
|---|---:|---:|---:|---:|
| Matched no-rules run | 0.4762 | 0.5763 | 0.4265 | 0.6797 |
| Light selective rules | **0.6220** | **0.6042** | 0.4084 | 0.4878 |
| Combined rules, weight 0.5 | 0.5950 | 0.6041 | 0.4368 | 0.6033 |
| Persistent-shock rules, weight 5.0 | 0.5264 | 0.6035 | 0.4447 | 0.6976 |
| Combined rules, weight 5.0 | 0.3750 | 0.5758 | 0.4518 | 0.8114 |

### Interpretation

Light rules produced a useful transition F1 and recall. Strong rules mostly made
the model more conservative, increasing stable accuracy while missing more
transitions.

However, the best selective-rule PR-AUC did not exceed the fundamentals-only
benchmark of `0.6145`. Small rule perturbations also changed which checkpoint
early stopping selected. Therefore, the apparent improvement cannot yet be
confidently attributed to the rules themselves.

### Decision

Present the selective rules as:

- auditable economic constraints;
- interpretable regularizers;
- tools for controlling the transition-versus-stability tradeoff.

Do not claim they have conclusively generated independent predictive improvement.

---

## 17. Current Best Models By Objective

There is no single universally best model.

### Best Transition Ranking

```text
Price + fundamentals
PR-AUC = 0.6145
```

This is currently the strongest model if the objective is ranking likely
transitions.

### Best Transition Sensitivity

```text
Price + filtered Qwen + fundamentals
transition F1 = 0.6220
transition accuracy = 0.3333
```

This detects more transitions but creates more false positives.

### Best Interpretable Stability/Transition Compromise

```text
Selective-all rules, weight 0.5
PR-AUC = 0.6041
transition F1 = 0.5950
stable accuracy = 0.6033
```

---

## 18. Main Research Conclusions So Far

1. Overall regime accuracy is misleading when stable days dominate.

2. Persistence is a strong baseline but detects no transitions.

3. The original three-class GRU mostly learned persistence.

4. Reducing persistence strength or applying class weights did not solve the
   underlying problem.

5. Revised labels produced a much more meaningful and balanced task.

6. Raw macroeconomic level features were harmful and caused model collapse.

7. Qwen predicates increase transition sensitivity but also increase false
   positives.

8. Filtering low-confidence and high-uncertainty Qwen predicates is preferable
   to using all generated predicates.

9. Company fundamentals provide the strongest current improvement in transition
   ranking.

10. Combining fundamentals and Qwen improves transition recall and F1 but reduces
    stable accuracy.

11. Two-stage transition detection is a more suitable problem formulation than
    direct three-class prediction.

12. Generic LTN-style rules are too broad to improve prediction materially.

13. Selective LTN-style rules improve interpretability and allow control over
    model behavior, but their independent predictive benefit is not yet proven.

14. The primary remaining bottleneck is correctly ranking transition risk.

---

## 19. Current Recommended Dissertation Framing

> This dissertation develops a neuro-symbolic market-regime transition pipeline
> for MAG7 equities. An instruction-tuned Qwen model converts daily ticker-level
> news packets into structured contextual predicates. A recurrent model combines
> price context, selected predicates, and leakage-safe quarterly fundamentals to
> estimate regime-transition risk and destination. Differentiable fuzzy rules
> encode economically motivated constraints, improving interpretability and
> allowing controlled regularization. Experiments show that company fundamentals
> contribute the strongest transition-ranking improvement, while contextual Qwen
> predicates increase transition sensitivity at the cost of false positives.
> Symbolic rules are most useful as auditable regularizers rather than a
> guaranteed source of predictive uplift.

Avoid claiming:

- that the LTN rules dramatically improve prediction;
- that the rules explain Qwen's internal reasoning;
- that Qwen is trained end-to-end with the symbolic system;
- that overall accuracy alone demonstrates useful regime prediction.

---

## 20. Important Current Files

Core extraction and data files:

```text
src/data/extract_daily_contextual_drivers.py
src/data/build_latent_state_dataset.py
src/data/enrich_daily_contextual_driver_macro_channels.py
src/data/build_filtered_qwen_transition_datasets.py
src/data/fetch_mag7_alphavantage_fundamentals.py
src/data/fetch_mag7_alphavantage_earnings_dates.py
src/data/enrich_selective_ltn_rule_features.py
```

Core model files:

```text
src/models/train_latent_state_gru.py
src/models/train_two_stage_transition_gru.py
src/models/explain_market_state_prediction.py
src/models/README.md
```

Important datasets and result folders:

```text
data/processed/daily_contextual_driver_predicates_mag7_qwen_sanitized.parquet
data/processed/qwen_quality_filter_fundamentals_transition_ablation
data/processed/qwen_quality_filter_fundamentals_selective_ltn/horizon_20
models/qwen_quality_filter_fundamentals_transition_ltn_ablation/horizon_20
models/qwen_quality_filter_fundamentals_selective_ltn_ablation/horizon_20
```

Latest selective-rule summary:

```text
models/qwen_quality_filter_fundamentals_selective_ltn_ablation/horizon_20/selective_ltn_ablation_summary.csv
```

---

## 21. Work That Should Be Done Next

### Priority 1: Repeated-Seed Experiments

The current LTN results are sensitive to checkpoint selection. Repeat the main
20-day experiments using:

```text
seeds = 7, 21, 42, 84, 123
```

Compare:

```text
price_fundamentals, no rules
price_qwen_fundamentals, no rules
price_qwen_fundamentals, selective_all weight 0.5
price_qwen_fundamentals, selective_persistence weight 5.0
```

Aggregate mean and standard deviation for:

```text
transition PR-AUC
transition precision
transition recall
transition F1
final accuracy
macro-F1
transition accuracy
stable accuracy
```

This is required before claiming that rules provide a reliable improvement.

### Priority 2: Walk-Forward Validation

Use several chronological train/validation/test folds. This will determine whether
results generalize across different market periods rather than depending on one
validation block.

### Priority 3: Ticker-Specific Transition Thresholds

MAG7 stocks have very different volatility and transition behavior. Tune
transition thresholds separately for each ticker using validation data only.

This may improve the stable-versus-transition tradeoff without changing the model.

### Priority 4: Transition Error Review

Create an inspection table for:

- false-positive transitions;
- missed transitions;
- correctly detected transitions.

Include:

```text
ticker
date
current and future regime
transition probability
price changes and drawdown
days since earnings release
fundamental surprises
Qwen narrative and top drivers
Qwen confidence and uncertainty
activated rules
```

This will show whether missed transitions were:

- gradual price moves with no obvious news catalyst;
- caused by poor Qwen predicates;
- driven by delayed earnings reactions;
- label artifacts;
- ticker-specific behavior.

### Priority 5: Richer Leakage-Safe Fundamental Features

Potential additions:

```text
EPS surprise
revenue acceleration
gross-margin change
free-cash-flow change
operating-income growth
capital-expenditure growth
R&D growth
cash change
debt change
```

Each should use public release timing and ticker-relative historical changes.

### Priority 6: Improve Qwen Filtering Before Regeneration

Test deterministic filtering using:

```text
confidence
uncertainty
scope
intensity
novelty
materiality
```

Potential scope weights:

```text
ticker        = 1.00
sector        = 0.60
macro         = 0.50
broad_market  = 0.35
irrelevant    = 0.00
```

Do not regenerate all Qwen predicates until deterministic filtering has been
fully evaluated.

### Priority 7: Calibration And Precision-Recall Analysis

Calibrate transition probabilities using validation data and report:

- precision-recall curves;
- PR-AUC;
- results at several operating thresholds;
- false positives per ticker/month.

Thresholds must never be selected using test data.

### Priority 8: Simpler Baseline Models

Compare the GRU against:

```text
logistic regression
random forest
XGBoost or LightGBM
simple MLP
```

Use identical chronological splits and leakage controls. If simpler models perform
similarly, that is still a valid and useful dissertation finding.

### Priority 9: Final Explainability Outputs

For each transition prediction, generate:

```text
current regime
transition probability
destination probabilities
price context
earnings-release proximity
ticker-relative fundamental surprises
retained Qwen drivers
filtered-out Qwen drivers
activated fuzzy rules
rule satisfaction values
final decision
```

This is where the neuro-symbolic contribution is strongest, even if predictive
improvement from rules remains modest.

---

## 22. Recommended Immediate Order

```text
1. Commit the latest selective-rule code.
2. Run repeated-seed 20-day experiments.
3. Aggregate mean and standard-deviation metrics.
4. Add ticker-specific validation thresholds.
5. Review false-positive and missed transitions.
6. Add richer leakage-safe earnings features.
7. Run chronological walk-forward evaluation.
8. Improve deterministic Qwen filtering.
9. Compare against simpler transition models.
10. Produce final explanation reports and dissertation tables.
```

Suggested commit message for the latest selective-rule changes:

```text
Add leakage-safe selective LTN transition rules
```

