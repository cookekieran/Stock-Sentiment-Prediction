# Dissertation Project Progress and Decision Log

**Project:** Neuro-symbolic market-regime and transition forecasting  
**Period summarised:** Initial proposal through 11 June 2026  
**Purpose:** A chronological record of what was attempted, what was learned, why the approach changed, and the current research position.

## Executive Summary

The project began as a relatively simple binary classification problem: use financial news sentiment and symbolic logic to classify the Magnificent Seven stocks as bullish or bearish. The intended contribution was to test whether a symbolic materiality layer could prevent a neural sentiment model from overreacting to financially irrelevant news.

During implementation, the research question became more precise and technically ambitious. Static article-level sentiment classification was replaced by daily ticker-level forecasting, temporal GRU models, three market states, explicit transition detection, leakage-safe fundamentals, Qwen-generated semantic predicates, differentiable Logic Tensor Network (LTN) rules, and finally a jointly trained Qwen2.5-7B + QLoRA + GRU + LTN architecture.

The major empirical finding so far is that price and leakage-safe fundamental features are the most reliable predictors on average. Qwen news predicates and LTN constraints improve interpretability and sometimes change the precision/recall trade-off, but they have not consistently improved predictive performance across seeds. A sampled end-to-end comparison produced effectively identical predictions with and without LTN pressure.

This negative result is still valuable. It suggests that economically plausible rules and highly satisfied logical formulas do not automatically provide predictive value or materially influence a large neural model. The dissertation can therefore evaluate both the promise and practical limitations of neuro-symbolic financial forecasting.

## Original Research Direction

### Initial aim

The original project definition proposed:

- collecting long-term financial news and stock-price data;
- using an open-source sentiment model such as FinBERT, FinGPT, LLaMA, or DeepSeek;
- adding a symbolic reasoning layer using static rules or differentiable LTNs;
- comparing a neural-only model against a neuro-symbolic model;
- evaluating accuracy, F1-score, and interpretability.

The initial target was a binary bullish/bearish label based on large historical price movements. The underlying hypothesis was that neural models respond to narrative sentiment, while symbolic rules could check whether the narrative was financially material.

### Why the original direction changed

Simple sentiment accuracy did not directly answer the important financial question. A model can classify the emotional tone of an article correctly without predicting whether the article affects future prices. The project therefore shifted from asking whether the language model understood sentiment to asking whether news and symbolic constraints improved out-of-time market forecasting.

## Chronological Development and Decisions

## 1. Early Model and Data Exploration

### What was attempted

- Tested FinGPT on FiQA, Twitter Financial News Sentiment, and Financial PhraseBank.
- Investigated alternative models including DeepSeek and Qwen.
- Built and renamed collection scripts for news, daily prices, intraday prices, and futures.
- Imported news and stock data into experimental notebooks.

### What was learned

Sentiment-classification accuracy alone was not sufficient evidence of market usefulness. The available sentiment datasets measured agreement with human sentiment labels, not whether sentiment predicted future stock behaviour.

Running several large decoder-only models also created storage, dependency, and environment-management problems.

### Decision

Move away from selecting a model solely through sentiment benchmark accuracy. Evaluate models through their usefulness in a downstream financial forecasting task.

## 2. Article-Level DeepSeek Baseline and LTN Model

### What was attempted

- Built an article-level DeepSeek baseline.
- Built an article-level DeepSeek + LTN model.
- Added comparison scripts and a reproducible experiment workflow.

### Why this approach was insufficient

Treating every article as an independent supervised observation did not match the intended theory of market state. One article should usually nudge an existing state rather than independently flip an entire market regime.

Article-level rows also risked over-representing days with many articles and did not naturally model persistent temporal behaviour.

### Decision

Replace independent article classification with a daily latent-state model that aggregates all news for each ticker and trading day.

## 3. Daily Latent-State GRU

### What was attempted

- Built a daily ticker-level dataset.
- Aggregated article information into daily signals.
- Added a GRU to maintain a persistent hidden market state.
- Added smoothness and logic losses to discourage implausible daily state changes.
- Added class-imbalance handling and expanded evaluation metrics.
- Added scripts to explain individual ticker/date predictions.

### Why this was an improvement

The GRU aligned better with the conceptual model: news modifies a persistent state over time. It also allowed the project to evaluate sequences instead of isolated articles.

### Problems discovered

- Stable regimes dominated the objective.
- A model could achieve apparently good accuracy by rarely predicting transitions.
- Raw market and macroeconomic levels encouraged shortcuts and chronological distribution-shift problems.
- The original 120-day regime definition was too slow and disconnected from individual news events.

### Decisions

- Use stationary inputs such as returns, volatility, drawdowns, year-on-year growth, and month-on-month macroeconomic changes.
- Exclude forward-looking diagnostic variables from training inputs.
- Change the regime definition from a 120-day framework to a more responsive 20-trading-day framework.
- Introduce explicit transition-focused evaluation.

## 4. Contextual News Predicates

### What was attempted

- Used DeepSeek/Qwen to interpret all daily news for a ticker together.
- Extracted interpretable predicates and narratives.
- Added semantic dimensions including relevance, materiality, risk-on pressure, risk-off pressure, confidence, novelty, and uncertainty.
- Mapped open-ended contextual drivers into auditable macroeconomic channels.
- Added sanitation for conflicting or incorrectly scoped LLM outputs.

### Why this was attempted

Simple sentiment was too coarse. Negative legal news, for example, may sound severe without being financially material. The project needed predicates representing why a story might matter, not merely whether its wording was positive or negative.

### Problems discovered

The LLM predicates were noisy and sometimes conflicting. Adding more news-derived features did not guarantee better transition prediction, and low-quality packets could encourage false transitions.

### Decision

Test quality filtering and controlled ablations instead of assuming all Qwen outputs were useful.

## 5. Two-Stage Transition Forecasting

### What was attempted

The three-class prediction problem was split into:

1. detecting whether a regime transition occurs; and
2. predicting the destination regime only when a transition is proposed.

The transition threshold was selected on validation data, with constraints to prevent the model from predicting transitions constantly.

### Why this change was made

The original three-class GRU was dominated by stable observations. A dedicated transition detector made the rare and important event explicit and allowed transition PR-AUC, precision, recall, and F1 to become central metrics.

### Additional analysis added

- Audited Qwen-driven overrides against persistence.
- Exported false positives, missed transitions, correct transitions, and correct persistence cases.
- Compared multiple forecast horizons.

### Problems discovered

Regime transitions based on rolling windows can occur because an old price movement leaves the lookback window, rather than because new information causes a genuine repricing event.

### Decision

Test a separate material-repricing-event formulation.

## 6. Material Repricing Event Experiment

### What was attempted

- Defined five-day risk-on and risk-off events using future return thresholds.
- Compared price-only, Qwen-only, fundamentals-only, and combined feature sets.

### Key results

| Model | Accuracy | Macro-F1 | Event F1 | Event PR-AUC |
|---|---:|---:|---:|---:|
| Price only | 0.692 | 0.304 | 0.098 | 0.384 |
| Price + Qwen | 0.579 | 0.346 | 0.394 | 0.408 |
| Fundamentals only | 0.511 | 0.388 | 0.514 | 0.397 |
| Price + fundamentals | 0.482 | 0.386 | **0.519** | **0.441** |
| Price + Qwen + fundamentals | 0.600 | **0.404** | 0.366 | 0.355 |

### What was learned

Price-only accuracy was misleading because it mostly predicted no event. Price and fundamentals produced the strongest event F1 and PR-AUC, while Qwen did not consistently add value.

### Decision

Retain material-event results as useful supporting evidence, but focus the main dissertation evaluation on transition forecasting, where the neuro-symbolic architecture and temporal state are most directly evaluated.

## 7. Leakage-Safe Fundamentals

### What was attempted

- Collected Magnificent Seven income statements and balance sheets.
- Collected earnings-release dates.
- Joined quarterly statements only after their public reported date, with an additional delay where announcement timing was uncertain.
- Used ratios and year-on-year changes rather than raw financial levels.

### Why this was necessary

Using the fiscal quarter-end date would reveal information before it became publicly available. This would create serious data leakage and invalidate results.

### Decision

Use public-availability dates and stationary fundamental transformations throughout the later experiments.

## 8. Qwen Quality Filtering and Forecast-Horizon Ablations

### What was attempted

- Neutralised Qwen packets with low confidence or high uncertainty.
- Compared raw Qwen, filtered Qwen, and price-only models over 5-, 10-, 20-, and 45-day horizons.
- Added leakage-safe fundamentals to matched experiments.

### Key findings

- Filtering improved Qwen transition PR-AUC at the 20-day horizon from approximately `0.524` to `0.584`.
- At 20 days, price + fundamentals achieved transition PR-AUC `0.615`.
- Filtered Qwen + fundamentals achieved transition PR-AUC `0.604` and higher transition recall, but lower final accuracy and stability.
- Short-horizon Qwen models often became overly conservative and predicted almost no transitions.
- The 20-day horizon provided the most useful balance for the main study.

### Decision

Use the 20-day transition task as the main experimental setting and retain price + fundamentals as the strongest conventional control.

## 9. Hand-Designed LTN Rules

### What was attempted

Added differentiable fuzzy rules such as:

- strong, material, confident news implies a transition;
- weak or irrelevant news implies persistence;
- conflicting news implies a sideways destination;
- deteriorating fundamentals plus risk-off news imply a bearish transition;
- improving fundamentals plus risk-on news imply a bullish transition.

Different rule families and logic weights were tested.

### Key findings

Many low and moderate LTN weights produced nearly identical outputs to the no-LTN baseline. Larger weights changed the precision/recall and stability trade-off but generally reduced PR-AUC.

For example:

- no-LTN PR-AUC-selected baseline: PR-AUC `0.576`, macro-F1 `0.414`;
- all-rules weight `0.5`: PR-AUC `0.576`, macro-F1 `0.458`, but much lower transition recall;
- all-rules weight `1.0`: PR-AUC `0.577`, macro-F1 `0.428`.

### What was learned

The rules could alter model behaviour, especially conservativeness and stability, but did not clearly improve transition-ranking performance. Broad rules were often too easy to satisfy or too weak to affect the learned decision boundary.

### Decision

Develop more selective rules tied to specific events, surprises, and repeated semantic shocks.

## 10. Selective and Learnable LTN Rules

### What was attempted

Created selective rule families:

- recent public earnings-release rules;
- ticker-relative fundamental surprise rules;
- repeated risk-on/risk-off semantic persistence rules;
- combined selective rules.

The generated rule features were used only by the fuzzy regulariser and were excluded from the GRU input vector.

### Key findings

On a single seed, some selective configurations improved transition F1 or stability. However, repeated-seed testing showed that these gains were not robust.

Five-seed results:

| Variant | Transition PR-AUC | Transition F1 | Macro-F1 |
|---|---:|---:|---:|
| Price + fundamentals | **0.615 +/- 0.006** | **0.365 +/- 0.189** | 0.449 +/- 0.039 |
| Price + Qwen + fundamentals | 0.570 +/- 0.023 | 0.301 +/- 0.230 | 0.436 +/- 0.033 |
| Qwen + all selective LTN rules | 0.567 +/- 0.023 | 0.287 +/- 0.217 | **0.450 +/- 0.017** |
| Qwen + persistence LTN rules | 0.576 +/- 0.027 | 0.302 +/- 0.239 | 0.439 +/- 0.031 |

### What was learned

- Price + fundamentals was the strongest and most stable transition-ranking model.
- Qwen features increased variability and did not consistently improve average performance.
- Selective LTN rules did not recover the lost PR-AUC.
- The all-rule selective LTN model had similar average macro-F1 with lower variance, suggesting a possible regularisation benefit rather than a predictive-performance benefit.

### Decision

Treat the repeated-seed result as the strongest quantitative evidence. Do not claim that Qwen or LTN consistently improves accuracy.

## 11. End-to-End Qwen + QLoRA + GRU + LTN

### Why the end-to-end approach was built

The offline pipeline used previously generated Qwen predicates as fixed inputs. This limited the ability of downstream losses and logical constraints to shape the language model's semantic representations.

The integrated architecture therefore jointly trained:

- live Qwen2.5-7B-Instruct with QLoRA adapters;
- five differentiable semantic predicate heads;
- a ticker-sequence GRU;
- transition, destination, and reaction heads;
- quantified fuzzy-LTN loss.

### Completed integrated result

The first completed integrated model achieved:

- transition PR-AUC: `0.625`;
- transition precision: `0.610`;
- transition recall: `0.379`;
- transition F1: `0.468`;
- final accuracy: `0.449`;
- final macro-F1: `0.383`;
- stable accuracy: `0.743`.

The encoded formulas had very high satisfaction values, generally above `0.94`.

### Interpretation

The integrated model achieved a competitive single-run transition PR-AUC. However, high formula satisfaction does not prove that the rules improved predictions. A matched ablation is needed to isolate the LTN contribution.

## 12. Candidate Rule Mining and Human Review

### What was attempted

- Mined candidate fuzzy implications using training data.
- Scored candidates using validation data only.
- Applied bootstrap analysis.
- Added human economic review before allowing rules into the knowledge base.

### Review outcome

From 500 candidate rules:

- `26` were marked **KEEP**;
- `190` were marked **MAYBE**;
- `284` were marked **REJECT**.

### What was learned

Many statistically strong rules were economically implausible, redundant, or simply encoded mean reversion and current-state information. This demonstrated that rule mining alone cannot replace domain review.

### Decision

Use only economically reviewed `KEEP` rules in the matched end-to-end LTN comparison.

## 13. Sampled End-to-End LTN Ablation

### Why a sampled experiment was used

A full integrated run takes approximately ten hours. Running several full seeds or a large hyperparameter grid was not practical. A capped, one-epoch matched comparison was therefore run before committing further GPU time.

The experiment compared identical data, seed, architecture, and training settings:

- `no_ltn`: the 26 approved rules were audited, but logic weight was zero;
- `keep_ltn`: the same rules contributed with logic weight `0.2`.

### Results

| Metric | No LTN | Approved LTN | Difference |
|---|---:|---:|---:|
| Transition PR-AUC | 0.496288 | 0.496079 | -0.000209 |
| Transition precision | 0.542373 | 0.542373 | 0 |
| Transition recall | 0.310680 | 0.310680 | 0 |
| Transition F1 | 0.395062 | 0.395062 | 0 |
| Final accuracy | 0.420000 | 0.420000 | 0 |
| Macro-F1 | 0.407505 | 0.407505 | 0 |
| Stable accuracy | 0.721649 | 0.721649 | 0 |
| Predicted transitions | 59 | 59 | 0 |

### What was learned

The LTN loss was calculated and added correctly, but at weight `0.2` it did not meaningfully alter the model during the sampled run. The two models produced exactly the same thresholded predictions.

### Decision

Do not immediately spend substantial GPU time on multiple full end-to-end ablations. The sampled result, repeated-seed offline experiments, and existing full integrated run provide enough evidence to discuss the limited practical influence of LTN under the tested settings.

## Current Research Position

The project should no longer be framed as an attempt to prove that neuro-symbolic logic necessarily improves financial prediction accuracy.

A defensible final research question is:

> Does integrating learned semantic predicates and differentiable logical constraints improve the accuracy, reliability, and interpretability of market-regime transition forecasting?

The evidence currently supports the following answer:

- **Accuracy and transition ranking:** not consistently; price and fundamentals are stronger on average.
- **Reliability:** LTN rules can alter stability and reduce variability in some settings, but the effect depends strongly on rule weight and seed.
- **Interpretability:** yes; semantic predicates, rule satisfaction, reviewed rules, and exported audit trails provide explanations unavailable from the conventional baseline.
- **Practical limitation:** highly satisfied logical constraints may have little influence on predictions, and economically plausible rules may not contain additional predictive information.

## Major Methodological Contributions

- Reframed sentiment classification as out-of-time transition forecasting.
- Built daily ticker-level temporal datasets rather than treating articles independently.
- Added chronological train, validation, and test splits.
- Removed unsafe forward-looking and non-stationary inputs.
- Built leakage-safe quarterly fundamental features using public release dates.
- Developed interpretable daily Qwen semantic predicates.
- Implemented two-stage transition and destination forecasting.
- Added transition-specific audits and case-study outputs.
- Tested raw, filtered, selective, learnable, mined, and jointly trained neuro-symbolic rules.
- Conducted repeated-seed robustness analysis.
- Demonstrated the importance of human economic review for mined rules.
- Produced evidence about the limitations of logical constraints in large neural financial models.

## Important Results Files

### Strongest repeated-seed evidence

`Sentiment project/models/qwen_quality_filter_fundamentals_repeated_seed_ablation/horizon_20/repeated_seed_summary.csv`

### Selective LTN comparison

`Sentiment project/models/qwen_quality_filter_fundamentals_selective_ltn_ablation/horizon_20/selective_ltn_ablation_summary.csv`

### Feature and horizon comparison

`Sentiment project/models/qwen_quality_filter_fundamentals_transition_ablation/full_ablation_summary.csv`

### Material-event experiment

`Sentiment project/models/material_events_horizon_5_fundamentals/ablation_summary.csv`

### General LTN weight comparison

`Sentiment project/models/qwen_quality_filter_fundamentals_transition_ltn_ablation/horizon_20/ltn_ablation_summary.csv`

### Integrated Qwen + QLoRA + GRU + LTN result

`Sentiment project/data/processed/end_to_end_neurosymbolic/qwen_qlora_adapter/test_metrics.json`

### Sampled approved-rule ablation

This was generated in the Jupyter repository:

`models/end_to_end_neurosymbolic/keep_rule_ltn_ablation_sample/ltn_ablation_summary.csv`

### Reviewed rules

`Sentiment project/outputs/end_to_end_neurosymbolic/rule_review/candidate_ltn_rules_reviewed.csv`

## Recommended Next Steps

1. Freeze the main experimental design and avoid adding further architectures.
2. Preserve the sampled end-to-end ablation outputs from Jupyter.
3. Use the repeated-seed experiment as the principal quantitative result.
4. Calculate bootstrap confidence intervals from saved predictions where useful.
5. Select representative prediction and rule cases for interpretability analysis.
6. Write the methodology and results chapters before undertaking further expensive runs.
7. Frame negative results clearly: adding semantic and logical complexity did not reliably outperform the simpler price-and-fundamentals control.

## Concise Meeting Summary

The project progressed from basic sentiment classification to a leakage-aware, temporal, end-to-end neuro-symbolic transition forecasting system. Each change addressed a specific weakness: article independence led to daily aggregation and GRUs; stable-class dominance led to two-stage transition detection; rolling-label artefacts led to material-event testing; noisy Qwen outputs led to filtering; generic rules led to selective and mined rules; fixed predicates led to a jointly trained Qwen + QLoRA architecture.

The strongest repeated-seed result is that price and fundamentals outperform Qwen and LTN variants on transition PR-AUC. The integrated neuro-symbolic model is competitive in one full run and provides much stronger interpretability, but a sampled matched ablation showed no measurable difference between approved LTN rules and no LTN. The emerging dissertation contribution is therefore a rigorous evaluation of when neuro-symbolic constraints help, when they do not, and why interpretability does not necessarily translate into improved predictive accuracy.
