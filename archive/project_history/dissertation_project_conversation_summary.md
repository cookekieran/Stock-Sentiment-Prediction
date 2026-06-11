# Dissertation Project: Decision History and Handover Summary

## 1. Project Objective

The project aims to build an explainable neuro-symbolic market-regime
transition prediction system for the MAG7 stocks:

- AAPL
- AMZN
- GOOGL
- META
- MSFT
- NVDA
- TSLA

The target is the ticker-level future market regime:

- `0`: bear drawdown
- `1`: sideways
- `2`: bull rally

The project evolved from a simple news-sentiment classifier into a
transition-focused neuro-symbolic system combining:

- daily ticker-level news packets;
- price and market-regime context;
- leakage-safe company fundamentals;
- a Qwen language model;
- a persistent GRU ticker state;
- transition and destination prediction heads;
- differentiable fuzzy logic rules.

The central research question became:

> Can contextual language-model representations, company fundamentals, and
> economically motivated fuzzy logic improve the detection and explanation of
> market-regime transitions beyond price persistence?

---

## 2. Why the Article-Level Approach Was Rejected

The original concept treated individual articles as prediction inputs.
However, one article should not normally cause an entire ticker's market
regime to change.

This created a mismatch between the unit of information and the unit of
prediction:

- the prediction target was a daily ticker-level regime;
- the input was a single article;
- several articles could describe the same event;
- irrelevant or minor articles could create false transition signals.

### Decision

News was grouped into one packet per ticker and trading day. The model therefore
sees the complete daily narrative before updating the ticker state.

### Revised input structure

```text
daily ticker news packet
+ price context
+ macroeconomic context
+ company fundamentals
+ current ticker regime
        |
        v
contextual predicates / Qwen representation
        |
        v
GRU ticker state
        |
        v
transition detector and destination classifier
```

---

## 3. DeepSeek Was Replaced by Qwen

The first contextual extraction experiments used:

```text
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

DeepSeek frequently returned long reasoning traces instead of reliable
schema-constrained JSON. This made automated downstream parsing unreliable.

The model was replaced with:

```text
Qwen/Qwen2.5-7B-Instruct
```

Qwen was substantially more reliable at producing structured outputs.

### Decision

Use Qwen as the contextual language layer because instruction-following and
structured extraction were more important than verbose reasoning.

### Defensible dissertation wording

Reasoning-style DeepSeek-R1 distilled models were tested but found unreliable
for schema-constrained predicate extraction. An instruction-tuned Qwen model
was therefore used to operationalise the contextual reasoning layer.

---

## 4. Offline Daily Qwen Predicate Extraction

The first Qwen-based architecture used an offline extraction stage:

```text
Qwen daily packet extraction
        |
        v
saved structured contextual predicates
        |
        v
GRU and fuzzy-rule experiments
```

The extraction script grouped news by ticker and anchor trading date and
generated one row per ticker/day.

Completed extraction:

- 5,115 ticker/day packets;
- all rows contained extracted drivers;
- no blank narratives;
- output stored as Parquet.

### Apparent Invalid JSON Problem

Hundreds of saved `raw_generation` values appeared to be invalid JSON.
Investigation showed that the diagnostic string had been deliberately
truncated:

```python
row["raw_generation"] = generated[:2000]
```

Every apparently invalid raw generation had exactly 2,000 characters. The
parsed structured fields were still valid.

### Decision

Do not regenerate Qwen outputs. Preserve the original output and treat the
truncated raw text only as an incomplete diagnostic preview.

### Remaining quality problems

- unrelated-company news was sometimes treated as ticker-level news;
- some variables were invented or inconsistently named;
- neutral days were not always represented clearly;
- predicates were semantically useful but noisy.

---

## 5. Initial Neuro-Symbolic Framing

The initial practical pipeline was:

```text
offline Qwen extraction
        |
        v
structured predicates
        |
        v
GRU prediction
        |
        v
differentiable fuzzy-rule regularisation
```

The GRU and fuzzy rule losses were differentiable, but gradients did not pass
back into Qwen because Qwen outputs had already been saved offline.

### Decision

Describe this architecture as an:

> LTN-style fuzzy-regularised neuro-symbolic regime-transition model.

It was not described as a fully end-to-end LTN because:

- Qwen was not updated during GRU training;
- logic rules did not affect Qwen representations;
- the implementation used custom fuzzy Real Logic operations rather than a
  complete external LTN framework;
- predicates were mostly fixed outputs rather than trainable grounded
  predicates.

---

## 6. Initial GRU Failure: Learning Persistence

The first three-class GRU directly predicted the future bear, sideways, or
bull regime.

Initial persistence baseline:

| Metric | Result |
|---|---:|
| Test accuracy | 0.7620 |
| Test macro-F1 | 0.6420 |

Initial GRU without rules:

| Metric | Result |
|---|---:|
| Test accuracy | 0.7539 |
| Test macro-F1 | 0.6338 |
| Transition accuracy | 0.0000 |
| Stable accuracy | 0.9894 |

The model appeared accurate but simply copied the current regime. It almost
never detected a transition.

Changing the persistence prior and applying class weighting caused unstable
behaviour and reduced general performance.

### Decision

Stop treating this as a normal three-class classification problem. Evaluate
transition and stable rows separately.

### Important lesson

Overall accuracy can hide complete transition-detection failure. Persistence is
a difficult baseline because many market-regime days remain stable.

---

## 7. Macro Feature Experiments

Qwen generated semantic macro channels such as inflation pressure, interest
rates, Federal Reserve policy, market volatility, and oil prices. A
deterministic mapping was created to connect these concepts to available macro
variables.

This improved semantic consistency and auditability, but did not materially
solve transition prediction.

Raw macroeconomic levels later caused a severe collapse:

```text
price_macro test predictions:
bear_drawdown = 1357
sideways = 0
bull_rally = 0
```

### Interpretation

Raw macro levels encoded nonstationary time structure and encouraged the model
to associate periods with regimes rather than learn useful transition signals.

### Decision

- avoid inappropriate raw macro levels;
- prefer MoM, YoY, changes, and spreads;
- treat macro features as a separate ablation;
- do not assume that adding more economic variables improves prediction.

---

## 8. Regime Labels Were Revised

The initial labels were excessively bullish and poorly suited to transition
prediction.

The revised label rule was:

```python
if return_20d <= -0.05 or drawdown_60d <= -0.10:
    return "bear_drawdown"
if return_20d >= 0.05:
    return "bull_rally"
return "sideways"
```

This produced a substantially more balanced overall distribution:

| Regime | Approximate proportion |
|---|---:|
| Bull rally | 0.353 |
| Bear drawdown | 0.324 |
| Sideways | 0.323 |

### Decision

Use the revised regime definition for later experiments, while acknowledging
that some gradual transitions may still be difficult to predict from discrete
news events.

---

## 9. Feature-Set Ablations

Explicit feature-set ablations were added:

- `price_only`
- `price_macro`
- `price_qwen`
- `price_fundamentals`
- `price_qwen_fundamentals`
- `full`

Selected findings:

| Feature set | Accuracy | Macro-F1 | Transition accuracy | Stable accuracy |
|---|---:|---:|---:|---:|
| Price only | 0.4495 | 0.4365 | 0.2295 | 0.6882 |
| Price plus macro | 0.3773 | 0.1826 | 0.3258 | 0.4332 |
| Price plus Qwen | 0.4031 | 0.3694 | 0.4136 | 0.3917 |

### Interpretation

- price features produced the most stable predictions;
- Qwen predicates increased transition sensitivity;
- Qwen also created many false positives;
- raw macro inputs were harmful;
- the main difficulty was balancing transition recall against stable-day
  accuracy.

---

## 10. Hybrid Persistence Override

A hybrid model was tested:

1. predict persistence by default;
2. allow the GRU to override only when its alternative-regime confidence margin
   exceeds a threshold.

Example:

| Model | Accuracy | Macro-F1 | Transition accuracy | Stable accuracy |
|---|---:|---:|---:|---:|
| Persistence | 0.4797 | 0.4725 | 0.0000 | 1.0000 |
| Raw price plus Qwen GRU | 0.4031 | 0.3694 | 0.4136 | 0.3917 |
| Hybrid threshold 0.21 | 0.4886 | 0.4729 | 0.0977 | 0.9124 |

### Decision

The hybrid slightly beat persistence overall, but transition accuracy remained
limited. This confirmed that transition detection should become an explicit
task rather than an indirect confidence override.

---

## 11. Forecast-Horizon Experiments

Forecast horizons of 5, 10, and 20 trading days were tested.

### Findings

- shorter horizons were easier overall;
- persistence remained difficult to beat directly;
- Qwen improved transition sensitivity but created false positives;
- the stable-versus-transition conflict remained at every horizon.

### Decision

Use a two-stage transition architecture and retain the 20-day horizon as the
main dissertation setting, while using shorter horizons as supporting
ablations.

The sequence length and forecast horizon were also clarified:

- sequence length 10 means the GRU observes the previous 10 ticker-days;
- forecast horizon 20 means it predicts the regime 20 trading days ahead;
- these values do not need to be equal.

---

## 12. Two-Stage Transition Architecture

The model was redesigned:

### Stage 1

Will the ticker's regime change within the forecast horizon?

### Stage 2

If the regime changes, which regime is the destination?

This architecture exposed the true bottleneck and made evaluation more
meaningful.

Important metrics became:

- transition precision;
- transition recall;
- transition F1;
- transition PR-AUC;
- destination accuracy on true transitions;
- final regime accuracy;
- final macro-F1;
- transition accuracy;
- stable accuracy.

### Why PR-AUC became central

The most useful question is whether likely transition rows are ranked above
stable rows. PR-AUC measures this under class imbalance without requiring one
fixed operating threshold.

---

## 13. Oracle Diagnostics

Oracle experiments replaced either transition detection or destination
classification with the true answer.

The oracle transition detector produced much larger gains than the oracle
destination classifier.

### Conclusion

Destination prediction was imperfect but usable. The primary bottleneck was
detecting whether a transition would happen.

This also suggested that daily news cannot explain every transition. Some
regime changes are gradual price trends without a discrete public catalyst.

---

## 14. Qwen Quality Filtering

Because using all Qwen predicates increased false positives, low-quality rows
were neutralised rather than deleted.

Initial filter:

```text
mean driver confidence >= 0.70
driver uncertainty <= 0.30
```

Approximately 58.5% of ticker/day packets were retained.

### Decision

Use deterministic filtering before regenerating Qwen predicates.

### Reasoning

Regeneration is expensive and does not guarantee improved quality. Filtering
allows controlled ablations and preserves the original extraction.

---

## 15. Leakage-Safe Company Fundamentals

Company fundamentals were introduced because quarterly financial statements
could provide ticker-specific transition signals missing from daily news.

Yahoo Finance did not provide sufficiently reliable historical quarterly depth,
so Alpha Vantage was selected.

### Critical leakage decision

Fundamentals become available to the model only from their actual public
`reportedDate`, not from the fiscal quarter end.

This prevents the model from seeing information before the market could have
known it.

### Key 20-day results

| Variant | Precision | Recall | F1 | PR-AUC | Final accuracy | Stable accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Price only | 0.4138 | 0.0184 | 0.0353 | 0.5694 | 0.4771 | 0.9724 |
| Filtered Qwen | 0.5856 | 0.3257 | 0.4186 | 0.5838 | 0.4652 | 0.7561 |
| Price plus fundamentals | 0.6022 | 0.5115 | 0.5532 | **0.6145** | 0.4400 | 0.6423 |
| Filtered Qwen plus fundamentals | 0.5806 | 0.6697 | **0.6220** | 0.6042 | 0.4084 | 0.4878 |

### Interpretation

- fundamentals provided the strongest transition-ranking improvement;
- Qwen increased recall and transition F1;
- Qwen also reduced stability through false positives;
- price plus fundamentals was the best ranking model;
- Qwen plus fundamentals was the best high-recall model.

---

## 16. Generic and Selective LTN-Style Rules

Generic fuzzy rules encoded ideas such as:

- material, novel, confident news implies transition;
- weak or irrelevant news implies persistence;
- conflicting risk-on and risk-off news implies sideways destination;
- negative fundamentals plus risk-off news implies a bearish transition.

These rules were economically reasonable but mostly flat in predictive
performance.

### Why generic rules did not work well

- antecedents were too broad;
- fuzzy implications were often vacuously satisfied;
- predicates were noisy;
- a single global logic weight could not represent different rule strengths;
- stronger weights often reduced ranking performance.

### Selective rule refinement

Leakage-safe rule features were added:

- days since public fundamentals release;
- release proximity;
- ticker-relative revenue and margin surprise z-scores;
- trailing risk-on and risk-off shock persistence.

Selective rule families included release, surprise, and persistence rules.

### Result

Selective rules allowed a useful stability-recall trade-off but did not
conclusively beat the fundamentals-only PR-AUC benchmark.

### Decision

Present symbolic rules primarily as auditable regularisers and explanation
mechanisms, not as proven independent predictive alpha.

---

## 17. Learnable Predicate Calibration Versus Rule Discovery

The project considered making rule parameters learnable:

- learn separate rule weights;
- learn revenue and margin surprise thresholds;
- learn release-decay strength;
- compare 3-, 5-, and 10-day news persistence;
- log each rule's satisfaction and contribution.

This is differentiable fuzzy-rule calibration. Gradients can tune how strongly
known formulas influence predictions.

It is not automatic symbolic rule discovery.

### Difference

**Learnable fuzzy calibration**

- the formula structure is supplied by the researcher;
- gradients tune thresholds, predicate grounding, and weights;
- the final rules remain interpretable.

**Automatic symbolic rule discovery**

- the system searches for the logical formula structure itself;
- candidate antecedents and consequents are generated automatically;
- discovered rules still require validation for stability and economic
  plausibility.

---

## 18. Decision to Build an End-to-End Neuro-Symbolic Pipeline

The offline architecture did not allow logic gradients to affect Qwen. The
project therefore moved to a separate end-to-end implementation.

### End-to-end architecture

```text
raw daily news packet
        |
        v
live Qwen model with QLoRA adapter
        |
        v
trainable semantic predicate head
        |
        +--------------------+
        |                    |
        v                    v
fuzzy logic rules       GRU ticker sequence state
        |                    |
        +---------+----------+
                  |
                  v
transition, destination, and reaction heads
```

### Why QLoRA was used

Full Qwen fine-tuning would be expensive and unnecessary. QLoRA:

- loads Qwen in 4-bit quantised form;
- freezes the base model;
- inserts small trainable low-rank adapters;
- allows gradients from prediction and logic losses to modify Qwen's
  representations efficiently.

The implementation exposed approximately:

```text
10,343,466 trainable parameters
of 3,818,318,378 exposed parameters
```

### Five initial semantic predicates

The end-to-end predicate head learned:

- ticker relevance;
- materiality;
- risk-on pressure;
- risk-off pressure;
- uncertainty.

Five predicates were chosen as a compact initial semantic interface. More
predicates could be added later, but adding many weakly supervised concepts
would increase noise and training difficulty.

---

## 19. End-to-End Qwen QLoRA-GRU-LTN Results

### Smoke test

A tiny smoke test verified that:

- Qwen loaded correctly;
- the QLoRA adapter was trainable;
- gradients passed through Qwen, predicate heads, GRU, and logic loss;
- model artefacts were saved.

The smoke-test PR-AUC of zero was not meaningful because the tiny validation
sample contained no positive transitions.

### Pilot

A 100-sequence pilot produced:

```text
Validation PR-AUC: 0.5838
Test PR-AUC:       0.5831
```

This demonstrated that the integrated pipeline could learn a non-trivial
transition ranking signal.

### Full seed-42 run

The full three-epoch result was:

| Metric | Result |
|---|---:|
| Best validation PR-AUC | 0.5467 |
| Test transition PR-AUC | **0.6253** |
| Transition precision | 0.6099 |
| Transition recall | 0.3794 |
| Transition F1 | 0.4678 |
| Selected threshold | 0.59 |
| Final accuracy | 0.4494 |
| Macro-F1 | 0.3834 |
| Transition accuracy | 0.1720 |
| Stable accuracy | 0.7431 |

### Interpretation

The single end-to-end seed achieved a slightly higher test PR-AUC than the
previous price-plus-fundamentals benchmark of approximately 0.6145. However:

- it was only one seed;
- final macro-F1 was weaker;
- transition accuracy remained limited;
- the result does not prove that the LTN rules caused the improvement;
- QLoRA, GRU learning, and fundamentals may explain most of the gain.

---

## 20. Why the Logic Loss Was Small

Observed logic losses were often around `0.02` to `0.03` in the original
end-to-end rules.

A low logic loss can mean:

- the model already satisfies the rules;
- the rules are too easy;
- fuzzy implication antecedents are usually weak;
- rules are vacuously satisfied because the antecedent truth is near zero.

It does not automatically mean that the model has learned economically useful
constraints.

### Pilot LTN versus no-LTN comparison

No-LTN pilot:

```text
Validation PR-AUC: 0.5613
Test PR-AUC:       0.5341
Final accuracy:    0.4543
Macro-F1:          0.4418
```

LTN weight `0.2` pilot:

```text
Validation PR-AUC: 0.5613
Test PR-AUC:       0.5340
Final accuracy:    0.4571
Macro-F1:          0.4441
```

### Conclusion

The original LTN rules had almost no measurable effect at pilot scale. This
motivated changing the rules rather than simply increasing their global weight.

---

## 21. Automatic Candidate Rule Mining

To explore better rules without manually inventing every formula, a candidate
rule miner was created.

It:

- constructs fuzzy atoms from price, fundamentals, regime, and news features;
- mines combinations of up to three antecedents;
- scores rules on training and validation only;
- calculates support, confidence, lift, satisfaction, and bootstrap
  uncertainty;
- avoids using the test set for rule discovery.

Five hundred candidate rules were generated.

### Important finding

The highest-ranked statistical rules were not always economically defensible.
Examples included:

```text
current_bear_regime => bull_transition
drawdown_stress => bull_transition
```

These may describe mean reversion during the validation period rather than a
stable causal or economic relationship.

### Decision

Use automatic mining to propose candidate rules, but retain human economic
review before adding them to the LTN.

---

## 22. Candidate Rule Review

The mined rules were classified as:

```text
KEEP   = 26
MAYBE  = 190
REJECT = 284
```

### Examples of retained rules

```text
negative_price_momentum => bear_destination
current_bear_regime AND negative_price_momentum => bear_destination
negative_price_momentum AND volatility_spike => bear_destination
negative_price_momentum AND negative_revenue_growth => bear_destination
current_bull_regime AND negative_revenue_growth AND volatility_spike
    => bear_transition
positive_price_momentum AND positive_revenue_growth => bull_destination
positive_revenue_growth AND risk_on_news => bull_destination
current_bull_regime AND risk_on_news => persistence
```

### Examples of rejected rules

```text
current_bear_regime => bull_transition
drawdown_stress => bull_transition
high_ticker_relevance => transition
```

### Reasoning

Rules were retained when they had:

- a defensible economic mechanism;
- useful validation support and lift;
- acceptable bootstrap stability;
- a combination of confirming signals.

Rules were rejected when they were:

- broad regime-only mean-reversion patterns;
- based on relevance rather than direction;
- statistically attractive but economically unsupported;
- redundant or weak.

---

## 23. Approved-Rule LTN Ablation

The end-to-end trainer was extended with:

```text
--logic-rule-set mined_keep
--approved-rules-path <reviewed CSV>
```

Only rules marked `KEEP` are grounded into differentiable fuzzy formulas.

Raw tabular values are retained separately for rule grounding, while normalised
values continue to feed the GRU. This ensures that a predicate such as
`negative_price_momentum` means negative return rather than merely below-average
normalised return.

### Fair comparison design

Two otherwise identical models are compared:

1. **No LTN**
   - approved rules are evaluated and audited;
   - `logic_loss_weight = 0`;
   - rules do not affect gradients.

2. **KEEP-rule LTN**
   - the same approved rules are evaluated;
   - `logic_loss_weight = 0.2`;
   - rule violations affect gradients.

### Sampled runner

A sampled experiment runner was created to avoid repeatedly spending around
16 hours on full runs.

Default sample:

- 200 training sequences;
- 200 validation sequences;
- 200 test sequences;
- one epoch;
- batch size 2.

This is intended for directional diagnostics only. Full-data and repeated-seed
experiments remain necessary before dissertation conclusions are made.

---

## 24. Current Conclusions

### Price and persistence

- price context is the most reliable source of stable-regime information;
- persistence is a strong baseline;
- a model can appear accurate while detecting no transitions.

### Fundamentals

- leakage-safe company fundamentals provide the strongest current improvement
  in transition ranking;
- price plus fundamentals remains the strongest established PR-AUC benchmark;
- public release dates are essential to avoid leakage.

### Qwen and news

- Qwen predicates increase transition sensitivity and recall;
- noisy news produces false positives and reduces stable accuracy;
- quality filtering is better than indiscriminately using all Qwen outputs;
- QLoRA may help Qwen learn which news is material, but this requires repeated
  controlled comparisons.

### Macroeconomic variables

- raw macro levels were harmful;
- stationary transformations and selective inclusion are required;
- semantic mapping improved explainability but not transition prediction.

### LTN and fuzzy rules

- generic rules were too broad and often vacuously satisfied;
- selective rules improved interpretability and operating-point control;
- current evidence does not prove independent predictive uplift from LTN
  regularisation;
- automatically mined rules require human economic review;
- approved rules should be evaluated against a no-LTN model using identical
  data and seeds.

---

## 25. Recommended Dissertation Framing

The following framing is defensible:

> This dissertation develops a neuro-symbolic market-regime transition
> pipeline for MAG7 equities. An instruction-tuned Qwen model converts daily
> ticker-level news packets into contextual representations and trainable
> semantic predicates. A recurrent model combines these representations with
> price context and leakage-safe quarterly fundamentals to estimate transition
> risk and destination. Differentiable fuzzy rules encode economically
> motivated constraints and provide an auditable explanation layer.
> Experiments show that fundamentals contribute the strongest established
> transition-ranking improvement, while Qwen-derived news information
> increases transition sensitivity at the cost of false positives. Symbolic
> rules are useful as interpretable regularisers, although their independent
> predictive benefit has not yet been conclusively established.

### Claims to avoid

- LTN rules dramatically improve prediction.
- The system fully explains Qwen's internal reasoning.
- Candidate mined rules represent causal economic relationships.
- One successful seed proves that the end-to-end architecture is superior.
- The model's PR-AUC implies that it would necessarily make money.

---

## 26. Recommended Next Steps

1. Complete the sampled KEEP-rule LTN versus no-LTN comparison.
2. Inspect whether approved rules change formula satisfaction and gradients.
3. Reject the approved-rule set if it does not improve validation PR-AUC,
   transition F1, or stability consistently.
4. Repeat the most promising end-to-end variants using multiple seeds.
5. Compare against price-plus-fundamentals and simpler models using identical
   chronological splits.
6. Add ticker-specific validation thresholds without tuning on the test set.
7. Generate false-positive and false-negative transition review tables.
8. Investigate whether regime labels create unforecastable gradual
   transitions.
9. Add richer leakage-safe earnings features and earnings-surprise rules.
10. Add rolling or walk-forward chronological validation.
11. Build final explanation reports showing predicates, fundamentals, rule
    activation, rule satisfaction, transition probability, and destination.

---

## 27. Final Project Position

The project moved through several increasingly precise formulations:

```text
individual article sentiment
        |
        v
daily ticker-level Qwen predicates
        |
        v
three-class GRU regime prediction
        |
        v
explicit transition detection and destination classification
        |
        v
filtered Qwen plus leakage-safe fundamentals
        |
        v
selective differentiable fuzzy rules
        |
        v
end-to-end Qwen QLoRA, GRU, and LTN-style fuzzy regularisation
        |
        v
automatically mined and economically reviewed candidate rules
```

Each pivot responded to an observed failure:

- article-level inputs were too noisy;
- DeepSeek was unreliable for structured extraction;
- direct three-class prediction learned persistence;
- raw macro levels caused nonstationary collapse;
- Qwen increased false positives;
- generic rules were too weak and broad;
- offline Qwen prevented end-to-end gradients;
- automatic rule ranking produced economically questionable patterns;
- full GPU experiments were too slow for rapid iteration.

The current approach is therefore not simply a larger model. It is a more
carefully scoped research system that separates transition detection from
destination classification, controls leakage, evaluates each information
source separately, and treats symbolic logic as an auditable hypothesis that
must be experimentally validated.
