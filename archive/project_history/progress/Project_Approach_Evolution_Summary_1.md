# Project Approach Evolution Summary

## 1. Project Objective

The dissertation project aims to investigate whether neuro-symbolic artificial
intelligence can improve market-regime prediction using:

- financial news interpreted by DeepSeek;
- macroeconomic and market data;
- a temporal hidden state representing the evolving market condition;
- Logic Tensor Network (LTN) constraints;
- ticker-level price regimes for the MAG7 stocks.

The target classes are:

| ID | Regime |
|---:|---|
| 0 | `bear_drawdown` |
| 1 | `sideways` |
| 2 | `bull_rally` |

The central hypothesis evolved into:

> Market regimes are persistent latent states that change gradually. News,
> macroeconomic conditions, sector conditions, and ticker-specific events should
> update the model's belief about the current state, rather than allowing one
> article to cause a sudden regime change.

---

## 2. Initial DeepSeek and LTN Experiment

### Initial implementation

The original notebook, `DeepSeek_LTN_Market_Regime_Experiment.ipynb`, treated
each article/ticker combination as an independent supervised example.

It:

- used a frozen DeepSeek 8B model to create an article representation;
- combined the representation with structured price, sentiment, and macro data;
- trained a small classification head;
- compared a baseline against a version with manually defined LTN rules.

Example initial rules included:

```text
material and high relevance and easing -> bull
material and high relevance and tightening -> bear
high inflation and rising rates -> not bull
low relevance -> sideways
```

### Problems identified

#### Very slow training

Although DeepSeek was frozen, its embeddings were recomputed during every
training epoch. Therefore, most training time was repeatedly spent running the
8B model instead of training the classifier.

One epoch took approximately five minutes.

#### Poor predictive performance

Validation accuracy was often below 33%, and the model behaved inconsistently
between epochs.

#### Severe split imbalance

The observed class distributions were approximately:

| Split | Bear | Sideways | Bull |
|---|---:|---:|---:|
| Train | 11.96% | 30.07% | 57.97% |
| Validation | 17.88% | 4.82% | 77.30% |
| Test | 12.23% | 40.02% | 47.75% |

This demonstrated that:

- 33% uniform random accuracy was not an appropriate benchmark;
- the validation period was overwhelmingly bullish;
- raw accuracy could be misleading;
- macro-F1 and per-class F1 were more appropriate metrics.

#### LTN misunderstanding

It was clarified that a standard LTN does not automatically discover symbolic
rules from nothing.

Instead:

- humans provide logical formulas;
- neural predicates and model parameters learn while being encouraged to satisfy
  those formulas;
- incorrect or overly broad rules can reduce predictive performance.

### Initial notebook improvements

The notebook was changed to:

- cache frozen DeepSeek embeddings;
- reuse the DeepSeek model between experiments;
- use shuffled and balanced batches;
- reduce the initial logic weight;
- report and select models using macro-F1;
- display split distributions;
- preserve the best validation checkpoint.

These changes improved training practicality but did not resolve the fundamental
article-level modelling problem.

---

## 3. Shift to a Persistent Latent Market State

### Motivation

The article-level approach did not match the intended market behaviour.

The revised idea was:

```text
previous hidden state
+ new daily news signal
+ macroeconomic context
+ current price/market context
= updated hidden state
```

Instead of asking:

```text
Does this article imply bull, bear, or sideways?
```

the model should ask:

```text
How should today's evidence update the existing market-state belief?
```

### Daily latent-state dataset

A new daily dataset builder was created:

```text
src/data/build_latent_state_dataset.py
```

It aggregated approximately:

```text
84,214 article/ticker rows
into
5,115 ticker/day rows
```

The daily features initially included:

- aggregated article count and relevance;
- Alpha Vantage sentiment scores;
- macroeconomic indicators;
- current price-trend features;
- volatility, rally, and drawdown features.

Forward-looking fields such as `next_day_return`, `three_day_return`, and
market-material labels derived from future returns were excluded from the new
model inputs to reduce leakage.

### GRU hidden-state model

A Gated Recurrent Unit (GRU) model was added:

```text
src/models/train_latent_state_gru.py
```

GRU stands for **Gated Recurrent Unit**. It is a recurrent neural network that
updates a hidden state across a sequence.

The model consumed rolling daily ticker sequences and predicted the future
regime from the final hidden state.

The training loss included:

```text
classification loss
+ smoothness loss
+ optional logic loss
```

The smoothness loss discouraged abrupt probability changes when the incoming
daily signal appeared calm.

### Initial GRU results

The first GRU experiments still performed poorly:

| Experiment | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| Initial GRU | 0.2735 | 0.2202 |
| Smaller, more regularised GRU | 0.2682 | 0.2421 |

The training loss became very low while validation remained poor, indicating
overfitting.

---

## 4. Diagnosing Class Bias and Model Collapse

Diagnostic output was added to show:

- train, validation, and test label counts;
- validation prediction counts;
- confusion matrices and per-class performance;
- balanced versus natural sampling behaviour;
- class-weighting options.

The following options were introduced:

```text
--sampler natural|balanced
--class-weighting none|inverse|sqrt_inverse
--diagnostics-every N
```

### Findings

The model frequently predicted mostly `bear_drawdown` or `sideways` and almost
never predicted `bull_rally` during the heavily bullish validation period.

Balanced sampling and class weighting often overcorrected the class imbalance.

Using natural sampling with no class weighting improved the test result:

```text
test accuracy = 0.4945
test macro-F1 = 0.3559
```

However, the model was still significantly weaker than a simple persistence
strategy.

---

## 5. Discovery of the Strong Persistence Baseline

A current-regime persistence baseline was introduced:

```text
predicted future regime = current regime
```

This baseline does not use DeepSeek, news, macro data, an LTN, or a GRU. It is a
sanity-check benchmark measuring how persistent the labels are.

Its test performance was:

```text
accuracy = 0.7620
macro-F1 = 0.6420
```

This was a crucial finding.

It demonstrated that:

- market-regime labels are highly persistent;
- random chance is an unsuitable benchmark;
- any learned model must beat persistence to demonstrate incremental value;
- news and macro data should modify persistence, not replace it.

---

## 6. Adding an Explicit Persistence Prior

The GRU was originally trying to predict regimes from scratch and frequently
overrode useful persistence.

The model was changed to include:

```text
--persistence-prior-strength
```

Conceptually:

```text
GRU logits
+ positive logit bias for the current regime
= final regime logits
```

This means:

> Assume the current regime continues unless the learned evidence is strong
> enough to override it.

### Results

With a persistence prior strength of `2.0`:

```text
test accuracy = 0.7480
test macro-F1 = 0.6282
```

This approached the persistence baseline:

```text
persistence baseline macro-F1 = 0.6420
GRU with persistence prior macro-F1 = 0.6282
```

This supported the project's central hidden-state hypothesis.

---

## 7. Fair Comparison of Models With and Without Logic

The correct LTN comparison was clarified as:

```text
same DeepSeek/GRU architecture without logic rules
versus
same DeepSeek/GRU architecture with logic rules
```

The persistence baseline remained a separate sanity-check benchmark.

Initial comparisons using the early features showed:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Persistence baseline | 0.7620 | 0.6420 |
| GRU without rules | 0.7362 | 0.6175 |
| GRU with rules | 0.7074 | 0.5960 |

The initial logic rules reduced performance.

The conclusion was:

> The rules were probably too broad, incorrectly specified, or misaligned with
> the actual regime-transition process.

This was not considered evidence that neuro-symbolic AI cannot work. It showed
that rule design and predicate construction were inadequate.

---

## 8. Realisation That the GRU Was Not Actually Using DeepSeek Interpretation

A major conceptual issue was identified:

The early daily GRU primarily used:

- Alpha Vantage sentiment and relevance scores;
- price features;
- macro features.

It did not genuinely use DeepSeek to interpret the news text.

Therefore, poor GRU performance could not be interpreted as evidence that
DeepSeek added no value.

### DeepSeek embedding pipeline

A DeepSeek feature extraction step was introduced:

```text
src/data/build_deepseek_news_features.py
```

It:

- runs a frozen DeepSeek model over unique article text;
- obtains hidden representations;
- saves compact projected numerical features;
- allows the daily GRU to use DeepSeek-derived semantic features without
  repeatedly running DeepSeek during training.

### DeepSeek-feature experiment results

Using DeepSeek features produced:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Persistence baseline | 0.7620 | 0.6420 |
| DeepSeek GRU without rules | 0.7067 | 0.5955 |
| DeepSeek GRU with rules | 0.7472 | 0.6275 |

This was a stronger result.

In this run:

- rules improved macro-F1 by approximately `0.032`;
- rules improved accuracy by approximately `0.041`;
- the full model approached, but did not exceed, persistence.

The interpretation was:

> DeepSeek-derived news information and symbolic rules showed potential, but
> regime persistence remained the dominant predictor.

---

## 9. From DeepSeek Embeddings to Contextual Market Drivers

### Problem with embeddings alone

DeepSeek embeddings contain semantic information, but they do not provide a
clear human-readable explanation of what the news means.

The intended system should be able to describe drivers such as:

```text
Iran conflict escalation
AI capital-expenditure optimism
regional bank credit stress
semiconductor export restrictions
Fed pivot uncertainty
Tesla delivery weakness
```

The system should not rely on a permanently hardcoded checklist such as:

```text
geopolitical risk
inflation pressure
rate-cut probability
earnings momentum
```

Such a fixed checklist could become brittle across different market periods.

### Open-ended contextual driver extraction

A contextual-driver extraction script was added:

```text
src/data/extract_contextual_drivers.py
```

DeepSeek was asked to return open-ended driver information such as:

```json
{
  "driver_label": "Iran conflict escalation",
  "driver_summary": "Escalation creates oil-supply and inflation concerns.",
  "directional_pressure": "risk_off",
  "intensity": 0.82,
  "novelty": 0.76,
  "confidence": 0.81,
  "time_horizon": "weeks",
  "market_scope": "broad_market",
  "evidence": "..."
}
```

These open-ended labels preserve flexibility, while numeric fields can be used
by the temporal model and LTN.

### Explainability support

The training script was changed to optionally save predictions:

```text
--save-predictions
```

An explanation utility was added:

```text
src/models/explain_market_state_prediction.py
```

The intended explanation combines:

1. DeepSeek's human-readable contextual drivers;
2. recent GRU regime-probability movement;
3. LTN-style rule activation scores;
4. the final market-regime interpretation.

Example intended explanation:

```text
DeepSeek identified repeated risk-off drivers from semiconductor export
restrictions and delayed rate-cut expectations.

The GRU hidden state reduced bull confidence over the last eight trading days.

The LTN showed strong activation of the persistent risk-off evidence rule and
moderate activation of the uncertainty rule.

Therefore, the model predicts rising flip risk from bull_rally toward sideways.
```

---

## 10. Runtime Problems With Article-Level Contextual Extraction

The article-level contextual extraction attempted to process:

```text
84,214 article/ticker rows
```

At batch size one, this was far too slow for practical experimentation.

The first 500 extracted rows were mostly:

```text
driver_label = no clear driver
directional_pressure = neutral
intensity = 0
confidence = 0
```

They were also simply the first chronological rows rather than a meaningful
sample.

This demonstrated two problems:

- article-by-article generation is computationally expensive;
- individual articles are often noisy or non-material.

Filtering controls were considered or added, including:

```text
minimum relevance
top articles per ticker/day
split/date filters
maximum rows
```

However, static Alpha Vantage relevance was recognised as an imperfect final
solution.

---

## 11. Dynamic Relevance and Conditional Materiality

A key conceptual refinement was:

> Article relevance is not static. It depends on the current market state and
> what the market already knows.

For example:

- a CEO resignation may be highly relevant on the announcement day;
- repeated coverage several days later may add almost no new information;
- new details could make it relevant again.

Therefore:

```text
ticker relevance != hidden-state update relevance
```

The desired quantity is **conditional materiality**:

> Given the current hidden state and recent news memory, does this article add
> new evidence that should update the market-state belief?

The ideal future model would use a trainable relevance/update gate:

```text
DeepSeek article representation
+ current hidden state
+ macro/market context
+ recent news memory
-> learned update weight between 0 and 1
```

Articles that improve regime prediction would receive larger learned weights,
while repeated or noisy articles would receive smaller weights.

This is related to attention and gating.

---

## 12. Pivot From Article-Level to Daily News Interpretation

The strongest architectural decision was to stop treating each article as the
primary DeepSeek input.

The revised design is:

```text
all news for one ticker on one trading day
-> one DeepSeek daily interpretation
-> one LTN reasoning step
-> one hidden-state update at end of day
```

This is better because it:

- reduces DeepSeek calls from approximately 84,000 article/ticker rows to
  approximately 5,000 ticker/day rows;
- reduces noise from individual articles;
- allows DeepSeek to identify the dominant daily narrative;
- matches the intended once-per-day hidden-state update;
- makes LTN predicates and explanations easier to understand.

An example daily DeepSeek output might be:

```json
{
  "daily_market_narrative": "Export restrictions weakened the AI growth narrative, while strong demand offered partial support.",
  "dominant_drivers": [
    {
      "driver": "semiconductor export restriction concerns",
      "directional_pressure": "risk_off",
      "intensity": 0.70,
      "novelty": 0.60,
      "confidence": 0.80
    },
    {
      "driver": "AI demand resilience",
      "directional_pressure": "risk_on",
      "intensity": 0.50,
      "novelty": 0.40,
      "confidence": 0.70
    }
  ],
  "net_pressure": "mixed",
  "update_strength": 0.45,
  "uncertainty": 0.62
}
```

---

## 13. Context-Conditioned Macro Relevance

Another important refinement was:

> The daily news context should determine which macroeconomic variables matter
> at that moment.

For example, an Iran conflict could make the following highly relevant:

- oil prices;
- inflation expectations;
- volatility;
- interest-rate expectations.

Unemployment may be much less relevant to that specific driver.

Therefore, macro data should not be treated as a flat collection of equally
important features.

The desired process is:

```text
daily news context
-> DeepSeek identifies contextual driver
-> DeepSeek identifies relevant macro/market channels
-> LTN reasons over the selected channels and observed values
-> GRU hidden state updates
```

The intended rule is not:

```text
high inflation -> bear
```

It is:

```text
high inflation matters when the current driver makes inflation relevant
```

This represents context-conditioned macro attention.

---

## 14. MAG7-Specific Versus Broad-Market Drivers

A further problem was recognised:

The target is not the overall S&P 500 regime. It is ticker-level regimes for
MAG7 stocks.

MAG7 stocks are heavily affected by technology-sector and company-specific
drivers, including:

- AI demand and capital expenditure;
- cloud growth;
- semiconductor supply;
- export controls;
- advertising demand;
- earnings guidance;
- regulatory and antitrust issues;
- product cycles;
- valuation and discount rates;
- company-specific management risks.

Therefore, broad macro conditions may not always dominate.

The system should distinguish:

```text
ticker-specific driver
technology-sector driver
broad-market driver
macro driver
irrelevant/general news
```

Macro variables should be treated as contextual background, not universal
drivers.

The LLM should determine the scope of each daily driver and which channels are
relevant to the particular ticker.

---

## 15. How the Neuro-Symbolic Components Should Integrate

The project initially felt only weakly neuro-symbolic because:

```text
DeepSeek interpreted news
GRU predicted regimes
LTN-style rules sat beside them as a separate regulariser
```

The intended stronger neuro-symbolic design is:

```text
DeepSeek extracts contextual symbolic predicates
-> LTN reasons directly over those predicates
-> GRU hidden-state updates are constrained by the reasoning
```

The bridge between the neural and symbolic components must be:

```text
DeepSeek output becomes LTN predicates
```

Potential predicates include:

```text
RiskOff(driver)
RiskOn(driver)
HighNovelty(driver)
HighConfidence(driver)
TickerSpecific(driver)
SectorWide(driver)
MacroRelevant(driver, channel)
PersistentEvidence(driver, time)
ConflictingEvidence(day)
LargeStateUpdate(day)
BullConfidenceFalling(day)
```

General LTN rules should focus on evidence and belief updating rather than
brittle universal economic assumptions.

Examples:

```text
low novelty -> small hidden-state update

low relevance or irrelevant scope -> small hidden-state update

high novelty and high confidence and high intensity -> larger update permitted

persistent risk-off evidence -> bull confidence should gradually decrease

persistent risk-on evidence -> bear confidence should gradually decrease

conflicting high-confidence drivers -> uncertainty/sideways probability should rise

ticker-specific drivers should affect that ticker more than broad-market drivers

selected macro channels confirming a driver -> stronger update

selected channels contradicting a driver -> weaker update or higher uncertainty
```

This would make the LTN a true reasoning and interpretability layer rather than
an unrelated loss term.

---

## 16. Current Intended Final Architecture

The intended final pipeline is now:

```text
1. Collect article-level news, prices, ticker mappings, and macro data.

2. Group all news by ticker and trading day.

3. Build a daily news packet for each ticker/day.

4. DeepSeek reads the complete daily packet.

5. DeepSeek outputs open-ended contextual driver predicates, including:
   - dominant driver labels and summaries;
   - scope: ticker, sector, broad market, macro, or irrelevant;
   - risk-on/risk-off/mixed pressure;
   - intensity, novelty, confidence, and horizon;
   - relevant company, sector, market, or macro channels;
   - update strength and uncertainty.

6. Combine the predicates with:
   - selected context-relevant macro variables;
   - technology-sector information;
   - ticker-specific price state;
   - current regime and persistence prior.

7. The LTN reasons over the DeepSeek predicates, selected context channels, and
   hidden-state transition.

8. The GRU updates the persistent ticker-level hidden state once per day.

9. The classifier outputs future probabilities for bear, sideways, and bull.

10. Save probabilities, driver summaries, rule activations, and explanations.
```

---

## 17. Required Experimental Comparisons

The core comparison should be:

| Experiment | Purpose |
|---|---|
| Persistence baseline | Measures label persistence and establishes the benchmark to beat |
| DeepSeek daily-driver GRU without LTN rules | Measures the value of learned semantic and temporal features |
| DeepSeek daily-driver GRU with LTN rules | Measures the impact of the neuro-symbolic constraints |

Additional ablations should include:

```text
price/current regime only
price + macro only
price + DeepSeek daily drivers
price + DeepSeek daily drivers + context-selected macro channels
full model with LTN
```

Important metrics:

- accuracy;
- macro-F1;
- per-class F1;
- confusion matrix;
- transition-case performance;
- performance specifically where persistence is wrong;
- rule satisfaction and violation scores;
- qualitative explanation case studies.

Transition-case performance is particularly important:

```text
current regime != future regime
```

These are the cases where news interpretation should add value over persistence.

---

## 18. Current Code and Artifacts

The implementation created or modified the following major files:

| File | Purpose |
|---|---|
| `notebooks/DeepSeek_LTN_Market_Regime_Experiment.ipynb` | Original experiment, diagnostics, and embedding caching improvements |
| `src/data/build_ltn_dataset.py` | Builds the original article-level supervised dataset |
| `src/data/build_latent_state_dataset.py` | Builds daily ticker-level latent-state features |
| `src/data/build_deepseek_news_features.py` | Extracts compact DeepSeek semantic article features |
| `src/data/extract_contextual_drivers.py` | Extracts open-ended contextual drivers from individual articles; now considered an intermediate prototype |
| `src/models/train_latent_state_gru.py` | Trains the GRU with persistence, smoothness, and optional LTN-style constraints |
| `src/models/explain_market_state_prediction.py` | Produces qualitative explanations from saved prediction rows and contextual drivers |
| `src/models/compare_experiments.py` | Compares baseline, LTN, and latent-state model metrics |
| `src/models/README.md` | Documents the evolving workflow |

---

## 19. Main Lessons Learned

1. **Article-level regime prediction is too noisy.**  
   The hidden state should update from aggregated daily evidence.

2. **Repeated frozen-LLM inference makes training unnecessarily slow.**  
   DeepSeek outputs should be precomputed or generated once per daily packet.

3. **Class imbalance makes raw accuracy misleading.**  
   Macro-F1 and per-class metrics are required.

4. **Persistence is the strongest baseline.**  
   The model should learn when to override persistence, not predict from scratch.

5. **Alpha Vantage sentiment is not sufficient for the dissertation claim.**  
   DeepSeek must genuinely interpret the news.

6. **DeepSeek embeddings improve semantic representation but are not inherently
   explainable.**  
   Human-readable contextual predicates are needed.

7. **Individual contextual-driver generation is too slow and noisy.**  
   DeepSeek should interpret all ticker/day news together.

8. **Static relevance is inadequate.**  
   Relevance and materiality depend on the current state, novelty, and recent
   evidence.

9. **Broad macro rules are inadequate for MAG7 ticker regimes.**  
   The system must distinguish ticker, sector, broad-market, and macro scope.

10. **The LTN must reason over DeepSeek-derived predicates.**  
    Otherwise, the neural interpretation and symbolic reasoning remain weakly
    connected.

11. **Explainability should be designed into the architecture.**  
    The final explanation should combine DeepSeek drivers, GRU probability
    movement, and LTN rule activations.

---

## 20. Current Recommended Next Step

The next implementation step should be to replace article-level contextual
driver extraction with a daily DeepSeek interpretation pipeline:

```text
extract_daily_contextual_drivers.py
```

It should:

1. group articles by ticker and trading day;
2. create one bounded daily news packet;
3. ask DeepSeek to identify the dominant open-ended drivers;
4. classify driver scope and relevant channels;
5. output numeric predicates and human-readable evidence;
6. merge directly into the daily latent-state dataset;
7. feed those predicates into both the GRU and LTN;
8. support saved explanations.

This daily-packet architecture is the best match for the final dissertation
hypothesis and is substantially more computationally practical than extracting
one generated interpretation for every article/ticker row.

---

## 21. Concise Dissertation Framing

> This dissertation proposes a neuro-symbolic temporal model for ticker-level
> market-regime prediction. DeepSeek interprets the complete daily news context
> for each MAG7 stock and extracts flexible contextual driver predicates,
> including driver scope, directional pressure, novelty, confidence, and
> context-relevant market channels. A Logic Tensor Network reasons over these
> predicates, relevant macro/sector/ticker conditions, and market-state
> transitions. A GRU maintains a persistent latent market state that updates
> once per trading day. The model is evaluated against a strong current-regime
> persistence baseline and an otherwise identical neural model without symbolic
> constraints. Explanations combine DeepSeek driver narratives, GRU probability
> movements, and LTN rule activations.

