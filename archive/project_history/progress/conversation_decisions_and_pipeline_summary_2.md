# Dissertation Project: Conversation, Decisions, Experiments, and Current Pipeline

## 1. Project Objective

This dissertation investigates neuro-symbolic AI for market regime prediction
using financial news, macroeconomic data, and MAG7 stock data.

The target is future ticker-level market regime classification:

| ID | Regime |
|---:|---|
| 0 | `bear_drawdown` |
| 1 | `sideways` |
| 2 | `bull_rally` |

The central research idea is that the market has a persistent latent state.
News should not predict a regime independently from scratch. Instead, daily news
and relevant economic context should provide evidence that gradually updates the
existing market state.

The desired conceptual pipeline is:

```text
Daily ticker news
+ macro, sector, market, and ticker context
+ current ticker regime
        |
        v
LLM interprets the complete daily news context
        |
        v
LLM produces contextual driver predicates
        |
        v
LTN-style rules reason over predicates and relevant context
        |
        v
GRU updates the persistent latent ticker state once per day
        |
        v
Future ticker regime probabilities
```

---

## 2. Original Article-Level Approach

### Initial idea

The original implementation treated each article as a separate supervised
example. DeepSeek embeddings or article-level contextual driver extraction would
be combined with market and macroeconomic features to predict a future regime.

The initial proposed workflow included:

1. Build an article-level dataset.
2. Generate DeepSeek article embeddings.
3. Extract one contextual driver per article.
4. Aggregate article features into daily features.
5. Train a GRU with and without LTN-style logic.
6. Compare both models against a persistence baseline.

### Why this approach was rejected

Article-level prediction was considered too noisy and conceptually inconsistent
with the dissertation claim.

A single article should not independently flip the market regime. Articles from
the same day can be repetitive, contradictory, irrelevant, or meaningful only
when interpreted together.

The article-level approach also made contextual driver extraction unnecessarily
slow because the language model had to process every article separately.

### Decision

The project pivoted from:

```text
one article -> one interpretation -> one prediction/update
```

to:

```text
all ticker/day news -> one daily interpretation -> one hidden-state update
```

---

## 3. Relevance Filtering Discussion

### Initial filtering proposal

An early practical proposal was to process only the top N articles per
ticker/day.

### Why fixed top-N filtering was rejected

The number of meaningful articles varies substantially by day:

- A busy day may contain many genuinely important articles.
- A quiet day may contain no important articles.
- Fixed top-N filtering can discard important news on busy days.
- Fixed top-N filtering can force irrelevant articles into quiet days.

### Relevance threshold proposal

It was then proposed that articles below a relevance threshold, initially `0.5`,
should be ignored.

### Important conceptual correction

Using Alpha Vantage's supplied relevance and sentiment scores was rejected.
Alpha Vantage should primarily provide the news headlines and article text.

The language model should determine:

- whether the daily news is relevant;
- which drivers matter;
- which scope each driver belongs to;
- which contextual channels matter;
- how strongly the hidden state should update.

### Final position

The core intelligence should come from the language model's interpretation of
the complete daily news packet, not Alpha Vantage sentiment or relevance
features.

---

## 4. Knowledge Graph Idea

A market knowledge graph was briefly considered.

Example relationships included:

```text
Fed policy -> interest rates -> growth-stock valuations
Geopolitical conflict -> oil prices -> inflation -> rate expectations
AI capex -> semiconductor demand -> NVDA/AMD/TSM
```

### Potential benefit

A knowledge graph could explain how a contextual driver propagates through
macro, sector, and ticker-level channels.

### Why it was postponed

A full market knowledge graph would substantially increase the dissertation
scope and could become a separate research project.

### Decision

The knowledge graph was postponed. The immediate priority became a working
daily LLM-predicate, LTN, and GRU pipeline.

---

## 5. First Small Prototype and Why It Was Rejected

### Prototype attempted

A lightweight 100-article prototype used:

```text
Alpha Vantage relevance score * Alpha Vantage ticker sentiment score
```

as a proxy for news pressure.

### Results

The balanced prototype showed weak performance:

```text
Article-level accuracy: approximately 0.33
Daily-level accuracy: approximately 0.37
Bull/bear transition recall: approximately 0.20
```

### Why it was rejected

This prototype did not test the dissertation claim. It tested Alpha Vantage's
precomputed sentiment and relevance, not whether a language model could
understand news and extract contextual drivers.

### Decision

The proxy prototype and its generated outputs were removed. The project returned
to an LLM-led interpretation design.

---

## 6. DeepSeek Article-Level Contextual Driver Extraction

### DeepSeek requirement

DeepSeek was initially considered important because the dissertation claim was
framed around DeepSeek interpreting financial news.

The extraction output was intended to contain fields such as:

```json
{
  "driver_label": "semiconductor export restriction concerns",
  "driver_summary": "...",
  "scope": "ticker",
  "directional_pressure": "risk_off",
  "intensity": 0.8,
  "novelty": 0.7,
  "confidence": 0.9
}
```

### Local execution problem

The local Windows environment did not initially have `torch` or `transformers`.
A project-local virtual environment also encountered installation issues under a
long OneDrive path.

A short-path environment was successfully created at:

```text
C:\codex_deepseek_venv
```

However, running an 8B DeepSeek model on the local CPU was recognised as
impractical.

### Decision

The language-model extraction would be run in the university container, which
has approximately 45 GB of VRAM.

---

## 7. Pivot to Daily Ticker/Day Contextual Interpretation

The final architectural pivot was formalised:

1. Build the article-level dataset for joining news, prices, macro data, and
   labels.
2. Group articles by `ticker + anchor_trading_date`.
3. Construct one daily news packet for each ticker/day.
4. Ask the language model to interpret the full packet.
5. Produce one structured contextual predicate row per ticker/day.
6. Merge the predicates into the daily latent-state dataset.
7. Apply LTN-style logic directly to the predicates and hidden-state
   transitions.
8. Update the GRU hidden state once per ticker/day.
9. Predict the future ticker regime.
10. Compare against the persistence baseline.

### Desired daily predicate structure

```json
{
  "daily_market_narrative": "...",
  "dominant_drivers": [
    {
      "driver_label": "semiconductor export restriction concerns",
      "driver_summary": "...",
      "scope": "ticker|sector|broad_market|macro|irrelevant",
      "directional_pressure": "risk_on|risk_off|mixed|neutral",
      "intensity": 0.0,
      "novelty": 0.0,
      "confidence": 0.0,
      "time_horizon": "days|weeks|months",
      "relevant_channels": [
        {
          "channel": "China revenue exposure",
          "importance": 0.9,
          "available_variable": "optional mapped variable"
        }
      ]
    }
  ],
  "net_pressure": "risk_on|risk_off|mixed|neutral",
  "update_strength": 0.0,
  "uncertainty": 0.0
}
```

---

## 8. DeepSeek-R1 JSON Output Problem

### Model tested

```text
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

### Observed behaviour

The model repeatedly produced long reasoning text instead of valid raw JSON.
Example generations began by explaining how the model intended to solve the
task rather than returning the requested schema.

This produced daily predicate rows with:

```text
blank daily narratives
neutral pressure
zero update strength
zero uncertainty
missing top drivers
```

### Attempted fixes

The extraction code was hardened to:

- search for the last valid JSON object;
- request JSON-only output;
- use explicit JSON tags;
- provide fallback "no clear market-moving driver" predicates;
- preserve longer raw generations for debugging;
- use chat templates where available.

### Why DeepSeek-R1 was still rejected

DeepSeek-R1 distilled models are reasoning-oriented. Their tendency to produce
chain-of-thought-style text makes them unreliable for schema-constrained
structured predicate extraction.

### Decision

Use an instruction-tuned model that is strong at financial interpretation and
reliable structured output.

---

## 9. Switching to Qwen

### Selected model

```text
Qwen/Qwen2.5-7B-Instruct
```

### Why Qwen was selected

Qwen provided:

- reliable structured output;
- strong instruction following;
- good contextual reasoning;
- practical execution on the university GPU;
- significantly better suitability for JSON predicate extraction than
  DeepSeek-R1.

### Dissertation framing

DeepSeek-R1 was tested but found unsuitable for direct structured extraction.
An instruction-tuned Qwen model was selected to operationalise the contextual
reasoning layer into structured daily predicates.

This is an engineering and model-selection result rather than a failure of the
research approach.

---

## 10. Qwen Daily Predicate Smoke Test

### Smoke-test command

The model processed 25 ticker/day packets with up to five articles per packet.

```bash
/home/jovyan/.venv/bin/python src/data/extract_daily_contextual_drivers.py \
  --input-path data/processed/ltn_all.parquet \
  --output-path data/processed/daily_contextual_driver_predicates_smoke25_qwen.parquet \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --max-days 25 \
  --max-articles-per-day 5 \
  --max-article-chars 500 \
  --max-input-length 1536 \
  --max-new-tokens 500 \
  --batch-size 1 \
  --checkpoint-every 5 \
  --force
```

### Result

Qwen successfully produced usable daily predicates, including:

- daily market narratives;
- net risk pressure;
- update strength;
- uncertainty;
- contextual driver labels;
- ticker, sector, broad-market, macro, and irrelevant scopes;
- driver shock scores.

Example:

```text
Ticker: AAPL
Narrative: Apple reports a revenue decline due to exchange rates,
supply-chain issues, and macroeconomic challenges.
Net pressure: risk_off
Update strength: 0.5
Top driver: Apple's Q1 2023 Financial Results
Scope: ticker
Driver shock: -0.28
```

### Remaining quality concerns

Some outputs assigned imperfect scopes:

- unrelated company news was occasionally marked as ticker-specific;
- broad technology news was occasionally marked irrelevant;
- neutral days sometimes had missing top-driver text;
- some drivers had non-neutral pressure despite irrelevant scope.

These are prompt-quality and validation issues to investigate before final
experiments.

---

## 11. Implemented Pipeline Changes

### Added daily extraction script

```text
src/data/extract_daily_contextual_drivers.py
```

It:

- groups news by ticker/day;
- builds daily news packets;
- sends one packet per inference call;
- outputs one predicate row per ticker/day;
- checkpoints progress;
- resumes when rerun without `--force`.

### Updated daily dataset builder

```text
src/data/build_latent_state_dataset.py
```

Added:

```text
--daily-contextual-drivers-path
```

It merges the daily Qwen predicates after article rows have been aggregated to
daily ticker rows.

New driver features include:

```text
driver_update_strength
driver_uncertainty
net_pressure_score
driver_mean_direction_score
driver_mean_intensity
driver_mean_novelty
driver_mean_confidence
driver_mean_materiality
driver_mean_shock_score
driver_max_risk_on_shock
driver_max_risk_off_shock
driver_ticker_scope_share
driver_sector_scope_share
driver_broad_market_scope_share
driver_macro_scope_share
driver_irrelevant_scope_share
driver_channel_count
driver_mean_channel_importance
driver_macro_variable_channel_importance
driver_price_variable_channel_importance
```

### Updated LTN-style GRU training

```text
src/models/train_latent_state_gru.py
```

The LTN-style logic now reasons directly over daily LLM predicates.

General rules include:

```text
low update strength or irrelevant scope -> preserve persistence
low novelty -> discourage large hidden-state movement
persistent risk-off evidence -> reduce bull confidence
persistent risk-on evidence -> reduce bear confidence
conflicting or uncertain drivers -> increase sideways probability
macro rules -> active only when the LLM identifies relevant macro channels
```

### Updated explanation script

```text
src/models/explain_market_state_prediction.py
```

Explanations can combine:

- daily Qwen narrative;
- dominant contextual drivers;
- driver pressure and scope;
- probability movement;
- LTN-style rule activations;
- final regime interpretation.

### Updated documentation

```text
src/models/README.md
```

The documented workflow now prioritises daily contextual predicates rather than
article-level extraction.

### Suggested commit message

```text
Add daily LLM driver predicates for neuro-symbolic regime modelling
```

---

## 12. Full MAG7 Qwen Extraction

### Overnight extraction command

The full extraction was run as a background process with checkpointing:

```bash
mkdir -p logs

LOG=logs/qwen_mag7_daily_$(date +%Y%m%d_%H%M%S).log

nohup /home/jovyan/.venv/bin/python -u src/data/extract_daily_contextual_drivers.py \
  --input-path data/processed/ltn_all.parquet \
  --output-path data/processed/daily_contextual_driver_predicates_mag7_qwen.parquet \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --max-articles-per-day 5 \
  --max-article-chars 500 \
  --max-input-length 1536 \
  --max-new-tokens 500 \
  --batch-size 1 \
  --checkpoint-every 25 \
  --force \
  > "$LOG" 2>&1 &
```

### Result

The full extraction completed successfully:

```text
Processed 5,100/5,115 new ticker/day packets
Wrote daily contextual driver predicates to
data/processed/daily_contextual_driver_predicates_mag7_qwen.parquet
```

Final output:

```text
data/processed/daily_contextual_driver_predicates_mag7_qwen.parquet
```

Total:

```text
5,115 unique ticker/trading-day packets
```

Approximately:

```text
5,115 / 7 tickers = 731 ticker/day rows per ticker
```

Weekend news is assigned to the next available trading day through
`anchor_trading_date`.

---

## 13. Daily Dataset Merge

The daily contextual predicates were successfully merged into the latent-state
dataset.

The resulting full daily dataset contained:

```text
Rows: 5,115
Tickers: AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
Date range: 2023-02-01 to 2026-04-22
```

Splits:

```text
train:      3,274
validation:   421
test:       1,420
```

Future regime distribution:

| Split | Bear | Bull | Sideways |
|---|---:|---:|---:|
| Train | 339 | 2,086 | 849 |
| Validation | 110 | 286 | 25 |
| Test | 125 | 848 | 447 |

---

## 14. AAPL-Only GRU/LTN Experiment

Before the full MAG7 extraction completed, an AAPL-only experiment was run.

### Important correction

The first file named `aapl_qwen` still contained all MAG7 rows, while only AAPL
had Qwen driver values. This contaminated the experiment.

The dataset was then filtered to AAPL-only before retraining.

### AAPL-only label distribution

```text
Train:
bear_drawdown = 14
sideways = 243
bull_rally = 214

Validation:
bear_drawdown = 22
sideways = 4
bull_rally = 24

Test:
bear_drawdown = 0
sideways = 62
bull_rally = 131
```

### Models tested

1. AAPL Qwen GRU without LTN rules.
2. AAPL Qwen GRU with LTN rules.

Both used:

```text
sequence_length = 10
epochs = 10
persistence_prior_strength = 2.0
sampler = natural
class_weighting = none
```

### Results

Persistence baseline:

```text
test accuracy = 0.8497
test macro-F1 = 0.5457
```

AAPL Qwen GRU without rules:

```text
test accuracy = 0.8497
test macro-F1 = 0.5457
```

AAPL Qwen GRU with rules:

```text
test accuracy = 0.8497
test macro-F1 = 0.5457
```

### Interpretation

Both models exactly matched the persistence baseline. They did not demonstrate
additional predictive value.

This result was not considered a meaningful final failure because:

- the AAPL test split contained no bear examples;
- the validation set was very small;
- the target was highly imbalanced;
- the persistence baseline was extremely strong;
- AAPL-only data was too narrow for evaluating all regimes.

### Decision

Use the full MAG7 dataset for the primary comparison.

---

## 15. Main Evaluation Design

The final comparison should be:

| Model | Purpose |
|---|---|
| Persistence baseline | Future regime equals current regime |
| Qwen daily-driver GRU without LTN | Tests daily contextual predicates and recurrent latent state |
| Qwen daily-driver GRU with LTN | Tests whether neuro-symbolic constraints improve the same model |

Main metrics:

```text
accuracy
macro-F1
per-class precision, recall, and F1
confusion matrix
transition-case accuracy
transition-case macro-F1
```

Transition cases are especially important:

```text
current regime != future regime
```

The persistence baseline will naturally perform well on stable regimes. The
research question is whether daily contextual predicates and LTN reasoning add
value when the regime changes.

---

## 16. Current State

Completed:

- article, price, and macro dataset construction;
- ticker/day daily grouping;
- Qwen contextual driver extraction design;
- successful 25-day Qwen smoke test;
- successful full MAG7 extraction;
- 5,115 ticker/day contextual predicate rows generated;
- daily predicate merge support;
- GRU persistence prior;
- LTN-style predicate rules;
- explanation support;
- AAPL-only preliminary experiment.

Current full predicate file:

```text
data/processed/daily_contextual_driver_predicates_mag7_qwen.parquet
```

The immediate priority is to validate the full extracted predicates before
training the final MAG7 models.

---

## 17. Immediate Next Steps

### Step 1: Validate full predicate coverage

Check:

- row count per ticker;
- missing narratives;
- missing drivers;
- pressure distribution;
- scope distribution;
- update-strength distribution;
- uncertainty distribution;
- suspicious ticker/scope combinations.

```bash
/home/jovyan/.venv/bin/python - <<'PY'
import pandas as pd

path = "data/processed/daily_contextual_driver_predicates_mag7_qwen.parquet"
df = pd.read_parquet(path)

print("rows:", len(df))
print("\nrows per ticker:")
print(df["ticker"].value_counts())

print("\nnet pressure:")
print(df["net_pressure"].value_counts(dropna=False))

print("\nrows with narratives:", int(df["daily_market_narrative"].notna().sum()))
print("rows with drivers:", int((df["driver_count"] > 0).sum()))
print("rows without drivers:", int((df["driver_count"] == 0).sum()))

print("\nmissing values:")
cols = [
    "daily_market_narrative",
    "net_pressure",
    "driver_update_strength",
    "driver_uncertainty",
    "driver_mean_shock_score",
    "top_driver_1_label",
]
print(df[cols].isna().sum())

print("\nsample:")
print(df[[
    "ticker",
    "anchor_trading_date",
    "daily_market_narrative",
    "net_pressure",
    "driver_update_strength",
    "driver_uncertainty",
    "driver_mean_shock_score",
    "top_driver_1_label",
    "top_driver_1_scope",
    "top_driver_1_pressure",
]].sample(30, random_state=42).to_string(index=False))
PY
```

### Step 2: Build the full merged daily dataset

```bash
/home/jovyan/.venv/bin/python src/data/build_latent_state_dataset.py \
  --input-path data/processed/ltn_all.parquet \
  --daily-contextual-drivers-path data/processed/daily_contextual_driver_predicates_mag7_qwen.parquet \
  --output-dir data/processed/mag7_qwen \
  --schema-path data/processed/mag7_qwen/latent_state_schema.json \
  --require-contextual-drivers
```

### Step 3: Train the MAG7 model without LTN rules

```bash
/home/jovyan/.venv/bin/python src/models/train_latent_state_gru.py \
  --daily-path data/processed/mag7_qwen/latent_state_daily.parquet \
  --schema-path data/processed/mag7_qwen/latent_state_schema.json \
  --sequence-length 10 \
  --epochs 15 \
  --batch-size 32 \
  --hidden-size 32 \
  --dropout 0.35 \
  --learning-rate 0.0003 \
  --smoothness-weight 0.05 \
  --logic-weight 0 \
  --persistence-prior-strength 2.0 \
  --sampler natural \
  --class-weighting none \
  --require-contextual-drivers \
  --save-predictions \
  --output-dir models/mag7_qwen_gru_no_rules
```

### Step 4: Train the MAG7 model with LTN rules

```bash
/home/jovyan/.venv/bin/python src/models/train_latent_state_gru.py \
  --daily-path data/processed/mag7_qwen/latent_state_daily.parquet \
  --schema-path data/processed/mag7_qwen/latent_state_schema.json \
  --sequence-length 10 \
  --epochs 15 \
  --batch-size 32 \
  --hidden-size 32 \
  --dropout 0.35 \
  --learning-rate 0.0003 \
  --smoothness-weight 0.05 \
  --logic-weight 0.02 \
  --persistence-prior-strength 2.0 \
  --sampler natural \
  --class-weighting none \
  --require-contextual-drivers \
  --save-predictions \
  --output-dir models/mag7_qwen_gru_with_rules
```

### Step 5: Analyse transition cases

The final analysis must explicitly compare performance where:

```text
current regime != future regime
```

This is the most important test of whether news interpretation and
neuro-symbolic logic add value beyond persistence.

---

## 18. Final Research Framing

A suitable final dissertation claim is:

> An instruction-tuned language model interprets complete daily ticker-level
> news contexts and extracts open-ended contextual market driver predicates.
> These predicates identify driver direction, intensity, novelty, confidence,
> scope, uncertainty, update strength, and relevant economic channels. The
> predicates are merged with contemporaneously available market and
> macroeconomic context. LTN-style logic regularises economically plausible
> hidden-state updates, while a GRU represents the persistent latent ticker
> market condition over time. The system is evaluated against a strong regime
> persistence baseline, with particular attention to regime-transition cases.

