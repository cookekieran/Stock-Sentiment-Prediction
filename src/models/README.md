# Market Regime Experiment Workflow

Run these commands from the project root.

## 1. Build the article-level dataset

Local files:

```powershell
python src\data\build_ltn_dataset.py
```

By default, the split is date-bounded:

- train: before `2025-04-01`
- validation: `2025-04-01` to `2025-06-30`
- test: from `2025-07-01` onward

This keeps the evaluation out-of-time and prevents later macro/market regimes
from leaking into model development.

With Hugging Face inputs, provide the dataset repos and paths/patterns:

```powershell
python src\data\build_ltn_dataset.py `
  --articles-hf-repo cookekieran/alphavantage-market-news `
  --articles-hf-pattern "articles/*.parquet" `
  --tickers-hf-repo cookekieran/alphavantage-market-news `
  --tickers-hf-pattern "tickers/*.parquet" `
  --prices-hf-repo cookekieran/mag7_prices `
  --prices-hf-path "mag7_daily_prices.parquet" `
  --macro-hf-repo cookekieran/fred-macro-data `
  --macro-hf-path "macro_monthly.parquet"
```

If the Hugging Face file paths differ, inspect them with:

```python
from huggingface_hub import HfApi
api = HfApi()
print(api.list_repo_files("cookekieran/alphavantage-market-news", repo_type="dataset"))
print(api.list_repo_files("cookekieran/mag7_prices", repo_type="dataset"))
print(api.list_repo_files("cookekieran/fred-macro-data", repo_type="dataset"))
```

The supervised target is `future_price_trend_id`:

- `0`: `bear_drawdown`
- `1`: `sideways`
- `2`: `bull_rally`

The current regime is based on backward-looking price context:

- `bull_rally`: 20-trading-day return is at least `+5%`
- `bear_drawdown`: 20-trading-day return is at most `-5%`, or the price is at
  least `10%` below its previous 60-trading-day high
- `sideways`: neither directional condition is active

The supervised target is that regime shifted 20 trading days into the future.
The older 120-day rally and drawdown features remain available as diagnostics,
but they no longer define the target label.

## 2. Build the latent-state daily dataset

This is the newer project direction. It aggregates many article rows into one
daily ticker signal, then lets a temporal model update a hidden market state over
time. This better matches the idea that one article should nudge the state,
rather than single-handedly flip the regime.

The main DeepSeek interpreter is the daily contextual driver extraction below.
Article-level DeepSeek embeddings are optional numeric features for ablation,
not the core explainability mechanism. If you want that optional embedding
baseline, run:

```powershell
python src\data\build_deepseek_news_features.py `
  --input-path data\processed\ltn_all.parquet `
  --output-path data\processed\deepseek_article_features.parquet `
  --batch-size 8 `
  --max-length 256 `
  --projection-dim 32
```

For a tiny smoke test:

```powershell
python src\data\build_deepseek_news_features.py `
  --max-articles 100 `
  --output-path data\processed\deepseek_article_features_smoke.parquet
```

For the explainable dissertation pipeline, DeepSeek should read all news for a
ticker/trading day together and output one daily contextual predicate set. This
keeps the model aligned with the latent-state idea: one article should not flip
the regime by itself; the hidden state updates once per ticker/day from the full
daily news context.

```powershell
python src\data\extract_daily_contextual_drivers.py `
  --input-path data\processed\ltn_all.parquet `
  --output-path data\processed\daily_contextual_driver_predicates.parquet `
  --batch-size 1
```

For a tiny smoke test:

```powershell
python src\data\extract_daily_contextual_drivers.py `
  --max-days 25 `
  --output-path data\processed\daily_contextual_driver_predicates_smoke.parquet
```

For a practical case-study extraction, restrict the expensive daily DeepSeek
step to a split, ticker, or date window:

```powershell
python src\data\extract_daily_contextual_drivers.py `
  --input-path data\processed\ltn_all.parquet `
  --output-path data\processed\daily_contextual_driver_predicates_case_study.parquet `
  --split test `
  --ticker NVDA `
  --start-date 2025-07-01 `
  --end-date 2026-04-30 `
  --batch-size 1
```

Map the open-ended daily drivers onto the controlled macro variables used by
the LTN-style rules. This is deterministic and does not call the language model:

```powershell
python src\data\enrich_daily_contextual_driver_macro_channels.py `
  --input-path data\processed\daily_contextual_driver_predicates.parquet `
  --output-path data\processed\daily_contextual_driver_predicates_macro_mapped.parquet
```

The command prints activation counts and sample matched drivers for inflation,
interest rates, the yield curve, volatility, oil prices, and unemployment. The
output includes the matched patterns so the mapping remains auditable.

Then build the daily latent-state dataset with the daily DeepSeek predicates
merged in. The model can be run without article-level embeddings; the key
interpretable DeepSeek input is the ticker/day predicate file:

```powershell
python src\data\build_latent_state_dataset.py `
  --daily-contextual-drivers-path data\processed\daily_contextual_driver_predicates_macro_mapped.parquet `
  --require-contextual-drivers
```

Outputs:

- `data/processed/latent_state_daily.parquet`
- `data/processed/latent_state_train.parquet`
- `data/processed/latent_state_validation.parquet`
- `data/processed/latent_state_test.parquet`
- `data/processed/latent_state_schema.json`

The latent-state features deliberately exclude forward-looking inputs such as
`next_day_return`, `three_day_return`, `market_sentiment_id`, and
`is_market_material`. Those fields are useful diagnostics, but using them as
inputs makes the task less honest.

The GRU schema also excludes raw price, volume, and non-stationary macro levels
such as CPI, GDP, VIX, and oil-price levels. The parquet retains them for
diagnostics. Training uses returns, drawdowns, volatility, YoY growth, monthly
macro changes, spreads, and regime indicators to reduce chronological
distribution-shift shortcuts.

## 3. Train the latent-state GRU model

```powershell
python src\models\train_latent_state_gru.py `
  --sequence-length 20 `
  --epochs 25 `
  --smoothness-weight 0.05 `
  --logic-weight 0.05 `
  --require-contextual-drivers `
  --save-predictions
```

The GRU hidden state is updated daily from:

- DeepSeek daily contextual driver predicates
- macroeconomic features
- known price/trend context at the anchor date

The additional losses encode the project idea:

- `smoothness_weight`: discourages abrupt regime probability jumps when the
  daily signal is calm
- `logic_weight`: softly regularises economically plausible transitions, e.g.
  persistent risk-off daily drivers should reduce bull confidence, conflicting
  drivers should raise uncertainty, and irrelevant/low-novelty contexts should
  preserve the current regime.

For a quick smoke test:

```powershell
python src\models\train_latent_state_gru.py `
  --sequence-length 5 `
  --epochs 1 `
  --hidden-size 16 `
  --batch-size 16
```

After training with `--save-predictions`, explain a specific ticker/date:

```powershell
python src\models\explain_market_state_prediction.py `
  --predictions-path models\latent_state_gru\test_predictions.parquet `
  --ticker NVDA `
  --date 2026-04-15
```

Audit Qwen-driven transition overrides against the persistence baseline:

```powershell
python src\models\audit_qwen_transition_overrides.py `
  --predictions-path models\revised_regime_price_qwen_no_rules\test_predictions.parquet `
  --output-dir models\revised_regime_price_qwen_no_rules\transition_audit `
  --threshold 0.21
```

The audit exports summaries and samples for true-positive overrides,
false-positive overrides, missed transitions, and correct persistence cases.
Each sample includes the Qwen narrative and top contextual driver.

To compare forecast horizons, rebuild the article-level and daily datasets with
`--trend-forward-days 5`, `10`, and `20`. Keep each horizon in its own output
folder, then train the same `--feature-set price_qwen` configuration against
each daily dataset.

## 4. Train the two-stage transition model

The three-class GRU is useful as a baseline, but stable regimes dominate its
objective. The two-stage trainer reframes the task around market-regime changes:

1. detect whether the current regime changes at the forecast horizon;
2. if a transition is proposed, classify the destination regime.

The destination loss is calculated only for genuine transition rows. At
inference, the model preserves the current regime unless the binary transition
detector crosses a threshold selected using validation transition F1. The
destination head is masked so a proposed transition cannot point back to the
current regime.

Run a price-plus-Qwen experiment for each forecast horizon:

```powershell
foreach ($horizon in 5, 10, 20) {
  python src\models\train_two_stage_transition_gru.py `
    --daily-path "data\processed\horizon_$horizon\daily\latent_state_daily.parquet" `
    --schema-path "data\processed\horizon_$horizon\daily\latent_state_schema.json" `
    --feature-set price_qwen `
    --sequence-length 10 `
    --epochs 25 `
    --early-stopping-patience 5 `
    --seed 42 `
    --batch-size 32 `
    --hidden-size 32 `
    --dropout 0.35 `
    --learning-rate 0.0003 `
    --require-contextual-drivers `
    --save-predictions `
    --output-dir "models\horizon_$horizon\two_stage_price_qwen"
}
```

Repeat with `--feature-set price_only` and a separate output folder to measure
whether Qwen predicates add signal beyond recent price dynamics.

To use recent price dynamics for transition timing while restricting Qwen
predicates to the destination classifier, run:

```powershell
python src\models\train_two_stage_transition_gru.py `
  --daily-path data\processed\horizon_5\daily\latent_state_daily.parquet `
  --schema-path data\processed\horizon_5\daily\latent_state_schema.json `
  --detector-feature-set price_only `
  --destination-feature-set price_qwen `
  --minimum-stable-accuracy 0.70 `
  --sequence-length 10 `
  --epochs 25 `
  --early-stopping-patience 5 `
  --seed 42 `
  --batch-size 32 `
  --hidden-size 32 `
  --dropout 0.35 `
  --learning-rate 0.0003 `
  --require-contextual-drivers `
  --save-predictions `
  --output-dir models\horizon_5\two_stage_split_price_detector_qwen_destination
```

The transition threshold is selected on validation final macro-F1 subject to
the minimum stable-accuracy requirement. This prevents the binary detector from
improving its apparent recall by proposing a transition almost every day.

Audit false positives and missed transitions against the forward price path:

```powershell
python src\models\audit_two_stage_transition_errors.py `
  --predictions-path models\horizon_5\two_stage_split_price_detector_price_destination\test_predictions.parquet `
  --daily-path data\processed\horizon_5\daily\latent_state_daily.parquet `
  --forecast-horizon 5 `
  --output-dir models\horizon_5\two_stage_split_price_detector_price_destination\transition_error_audit
```

The audit exports ticker-level summaries and samples with forward returns,
largest one-day moves, gradual-move flags, Qwen narratives, and top drivers.

## 5. Train the old DeepSeek-only baseline

```powershell
python src\models\train_deepseek_baseline.py `
  --batch-size 2 `
  --epochs 3
```

For a smoke test:

```powershell
python src\models\train_deepseek_baseline.py `
  --max-train-rows 20 `
  --max-validation-rows 20 `
  --max-test-rows 20 `
  --batch-size 1 `
  --epochs 1
```

## 6. Train the old DeepSeek+LTN model

```powershell
python src\models\train_deepseek_ltn.py `
  --batch-size 2 `
  --epochs 3 `
  --logic-weight 0.2
```

The DeepSeek+LTN model is still useful as a comparison, but it is article-level:
each article is treated as a separate supervised example. It does not maintain a
continuous hidden market state.

## 6. Compare experiments

```powershell
python src\models\compare_experiments.py
```

## Jupyter/Jovyan Container

The simplest container workflow is:

1. Upload one notebook.
2. In the first cell, install dependencies:

```python
%pip install -q pandas pyarrow numpy huggingface_hub torch transformers accelerate
```

3. Use `huggingface_hub` to download the three datasets into the notebook's
temporary storage.
4. Run the dataset builder code in the notebook.
5. Train baseline and LTN models against the generated `data/processed/*.parquet`.

Avoid relying on local project files in the container. Treat Hugging Face as the
source of truth for data and the notebook storage as disposable working space.
