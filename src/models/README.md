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

## 2. Build the latent-state daily dataset

This is the newer project direction. It aggregates many article rows into one
daily ticker signal, then lets a temporal model update a hidden market state over
time. This better matches the idea that one article should nudge the state,
rather than single-handedly flip the regime.

To use DeepSeek as the news interpreter, first create article-level DeepSeek
features. This runs the frozen model once per unique article and saves compact
numeric features:

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

For the explainable dissertation pipeline, extract open-ended contextual market
drivers. This is not just an optional add-on: it is the part where DeepSeek
interprets the news and produces human-readable market-state update evidence.
It lets DeepSeek name period-specific drivers such as "Iran conflict
escalation", "AI capex optimism", or "regional bank credit stress", while saving
general numeric update signals such as risk direction, novelty, intensity, and
confidence:

```powershell
python src\data\extract_contextual_drivers.py `
  --input-path data\processed\ltn_all.parquet `
  --output-path data\processed\contextual_driver_features.parquet `
  --batch-size 1 `
  --min-relevance 0.25 `
  --top-n-per-ticker-day 3
```

For a tiny smoke test:

```powershell
python src\data\extract_contextual_drivers.py `
  --max-rows 25 `
  --min-relevance 0.25 `
  --output-path data\processed\contextual_driver_features_smoke.parquet
```

For a practical case-study extraction, restrict the expensive contextual driver
step to a split, date window, or top articles:

```powershell
python src\data\extract_contextual_drivers.py `
  --input-path data\processed\ltn_all.parquet `
  --output-path data\processed\contextual_driver_features_case_study.parquet `
  --split test `
  --start-date 2025-07-01 `
  --end-date 2026-04-30 `
  --min-relevance 0.35 `
  --top-n-per-ticker-day 2 `
  --batch-size 1
```

Then build the daily latent-state dataset with those DeepSeek features merged in:

```powershell
python src\data\build_latent_state_dataset.py `
  --deepseek-features-path data\processed\deepseek_article_features.parquet `
  --contextual-drivers-path data\processed\contextual_driver_features.parquet `
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

- aggregated news intensity/relevance/sentiment
- macroeconomic features
- known price/trend context at the anchor date

The additional losses encode the project idea:

- `smoothness_weight`: discourages abrupt regime probability jumps when the
  daily signal is calm
- `logic_weight`: softly regularises economically plausible transitions, e.g.
  tightening plus negative relevant DeepSeek contextual drivers should raise
  bearish probability

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

## 4. Train the old DeepSeek-only baseline

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

## 5. Train the old DeepSeek+LTN model

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
