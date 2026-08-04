# Stock Sentiment Prediction

Dissertation research project exploring whether MAG7 financial news can be
converted into structured, firm-specific event labels and combined with market
context to predict stock-market reactions.

The project combines data engineering, LLM-based event extraction, and
neuro-symbolic modelling. Rather than assigning a single positive/negative
sentiment score to every headline, the pipeline first decides whether an
article describes a genuine, firm-specific market-moving event for its target
MAG7 company, then classifies that event, and finally feeds it into a softmax
neural classifier trained jointly with a Logic Tensor Network (LTN) rule layer.

## Pipeline

```text
Alpha Vantage news
        -> local PostgreSQL archive
        -> cleaned article / ticker / topic tables
        -> Hugging Face dataset snapshot
        -> prompt selection (few-shot relevance judge)
        -> article-event label extraction (LLM classifier + leakage-safe event date)
        -> interaction model: article-event labels + market context + LTN rules
```

## Repository Structure

```text
data/
  pipeline/                                   # scripts used to collect, clean, and export the raw news data
    config.py                                 # environment variable loading
    collect_news_by_ticker.py                 # ticker-based Alpha Vantage news backfill
    collect_news_by_topic.py                  # topic-based Alpha Vantage news backfill
    backfill_news_windows.py                  # repair/backfill utility for missed windows
    build_clean_news_tables.py                # PostgreSQL -> cleaned article/ticker/topic tables
    transform_alpha_vantage.py                # shared Alpha Vantage transformation logic
    fetch_mag7_alphavantage_earnings_dates.py # earnings-date calendar per MAG7 ticker
    fetch_mag7_alphavantage_fundamentals.py   # supplementary MAG7 fundamentals data
  mag7_daily_prices.parquet                   # daily OHLCV prices for the MAG7 tickers
  sp500_daily_prices.parquet                  # daily S&P 500 prices, used as market context
  mag7_headline_relevance_hard_audit_420_relabelled_clean.csv   # manual audit set used by the prompt evaluator
  mag7_headline_relevance_hard_audit_420.csv                    # pre-relabel version, kept for provenance
  article_event_labels.parquet                # output of scripts/1_run_mag7_strict_event_study_v3.py; consumed by the modelling notebook

outputs/
  prompt_eval_market_reaction_relevance/      # results of scripts/2_evaluate_prompt_variants_v3.py (per-variant metrics, best-prompt selection)
  mag7_strict_event_study/                    # provenance for article_event_labels.parquet (audit csv, manifest, run log)

scripts/
  1_run_mag7_strict_event_study_v3.py         # LLM article-event label extraction (production classifier)
  2_evaluate_prompt_variants_v3.py            # 0-6 shot prompt selection for the relevance judge

notebooks/
  mag7_article_event_label_interaction_FINAL_AUGUST.ipynb       # final modelling notebook
  FINAL_17_mag7_article_event_label_interaction_clean_no_robust/  # the notebook's own output (metrics, confusion matrices, LTN rules)
```

The separation is intentional:

- `data/pipeline/` contains everything used to collect, clean, and export the
  raw data the study is built on.
- `scripts/` contains the two-stage LLM event-extraction pipeline, run in a
  GPU container.
- `outputs/` contains the results of the two `scripts/` stages, for provenance.
- `notebooks/` contains the final modelling notebook and its own output.

## Data Pipeline

The Alpha Vantage free tier is limited, so the collector scripts fetch data in
chunks and write progress to checkpoint files. This made it possible to build a
larger local dataset over multiple sessions without paid access.

1. `data/pipeline/collect_news_by_ticker.py` or
   `data/pipeline/collect_news_by_topic.py` requests a small date window from
   Alpha Vantage.
2. Raw article fields, ticker sentiment, and topic metadata are inserted into a
   local PostgreSQL table.
3. `data/pipeline/build_clean_news_tables.py` loads the stored rows and uses
   `data/pipeline/transform_alpha_vantage.py` to build a main article table
   plus exploded ticker and topic metadata, which is exported as a Hugging Face
   dataset snapshot for the extraction stage below.
4. `data/pipeline/fetch_mag7_alphavantage_earnings_dates.py` and
   `fetch_mag7_alphavantage_fundamentals.py` pull supplementary MAG7 data used
   as model context/features.

Large raw exports are not committed to GitHub. The repository keeps the code
and project structure public while leaving full data snapshots private, which
avoids repository bloat and respects third-party data access limits.

## Event-Extraction Pipeline

1. `scripts/2_evaluate_prompt_variants_v3.py` evaluates 0-shot through 6-shot
   variants of the market-reaction relevance prompt against a manually
   labelled audit set, and selects the best-performing variant by F1 on a
   held-out development split.
2. `scripts/1_run_mag7_strict_event_study_v3.py` runs the selected prompt
   against every MAG7 ticker-article using a local Qwen2.5-7B-Instruct model.
   Each kept article is classified into one primary event type and direction,
   and mapped to a leakage-safe trading `event_date` (i.e. the next trading day
   the market could actually have reacted to the news). Output:
   `article_event_labels.parquet`, one row per kept ticker-article label.
3. `notebooks/mag7_article_event_label_interaction_FINAL_AUGUST.ipynb` fuses
   `article_event_labels.parquet` with market context (`mag7_daily_prices.parquet`,
   `sp500_daily_prices.parquet`) and interaction features to classify each
   event as bullish, bearish, or neutral, using a softmax classifier trained
   jointly with an LTN rule layer, then evaluates it on a held-out test period.
   Results are written to `notebooks/FINAL_17_mag7_article_event_label_interaction_clean_no_robust/`.

The notebook expects `article_event_labels.parquet` (and the price parquets) in
its own working directory; when running it, either copy `data/article_event_labels.parquet`
next to the notebook or point the `ARTICLE_EVENTS_FILE`/`WORK_DIR` variables at `data/`.

## Setup

Create an environment and install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebooks]"
```

Copy `.env.example` to `.env` and fill in only the credentials needed for the
part of the project you are running.

## Environment Variables

- `ALPHA_VANTAGE_API_KEY`: required for collecting new Alpha Vantage data.
- `POSTGRES_PASS`: required for local PostgreSQL collection/export scripts.
- `HF_TOKEN`: required for the event-extraction scripts, which load the news
  dataset and price calendar from private Hugging Face datasets and download
  the Qwen model weights.

## Current Status

This is an in-progress dissertation/research project. The MAG7 study pipeline
above is the finalised version referenced in the dissertation; earlier
iterations (including a parallel oil-market transfer study) have been moved out
of the repository.

## Skills Demonstrated

- Python data pipeline development
- API collection under quota constraints
- PostgreSQL storage and export workflows
- LLM-based financial event classification and prompt evaluation
- Leakage-safe event-to-trading-date mapping
- Neuro-symbolic (LTN) modelling combined with a neural classifier
- Financial NLP dataset preparation
- Parquet/private dataset handling
