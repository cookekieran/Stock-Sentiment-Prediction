# Stock Sentiment Prediction

Research project exploring whether financial news can be transformed into
structured event signals for stock-market prediction.

The project combines data engineering, NLP, and neuro-symbolic modelling. The
current focus is not just assigning a positive/negative sentiment label to each
article, but building a cleaner pipeline where raw news is collected, stored,
converted into event-level information, and then used for downstream market
reasoning.

## Why This Project

Financial news is noisy. A single ticker can appear in broad ETF articles,
competitor stories, market roundups, or repeated coverage of the same event.
Treating every article as an independent sentiment signal risks double-counting
and false positives.

This project investigates a more structured approach:

```text
Alpha Vantage news
        -> local PostgreSQL archive
        -> cleaned article / ticker / topic tables
        -> event extraction and event memory
        -> daily predicates for market modelling
```

## What I Have Built

- Backfilled Alpha Vantage market-news data in small time windows to work within
  the free API limit of 25 calls per day.
- Stored collected news in PostgreSQL so repeated collection sessions can resume
  from checkpoints instead of starting again.
- Normalised raw Alpha Vantage responses into article, ticker, and topic tables.
- Exported cleaned research snapshots to Parquet/private storage for notebook
  analysis.
- Built experimental notebooks that map news articles into event state and then
  into daily predicates for neuro-symbolic reasoning.

## Repository Structure

```text
data/
  pipeline/
    config.py                      # environment variable loading
    collect_news_by_ticker.py      # ticker-based Alpha Vantage news backfill
    collect_news_by_topic.py       # topic-based Alpha Vantage news backfill
    backfill_news_windows.py       # repair/backfill utility for missed windows
    build_clean_news_tables.py     # PostgreSQL -> cleaned article/ticker/topic tables
    transform_alpha_vantage.py     # shared Alpha Vantage transformation logic

  inspect_data/
    exploratory notebooks for checking uploaded/private datasets

notebooks/
  news_event_notebook_gemini.ipynb
  news_event_notebook_deepseek.ipynb
```

The separation is intentional:

- `data/pipeline/` contains scripts used to collect, store, clean, and export
  data.
- `data/inspect_data/` contains audit notebooks used to inspect saved datasets.
- `notebooks/` contains the research layer: event extraction, event memory, and
  predicate generation.

## Data Pipeline

The Alpha Vantage free tier is limited, so the collector scripts fetch data in
chunks and write progress to checkpoint files. This made it possible to build a
larger local dataset over multiple sessions without paid access.

The basic flow is:

1. `data/pipeline/collect_news_by_ticker.py` or
   `data/pipeline/collect_news_by_topic.py` requests a small date window from
   Alpha Vantage.
2. Raw article fields, ticker sentiment, and topic metadata are inserted into a
   local PostgreSQL table.
3. `data/pipeline/build_clean_news_tables.py` loads the stored rows and uses
   `data/pipeline/transform_alpha_vantage.py` to create:
   - a main article table;
   - exploded ticker metadata;
   - exploded topic metadata.
4. Cleaned tables can be exported to Parquet for research notebooks.

Large raw exports are not committed to GitHub. The repository keeps the code and
project structure public while leaving full data snapshots private, which avoids
repository bloat and respects third-party data access limits.

## Research Direction

The notebook work is moving toward an event-based representation of news.
Instead of asking only whether an article is bullish or bearish, the pipeline
asks:

- Is this article actually relevant to the target ticker?
- Does it describe a new event or repeat an existing event?
- What type of event is it: supply chain, demand, regulation, product,
  earnings, macro pressure, or something else?
- Can the latest event state be converted into daily predicates for modelling?

This creates a clearer bridge between unstructured news and structured market
features.

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
- `GEMINI_API_KEY`: required for the Gemini event-extraction notebook.
- `GEMINI_MODEL_ID`: optional Gemini model override.
- `GEMINI_BATCH_SIZE`: optional batching control for notebook calls.
- `GEMINI_SLEEP_SECONDS`: optional delay between Gemini requests.
- `HF_TOKEN`: required only if notebooks need to read private Hugging Face
  research snapshots.

## Current Status

This is an in-progress dissertation/research project. The data collection and
cleaning pipeline is the most developed part of the repository. The event-based
NLP and neuro-symbolic modelling work is active research and is being iterated
in the notebooks.

Near-term improvements:

- make the data collection scripts easier to run from a clean machine;
- add a small public sample dataset for demonstration;
- add tests around the Alpha Vantage transformation functions;
- evaluate whether event-level predicates improve over article-level sentiment
  baselines.

## Skills Demonstrated

- Python data pipeline development
- API collection under quota constraints
- PostgreSQL storage and export workflows
- Financial NLP dataset preparation
- Parquet/private dataset handling
- Notebook-based research iteration
- Event extraction and neuro-symbolic modelling concepts
