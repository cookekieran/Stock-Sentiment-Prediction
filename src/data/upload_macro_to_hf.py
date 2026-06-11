"""
Upload the FRED macroeconomic dataset artifacts to a Hugging Face dataset repo.

Example:
    python src/data/upload_macro_to_hf.py --repo-id username/mag7-macro-fred
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_FILES = (
    Path("data/raw/macro/fred_macro_raw.parquet"),
    Path("data/raw/macro/fred_macro_raw.csv"),
    Path("data/raw/macro/fred_series.json"),
    Path("data/processed/macro_monthly.parquet"),
    Path("data/processed/macro_monthly.csv"),
)


DATASET_CARD = """---
license: cc-by-4.0
pretty_name: FRED Macroeconomic Features for Mag7 News Trend Modelling
tags:
- finance
- macroeconomics
- fred
- stock-market
- neurosymbolic-ai
- logic-tensor-networks
---

# FRED Macroeconomic Features for Mag7 News Trend Modelling

This dataset contains broad U.S. macroeconomic indicators fetched from FRED for
use in a neuro-symbolic financial trend modelling project.

The model-ready file is `macro_monthly.parquet`. It contains monthly macro
features and derived regime indicators intended to be joined to article-level
market news examples by month.

## Included feature groups

- Inflation: CPI, core CPI, PCE, core PCE, year-over-year inflation features
- Rates: effective federal funds rate, Fed target range, 2Y and 10Y Treasury yields
- Yield curve: 10Y-2Y spread, inverted-yield-curve indicator
- Labour and growth: unemployment, nonfarm payrolls, real GDP, industrial production
- Stress/liquidity: VIX, trade-weighted dollar, WTI oil, BAA-10Y credit spread
- Macro regimes: high inflation, rising/falling rates, tightening, easing, yield-curve stress

## Files

- `macro_monthly.parquet`: monthly model-ready macro features
- `macro_monthly.csv`: CSV copy of the model-ready features
- `fred_macro_raw.parquet`: raw long-form FRED observations
- `fred_macro_raw.csv`: CSV copy of the raw observations
- `fred_series.json`: FRED series metadata used by the fetch script

## Intended use

These features are intended for research on combining financial news text,
ticker relevance, price trend labels, macro regime facts, and LTN constraints.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo, e.g. user/name")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        default=list(DEFAULT_FILES),
        help="Files to upload. Defaults to the FRED macro raw and processed outputs.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload FRED macroeconomic dataset",
    )
    return parser.parse_args()


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def validate_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot upload missing files: {missing}")


def main() -> None:
    args = parse_args()
    load_dotenv()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add it to .env or your environment.")

    validate_files(args.files)

    create_repo(
        repo_id=args.repo_id,
        token=token,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api = HfApi(token=token)
    for path in args.files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.commit_message,
        )
        print(f"Uploaded {path} -> {args.repo_id}/{path.name}")

    api.upload_file(
        path_or_fileobj=io.BytesIO(DATASET_CARD.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print(f"Uploaded dataset card -> {args.repo_id}/README.md")


if __name__ == "__main__":
    main()
