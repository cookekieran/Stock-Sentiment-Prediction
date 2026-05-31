"""Add deterministic macro-channel mappings to cached daily Qwen predicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from contextual_macro_channels import MACRO_CHANNEL_FEATURES, enrich_macro_channel_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = enrich_macro_channel_features(pd.read_parquet(args.input_path))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"Wrote {len(df):,} rows to {args.output_path}")
    print("\nMacro channel activations")
    for column in MACRO_CHANNEL_FEATURES:
        active = df[pd.to_numeric(df[column], errors="coerce").fillna(0.0) > 0.0]
        print(f"\n{column}: {len(active):,} ticker/day rows")
        for _, row in active.head(args.sample_size).iterrows():
            matches = json.loads(row["driver_macro_channel_matches_json"])
            labels = [
                match["driver_label"]
                for match in matches
                if f"driver_{match['macro_variable']}_channel_importance" == column
            ]
            print(f"  {row['ticker']} {row['anchor_trading_date'].date()}: {labels}")


if __name__ == "__main__":
    main()
