"""Conservatively sanitize scope features from an older daily predicate checkpoint.

Older checkpoints treated composite scopes such as ``ticker|sector`` as
``irrelevant``. The original scope text is not recoverable for every row because
the debug generation was truncated after parsing. Disable that unreliable
feature before training instead of applying a false persistence signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    column = "driver_irrelevant_scope_share"
    if column not in df.columns:
        raise ValueError(f"Missing required scope feature: {column}")

    affected = int((pd.to_numeric(df[column], errors="coerce").fillna(0.0) > 0.0).sum())
    df[column] = 0.0
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"Wrote {len(df):,} rows to {args.output_path}")
    print(f"Disabled unreliable irrelevant-scope signal on {affected:,} rows")


if __name__ == "__main__":
    main()
