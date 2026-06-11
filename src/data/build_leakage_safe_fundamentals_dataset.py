"""Join quarterly fundamentals to a daily dataset after their public release dates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATEMENT_FIELDS = {
    "income_statement": [
        "totalRevenue",
        "grossProfit",
        "operatingIncome",
        "netIncome",
        "ebitda",
    ],
    "balance_sheet": [
        "totalAssets",
        "totalCurrentAssets",
        "totalCurrentLiabilities",
        "totalLiabilities",
        "totalShareholderEquity",
        "cashAndCashEquivalentsAtCarryingValue",
        "inventory",
        "currentNetReceivables",
        "shortLongTermDebtTotal",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-path", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument(
        "--raw-fundamentals-dir",
        type=Path,
        default=Path("data/raw/company_fundamentals/alphavantage"),
    )
    parser.add_argument(
        "--earnings-dates-path",
        type=Path,
        default=Path(
            "data/processed/company_fundamentals/"
            "mag7_quarterly_earnings_dates_alphavantage.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-schema-path", type=Path, required=True)
    parser.add_argument("--output-stem", default="daily_context")
    parser.add_argument("--release-delay-days", type=int, default=1)
    parser.add_argument("--fallback-reporting-lag-days", type=int, default=45)
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"None": None, "": None}), errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def load_statement_history(raw_dir: Path) -> pd.DataFrame:
    statement_frames = []
    for ticker_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        ticker = ticker_dir.name.upper()
        merged = None
        for statement_type, fields in STATEMENT_FIELDS.items():
            payload = json.loads((ticker_dir / f"{statement_type}_raw.json").read_text(encoding="utf-8"))
            reports = pd.DataFrame(payload.get("quarterlyReports", []))
            reports["ticker"] = ticker
            reports["fiscal_period_end"] = pd.to_datetime(reports.pop("fiscalDateEnding")).dt.normalize()
            available = [field for field in fields if field in reports.columns]
            reports = reports[["ticker", "fiscal_period_end", *available]].copy()
            for field in available:
                reports[field] = numeric(reports[field])
            reports = reports.rename(
                columns={field: f"raw_{statement_type}_{field}" for field in available}
            )
            merged = reports if merged is None else merged.merge(
                reports,
                on=["ticker", "fiscal_period_end"],
                how="outer",
                validate="one_to_one",
            )
        statement_frames.append(merged)
    return pd.concat(statement_frames, ignore_index=True).sort_values(
        ["ticker", "fiscal_period_end"]
    )


def derive_features(statements: pd.DataFrame) -> pd.DataFrame:
    statements = statements.copy()
    income = "raw_income_statement_"
    balance = "raw_balance_sheet_"
    revenue = statements[f"{income}totalRevenue"]
    gross_profit = statements[f"{income}grossProfit"]
    operating_income = statements[f"{income}operatingIncome"]
    net_income = statements[f"{income}netIncome"]
    ebitda = statements[f"{income}ebitda"]
    assets = statements[f"{balance}totalAssets"]
    current_assets = statements[f"{balance}totalCurrentAssets"]
    current_liabilities = statements[f"{balance}totalCurrentLiabilities"]
    liabilities = statements[f"{balance}totalLiabilities"]
    equity = statements[f"{balance}totalShareholderEquity"]
    cash = statements[f"{balance}cashAndCashEquivalentsAtCarryingValue"]
    inventory = statements[f"{balance}inventory"]
    receivables = statements[f"{balance}currentNetReceivables"]
    debt = statements[f"{balance}shortLongTermDebtTotal"]

    statements["fundamental_gross_margin"] = safe_divide(gross_profit, revenue)
    statements["fundamental_operating_margin"] = safe_divide(operating_income, revenue)
    statements["fundamental_net_margin"] = safe_divide(net_income, revenue)
    statements["fundamental_ebitda_margin"] = safe_divide(ebitda, revenue)
    statements["fundamental_current_ratio"] = safe_divide(current_assets, current_liabilities)
    statements["fundamental_liabilities_to_assets"] = safe_divide(liabilities, assets)
    statements["fundamental_debt_to_equity"] = safe_divide(debt, equity)
    statements["fundamental_cash_to_assets"] = safe_divide(cash, assets)

    yoy_sources = {
        "revenue": revenue,
        "gross_profit": gross_profit,
        "operating_income": operating_income,
        "net_income": net_income,
        "ebitda": ebitda,
        "assets": assets,
        "cash": cash,
        "inventory": inventory,
        "receivables": receivables,
        "debt": debt,
    }
    for name, source in yoy_sources.items():
        statements[f"fundamental_{name}_yoy"] = source.groupby(statements["ticker"]).pct_change(
            4,
            fill_method=None,
        )
    for name in ["gross_margin", "operating_margin", "net_margin", "ebitda_margin"]:
        column = f"fundamental_{name}"
        statements[f"{column}_yoy_change"] = statements.groupby("ticker")[column].diff(4)
    feature_columns = [column for column in statements.columns if column.startswith("fundamental_")]
    statements[feature_columns] = statements[feature_columns].replace([np.inf, -np.inf], np.nan)
    return statements[["ticker", "fiscal_period_end", *feature_columns]]


def attach_release_dates(
    features: pd.DataFrame,
    earnings_dates_path: Path,
    release_delay_days: int,
    fallback_reporting_lag_days: int,
) -> pd.DataFrame:
    earnings = pd.read_parquet(earnings_dates_path).copy()
    earnings["ticker"] = earnings["ticker"].astype(str).str.upper()
    earnings["fiscal_period_end"] = pd.to_datetime(earnings["fiscal_period_end"]).dt.normalize()
    earnings["reported_date"] = pd.to_datetime(earnings["reported_date"]).dt.normalize()
    release_dates = earnings[["ticker", "fiscal_period_end", "reported_date"]].drop_duplicates(
        ["ticker", "fiscal_period_end"]
    )
    features = features.merge(
        release_dates,
        on=["ticker", "fiscal_period_end"],
        how="left",
        validate="one_to_one",
    )
    fallback = features["fiscal_period_end"] + pd.to_timedelta(fallback_reporting_lag_days, unit="D")
    features["fundamental_release_date_source"] = np.where(
        features["reported_date"].notna(),
        "alphavantage_earnings_reported_date",
        f"fallback_fiscal_end_plus_{fallback_reporting_lag_days}d",
    )
    features["fundamental_reported_date"] = features["reported_date"].fillna(fallback)
    features["fundamental_available_from_date"] = (
        features["fundamental_reported_date"]
        + pd.to_timedelta(release_delay_days, unit="D")
    ).dt.normalize()
    return features.drop(columns=["reported_date"])


def join_daily(daily: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["ticker"] = daily["ticker"].astype(str).str.upper()
    daily["anchor_trading_date"] = pd.to_datetime(daily["anchor_trading_date"]).dt.normalize()
    daily["_daily_order"] = np.arange(len(daily))
    daily = daily.sort_values(["anchor_trading_date", "ticker"])
    quarterly = quarterly.sort_values(["fundamental_available_from_date", "ticker"])
    joined = pd.merge_asof(
        daily,
        quarterly,
        by="ticker",
        left_on="anchor_trading_date",
        right_on="fundamental_available_from_date",
        direction="backward",
        allow_exact_matches=True,
    )
    joined["fundamental_quarter_age_days"] = (
        joined["anchor_trading_date"] - joined["fundamental_available_from_date"]
    ).dt.days
    joined["fundamental_has_public_statement"] = joined["fundamental_available_from_date"].notna().astype(float)
    leaked = joined["fundamental_available_from_date"].notna() & (
        joined["fundamental_available_from_date"] > joined["anchor_trading_date"]
    )
    if leaked.any():
        raise RuntimeError(f"Leakage check failed for {int(leaked.sum())} daily rows.")
    return joined.sort_values("_daily_order").drop(columns=["_daily_order"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))
    daily = pd.read_parquet(args.daily_path)
    history = load_statement_history(args.raw_fundamentals_dir)
    quarterly = attach_release_dates(
        derive_features(history),
        args.earnings_dates_path,
        args.release_delay_days,
        args.fallback_reporting_lag_days,
    )
    joined = join_daily(daily, quarterly)
    feature_columns = [column for column in joined.columns if column.startswith("fundamental_")]
    excluded_metadata = {
        "fundamental_release_date_source",
        "fundamental_reported_date",
        "fundamental_available_from_date",
    }
    numeric_features = [column for column in feature_columns if column not in excluded_metadata]
    output_schema = {
        **schema,
        "feature_columns": list(dict.fromkeys([*schema["feature_columns"], *numeric_features])),
        "fundamental_feature_columns": numeric_features,
        "fundamental_release_delay_days": args.release_delay_days,
        "fundamental_fallback_reporting_lag_days": args.fallback_reporting_lag_days,
        "fundamental_leakage_control": (
            "Quarterly features become available one day after Alpha Vantage EARNINGS reportedDate; "
            "unmatched periods use a flagged conservative lag after fiscal period end."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.output_dir / f"{args.output_stem}_daily.parquet", index=False)
    quarterly.to_parquet(args.output_dir / "fundamentals_quarterly_features.parquet", index=False)
    for split, part in joined.groupby("split", sort=False):
        part.to_parquet(args.output_dir / f"{args.output_stem}_{split}.parquet", index=False)
    args.output_schema_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_schema_path.write_text(json.dumps(output_schema, indent=2), encoding="utf-8")

    known = joined["fundamental_available_from_date"].notna()
    print("Leakage-safe fundamentals dataset created")
    print(f"Rows: {len(joined):,}")
    print(f"Daily rows with a public statement available: {int(known.sum()):,}")
    print(f"Daily rows before the first public statement: {int((~known).sum()):,}")
    print(f"Fundamental model features: {len(numeric_features)}")
    print("\nRelease-date sources")
    print(quarterly["fundamental_release_date_source"].value_counts().to_string())
    print("\nRelease-date sources visible on daily model rows")
    print(joined["fundamental_release_date_source"].value_counts(dropna=False).to_string())
    print("\nFirst daily availability by ticker")
    print(
        joined.loc[known]
        .groupby("ticker")["fundamental_available_from_date"]
        .min()
        .to_string()
    )
    print("\nLeakage check: passed (no statement used before its availability date)")


if __name__ == "__main__":
    main()
