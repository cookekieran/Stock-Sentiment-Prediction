from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "ticker_daily_context.parquet"

load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")
RETURN_WINDOWS = (1, 5, 20, 45, 60, 90, 120)
STATE_WINDOWS = (20, 60, 90)


def load_parquet(repo_id, filename):
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        token=HF_TOKEN,
    )
    return pd.read_parquet(path)


prices = load_parquet("cookekieran/mag7_prices", "mag7_daily_prices.parquet")
market = load_parquet("cookekieran/us_market_prices", "sp500_daily_prices.parquet")
fundamentals = load_parquet(
    "cookekieran/mag7-fundamentals",
    "data/model_ready_fundamentals_wide.parquet",
)
earnings = load_parquet(
    "cookekieran/mag7-fundamentals",
    "data/earnings_dates_quarterly.parquet",
)

prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["ticker", "date"])

for days in RETURN_WINDOWS:
    prices[f"stock_return_{days}d"] = prices.groupby("ticker")["adj_close"].pct_change(days)

for days in STATE_WINDOWS:
    prices[f"stock_volatility_{days}d"] = prices.groupby("ticker")["stock_return_1d"].transform(
        lambda returns: returns.rolling(days).std()
    )
    prices[f"relative_volume_{days}d"] = prices.groupby("ticker")["volume"].transform(
        lambda volume: volume / volume.rolling(days).mean().shift(1)
    )
    prices[f"distance_from_{days}d_high"] = prices.groupby("ticker")["adj_close"].transform(
        lambda close: close / close.rolling(days).max() - 1
    )
    prices[f"distance_from_{days}d_low"] = prices.groupby("ticker")["adj_close"].transform(
        lambda close: close / close.rolling(days).min() - 1
    )
prices["distance_from_20d_moving_average"] = prices.groupby("ticker")["adj_close"].transform(
    lambda close: close / close.rolling(20).mean() - 1
)

market["date"] = pd.to_datetime(market["date"])
market = market.sort_values("date")
for days in RETURN_WINDOWS:
    market[f"sp500_return_{days}d"] = market["adj_close"].pct_change(days)

context = prices.merge(
    market[["date", *[f"sp500_return_{days}d" for days in RETURN_WINDOWS]]],
    on="date",
    how="left",
)
for days in RETURN_WINDOWS:
    context[f"stock_minus_sp500_return_{days}d"] = (
        context[f"stock_return_{days}d"] - context[f"sp500_return_{days}d"]
    )

fundamentals["fiscal_period_end"] = pd.to_datetime(fundamentals["fiscal_period_end"])
earnings["fiscal_period_end"] = pd.to_datetime(earnings["fiscal_period_end"])
earnings["reported_date"] = pd.to_datetime(earnings["reported_date"])

financials = earnings.merge(
    fundamentals,
    on=["ticker", "fiscal_period_end"],
    how="left",
).sort_values(["ticker", "fiscal_period_end"])

revenue = "fundamental_income_statement_totalRevenue"
operating_income = "fundamental_income_statement_operatingIncome"
gross_profit = "fundamental_income_statement_grossProfit"
research_and_development = "fundamental_income_statement_researchAndDevelopment"
cash = "fundamental_balance_sheet_cashAndShortTermInvestments"
debt = "fundamental_balance_sheet_shortLongTermDebtTotal"
assets = "fundamental_balance_sheet_totalAssets"

financials["revenue_growth_yoy"] = financials.groupby("ticker")[revenue].pct_change(
    4,
    fill_method=None,
)
financials["gross_margin"] = financials[gross_profit] / financials[revenue]
financials["operating_margin"] = financials[operating_income] / financials[revenue]
financials["operating_margin_change_yoy"] = financials.groupby("ticker")[
    "operating_margin"
].diff(4)
financials["r_and_d_intensity"] = financials[research_and_development] / financials[revenue]
financials["cash_to_debt"] = financials[cash] / financials[debt].where(financials[debt] != 0)
financials["debt_to_assets"] = financials[debt] / financials[assets]
financials["available_from"] = financials["reported_date"] + pd.Timedelta(days=1)

financial_columns = [
    "ticker",
    "available_from",
    "reported_date",
    "reportedEPS",
    "estimatedEPS",
    "surprisePercentage",
    "reportTime",
    revenue,
    operating_income,
    cash,
    debt,
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "operating_margin_change_yoy",
    "r_and_d_intensity",
    "cash_to_debt",
    "debt_to_assets",
]

context = pd.merge_asof(
    context.sort_values(["date", "ticker"]),
    financials[financial_columns].sort_values(["available_from", "ticker"]),
    left_on="date",
    right_on="available_from",
    by="ticker",
    direction="backward",
)

context["days_since_last_earnings_report"] = (
    context["date"] - context["reported_date"]
).dt.days

context = context.rename(columns={
    "date": "price_data_as_of",
    "reportedEPS": "latest_reported_eps",
    "estimatedEPS": "latest_estimated_eps",
    "surprisePercentage": "latest_eps_surprise_percentage",
    "reportTime": "latest_report_time",
    revenue: "latest_revenue",
    operating_income: "latest_operating_income",
    cash: "cash_and_short_term_investments",
    debt: "total_debt",
})

context_columns = [
    "ticker",
    "price_data_as_of",
    *[f"stock_return_{days}d" for days in RETURN_WINDOWS],
    *[f"stock_volatility_{days}d" for days in STATE_WINDOWS],
    *[f"relative_volume_{days}d" for days in STATE_WINDOWS],
    *[f"distance_from_{days}d_high" for days in STATE_WINDOWS],
    *[f"distance_from_{days}d_low" for days in STATE_WINDOWS],
    "distance_from_20d_moving_average",
    *[f"sp500_return_{days}d" for days in RETURN_WINDOWS],
    *[f"stock_minus_sp500_return_{days}d" for days in RETURN_WINDOWS],
    "days_since_last_earnings_report",
    "latest_reported_eps",
    "latest_estimated_eps",
    "latest_eps_surprise_percentage",
    "latest_report_time",
    "latest_revenue",
    "latest_operating_income",
    "cash_and_short_term_investments",
    "total_debt",
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "operating_margin_change_yoy",
    "r_and_d_intensity",
    "cash_to_debt",
    "debt_to_assets",
]

context[context_columns].sort_values(["ticker", "price_data_as_of"]).to_parquet(
    OUTPUT_PATH,
    index=False,
)

print(f"Saved {len(context):,} rows to {OUTPUT_PATH}")
