import yfinance as yf
import pandas as pd
import datetime as datetime
from functions import get_end_date


def get_stock_data(ticker, start_date):
    start_date, end_date = get_end_date(start_date)

    data = yf.Ticker(ticker)

    custom_period_prices = data.history(start=start_date, end=end_date, interval="1m")

    df = pd.DataFrame(custom_period_prices)

    df.index = pd.to_datetime(df.index, utc=True)

    return df