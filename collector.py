"""
This script runs daily at midday and takes all the stock data on my desired tickers from yesterday. Data is saved into an SQL database so can be accessed later.
"""

from yfinance_data import get_stock_data
from alpaca_api import get_news
from datetime import date, timedelta
import pandas as pd
from admin import POSTGRES_PASS, DB_USER, DB_PASSWORD, DB_HOST
from sqlalchemy import create_engine
import datetime
from sqlalchemy.exc import IntegrityError, ProgrammingError

day_of_week = datetime.datetime.now().weekday()
if day_of_week in [0,6]:
    print("No weekend data to collect.")
    exit()

today = date.today()
yesterday_object = today - timedelta(days=1)
yesterday = yesterday_object.strftime("%Y-%m-%d")

all_prices = []

tickers = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    "SPY", "QQQ"                                     
]

for ticker in tickers:
    stock_data = get_stock_data(ticker, yesterday)
    stock_data["ticker"] = ticker
    all_prices.append(stock_data)

master_df = pd.concat(all_prices)
master_df.drop_duplicates(inplace=True)
master_df.index.name = 'Datetime'
local_engine = create_engine(f'postgresql://postgres:{POSTGRES_PASS}@localhost:5432/stock_market')
cloud_engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/postgres")

for engine in [local_engine, cloud_engine]:
    try:
        master_df.to_sql('stock_prices', engine, if_exists='append', index=True)

    except IntegrityError:
        print("Data already exists...")
    except ProgrammingError:
        print("Column mismatch error")
    except Exception as e:
        print(f"An unexpected error ocured {e}")
    finally:
        print("Script finished.")