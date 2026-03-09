import pandas as pd
import requests
from admin import ALPACA_API_KEY, ALPACA_SECRET_KEY
from functions import get_end_date


def get_news(ticker, start_date, desired_timeframe_in_days=1):    
    url = "https://data.alpaca.markets/v1beta1/news"

    start_date, end_date = get_end_date(start_date, desired_timeframe_in_days)

    start_date = start_date.isoformat() + "Z"
    end_date = end_date.isoformat() + "Z"


    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY":ALPACA_SECRET_KEY
        }
    
    all_articles = []
    next_page_token = None
    
    while True:

        params = {
            "start": start_date,
            "end": end_date,
            "symbols": ticker,
            "limit": 50,
            "include_content": True,
            "page_token": next_page_token
        }

        response = requests.get(url, headers=headers, params=params)
        raw_json = response.json()

        sample = raw_json.get("news", [])
        all_articles.extend(sample)

        next_page_token = raw_json.get("next_page_token")

        if not next_page_token:
            break

    cleaned_news = [
        {
            "id": article["id"],
            "author": article["author"],
            "publish_time": article["created_at"],
            "headline": article["headline"],
            "source": article["source"],
            "related_tickers": article["symbols"],
            "summary": article["summary"]
        }
        for article in all_articles
    ]

    df = pd.DataFrame(cleaned_news)

    try:
        df['publish_time'] = pd.to_datetime(df['publish_time'], utc=True)
    except KeyError:
        print(f"No data found for ticker {ticker}")

    return df