import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from config import POSTGRES_PASS, ALPHA_VANTAGE_API_KEY


# ALL_TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SPY", "QQQ", "GLD", "SLV", "USO"]
# day_of_year = datetime.now().timetuple().tm_yday
# ticker_index = day_of_year % len(ALL_TICKERS)
# CURRENT_TICKER = ALL_TICKERS[ticker_index]

# manual override
CURRENT_TICKER = "TSLA"
day_of_year = datetime.now().timetuple().tm_yday

DB_URL = f'postgresql://postgres:{POSTGRES_PASS}@localhost:5432/stock_market'
engine = create_engine(DB_URL)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, f"checkpoint_{CURRENT_TICKER}.txt")

print(f"--- AUTOMATION: Today is day {day_of_year}. Target Ticker: {CURRENT_TICKER} ---")

def get_current_pointer():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            val = f.read().strip()
            return datetime.strptime(val, "%Y%m%dT%H%M")
    return datetime(2023, 1, 1)


def fetch_news_chunk(ticker, start_time, end_time):

    start_str = start_time.strftime("%Y%m%dT%H%M")
    end_str = end_time.strftime("%Y%m%dT%H%M")


    url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
           f"&tickers={ticker}&time_from={start_str}&time_to={end_str}"
           f"&sort=EARLIEST&limit=1000&apikey={ALPHA_VANTAGE_API_KEY}")
    try:
        response = requests.get(url)
        data = response.json()

        if "Note" in data or "Information" in data:
            print(f"API Limit reached: {data.get('Note', data.get('Information'))}")
            return None
        
        if "feed" not in data:
            print(f"No data for {start_time}, {end_time} retrieved")
            return True
        

        count = len(data["feed"])

        if count > 0:       
            processed_news = []

            for item in data["feed"]:
                dt_obj = datetime.strptime(item.get("time_published"), "%Y%m%dT%H%M%S")
                processed_news.append({
                    "requested_entity": ticker,
                    "title": item.get("title"),
                    "time_published": dt_obj,
                    "url": item.get("url"),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "overall_sentiment_score": float(item.get("overall_sentiment_score", 0)),
                    "overall_sentiment_label": item.get("overall_sentiment_label"),
                    "ticker_sentiment": str(item.get("ticker_sentiment", [])),
                    "topics": str(item.get("topics", [])) 
                })

            
            df = pd.DataFrame(processed_news)
            with engine.begin() as conn:
                stmt = text("""
                    INSERT INTO news_sentiment 
                    (requested_entity, title, time_published, url, summary, source, overall_sentiment_score, overall_sentiment_label, ticker_sentiment, topics)
                    VALUES (:requested_entity, :title, :time_published, :url, :summary, :source, :overall_sentiment_score, :overall_sentiment_label, :ticker_sentiment, :topics)
                    ON CONFLICT (url) DO NOTHING
                """)
                conn.execute(stmt, df.to_dict(orient='records'))
            
            print(f"Saved {count} articles ({start_str} -> {end_str})")

        
            if count >= 1000:
                duration = end_time - start_time
                if duration > timedelta(minutes=30): # Avoid infinite loops on extremely high volume
                    mid_point = start_time + (duration / 2)
                    print(f"1000 article API limit reached, halfing window to {duration/2}")
                    
                    time.sleep(15)
                    fetch_news_chunk(ticker, start_time, mid_point)
                    time.sleep(15) # Cooldown
                    fetch_news_chunk(ticker, mid_point, end_time)
                else:
                    print("Window too small to split further. Moving on.")

            return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False



current_pointer = get_current_pointer()
end_goal = datetime.now()

for i in range(25): 
    if current_pointer >= end_goal:
        print(f"Backfill complete!")
        break
    
    # request n days at a time
    n = 8
    window_end = min(current_pointer + timedelta(days=n), end_goal)

    t_from = current_pointer.strftime("%Y%m%dT%H%M")
    t_to = window_end.strftime("%Y%m%dT%H%M")
    
    print(f"Request {i+1}/25 | {CURRENT_TICKER} | {t_from} to {t_to}")
    
    success_bool = fetch_news_chunk(CURRENT_TICKER, current_pointer, window_end)
    
    if success_bool is None:
        break 

    current_pointer = window_end
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(current_pointer.strftime("%Y%m%dT%H%M"))
    
    time.sleep(15) 

print(f"\nSession finished. Current progress for {CURRENT_TICKER}: {current_pointer}")
