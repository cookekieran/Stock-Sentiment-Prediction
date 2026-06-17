import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from config import POSTGRES_PASS, ALPHA_VANTAGE_API_KEY

DB_URL = f'postgresql://postgres:{POSTGRES_PASS}@localhost:5432/stock_market'
engine = create_engine(DB_URL)

def repair_ticker(ticker, start_date_str, end_date_str, days_per_step=2):
   
    current_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    final_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    print(f"--- REPAIR START: {ticker} from {start_date_str} to {end_date_str} ---")
    
    while current_date < final_date:
        next_date = current_date + timedelta(days=days_per_step)

        if next_date > final_date:
            next_date = final_date
        
        t_from = current_date.strftime("%Y%m%dT%H%M")
        t_to = next_date.strftime("%Y%m%dT%H%M")
        
        print(f"Fetching: {t_from} to {t_to}...")

        url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
               f"&tickers={ticker}&time_from={t_from}&time_to={t_to}"
               f"&sort=EARLIEST&limit=1000&apikey={ALPHA_VANTAGE_API_KEY}")
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if "Note" in data or "Information" in data:
                print("API Limit reached.")
                break

            if "feed" not in data:
                print(f"No news found for this slice. Moving on.")
                current_date = next_date
                continue

            processed_news = []
            for item in data["feed"]:
                processed_news.append({
                    "requested_entity": ticker,
                    "title": item.get("title"),
                    "time_published": datetime.strptime(item.get("time_published"), "%Y%m%dT%H%M%S"),
                    "url": item.get("url"),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "overall_sentiment_score": float(item.get("overall_sentiment_score", 0)),
                    "overall_sentiment_label": item.get("overall_sentiment_label"),
                    "ticker_sentiment": str(item.get("ticker_sentiment", [])),
                    "topics": str(item.get("topics", [])) 
                })
            
            # Save to the SQL database
            if processed_news:
                df = pd.DataFrame(processed_news)
                with engine.begin() as conn:
                    stmt = text("""
                        INSERT INTO news_sentiment 
                        (requested_entity, title, time_published, url, summary, source, overall_sentiment_score, overall_sentiment_label, ticker_sentiment, topics)
                        VALUES (:requested_entity, :title, :time_published, :url, :summary, :source, :overall_sentiment_score, :overall_sentiment_label, :ticker_sentiment, :topics)
                        ON CONFLICT (url) DO NOTHING
                    """)
                    conn.execute(stmt, df.to_dict(orient='records'))
                
                status = "CAPPED AT 1000" if len(df) == 1000 else f"Saved {len(df)}"
                print(f"-> {status} articles.")

            
            current_date = next_date
            time.sleep(15) 

        except Exception as e:
            print(f"Error during repair: {e}")
            break

    print(f"--- REPAIR SESSION FINISHED ---")
