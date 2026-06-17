import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
from transform_alpha_vantage import process_alpha_vantage_data

load_dotenv()

POSTGRES_PASS = os.getenv("POSTGRES_PASS")

# Connect to pgAdmin4

try:
    conn = psycopg2.connect(
        dbname="stock_market",
        user="postgres",
        password=POSTGRES_PASS,
        host="localhost",
        port="5432"
    )
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()


query = "SELECT * FROM news_sentiment;"
raw_df = pd.read_sql_query(query, conn)
conn.close()
print("data loaded into pandas")


main_article_df, ticker_df, topic_df = process_alpha_vantage_data(raw_df)

def drop_error_col_if_clean(df, col):
    if col in df.columns:
        if not df[col].fillna(False).any():
            return df.drop(columns=[col])
    return df

ticker_df = drop_error_col_if_clean(ticker_df, "ticker_parse_error")
topic_df  = drop_error_col_if_clean(topic_df, "topic_parse_error")

main_article_df["month"] = main_article_df["time_published"].dt.strftime("%Y-%m")

ticker_df = ticker_df.merge(
    main_article_df[["article_id", "time_published"]],
    on="article_id",
    how="left"
)

topic_df = topic_df.merge(
    main_article_df[["article_id", "time_published"]],
    on="article_id",
    how="left"
)

os.makedirs("hf_dataset/articles", exist_ok=True)
os.makedirs("hf_dataset/tickers", exist_ok=True)
os.makedirs("hf_dataset/topics", exist_ok=True)

months = sorted(main_article_df["month"].unique())

for m in months:
    # filter each table
    art_m = main_article_df[main_article_df["month"] == m]
    tic_m = ticker_df[ticker_df["article_id"].isin(art_m["article_id"])]
    top_m = topic_df[topic_df["article_id"].isin(art_m["article_id"])]

    # save
    art_m.to_parquet(f"hf_dataset/articles/{m}.parquet", index=False)
    tic_m.to_parquet(f"hf_dataset/tickers/{m}.parquet", index=False)
    top_m.to_parquet(f"hf_dataset/topics/{m}.parquet", index=False)
