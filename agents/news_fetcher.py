import requests #makes web requests
import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv #loads secret keys

load_dotenv()
logging.basicConfig(level=logging.INFO)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1/company-news"


def fetch_finnhub_news(symbol: str, start_date: str = None, end_date: str = None, days_back: int = 7):
    """
    Fetch recent company news using Finnhub API.
    Provide either a date range (start_date, end_date) or days_back.
    By default,: Use today as the end date, Count 7 days back for the start date.
    Returns a list of cleaned article dictionaries.
    """
    if not start_date or not end_date:
        end_date_obj = datetime.today()
        start_date_obj = end_date_obj - timedelta(days=days_back)
    else:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    url = f"{BASE_URL}?symbol={symbol.upper()}&from={start_date_obj.date()}&to={end_date_obj.date()}&token={FINNHUB_API_KEY}" #builds the request URL

    try:
        response = requests.get(url) #Sends the request.
        response.raise_for_status() #Raises an error if the request fails
        articles = response.json() #Converts the response from JSON to a Python list of dictionaries.
        logging.info(f"Fetched {len(articles)} articles for {symbol}.")
        #returns a list of only useful articles
        return [
            {
                "headline": a.get("headline", ""),
                "summary": a.get("summary", ""),
                "datetime": datetime.fromtimestamp(a["datetime"]).strftime("%Y-%m-%d") if "datetime" in a else None,
                "source": a.get("source", ""),
                "url": a.get("url", "")
            }
            for a in articles
            if a.get("headline") and a.get("summary")
        ]

    #it logs an error and returns an empty list if it failed
    except requests.RequestException as e:
        logging.error(f"Error fetching news from Finnhub: {e}")
        return []


# Example
if __name__ == '__main__':
    symbol = "AAPL"
    start = "2025-01-01"
    end = "2025-07-10"
    news = fetch_finnhub_news(symbol, start_date=start, end_date=end)
    for item in news:
        print(f"{item['datetime']} - {item['headline']}\n{item['summary']}\n")
