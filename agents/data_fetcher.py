import yfinance as yf #Where we get our news
import pandas as pd #For tables of data
import logging #help track our system in case of error
from datetime import datetime, timedelta #for date and time import

logging.basicConfig(level=logging.INFO) #Keep track of our program for debugging process

# Defaults days if user didn't input properly
DEFAULT_INTERVAL = "1d"
DEFAULT_DAYS_BACK = 180


def fetch_stock_data(
    ticker: str,
    start: str = None,
    end: str = None,
    interval: str = DEFAULT_INTERVAL # How often we need to get the data
) -> pd.DataFrame:
    if not end:
        end = datetime.today().strftime('%Y-%m-%d') #If no input, it is today
    if not start:
        start = (datetime.today() - timedelta(days=DEFAULT_DAYS_BACK)).strftime('%Y-%m-%d') #Input for starting date, default is -180 days

    logging.info(f"Fetching {ticker} from {start} to {end} ({interval})")
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, interval=interval) #Fetching data by using yahoo finance
        if df.empty:
            logging.warning(f"No data returned for {ticker}. Check date range or symbol.") # For debugging
        else:
            logging.info(f"Fetched {len(df)} rows.") # If there is data, return how many rows it got
        return df #saved in dataframe table called df
    except Exception as e:
        logging.error(f"Error fetching {ticker}: {e}")# Return error if ticker name is wrong or something else
        return pd.DataFrame()


def get_company_name(ticker: str) -> str: #Getting the correct ticker name, for example: Apple Inc. for AAPL
    try:
        return yf.Ticker(ticker).info.get("shortName", "Unknown Company")
    except Exception as e:
        logging.error(f"Error getting company name for {ticker}: {e}")
        return "Unknown Company"
