import os #To use API Key
import pandas as pd
import logging
from openai import OpenAI
from agents.data_fetcher import fetch_stock_data
from agents.math_solver import calculate_all_indicators

#Importing OpenAI api key
logging.getLogger().setLevel(logging.WARNING)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Setting the default days of System's prediction
PROJECTION_DAYS = 30


def get_stock_with_indicators(ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    df = fetch_stock_data(ticker, start=start_date, end=end_date) #Get stock data within this time range
    if df.empty:
        logging.error("No stock data available.")
        return pd.DataFrame()

    df = calculate_all_indicators(df) #Calculate all the indicators using the math solver agent
    df.dropna(inplace=True)
    return df

def project_fallback_linear(df: pd.DataFrame, days: int) -> pd.DataFrame: #If AI fails, the data is going to be projected as a linear line
    last_close = df['Close'].iloc[-1]
    slope = df['Trend_Slope'].iloc[-1]
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
    projected_prices = [last_close + slope * i for i in range(1, days + 1)]
    logging.warning("Falling back to linear projection due to LLM error or invalid format.")
    return pd.DataFrame({'Price': projected_prices}, index=future_dates)

def project_future_prices_with_llm(ticker: str, df: pd.DataFrame, days: int = PROJECTION_DAYS) -> pd.DataFrame: # This uses ChatGPT to predict the next 30 days of prices.
    latest_data = df.iloc[-1]
    recent_prices = df['Close'].tail(60).tolist() #Takes the most recent 60 days of prices to give context.
    data_summary = f"""
    - Closing Price: {latest_data['Close']:.2f}
    - SMA: {latest_data['SMA_20']:.2f}, EMA: {latest_data['EMA_20']:.2f}
    - Bollinger: {latest_data['Lower_Band']:.2f} to {latest_data['Upper_Band']:.2f}
    - MACD: {latest_data['MACD_Line']:.2f}, Signal: {latest_data['Signal_Line']:.2f}
    - RSI: {latest_data['RSI_14']:.2f}, Trend: {latest_data['Trend_Slope']:.4f}
    """
    #Put all the data into data_summary and feed it in the prompt with words and send to chatgpt
    prompt = f"""
You are a financial modeler. Only base on the technical indicators and recent price movement, predict the next {days} daily closing prices for {ticker}.
Include realistic fluctuations. Avoid linear or flat patterns. Make the mathematical prediction from the data you have.

Technical Summary:
{data_summary}
Recent Prices (last few months): {recent_prices}

Respond with a Python-style list of {days} floats.
"""
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=prompt,
            temperature=0.9
        )
        text = response.output_text.strip() #Remove all the spaces between words
        #Tries to find the list inside the response (between [ and ]).
        start_idx = text.find("[")
        end_idx = text.find("]", start_idx)
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No valid list found in response.")

        #Extracts and evaluates the string into a real Python list of numbers.
        price_list_str = text[start_idx:end_idx + 1]
        predicted_prices = eval(price_list_str)
        if not isinstance(predicted_prices, list) or len(predicted_prices) != days:
            raise ValueError("Invalid format for predicted prices.")
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
        return pd.DataFrame({'Price': predicted_prices}, index=future_dates) #Creates a DataFrame of predicted prices with future dates.
    except Exception as e:
        logging.error(f"LLM price projection failed: {e}")
        return project_fallback_linear(df, days)

def get_llm_analysis(ticker: str, latest_data: pd.Series, projection: pd.DataFrame) -> str: #This asks ChatGPT to summarize the technical indicators + future predictions.
    recent_summary = projection['Price'].describe().to_dict() #Gives GPT a summary of the 30-day prediction (max, min, mean, etc.)
    data_summary = f"""
    - Closing Price: ${latest_data['Close']:.2f}
    - 20-Day SMA: ${latest_data['SMA_20']:.2f}
    - 20-Day EMA: ${latest_data['EMA_20']:.2f}
    - Bollinger Bands: ${latest_data['Lower_Band']:.2f} ~ ${latest_data['Upper_Band']:.2f}
    - MACD: {latest_data['MACD_Line']:.3f}, Signal: {latest_data['Signal_Line']:.3f}
    - RSI (14): {latest_data['RSI_14']:.2f}
    - Trend Slope: {latest_data['Trend_Slope']:.4f}
    """

    prompt = f"""
You are a senior financial analyst preparing a summary for investors.

Based on the following technical indicators and 30-day price projection, write a short outlook for {ticker}.

{data_summary}
Projected Price Stats: {recent_summary}

Include:
1. Market tone (stable, volatile, gaining/losing momentum).
2. Bullish/Bearish/Neutral short-term trend.
3. Justification using at least three indicators.
4. Confidence level (High / Medium / Low).
5. Key movements in projected price.
"""

    try:
        response = client.responses.create(
            model="gpt-4o",
            input=prompt,
            temperature=0.8
        )
        return response.output_text.strip() #Returns GPT’s answer as a summary paragraph.
    except Exception as e:
        logging.error(f"LLM analysis generation failed: {e}")
        return "Error: Could not generate AI analysis."

def analyze_and_visualize(ticker: str, start_date: str = "2024-01-01", end_date: str = None): #This runs the whole math analyzing function
    df = get_stock_with_indicators(ticker, start_date, end_date)
    if df.empty:
        return None, None, None, None

    #Get historical data + indicators.
    projection = project_future_prices_with_llm(ticker, df)
    latest_data = df.iloc[-1]
    analysis_text = get_llm_analysis(ticker, latest_data, projection)

    #Predict prices using GPT, and get a summary.
    future_start = projection.index[0]
    future_end = projection.index[-1]
    actual_future = fetch_stock_data(ticker, start=future_start.strftime("%Y-%m-%d"), end=future_end.strftime("%Y-%m-%d"))

    #Try to get the real future data (to compare later). Rename 'Close' to 'Actual' for easier graphing later.
    if not actual_future.empty and "Close" in actual_future.columns:
        actual_future = actual_future.rename(columns={"Close": "Actual"})

    return analysis_text, None, projection, actual_future #GPT’s summary, None(to add a chart later), GPT’s predicted prices, Actual future prices (if available)

