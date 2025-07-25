import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_summary(company: str, ticker: str, analysis_text: str, news_summaries: list) -> str: #Generate a written summary by GPT
    news_text = "\n\n".join(news_summaries) if news_summaries else "(No major news found.)"

    prompt = f"""
You are a financial assistant. Write a short investor-focused summary combining technical and news insights.

Company: {company} ({ticker})

--- Technical Analysis ---
{analysis_text}

--- News Summary ---
{news_text}

Write 1 concise paragraph (4–6 sentences) summarizing the stock's short-term outlook.
Mention if the trend is bullish or bearish, what indicators support this, and how news might impact it.
Avoid listing prices. Avoid headings. Avoid bullet points.
"""

    try:
        response = client.responses.create(
            model="gpt-4o",
            input=prompt,
            temperature=0.9
        )
        return response.output_text.strip()
    except Exception as e:
        logging.error(f"generate_summary() failed: {e}")
        return "Could not generate report."


def adjust_price_series_with_llm(ticker: str, projection_df: 'pd.DataFrame', news_summaries: list) -> 'pd.DataFrame':
    """
    Use GPT to generate news-adjusted prices based on original math predictions and news summaries.
    Returns a DataFrame with both 'Math Prediction' and 'News Adjusted Prediction' columns.
    """

    if projection_df is None or projection_df.empty or "Price" not in projection_df.columns:
        raise ValueError("Projection DataFrame is invalid.")

    math_prices = projection_df["Price"].tolist()
    news_text = "\n\n".join(news_summaries) if news_summaries else "(No significant news found.)"

    prompt = f"""
You are a financial analyst. Based on the following 30-day price predictions and recent company news, adjust the predictions to reflect investor sentiment and likely market impact.
Include realistic fluctuations. Avoid linear or flat patterns. Make the mathematical prediction from the data you have. Prefer a significant difference than the math-based prediction.
--- Math-Based Price Predictions (in USD) ---
{math_prices}

--- News Summary ---
{news_text}

Please return only the adjusted 30 prices as a Python-style list of floats. Do not include explanation or formatting.
"""

    try:
        response = client.responses.create(
            model="gpt-4o",
            input=prompt,
            temperature=1.2
        )
        raw = response.output_text.strip()

        start = raw.find("[")
        end = raw.find("]", start)
        if start == -1 or end == -1:
            raise ValueError("No valid list found in response.")

        price_list_str = raw[start:end + 1]
        adjusted_prices = eval(price_list_str)

        if not isinstance(adjusted_prices, list) or len(adjusted_prices) != 30:
            raise ValueError("Adjusted price list must have 30 values.")

        # Return new dataframe
        df = projection_df.copy()
        df = df.rename(columns={"Price": "Math Prediction"})
        df["News Adjusted Prediction"] = adjusted_prices
        return df

    except Exception as e:
        logging.error(f"adjust_price_series_with_llm() failed: {e}")
        # Fallback to no adjustment
        df = projection_df.copy()
        df = df.rename(columns={"Price": "Math Prediction"})
        df["News Adjusted Prediction"] = df["Math Prediction"]
        return df

