import os
from dotenv import load_dotenv
from openai import OpenAI
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_article(article: dict) -> str:
    """
    Summarizes a single news article for an investor.
    Requires: 'headline', 'summary', 'source', 'date'
    """
    title = article.get('headline', '')
    summary = article.get('summary', '')
    date = article.get('date', '')
    source = article.get('source', '')

    prompt = f"""
You are a financial assistant. Write a 1-2 sentence investor-focused summary of the following article.

Title: {title}
Summary: {summary}
Source: {source}
Date: {date}
"""
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt,
            temperature=0.9,
        )
        return response.output_text.strip()
    except Exception as e:
        logging.warning(f"Failed to summarize article: {e}")
        return "(Summary unavailable)"

def summarize_news_batch(filtered_articles: list) -> list:
    """
    Summarizes all filtered articles. Returns a list of strings.
    """
    summaries = []
    for article in filtered_articles:
        summary = summarize_article(article)
        summaries.append(summary)
    return summaries


# Example usage
if __name__ == '__main__':
    from news_fetcher import fetch_finnhub_news
    from news_filter import filter_relevant_articles

    symbol = "AAPL"
    company_name = "Apple"
    articles = fetch_finnhub_news(symbol, start_date="2024-07-01", end_date="2024-07-10")
    filtered = filter_relevant_articles(articles, company=company_name, max_articles=5)
    summaries = summarize_news_batch(filtered)

    print("\n--- News Summaries ---")
    for s in summaries:
        print(f"- {s}\n")