import os
from dotenv import load_dotenv
from openai import OpenAI
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

def filter_relevant_articles(articles, company: str, max_articles: int = 5): #We only need by default 5 useful articles by default
    """
    Uses GPT to filter and prioritize news articles that are relevant to investors.
    Returns up to `max_articles` results with explanations.
    """
    filtered = []
    idx = 0

    while len(filtered) < max_articles and idx < len(articles):
        article = articles[idx]
        idx += 1

        title = article.get("headline", "")
        summary = article.get("summary", "")
        source = article.get("source", "")
        date = article.get("datetime", "")
        url = article.get("url", "")

        if not title or not summary:
            continue

        temp_prompt = f"""
You are a financial assistant. Your task is to determine if a news article is relevant to an investor researching {company}.

If the article is NOT relevant, reply exactly: Not useful
If the article IS relevant, reply with one short sentence explaining why it matters.

Title: {title}
Summary: {summary}
Source: {source}
Date: {date}
"""

        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=temp_prompt,
                temperature=0.4,
            )
            decision = response.output_text.strip()

            if decision != "Not useful":
                filtered.append({
                    "headline": title,
                    "summary": summary,
                    "reason": decision,
                    "url": url,
                    "source": source,
                    "date": date
                })

        except Exception as e:
            logging.warning(f"OpenAI filter failed for article: {e}")
            continue

    logging.info(f"Selected {len(filtered)} relevant articles for {company}.")
    return filtered


# Example
if __name__ == '__main__':
    from news_fetcher import fetch_finnhub_news

    symbol = "AAPL"
    company_name = "Apple"
    articles = fetch_finnhub_news(symbol, start_date="2024-07-01", end_date="2024-07-10")
    filtered_articles = filter_relevant_articles(articles, company=company_name, max_articles=5)

    for item in filtered_articles:
        print(f"- {item['date']} - {item['headline']}\n  Reason: {item['reason']}\n")
