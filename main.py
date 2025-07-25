from agents.data_fetcher import get_company_name
from agents.news_fetcher import fetch_finnhub_news
from agents.news_filter import filter_relevant_articles
from agents.news_summarizer import summarize_news_batch
from agents.math_analyzer import analyze_and_visualize
from agents.reporter import (
    generate_summary,
    adjust_price_series_with_llm,
)
from datetime import datetime

def run_stock_analysis(ticker: str, start_date: str, end_date: str, max_articles: int):
    # Get company name using the stock ticker (e.g., AAPL → Apple Inc.)
    company_name = get_company_name(ticker)
    print(f"\nRunning Analysis for {company_name} ({ticker})")

    #Run technical analysis to generate trends and forecasts
    analysis_text, _, projection_df, actual_df = analyze_and_visualize(ticker, start_date, end_date)
    if projection_df is None or projection_df.empty or actual_df is None or actual_df.empty:
        print("Technical analysis failed or missing data.")
        return

    #Fetch recent news articles and summarize the relevant ones
    articles = fetch_finnhub_news(ticker, start_date=start_date, end_date=end_date)
    filtered = filter_relevant_articles(articles, company=company_name, max_articles=max_articles)
    news_summaries = summarize_news_batch(filtered)

    # Print each news summary with index
    print("\nNews Summaries:")
    for idx, summary in enumerate(news_summaries, 1):
        print(f"{idx}. {summary}")

    # Print GPT's summary of the technical indicators
    print("\nMath Model Analysis:")
    print(analysis_text)

    #Generate an investment-style summary from both math + news
    gpt_summary = generate_summary(company_name, ticker, analysis_text, news_summaries)
    print("\nGPT Investment Summary:\n" + gpt_summary)

    #Adjust math-only predictions using LLM to factor in news impact
    adjusted_df = adjust_price_series_with_llm(ticker, projection_df, news_summaries)

    #Join predictions with actual prices to prepare the comparison table
    adjusted_df.index = adjusted_df.index.strftime('%Y-%m-%d')
    adjusted_df.rename(columns={"Price": "Math Prediction"}, inplace=True)
    final_table = adjusted_df

    print("\nFinal 30-Day Prediction Table:")
    print(final_table.to_string())

    #Write the full report to a .txt file
    filename = f"final_30_day_report_{ticker}.txt"
    with open(filename, "w") as f:
        f.write(f"Final Stock Analysis Report for {company_name} ({ticker})\n")
        f.write("=" * 90 + "\n\n")

        f.write("News Summaries:\n")
        for idx, summary in enumerate(news_summaries, 1):
            f.write(f"{idx}. {summary}\n")
        f.write("\n")

        f.write("Technical Analysis Report:\n")
        f.write(analysis_text + "\n\n")

        f.write("GPT Investment Summary:\n")
        f.write(gpt_summary + "\n\n")

        f.write("30-Day Prediction Comparison:\n")
        f.write(final_table.to_string() + "\n")

    print(f"\nFinal report saved to: {filename}")


if __name__ == "__main__":
    print("Welcome to the Stock Report Generator")

    # Collect user input
    ticker = input("Enter the stock ticker (e.g. AAPL): ").upper()
    start_date = None  # use default of 180 days ago
    end_date = datetime.today().strftime('%Y-%m-%d')  # use today
    max_articles = input("Enter number of news articles to use (default 5): ")

    # Fallback to 5 if user enters nothing or non-numeric
    max_articles = int(max_articles) if max_articles.isdigit() else 5

    run_stock_analysis(ticker, start_date, end_date, max_articles)
