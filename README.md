# 📈 Multi-Agent Stock Analysis System

This project is an AI-powered multi-agent system that performs stock analysis using technical indicators, news summaries, and GPT-based forecasting. The system fetches, filters, and summarizes financial data to produce an investor-focused report.

## 💡 What It Does

- **Fetches** stock price history from Yahoo Finance.
- **Analyzes** the data using technical indicators (SMA, EMA, MACD, RSI, etc.).
- **Predicts** the next 30 days of prices using GPT-4.
- **Fetches and filters** recent news articles using the Finnhub API.
- **Summarizes** news relevant to investors using GPT-4.
- **Adjusts** technical predictions based on news sentiment.
- **Generates** a final summary and comparison table.
- **Evaluates** prediction performance against actual data.

---

## 🧠 Agents Overview

| Agent           | Responsibility                                                                 |
|----------------|----------------------------------------------------------------------------------|
| `data_fetcher`  | Pulls historical stock data and metadata from Yahoo Finance.                   |
| `news_fetcher`  | Uses the Finnhub API to pull company news.                                     |
| `news_filter`   | Filters and ranks news using GPT-4 for investor relevance.                      |
| `news_summarizer` | Summarizes relevant news articles into concise investor insights.             |
| `math_solver`   | Computes technical indicators like SMA, EMA, MACD, RSI, trend slope.            |
| `math_analyzer` | Projects future prices using GPT-4 and analyzes technical sentiment.            |
| `reporter`      | Combines news and math insights into a summary and adjusts predictions.         |
| `evaluate_prediction` | Compares predicted vs actual prices and visualizes the results.            |

---

## 🛠️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/Stock-Prediction-Agent.git
cd Stock-Prediction-Agent
