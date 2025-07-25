import os
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from agents.data_fetcher import get_company_name
from agents.math_analyzer import analyze_and_visualize
from agents.reporter import adjust_price_series_with_llm
from sklearn.metrics import mean_squared_error


def evaluate_predictions(df: pd.DataFrame) -> dict:
    """
    Compute RMSE and compare prediction trend direction to actual trend.
    """
    if not all(col in df.columns for col in ['News Adjusted Prediction', 'Actual']):
        raise ValueError("Missing required columns in DataFrame.")

    df = df.dropna(subset=['News Adjusted Prediction', 'Actual'])
    y_pred = df['News Adjusted Prediction'].values
    y_true = df['Actual'].values

    # Root Mean Squared Error
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    # Determine trend direction
    pred_trend = y_pred[-1] - y_pred[0]
    actual_trend = y_true[-1] - y_true[0]

    pred_direction = "Bullish" if pred_trend > 0 else "Bearish"
    actual_direction = "Bullish" if actual_trend > 0 else "Bearish"
    direction_match = pred_direction == actual_direction

    return {
        "RMSE": round(rmse, 2),
        "Predicted Trend": pred_direction,
        "Actual Trend": actual_direction,
        "Direction Match": direction_match
    }


def run_prediction(ticker: str, start_date: str, end_date: str, output_dir: str) -> dict:

    #Pull company name and do analysis
    company_name = get_company_name(ticker)
    print(f"\nRunning for {company_name} ({ticker}) from {start_date} to {end_date}")

    # Run the math analyzer to get projections and true future prices
    _, _, projection_df, actual_df = analyze_and_visualize(ticker, start_date, end_date)
    if projection_df is None or projection_df.empty or actual_df is None or actual_df.empty:
        print("Insufficient data.")
        return {"ticker": ticker, "start": start_date, "end": end_date, "error": True}

    # Use dummy news input to generate news-adjusted predictions
    dummy_news = ["(No major news)"]
    adjusted_df = adjust_price_series_with_llm(ticker, projection_df, dummy_news)

    # Format dates and join with actuals
    adjusted_df.index = adjusted_df.index.strftime('%Y-%m-%d')
    actual_df.index = actual_df.index.strftime('%Y-%m-%d')
    final_table = adjusted_df.join(actual_df[["Actual"]], how="inner")

    # Evaluate model performance
    evaluation = evaluate_predictions(final_table)
    evaluation.update({"ticker": ticker, "start": start_date, "end": end_date})

    # Write TXT output
    txt_path = os.path.join(output_dir, f"{ticker}_{start_date}_eval.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Summary for {company_name} ({ticker})\n")
        f.write(f"Date Range: {start_date} to {end_date}\n")
        f.write(f"RMSE: {evaluation['RMSE']}\n")
        f.write(f"Predicted Trend: {evaluation['Predicted Trend']}\n")
        f.write(f"Actual Trend: {evaluation['Actual Trend']}\n")
        f.write(f"Direction Match: {evaluation['Direction Match']}\n\n")
        f.write("Predicted vs Actual Table:\n")
        f.write(final_table.to_string())

    # Plot
    plt.figure(figsize=(10, 6))
    final_table[['News Adjusted Prediction', 'Actual']].plot(title=f"{ticker} - Predicted vs Actual", grid=True)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{ticker}_{start_date}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved evaluation to {txt_path}")
    print(f"Saved plot to {plot_path}")
    return evaluation


if __name__ == "__main__":
    from types import SimpleNamespace

    args = SimpleNamespace(
        tickers=["NVDA"],
        start=input("Enter start date (YYYY-MM-DD): "),
        interval_days=int(input("Enter interval between runs (days): ")),
        repeat=int(input("How many repeats: ")),
        outdir="evaluations_txt"
    )

    os.makedirs(args.outdir, exist_ok=True)
    results = []

    # Loop over time periods for backtesting
    for ticker in args.tickers:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")

        for i in range(args.repeat):
            s = start_date + timedelta(days=i * args.interval_days)
            e = s + timedelta(days=90)
            result = run_prediction(
                ticker=ticker,
                start_date=s.strftime("%Y-%m-%d"),
                end_date=e.strftime("%Y-%m-%d"),
                output_dir=args.outdir
            )
            results.append(result)

    # Summarize all batch results into a single file
    summary_path = os.path.join(args.outdir, "summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for result in results:
            if result.get("error"):
                continue
            f.write(f"{result['ticker']} ({result['start']} to {result['end']})\n")
            f.write(f"  RMSE: {result['RMSE']}\n")
            f.write(f"  Predicted: {result['Predicted Trend']}\n")
            f.write(f"  Actual: {result['Actual Trend']}\n")
            f.write(f"  Match: {result['Direction Match']}\n\n")

    print(f"\nSummary saved to {summary_path}")
