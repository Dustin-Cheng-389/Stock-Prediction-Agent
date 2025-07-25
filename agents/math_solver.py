import pandas as pd
import numpy as np #Help with numbers and arrays
import logging
from sklearn.linear_model import LinearRegression #Help to find a line that best fit


logging.basicConfig(level=logging.INFO)

# Setting how big each window is when calculating indicators:
DEFAULT_SMA_WINDOW = 20       # how many days to average for SMA
DEFAULT_EMA_SPAN = 20         # how many days for EMA
DEFAULT_BOLL_WINDOW = 20      # used for Bollinger Bands
DEFAULT_MACD_SHORT = 12       # fast EMA for MACD
DEFAULT_MACD_LONG = 26        # slow EMA for MACD
DEFAULT_MACD_SIGNAL = 9       # signal line for MACD
DEFAULT_RSI_PERIOD = 14       # how many days for RSI
DEFAULT_TREND_WINDOW = 30     # how many days to check trend

#Since we are not PROS at stocks, these few indicating methods are suggested by GPT

def calculate_moving_average(df: pd.DataFrame, window: int = DEFAULT_SMA_WINDOW) -> pd.Series:
    """Simple Moving Average over the past 20 days. To get general moving direction"""
    return df['Close'].rolling(window=window).mean() # Returns a new column of average prices.


def calculate_exponential_moving_average(df: pd.DataFrame, span: int = DEFAULT_EMA_SPAN) -> pd.Series:
    """Similar to SMA, but gives more weight to recent prices. Reacts faster to changes than SMA."""
    return df['Close'].ewm(span=span, adjust=False).mean()


def calculate_bollinger_bands(df: pd.DataFrame, window: int = DEFAULT_BOLL_WINDOW):
    """These are two lines above and below the moving average. They show when a stock might be too high or too low compared to its average."""
    sma = calculate_moving_average(df, window)
    std = df['Close'].rolling(window=window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return upper_band, lower_band


def calculate_macd(
    df: pd.DataFrame,
    span_short: int = DEFAULT_MACD_SHORT,
    span_long: int = DEFAULT_MACD_LONG,
    signal: int = DEFAULT_MACD_SIGNAL
):
    """Calculate MACD, Compares two EMAs (short and long). Shows trend direction and strength."""
    ema_short = calculate_exponential_moving_average(df, span_short)
    ema_long = calculate_exponential_moving_average(df, span_long)
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def linear_regression_trend(df: pd.DataFrame, window: int = 30):
    """
    Uses linear regression (a straight line) to find the trend of the stock over time. A positive slope means the stock is trending up.
    A negative slope means it’s trending down.
    """
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(df) + 1):
        y = df['Close'].iloc[i - window:i].values.reshape(-1, 1)
        x = np.arange(window).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        slopes.append(model.coef_[0][0])

    # Ensure slopes length matches df length
    slopes = slopes[:len(df)]  # trim if too long
    while len(slopes) < len(df):  # pad if too short
        slopes.append(np.nan)

    return pd.Series(slopes, index=df.index)



def calculate_rsi(df: pd.DataFrame, period: int = DEFAULT_RSI_PERIOD) -> pd.Series:
    """
    Tells if a stock is overbought or oversold.
    RSI above 70 → might be too high (overbought).
    RSI below 30 → might be too low (oversold).
    """
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_all_indicators(
    df: pd.DataFrame,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Adds: SMA EMA Bollinger Bands MACD RSI Trend slope Removes empty values if needed.
    """
    if 'Close' not in df.columns:
        logging.error("Missing 'Close' column in DataFrame.") #Debug
        return df

    logging.info("Calculating all technical indicators...")

    #All indicators are calculated here
    df['SMA_20'] = calculate_moving_average(df)
    df['EMA_20'] = calculate_exponential_moving_average(df)
    df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df)
    df['MACD_Line'], df['Signal_Line'], df['MACD_Histogram'] = calculate_macd(df)
    df['RSI_14'] = calculate_rsi(df)
    df['Trend_Slope'] = linear_regression_trend(df)

    if dropna:
        df.dropna(inplace=True)

    logging.info("Indicator calculation complete.")
    return df #return all the data with added columns
