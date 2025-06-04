import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yfinance as yf


print("Available Matplotlib styles:", plt.style.available)
plt.style.use('ggplot')  


DATA_PATH = "../data/yfinance_data/"
STOCK_TICKERS = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
OUTPUT_DIR = "notebooks"
Path(OUTPUT_DIR).mkdir(exist_ok=True)  
USE_YFINANCE = False  
START_DATE = "2024-12-01" 
END_DATE = "2024-12-31"   


def load_stock_data(ticker, use_yfinance=USE_YFINANCE):
    """
    Load stock price data either from CSV file or yfinance for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        use_yfinance (bool): If True, fetch from yfinance; if False, load from CSV
    
    Returns:
        pd.DataFrame: Stock price data with Date as datetime
    """
    if use_yfinance:
        try:
            stock = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if stock.empty:
                print(f"Error: No data fetched for {ticker} from yfinance")
                return None
            stock.reset_index(inplace=True)
            stock['Date'] = pd.to_datetime(stock['Date'])
            stock.set_index('Date', inplace=True)
            return stock
        except Exception as e:
            print(f"Error fetching {ticker} from yfinance: {e}")
            return None
    else:
        file_path = f"{DATA_PATH}{ticker}_historical_data.csv"
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        except FileNotFoundError:
            print(f"Error: File for {ticker} not found at {file_path}")
            return None

def calculate_technical_indicators(df):
    """
    Calculate technical indicators (SMA, RSI, MACD) for stock data.
    
    Args:
        df (pd.DataFrame): Stock price data with Close column
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicator columns
    """
    df = df.copy()  
    
    
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    

    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    

    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    return df


def visualize_stock_data(df, ticker):
    """
    Create visualizations for stock price and technical indicators.
    
    Args:
        df (pd.DataFrame): Stock data with technical indicators
        ticker (str): Stock ticker symbol
    """

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.plot(df.index, df['SMA_20'], label='20-Day SMA', color='orange')
    plt.title(f'{ticker} Stock Price with 20-Day SMA')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/quantitative_analysis_output/{ticker}_stock_price_sma.png')
    plt.close()
    

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['RSI_14'], label='RSI (14)', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f'{ticker} Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/quantitative_analysis_output/{ticker}_rsi_plot.png')
    plt.close()
    

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['MACD'], label='MACD', color='blue')
    plt.plot(df.index, df['MACD_Signal'], label='Signal Line', color='orange')
    plt.bar(df.index, df['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.5)
    plt.title(f'{ticker} MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/quantitative_analysis_output/{ticker}_macd_plot.png')
    plt.close()


def process_all_stocks(tickers, use_yfinance=USE_YFINANCE):
    """
    Load, analyze, and visualize data for all stock tickers.
    
    Args:
        tickers (list): List of stock ticker symbols
        use_yfinance (bool): If True, fetch from yfinance; if False, load from CSV
    """
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        stock_data = load_stock_data(ticker, use_yfinance)
        if stock_data is None:
            continue
        
        print(f"{ticker} Raw Data Preview:")
        print(stock_data.head())
        
        stock_data = calculate_technical_indicators(stock_data)
        
        stock_data.dropna(inplace=True)
        
        visualize_stock_data(stock_data, ticker)
        
        output_file = f"{OUTPUT_DIR}/processed_stock_data/{ticker}_processed_stock_data.csv"
        stock_data.to_csv(output_file)
        print(f"Processed data for {ticker} saved to {output_file}")

if __name__ == "__main__":
    process_all_stocks(STOCK_TICKERS, USE_YFINANCE)