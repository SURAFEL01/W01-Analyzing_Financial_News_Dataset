# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

# Set plotting style
plt.style.use('ggplot')

# Step 2: Define file paths and stock tickers
NEWS_DATA_PATH = "../data/raw_analyst_ratings.csv"
STOCK_DATA_PATH = "../notebooks/processed_stock_data/"
STOCK_TICKERS = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
OUTPUT_DIR = "notebooks/correlation_analysis_output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Step 3: Load datasets
def load_news_data():
    """
    Load FNSPID news dataset.
    
    Returns:
        pd.DataFrame: News data with normalized dates
    """
    try:
        df = pd.read_csv(NEWS_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date'] = df['date'].dt.tz_convert('America/New_York')
        df['date_only'] = df['date'].dt.date
        return df
    except FileNotFoundError:
        print(f"Error: News data not found at {NEWS_DATA_PATH}")
        return None

def load_stock_data(ticker):
    """
    Load stock price data for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        pd.DataFrame: Stock price data with Date as index
    """
    file_path = f"{STOCK_DATA_PATH}{ticker}_processed_stock_data.csv"
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: File for {ticker} not found at {file_path}")
        return None

# Step 4: Perform sentiment analysis
def calculate_sentiment(df):
    """
    Calculate sentiment scores for news headlines using VADER.
    
    Args:
        df (pd.DataFrame): News data with 'headline' column
    
    Returns:
        pd.DataFrame: Data with sentiment scores
    """
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['headline'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral'
    )
    return df

# Step 5: Calculate daily stock returns
def calculate_daily_returns(df):
    """
    Calculate daily percentage returns for stock data.
    
    Args:
        df (pd.DataFrame): Stock data with 'Close' column
    
    Returns:
        pd.DataFrame: Data with daily returns
    """
    df = df.copy()
    df['daily_return'] = df['Close'].pct_change() * 100
    return df

# Step 6: Align and aggregate data
def align_and_aggregate(news_df, stock_df, ticker):
    """
    Align news and stock data by date and aggregate sentiment.
    
    Args:
        news_df (pd.DataFrame): News data with sentiment
        stock_df (pd.DataFrame): Stock data with returns
        ticker (str): Stock ticker symbol
    
    Returns:
        pd.DataFrame: Aligned data with average sentiment and returns
    """
    # Filter news for the specific ticker
    news_ticker = news_df[news_df['stock'] == ticker].copy()
    if news_ticker.empty:
        print(f"No news data for {ticker}")
        return None
    
    # Aggregate sentiment by date
    sentiment_agg = news_ticker.groupby('date_only')['sentiment_score'].mean().reset_index()
    sentiment_agg['date_only'] = pd.to_datetime(sentiment_agg['date_only'])
    
    # Prepare stock data
    stock_df = stock_df.reset_index()
    stock_df['date_only'] = stock_df['Date'].dt.date
    stock_df['date_only'] = pd.to_datetime(stock_df['date_only'])
    
    # Merge on date
    merged_df = pd.merge(sentiment_agg, stock_df[['date_only', 'daily_return']], on='date_only', how='inner')
    
    return merged_df

# Step 7: Calculate correlation
def calculate_correlation(merged_df, ticker):
    """
    Calculate Pearson correlation between sentiment and stock returns.
    
    Args:
        merged_df (pd.DataFrame): Data with sentiment and returns
        ticker (str): Stock ticker symbol
    
    Returns:
        tuple: Correlation coefficient and p-value
    """
    if merged_df.empty or len(merged_df) < 2:
        print(f"Insufficient data for correlation analysis for {ticker}")
        return None, None
    
    corr, p_value = pearsonr(merged_df['sentiment_score'], merged_df['daily_return'])
    return corr, p_value

# Step 8: Visualize correlation
def visualize_correlation(merged_df, ticker):
    """
    Create scatter plot of sentiment vs. stock returns.
    
    Args:
        merged_df (pd.DataFrame): Data with sentiment and returns
        ticker (str): Stock ticker symbol
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sentiment_score', y='daily_return', data=merged_df)
    plt.title(f'{ticker} Sentiment vs. Daily Stock Return')
    plt.xlabel('Average Daily Sentiment Score')
    plt.ylabel('Daily Return (%)')
    plt.savefig(f'{OUTPUT_DIR}/{ticker}_sentiment_vs_return.png')
    plt.close()

# Step 9: Main function to process all stocks
def process_correlation_analysis(tickers):
    """
    Perform correlation analysis for all stocks.
    
    Args:
        tickers (list): List of stock ticker symbols
    """
    news_df = load_news_data()
    if news_df is None:
        return
    
    # Initialize results
    correlation_results = []
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # Load stock data
        stock_df = load_stock_data(ticker)
        if stock_df is None:
            continue
        
        # Calculate daily returns
        stock_df = calculate_daily_returns(stock_df)
        
        # Calculate sentiment for news
        news_df = calculate_sentiment(news_df)
        
        # Align and aggregate
        merged_df = align_and_aggregate(news_df, stock_df, ticker)
        if merged_df is None:
            continue
        
        # Calculate correlation
        corr, p_value = calculate_correlation(merged_df, ticker)
        if corr is not None:
            correlation_results.append({
                'ticker': ticker,
                'correlation': corr,
                'p_value': p_value,
                'data_points': len(merged_df)
            })
        
        # Visualize
        visualize_correlation(merged_df, ticker)
        
        # Save merged data
        output_file = f"{OUTPUT_DIR}/{ticker}_sentiment_returns.csv"
        merged_df.to_csv(output_file)
        print(f"Sentiment and returns data for {ticker} saved to {output_file}")
    
    # Save correlation results
    results_df = pd.DataFrame(correlation_results)
    results_df.to_csv(f"{OUTPUT_DIR}/correlation_results.csv", index=False)
    print(f"Correlation results saved to {OUTPUT_DIR}/correlation_results.csv")

# Step 10: Execute the analysis
if __name__ == "__main__":
    process_correlation_analysis(STOCK_TICKERS)