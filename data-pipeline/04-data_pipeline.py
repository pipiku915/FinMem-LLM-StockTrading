# !pip install yfinance
# !pip install pyarrow
# !pip install fastparquet
# You may need a VPN to connect to Yahoo Finance

import glob
import pickle
import datetime
import pandas as pd
import yfinance as yf
from typing import List

def download_data(path: str, start_day: str, end_day: str, tickers: List[str]) -> List[pd.DataFrame]:
    """Downloads adjusted close price data for given tickers within a date range."""
    df_list = []
    for ticker in tickers:
        print(f'Downloading data for {ticker}')
        data = yf.download(ticker, start=start_day, end=end_day)
        data = data.reset_index()
        data['Date'] = data['Date'].dt.date
        data = data[['Date', 'Adj Close']]
        data = data.rename(columns={'Date': 'date', 'Adj Close': ticker})
        df_list.append(data)
    return df_list

def combine_dataframes(df_list: List[pd.DataFrame], path: str, tickers: List[str]) -> dict:
    """Combines dataframes of different tickers into a single dictionary and saves it as a pickle file."""
    df_dicts = [dict(zip(df['date'], df[ticker])) for df, ticker in zip(df_list, tickers)]
    combined_dict = {date: {'price': {}} for df_dict in df_dicts for date in df_dict}
    for i, df_dict in enumerate(df_dicts):
        for date, price in df_dict.items():
            combined_dict[date]['price'][tickers[i]] = price
    pkl_filename = path + 'price.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(combined_dict, file)
    print(f'Price data saved to: {pkl_filename}')
    return combined_dict

def create_news_dict(news_path: str, csv_pattern: str, col_name: str) -> dict:
    """Creates a dictionary from news CSV files and saves it as a pickle file."""
    full_csv_pattern = news_path + csv_pattern
    csv_files = glob.glob(full_csv_pattern)
    combined_dict = {}
    for file in csv_files:
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date']).dt.date
        symbol = df.at[0, 'symbols']
        for date, summary in df.groupby('date')[col_name].apply(list).items():
            if date not in combined_dict:
                combined_dict[date] = {'news': {}}
            combined_dict[date]['news'][symbol] = summary
    
    # Make sure the directory exists
    if not os.path.exists(news_path):
        os.makedirs(news_path)
    
    pkl_filename = news_path + 'news.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(combined_dict, file)
    print(f'News data saved to: {pkl_filename}')
    return combined_dict

def process_filing_data(start_day: str, end_day: str, kq_path: str, filing_data: str, tickers: List[str]) -> (dict, dict):
    """
    Process filing data to filter, sort, and save in nested dictionary format.

    Parameters:
    start_day (str): The start date in 'YYYY-MM-DD' format.
    end_day (str): The end date in 'YYYY-MM-DD' format.
    kq_path (str): The path to save the output pickle files.
    filing_data (str): The path to the input .parquet file.
    tickers (List[str]): List of ticker symbols to filter by.
    """
    start = pd.to_datetime(start_day)
    end = pd.to_datetime(end_day)
    
    filingkq = pd.read_parquet(filing_data)
    filingkq = filingkq.drop(columns=['document_url', 'cik', 'utc_timestamp'])
    filingkq = filingkq.rename(columns={'est_timestamp': 'date'})
    filingkq = filingkq[['date', 'ticker', 'content', 'type']]
    filingkq['date'] = filingkq['date'].dt.date
    filingkq['date'] = pd.to_datetime(filingkq['date'])
    df_filteredkq = filingkq[filingkq['ticker'].isin(tickers)]
    df_10k = df_filteredkq[df_filteredkq['type'] == '10-K']
    df_10q = df_filteredkq[df_filteredkq['type'] == '10-Q']
    
    df_10k = df_10k.drop(columns=['type']).sort_values(by='date')
    df_10q = df_10q.drop(columns=['type']).sort_values(by='date')
    d = df_10k['date'].tolist()

    df_10k = df_10k[(df_10k['date'] >= start) & (df_10k['date'] <= end)]
    df_10q = df_10q[(df_10q['date'] >= start) & (df_10q['date'] <= end)]
    df_10k['date'] = df_10k['date'].dt.date
    df_10q['date'] = df_10q['date'].dt.date
    
    nested_10k = {date: {'filing_k': df_group.set_index('ticker')['content'].to_dict()} 
                  for date, df_group in df_10k.groupby('date')}
    nested_10q = {date: {'filing_q': df_group.set_index('ticker')['content'].to_dict()} 
                  for date, df_group in df_10q.groupby('date')}
    
    q_filename = kq_path + 'filing_q.pkl'
    k_filename = kq_path + 'filing_k.pkl'
    
    with open(q_filename, 'wb') as file:
        pickle.dump(nested_10q, file)
    with open(k_filename, 'wb') as file:
        pickle.dump(nested_10k, file)
    print(f'10-K data saved to: {k_filename}')
    print(f'10-Q data saved to: {q_filename}')
    return nested_10q,nested_10k
    
if __name__ == '__main__':
    base_path = '/home/yyu/YJ'
    price_path = '/home/yyu/YJ/price/'
    news_path = '/home/yyu/YJ/add_summary_data/'
    kq_path = '/home/yyu/YJ/10k10q/'
    filing_data = '/home/yyu/YJ/filing_data.parquet'
    start_day = '2021-08-01'
    end_day = '2023-06-01'
    tickers = ['BAC', 'DIS', 'GM', 'MRNA', 'NVDA', 'PFE']
    csv_files_pattern = '*.csv'
    col_name = 'summary'

    df_list = download_data(price_path, start_day, end_day, tickers)
    price = combine_dataframes(df_list, price_path, tickers)
    news = create_news_dict(news_path, csv_files_pattern, col_name)
    q, k = process_filing_data(start_day, end_day, kq_path, filing_data, tickers)


    # Update dictionaries
    for date in price.keys():
        q.setdefault(date, {'filing_q': {}})
        k.setdefault(date, {'filing_k': {}})
        news.setdefault(date, {'news': {ticker: [] for ticker in tickers}})


    # Update news dictionary
    for date, data in news.items():
        if 'news' in data:
            missing_tickers = [ticker for ticker in tickers if ticker not in data['news']]
            for ticker in missing_tickers:
                news[date]['news'][ticker] = []
        if len(list(news[date]['news'].keys())) != len(tickers):
            print('ERROR on:',date)

    # Sorting dictionaries
    filled_q = dict(sorted(q.items()))
    filled_k = dict(sorted(k.items()))

    # Combining data
    env_data = {key: (price[key], news[key], filled_q[key], filled_k[key]) for key in price.keys()}

    # Save the combined data
    output_path = f'{base_path}/env_data.pkl'
    with open(output_path, 'wb') as file:
        pickle.dump(env_data, file)

    print(f'Environment data saved to: {output_path}')


