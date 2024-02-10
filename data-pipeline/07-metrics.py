import numpy as np
import pandas as pd
import yfinance as yf

# Get daily stock price
def get_price(Start, End, Ticker):
    """
    Fetch daily adjusted closing prices of a stock from Yahoo Finance.

    Parameters:
    Start (str): Start date for the price data.
    End (str): End date for the price data.
    Ticker (str): Ticker symbol of the stock.

    Returns:
    list: List of daily adjusted closing prices.
    """
    df = yf.download(ticker, start=Start, end=End)
    return df['Adj Close'].tolist()

# Get actions for different models
def get_action(start, end, ticker, file_path, col):
    """
    Fetch action data from a specified CSV file.

    Parameters:
    start (str): Start date for the action data.
    end (str): End date for the action data.
    ticker (str): Ticker symbol of the stock (not used in the function).
    file_path (str): Path to the CSV file containing the action data.
    col (list): Column names to be used for date and action.

    Returns:
    list: List of actions corresponding to the specified date range.
    """
    df = pd.read_csv(file_path)
    df[col[0]] = pd.to_datetime(df[col[0]])
    action = df[(df[col[0]] >= pd.to_datetime(start)) & (df[col[0]] < pd.to_datetime(end))][col[1]].tolist()
    return action

def daily_reward(price, actions):
    """
    Calculate daily rewards based on price changes and actions.

    Parameters:
    price (list): List of prices.
    actions (list): List of actions taken.

    Returns:
    list: Daily rewards.
    """
    reward = []
    for i in range(len(price) - 1):
        r = actions[i] * np.log(price[i+1] / price[i])
        reward.append(r)
    return reward

def standard_deviation(reward):
    """
    Calculate the daily standard deviation of rewards.

    Parameters:
    reward (list): List of rewards.

    Returns:
    float: Standard deviation of the rewards.
    """
    mean = sum(reward) / len(reward)
    variance = sum((r - mean) ** 2 for r in reward) / (len(reward) - 1)
    return variance ** 0.5

def total_reward(price, actions):
    """
    Calculate the cumulative return over the trading period.

    Parameters:
    price (list): List of prices.
    actions (list): List of actions taken.

    Returns:
    float: Total cumulative reward.
    """
    reward = 0
    for i in range(len(price) - 1):
        reward += actions[i] * np.log(price[i+1] / price[i])
    return reward

def annualized_volatility(daily_std_dev, trading_days=252):
    """
    Calculate the annualized volatility from daily standard deviation.

    Parameters:
    daily_std_dev (float): Standard deviation of daily returns.
    trading_days (int): Number of trading days in a year, typically 252.

    Returns:
    float: Annualized volatility.
    """
    return daily_std_dev * (trading_days ** 0.5)

def calculate_sharpe_ratio(Rp, Rf, sigma_p, price):
    """
    Calculate the Sharpe ratio of an investment.

    Parameters:
    Rp (float): Return of the portfolio.
    Rf (float): Risk-free rate (annualized).
    sigma_p (float): Standard deviation of the portfolio's excess return.

    Returns:
    float: Sharpe ratio.
    """
    if sigma_p == 0:
        raise ValueError("Standard deviation cannot be zero.")
    Rp = Rp / (len(price)/252)
    
    return (Rp - Rf) / sigma_p

def calculate_max_drawdown(daily_returns):
    """
    Calculate the maximum drawdown of a portfolio.

    Parameters:
    daily_returns (list): List of daily returns.

    Returns:
    float: Maximum drawdown.
    """
    cumulative_returns = [1]
    for r in daily_returns:
        cumulative_returns.append(cumulative_returns[-1] * (1 + r))

    peak = cumulative_returns[0]
    max_drawdown = 0

    for r in cumulative_returns:
        if r > peak:
            peak = r
        drawdown = (peak - r) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown
    
def calculate_metrics(price, actions):
    """
    Calculate various financial metrics based on price and actions.

    Parameters:
    price (list): List of daily prices.
    actions (list): List of actions taken.

    Returns:
    tuple: A tuple containing calculated metrics (standard deviation, annualized volatility, cumulative return, Sharpe ratio, max drawdown).
    """
    daily_rw = daily_reward(price, actions)
    std_dev_r = standard_deviation(daily_rw)
    
    
    ann_vol = annualized_volatility(std_dev_r)
    cum_return = total_reward(price, actions)
    sharpe_ratio = calculate_sharpe_ratio(cum_return, 0, ann_vol, price)
    max_dd = calculate_max_drawdown(daily_rw)
    return cum_return, sharpe_ratio, std_dev_r, ann_vol, max_dd

def main(ticker, start, end, df_paths, col_names, save_path):
    """
    Main function to calculate metrics and save results to a CSV file.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start (str): Start date for analysis.
    end (str): End date for analysis.
    df_paths (dict): Dictionary of file paths for different models.
    col_names (dict): Dictionary of column names for different models.
    save_path (str): Path to save the results CSV file.
    """
    price = get_price(start, end, ticker)
    delta = max(price)-min(price)
    print('The price range of {0} in test period is {1}'.format(ticker, round(delta, 2)))
    metrics = ['Cumulative Return', 'Sharpe Ratio', 'Standard Deviation', 'Annualized Volatility', 'Max Drawdown']
    results = {'Buy & Hold': calculate_metrics(price, [1] * len(price))}
    daily_price_change_percentage = list(map(lambda p: (p[1] - p[0]) / p[0], zip(price[:-1], price[1:])))
    std_dev_p = standard_deviation(daily_price_change_percentage)
    for model, file_path in df_paths.items():
        if model == 'FinMe':
            print('Standard Deviation of Daily Price Change Percentage of {0}: {1:.2%}'.format(ticker, std_dev_p))
        col = col_names[model]
        actions = get_action(start, end, ticker, file_path, col)
        results[model] = calculate_metrics(price, actions)

    df_results = pd.DataFrame(results, index=metrics)
    df_results.to_csv(save_path)
    print(df_results)


if __name__ == '__main__':
    ticker = 'TSLA'
    start_time = '2022-10-06'
    end_time = '2023-04-10'
    
    df_paths = {
        'FinMe': '/Users/yuechenjiang/Desktop/CatMemo/result/Tsla-new-full.csv',
        'Park': '/Users/yuechenjiang/Desktop/CatMemo/result/action_df_tsla_park_v2.csv',
        'FinGPT': '/Users/yuechenjiang/Desktop/CatMemo/BenchMark/fingpt/tsla_curie.csv',
        'A2C': '/Users/yuechenjiang/Desktop/CatMemo/result/TSLA_A2C_summary_data_seed2_full.csv',
        'PPO': '/Users/yuechenjiang/Desktop/CatMemo/result/TSLA_PPO_summary_data_seed1_full.csv',
        'DQN': '/Users/yuechenjiang/Desktop/CatMemo/result/TSLA_DQN_summary_data_seed1_full.csv'
    }

    col_names = {
        'FinMe': ['date', 'direction'],
        'Park': ['date', 'direction'],
        'FinGPT': ['dates', 'actions'],
        'A2C': ['date', 'action'],
        'PPO': ['date', 'action'],
        'DQN': ['date', 'action']
    }

    save_path = '/Users/yuechenjiang/Desktop/CatMemo/Final_result/metrics/TSLA.csv'
    
    main(ticker, start_time, end_time, df_paths, col_names, save_path)
