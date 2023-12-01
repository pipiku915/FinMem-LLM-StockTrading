import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import wilcoxon

def get_price(start, end, ticker):
    """
    Fetch daily adjusted closing prices of a stock from Yahoo Finance.

    Parameters:
    Start (str): Start date for the price data.
    End (str): End date for the price data.
    Ticker (str): Ticker symbol of the stock.

    Returns:
    list: List of daily adjusted closing prices.
    """
    df = yf.download(ticker, start=start, end=end)
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

def reward_list(price:list, actions:list):
    reward = 0
    reward_list = []
    for i in range(len(price)-1):
        reward += actions[i] * np.log(price[i+1]/price[i])
        reward_list.append(reward)
    return reward_list

def calculate_metrics(price, actions):
    """
    Calculate various financial metrics based on price and actions.

    Parameters:
    price (list): List of daily prices.
    actions (list): List of actions taken.

    Returns:
    tuple: A tuple containing calculated metrics
    """
    daily_rw = daily_reward(price, actions)
    cum_return = reward_list(price, actions)
    return daily_rw, cum_return

def perform_wilcoxon_test(spr1, spr2):
    stat, p = wilcoxon(spr1, spr2)
    return stat, p