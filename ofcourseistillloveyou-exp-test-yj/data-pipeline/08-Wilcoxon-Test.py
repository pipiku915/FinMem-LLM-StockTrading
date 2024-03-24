import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import wilcoxon

def get_price(start, end, ticker):
    """
    Fetch daily adjusted closing prices of a stock from Yahoo Finance.
    """
    df = yf.download(ticker, start=start, end=end)
    return df['Adj Close'].to_numpy()

def get_action(start, end, file_path, col):
    """
    Fetch action data from a specified CSV file.
    """
    df = pd.read_csv(file_path)
    df[col[0]] = pd.to_datetime(df[col[0]])
    actions = df.loc[(df[col[0]] >= pd.to_datetime(start)) & (df[col[0]] <= pd.to_datetime(end)), col[1]].to_numpy()
    return actions

def calculate_cumulative_rewards(price, actions):
    """
    Calculate cumulative rewards based on price changes and actions.
    """
    reward = 0
    reward_list = []
    for i in range(len(price)-1):
        reward += actions[i] * np.log(price[i+1]/price[i])
        reward_list.append(reward)
    return reward_list

def perform_wilcoxon_test(data1, data2):
    """
    Perform Wilcoxon signed-rank test on two sets of data.
    """
    stat, p = wilcoxon(data1, data2)
    return stat, p

def main():
    ticker = 'TSLA'
    start = '2022-06-25'
    end = '2023-04-25'

    # Paths and columns setup
    file_paths = {
        'FinMe': ('/Users/yuechenjiang/Desktop/CatMemo/result/Tsla-new-full.csv', ['date', 'direction']),
        'Park': ('/Users/yuechenjiang/Desktop/CatMemo/result/action_df_tsla_park_v2.csv', ['date', 'direction']),
        'FinGPT': ('/Users/yuechenjiang/Desktop/CatMemo/BenchMark/fingpt/tsla_curie.csv', ['dates', 'actions']),
        'A2C': ('/Users/yuechenjiang/Desktop/CatMemo/result/TSLA_A2C_summary_data_seed2_full.csv', ['date', 'action']),
        'PPO': ('/Users/yuechenjiang/Desktop/CatMemo/result/TSLA_PPO_summary_data_seed1_full.csv', ['date', 'action']),
        'DQN': ('/Users/yuechenjiang/Desktop/CatMemo/result/TSLA_DQN_summary_data_seed1_full.csv', ['date', 'action'])
    }

    # Fetch price data
    price = get_price(start, end, ticker)

    results = {}
    for model, (path, cols) in file_paths.items():
        actions = get_action(start, end, path, cols)
        results[model] = calculate_cumulative_rewards(price, actions)

    # Perform Wilcoxon tests
    print(f'Wilcoxon Tests for {ticker} from {start} to {end}')
    model_keys = list(results.keys())
    for i in range(len(model_keys)):
        for j in range(i+1, len(model_keys)):
            model1 = model_keys[i]
            model2 = model_keys[j]
            rewards1 = results[model1]
            rewards2 = results[model2]
            statistic, pvalue = perform_wilcoxon_test(rewards1, rewards2)
            print(f'Wilcoxon Test between {model1} and {model2} - Statistic: {statistic}, P-Value: {pvalue}')

if __name__ == "__main__":
    main()