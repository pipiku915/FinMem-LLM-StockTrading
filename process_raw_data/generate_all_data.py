import pandas as pd
import numpy as np
import pickle
import datetime


class AgentEnvironment:
    def __init__(self, news_df_path, fillings_10k_q_df_path, econ_df_path, ark_records_df_path, daily_price_df_path):
        self.news_df = None
        self.fillings_10k_q_df = None
        self.econ_df = None
        self.ark_records_df = None 
        self.daily_price_df = None
        
        self.news_df_path = news_df_path
        self.fillings_10k_q_df_path = fillings_10k_q_df_path
        self.econ_df_path = econ_df_path
        self.ark_records_df_path  = ark_records_df_path
        self.daily_price_df_path = daily_price_df_path
        
        self.unique_trading_time = []
    
    def sync_dfs(self):    
          
        ### import dataframes:
        news_df = pd.read_parquet(self.news_df_path)
        fillings_10k_q_df = pd.read_parquet(self.fillings_10k_q_df_path)
        econ_df = pd.read_parquet(self.econ_df_path)
        ark_records_df = pd.read_parquet(self.ark_records_df_path)
        daily_price_df = pd.read_parquet(self.daily_price_df_path)
        # print('news: --------')
        # print(print(news_df.head(3)))
        # print('---------------------------------------')
        print('fillings: --------')
        print(fillings_10k_q_df.columns)
        print(fillings_10k_q_df['type'].unique())
        print(fillings_10k_q_df.head(3))
        
        # print('---------------------------------------')
        # print('econs: --------')
        # print(econ_df.head(3))
        # print('---------------------------------------')
        # print('ark:-----------')
        # print(ark_records_df.head(3))
        # print('---------------------------------------')
        print('stock:-----------')
        print(daily_price_df.head(3))
        # print(self.daily_price_df.dtypes)
        # print('-----------------------------------------')
        
        ### create trading ts unique list from earliest to latest
        unique_times = sorted(list(daily_price_df['Date'].dt.date.unique()))
        # print(unique_times)
        print('how many trading days?: ', len(unique_times))

        
        return unique_times, news_df, fillings_10k_q_df, econ_df, ark_records_df, daily_price_df
    
    
    def steps(self): 
         
        self.unique_trading_time, self.news_df, self.fillings_10k_q_df, \
        self.econ_df, self.ark_records_df, self.daily_price_df = self.sync_dfs()
        
        # Count the number of intervals that are more than one day using a for loop
        count = 0
        all_grouped_records_dict = {}
        for i in range(len(self.unique_trading_time) - 1):
            if (self.unique_trading_time[i + 1] - self.unique_trading_time[i]).days == 1:
                data_1d_dict = {}
                sub_news_1d_df = self.news_df[self.news_df['date'].dt.date == self.unique_trading_time[i + 1]]
                
                sub_fillings_10k_1d_df = self.fillings_10k_q_df[(self.fillings_10k_q_df['est_timestamp'].dt.date  == self.unique_trading_time[i + 1]) & (self.fillings_10k_q_df['type'] == '10-K')]
                sub_fillings_10q_1d_df = self.fillings_10k_q_df[(self.fillings_10k_q_df['est_timestamp'].dt.date  == self.unique_trading_time[i + 1]) & (self.fillings_10k_q_df['type'] == '10-Q')]
                # print('test-----------------------')
                # print(sub_fillings_10q_1d_df[['ticker', 'content']])
                
                sub_econ_1d_df = self.econ_df[self.econ_df['date'].dt.date  == self.unique_trading_time[i + 1]]
                sub_ark_records_1d_df = self.ark_records_df[self.ark_records_df['date'] == self.unique_trading_time[i + 1]]
                sub_daily_price_1d_df = self.daily_price_df[self.daily_price_df['Date'].dt.date == self.unique_trading_time[i + 1]]
                
                
                ### convert dataframes into the right formats
                sub_news_1d_df = sub_news_1d_df.rename(columns={'equity': 'ticker'})
                sub_news_1d_dict = sub_news_1d_df[['ticker', 'text']].to_dict(orient='records')
                sub_fillings_10k_1d_dict = sub_fillings_10k_1d_df[['ticker', 'content']].to_dict(orient='records')
                sub_fillings_10q_1d_dict = sub_fillings_10q_1d_df[['ticker', 'content']].to_dict(orient='records')
                # print('test-----------------------')
                # print(sub_fillings_10q_1d_dict)
                sub_econ_1d_dict = sub_econ_1d_df[['t10yff', 'permit', 'm2sl', 'ismnmdi', 'ICSA', 'cci', 'AWHAEMAN', 'ANDENO', 'amtmno']].to_dict(orient='records')
                sub_ark_records_1d_dict = sub_ark_records_1d_df[['date',  'direction', 'equity', 'quantity']].to_dict(orient='records')
                sub_daily_price_1d_df = sub_daily_price_1d_df.rename(columns={'equity': 'ticker'})
                sub_daily_price_1d_dict = sub_daily_price_1d_df[['symbol', 'Adj Close']].to_dict(orient='records')
                
                data_1d_dict['price'] = sub_daily_price_1d_dict
                data_1d_dict['economic_variable'] = sub_econ_1d_dict
                data_1d_dict['10k_fillings'] = sub_fillings_10k_1d_dict
                data_1d_dict['10q_fillings'] = sub_fillings_10q_1d_dict
                data_1d_dict['news'] = sub_news_1d_dict
                data_1d_dict['ark_record'] = sub_ark_records_1d_dict
                
                # print(sub_fillings_10k_q_1d_dict)
                # print(len(sub_fillings_10k_q_1d_dict))
                # print(sub_ark_records_1d_df.describe())
                # print(data_1d_dict['10k_fillings'])
                # print('--------------------------')
                all_grouped_records_dict[self.unique_trading_time[i + 1]] = data_1d_dict
                
            elif (self.unique_trading_time[i + 1] - self.unique_trading_time[i]).days > 1:
                data_nd_dict = {}
                sub_news_nd_df = self.news_df[(self.news_df['date'].dt.date <= self.unique_trading_time[i + 1]) & (self.news_df['date'].dt.date > self.unique_trading_time[i])]
                sub_fillings_10k_nd_df = self.fillings_10k_q_df[(self.fillings_10k_q_df['est_timestamp'].dt.date  <= self.unique_trading_time[i + 1]) & (self.fillings_10k_q_df['est_timestamp'].dt.date  > self.unique_trading_time[i]) & (self.fillings_10k_q_df['type'] == '10-K')]
                sub_fillings_10q_nd_df = self.fillings_10k_q_df[(self.fillings_10k_q_df['est_timestamp'].dt.date  <= self.unique_trading_time[i + 1]) & (self.fillings_10k_q_df['est_timestamp'].dt.date  > self.unique_trading_time[i]) & (self.fillings_10k_q_df['type'] == '10-Q')]
                sub_econ_nd_df = self.econ_df[(self.econ_df['date'].dt.date  <= self.unique_trading_time[i + 1]) & (self.econ_df['date'].dt.date  > self.unique_trading_time[i])]
                sub_ark_records_nd_df = self.ark_records_df[(self.ark_records_df['date'] <= self.unique_trading_time[i + 1]) & (self.ark_records_df['date'] > self.unique_trading_time[i])]
                sub_daily_price_nd_df = self.daily_price_df[self.daily_price_df['Date'].dt.date == self.unique_trading_time[i + 1]]
                
                sub_news_nd_df = sub_news_nd_df.rename(columns={'equity': 'ticker'})
                sub_news_nd_dict = sub_news_nd_df[['ticker', 'text']].to_dict(orient='records')
                sub_fillings_10k_nd_dict = sub_fillings_10k_nd_df[['ticker', 'content']].to_dict(orient='records')
                sub_fillings_10q_nd_dict = sub_fillings_10q_nd_df[['ticker', 'content']].to_dict(orient='records')
                sub_econ_nd_dict = sub_econ_nd_df[['t10yff', 'permit', 'm2sl', 'ismnmdi', 'ICSA', 'cci', 'AWHAEMAN', 'ANDENO', 'amtmno']].to_dict(orient='records')
                sub_ark_records_nd_dict = sub_ark_records_nd_df[['date',  'direction', 'equity', 'quantity']].to_dict(orient='records')
                sub_daily_price_nd_df = sub_daily_price_nd_df.rename(columns={'equity': 'ticker'})
                sub_daily_price_nd_dict = sub_daily_price_nd_df[['symbol', 'Adj Close']].to_dict(orient='records')
                
                data_nd_dict['price'] = sub_daily_price_nd_dict
                data_nd_dict['economic_variable'] = sub_econ_nd_dict
                data_nd_dict['10k_fillings'] = sub_fillings_10k_nd_dict
                data_nd_dict['10q_fillings'] = sub_fillings_10q_nd_dict
                data_nd_dict['news'] = sub_news_nd_dict
                data_nd_dict['ark_record'] = sub_ark_records_nd_dict
                
                # print('sub_fillings_10k_nd_dict', sub_fillings_10k_nd_dict)
                
                # print('news description: ', sub_news_nd_df.describe())
                # print('ark description: ', sub_ark_records_nd_df.describe())
                count += 1
                all_grouped_records_dict[self.unique_trading_time[i + 1]] = data_nd_dict
                # print('--------------------------------')

        print('Interval count: ', count)
        return all_grouped_records_dict
            

        
### dfs directories:      
environ = AgentEnvironment(news_df_path = "./data/04_input_data/news_data.parquet", \
                        fillings_10k_q_df_path = "./data/04_input_data/filing_data.parquet",\
                        econ_df_path = "./data/04_input_data/eco_data.parquet", \
                        ark_records_df_path = "./data/04_input_data/ark_record.parquet", \
                        daily_price_df_path = "./data/04_input_data/price_data.parquet"
                        )
all_grouped_records_dict = environ.steps()

print(all_grouped_records_dict.keys())
print(all_grouped_records_dict[ datetime.date(2023, 8, 7)]['10k_fillings'])
print(len(all_grouped_records_dict))

with open('./data/05_env_data/env_data.pkl', 'wb') as file:
    pickle.dump(all_grouped_records_dict, file)