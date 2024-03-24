import pickle
from tqdm import tqdm
import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import pandas as pd
import numpy as np


def subset_symbol_dict(input_dir, cur_symbol):
    new_dict = {}
    with open(input_dir, "rb") as f:
        data = pickle.load(f)
    # Now combined_dict contains all the data from the tuple
    # print(data.keys())
    # print(len(data[datetime.date(2023, 5, 17)]))
    # print(data[datetime.date(2023, 5, 17)][0])
    # print(data[datetime.date(2022, 8, 11)][1]['news']['NVDA'])
    # # print(data[datetime.date(2023, 5, 17)][2])
    # # print(data[datetime.date(2023, 5, 17)][3])
    # print(data[datetime.date(2023, 5, 17)][0]['price'].keys())
    new_dict = {}
    ticker_dict_byDate = {}
    for k, v in tqdm(data.items()):
        cur_price = v[0]['price']  # price
        cur_news = v[1]['news']   # news
        cur_filing_q = v[2]['filling_q']  # form q
        cur_filing_k = v[3]['filling_k']  # form k
        # print('Date: ---------', k)
        # print('Available tickers: ---------',cur_news.keys())

        new_price = {}
        new_filing_k = {}
        new_filing_q = {}
        new_news = {}
        if cur_symbol in list(cur_price.keys()):
            new_price[cur_symbol] = cur_price[cur_symbol]
        if cur_symbol in list(cur_filing_k.keys()):
            new_filing_k[cur_symbol] = cur_filing_k[cur_symbol]
        if cur_symbol in list(cur_filing_q.keys()):
            new_filing_q[cur_symbol] = cur_filing_q[cur_symbol]
        if cur_symbol in list(cur_news.keys()):
            new_news[cur_symbol] = cur_news[cur_symbol]
        else:
            continue

        new_dict[k] = {
            "price": new_price,
            "filing_k": new_filing_k,
            "filing_q": new_filing_q,
            "news": new_news
        }
        ticker_dict_byDate[k] = list(new_dict[k]["price"].keys())
        # print("On date: ", k, "ticker list: ---", ticker_dict_byDate[k])

    return new_dict, ticker_dict_byDate

#### finBERT
# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Function to analyze sentiment
def sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return scores.tolist()[0]

def assign_finBERT_scores(new_dict, cur_symbol):
    for i_date in tqdm(new_dict):
        i_date_dict = new_dict[i_date]
        if len(i_date_dict["news"]) != 0:
            i_date_news = i_date_dict["news"][cur_symbol]
            # i_date_news = i_date_news[: -3]
            
            j_new_news = []
            for j in range(len(i_date_news)):
                j_news = i_date_news[j]
                j_news_sentiment = sentiment_score(j_news)
                pos_score = j_news_sentiment[2]
                neu_score = j_news_sentiment[1]
                neg_score = j_news_sentiment[0]

                pos_sentence = f"The positive score for this news is {pos_score}."
                neu_sentence = f"The neutral score for this news is {neu_score}."
                neg_sentence = f"The negative score for this news is {neg_score}."

                j_combine_news_sentiment = f"{j_news} {pos_sentence} {neu_sentence} {neg_sentence}"
                # j_combine_news_sentiment = f"{j_news} {pos_sentence} {neg_sentence}"
                
                j_new_news.append(j_combine_news_sentiment)
            
            i_date_dict["news"][cur_symbol] = j_new_news


### vader
analyzer = SentimentIntensityAnalyzer()

def assign_vader_scores(new_dict, cur_symbol):

    for i_date in new_dict:
        i_date_dict = new_dict[i_date]
        if len(i_date_dict["news"]) != 0:
            i_date_news = i_date_dict["news"][cur_symbol]
            # i_date_news = i_date_news[: -3]
            
            j_new_news = []
            for j in range(len(i_date_news)):
                j_news = i_date_news[j]
                j_news_sentiment = analyzer.polarity_scores(j_news)
                pos_score = j_news_sentiment["pos"]
                neu_score = j_news_sentiment["neu"]
                neg_score = j_news_sentiment["neg"]

                pos_sentence = f"The positive score for this news is {pos_score}."
                neu_sentence = f"The neutral score for this news is {neu_score}."
                neg_sentence = f"The negative score for this news is {neg_score}."

                j_combine_news_sentiment = f"{j_news} {pos_sentence} {neu_sentence} {neg_sentence}"
                
                j_new_news.append(j_combine_news_sentiment)
            
            i_date_dict["news"][cur_symbol] = j_new_news



def export_sub_symbol(cur_symbol_lst, senti_model_type):
    print('Ticker list: ------', cur_symbol_lst)
    for cur_symbol_0 in cur_symbol_lst:
        new_dict, ticker_dict_byDate = subset_symbol_dict(input_dir, cur_symbol_0)
        
        if senti_model_type == 'FinBERT':
            assign_finBERT_scores(new_dict, cur_symbol_0)
            print('finBERT Date" 2023-05-30: ----- ', new_dict[datetime.date(2023, 5, 30)]['news'])
        else: 
            assign_vader_scores(new_dict, cur_symbol_0)   
            print('vader Date" 2023-05-30: ----- ', new_dict[datetime.date(2023, 5, 30)]['news'])
    
        out_dir = "./data/06_input/subset_symbols_"+ cur_symbol_0 + ".pkl"
        with open(out_dir, "wb") as f:
            pickle.dump(new_dict, f)
        print('*************---------------************')
    
    
cur_symbol_lst = ['BAC', 'DIS', 'GM', 'MRNA', 'NVDA', 'PFE']
input_dir = "./data/05_env_data/env_data.pkl"
#### option = 'FinBERT' or 'Vader'
export_sub_symbol(cur_symbol_lst, senti_model_type = 'FinBERT')