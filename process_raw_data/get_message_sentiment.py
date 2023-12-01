import pandas as pd
import numpy as np
import datetime
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# just need to change the variable here
file_name = '/workspaces/ArkGPT/data/06_input/subset_symbols.pkl'
ticker = "TSLA"



with open(file_name, 'rb') as f:
    data = pickle.load(f)


analyzer = SentimentIntensityAnalyzer()

for i_date in data:
    i_date_dict = data[i_date]
    if len(i_date_dict["news"]) != 0:
        i_date_news = i_date_dict["news"][ticker]
        i_date_news = i_date_news[: -3]
        
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
        
        i_date_dict["news"][ticker] = j_new_news


with open(file_name, 'wb') as file:
    pickle.dump(data, file)


# lagacy ===========
# for i_date in data:
#     i_date_dict = data[i_date]

#     if len(i_date_dict["news"]) != 0:
#         for i_ticker in list(i_date_dict["news"].keys()):
        
#             i_ticker_news_list = i_date_dict["news"][i_ticker]
#             i_news = " ".join(i_ticker_news_list)

#             i_sentiment = analyzer.polarity_scores(i_news)
#             pos_score = i_sentiment["pos"]
#             neu_score = i_sentiment["neu"]
#             neg_score = i_sentiment["neg"]
            
#             date = i_date.strftime("%Y%m%d")
            
#             i_date_dict["news"][i_ticker].append(f"The positive score of date {date} news about {i_ticker} is {pos_score}")
#             i_date_dict["news"][i_ticker].append(f"The neutral score of date {date} news about {i_ticker} is {neu_score}")
#             i_date_dict["news"][i_ticker].append(f"The negative score of date {date} news about {i_ticker} is {neg_score}")