# Generate *FinMem* input data

## Overview

This suite of Python scripts provides a comprehensive toolkit for generating *FinMem* input data, leveraging various data sources and analytical models. Part of the data is from **Private Dataset** You can obtain data through the open-source API, or you can use your own news data and process it through the pipeline, starting with section 5: `03-model_wrapper.py`.

**Note:** The performance of the model is highly dependent on the quality of the news data and the type of Large Language Model (LLM) used.

### Data Acquisition Scripts

1. **SEC API Data Download (`01_SEC_API_10k10q_download.py`):** 
   - Access U.S. Securities and Exchange Commission filings and data.
   - Obtain an API Key from [SEC API](https://sec-api.io/).

2. **Alpaca News API Data Download (`01_Alpaca_News_API_download.py`):**
   - Fetch real-time financial news and historical articles for market analysis.
   - API Key required, available at [Alpaca Markets](https://app.alpaca.markets/paper/dashboard/overview).

3. **Refinitiv Real-Time News Download (`01-Refinitiv_Real_Time_News_download.sql`):**
   - Gather news content from the Refinitiv Real-Time News (**Private Dataset**), mainly comprising **Reuters** articles.
   - Further summary needed.

### Data Processing Scripts

4. **Raw News Data Cleaning (`02-Raw_News_Data_Cleaning_Refinitiv.py`):**
   - Use this script to clean and preprocess your news data.

5. **Model Wrapper (`03-model_wrapper.py`):**
   - Rename to `model_wrapper.py` for use.
   - Select between ChatGPT or TogetherAI models from the model factory.

6. **News Summarization (`03-summary.py`):**
   - Summarize lengthy news data or 10k10q files.

### Data Pipeline

7. **Data Pipeline (`04-data_pipeline.py`):**
   - Handles two types of input files:
     a) News files: Summarized news for each stock ticker in CSV format, with 'date' (datetime.date), 'symbols', and 'summary' columns.
     b) 10k10q files: With columns 'document_url', 'content', 'ticker', 'cik', 'utc_timestamp', 'est_timestamp', 'type'.
   - Price Data Acquisition:
     a)This script automatically downloads price data using the Yahoo Finance API through the 'yfinance' package.
     b)The downloaded data is saved in the file `price.pkl`.
   - Produces five output files: `price.pkl`, `news.pkl`, `filing_q.pkl`, `filing_k.pkl`, and `env_data.pkl`.

### Sentiment Analysis

8. **Sentiment Analysis by Ticker (`05-get_sentiment_by_ticker.py`):**
   - Inputs: `env_data.pkl`.
   - Adds sentiment to each news item using either VADER or FinBERT.
   - VADER is recommended for limited computing resources, while FinBERT is preferred for financial applications.

### Visualization and Metrics

9. **Results Visualization (`06-Visualize-results.py`):**
   - Visualizes Cumulative Return and compares it with a buy-and-hold strategy.
   - Allows comparison of different model performances.

10. **Performance Metrics (`07-metrics.py`):**
    - Calculates Sharpe Ratio, Cumulative Return, Max Drawdown, Standard Deviation, and Annualized Volatility.
    - Inputs: model actions.

### Statistical Analysis

#### Wilcoxon Test for Different Models (`08-Wilcoxon-Test.py`)

This script conducts a Wilcoxon signed-rank test to evaluate statistically the performance differences between two models. The Wilcoxon test is a non-parametric test that does not assume a Gaussian distribution of data, making it particularly suitable for financial datasets, which often exhibit skewness, volatility, and outliers. This choice is predicated on the financial data's tendency to deviate from normality, rendering traditional parametric tests less appropriate.

**Key Features:**

- **Non-Gaussian Data Appropriateness:** Specifically chosen for financial datasets that do not follow a normal distribution, ensuring the analysis remains valid under non-standard data conditions.
- **Median-focused Analysis:** Unlike mean-focused tests, the Wilcoxon test examines differences in median values, offering a more accurate analysis for skewed data.
- **Outlier Impact Minimization:** Reduces the influence of outliers, providing a more stable and reliable comparison between model performances.

**Benefits:**

- Facilitates the comparison of two financial models' performances by examining their paired sample data, identifying whether their median ranks significantly differ.
- Delivers a robust statistical foundation for making informed decisions regarding model adjustments, strategy enhancements, or the introduction of new models.

## Toy Example for Data Pipeline

**Note:** The data used in this example is sourced from Kaggle ([TextDB3 Dataset](https://www.kaggle.com/datasets/hassanamin/textdb3)). Please note that this dataset **`does not`** contain real stock data; it is utilized solely as a test case to demonstrate the functionality of our data pipeline.

### Getting Started with the *FinMem* Data Pipeline

Embark on a journey to harness the full potential of the *FinMem* data pipeline by following these introductory steps:

1. **Initiate by Unzipping `Fake-Sample_data.zip`:**
   Uncover the foundational elements necessary for pipeline evaluation contained within this archive:
   - `example_input`: A collection of simulated input data that sets the stage for a genuine testing environment of the *FinMem* data pipeline.
   - `example_output`: This directory showcases the output expected from processing the provided input through the *FinMem* pipeline, enabling you to assess its efficiency and accuracy.

2. **Dive into the Sample Input Data:**
   The `example_input` folder houses the `Fake-News-Data-for-Each-Stock/` directory, featuring artificial financial news datasets for four pivotal stocks: Amazon (AMZN), Microsoft (MSFT), Netflix (NFLX), and Tesla (TSLA). Each dataset is meticulously designed to resemble authentic financial news, offering a varied testing landscape:

   - **AMZN News Data (`AMZN_fake.csv`):** 1245 articles from January 13, 2016, to December 30, 2017.
   - **MSFT News Data (`MSFT_fake.csv`):** 1342 articles from January 13, 2016, to December 27, 2017.
   - **NFLX News Data (`NFLX_fake.csv`):** 1220 articles from January 13, 2016, to December 27, 2017.
   - **TSLA News Data (`TSLA_fake.csv`):** 1313 articles from January 15, 2016, to December 27, 2017.

   Begin your journey with the `03-model_wrapper.py` script to summarize this news data effectively. After summarizing, utilize both the summarized news data and the sample `filing_data.parquet` file for 10k10q documents as inputs for further processing. This approach ensures a thorough understanding and utilization of the *FinMem* data pipeline, showcasing its ability to handle and analyze financial news and filings across varied timelines and companies.
   The breadth of these datasets across different sectors and time frames makes them an exemplary resource for evaluating the versatility and depth of the *FinMem* data pipeline's analytical capabilities.

4. **Workflow Integration:**
   - Commence with the `env_data.pkl` file located in the `sample_input` folder, employing it as the initial input for the `05-get_sentiment_by_ticker.py` script.
   - Subsequently, harness the output from `05-get_sentiment_by_ticker.py` as the requisite input for the *FinMem* model.

This streamlined approach elucidates the data progression within the *FinMem* data pipeline, illustrating the journey from preliminary input through to the conclusive model integration.
