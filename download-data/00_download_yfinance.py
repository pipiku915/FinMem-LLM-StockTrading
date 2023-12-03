import os
import yfinance as yf
import pandas as pd
import polars as pl
from tqdm import tqdm
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed


# start date
START_DATE = "2021-04-25"
END_DATE = "2023-08-15"


def download_one_symbol(symbol: str) -> Union[None, pl.DataFrame]:
    try:
        data = yf.download(
            tickers=[symbol], start=START_DATE, end=END_DATE, progress=False
        ).reset_index()
    except Exception:
        return None
    if data.shape[0] == 0:
        return None
    data = pl.from_pandas(data)
    return data.with_columns(
        [
            pl.lit(symbol).alias("symbol"),
        ]
    )


if __name__ == "__main__":
    # ticker download list
    ark_data = pl.read_csv(
        os.path.join("data", "02_intermediate", "parsed_ark.csv"),
        try_parse_dates=True,
    ).drop_nulls()
    unique_symbols = ark_data.select(pl.col("equity").unique())["equity"].to_list()

    # start download
    downloaded_dfs = []
    failed_downloaded_symbol = []
    counter = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(download_one_symbol, symbol): symbol
            for symbol in unique_symbols
        }

        for f in tqdm(as_completed(futures), total=len(futures)):
            symbol = futures[f]
            df = f.result()
            if df is not None:
                downloaded_dfs.append(df)
            else:
                failed_downloaded_symbol.append(symbol)
                counter += 1
                if counter % 10 == 0:
                    print(f"Failed to download {counter} symbols")

    df = pl.concat(downloaded_dfs)
    df.write_parquet(os.path.join("data", "02_intermediate", "yf_data.parquet"))
    with open(os.path.join("data", "02_intermediate", "failed_symbols.txt"), "w") as f:
        f.write("\n".join(failed_downloaded_symbol))
