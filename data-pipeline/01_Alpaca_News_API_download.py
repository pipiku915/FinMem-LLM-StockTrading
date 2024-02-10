# Arguments
END_POINT_TEMPLATE = "https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&limit=50&symbols={symbol}"
END_POINT_TEMPLATE_LINK_PAGE = "https://data.alpaca.markets/v1beta1/news?limit=50&symbol={symbol}&page_token={page_token}"
NUM_NEWS_PER_RECORD = 200
MAX_ATTEMPTS = 5
WAIT_TIME = 60
MAX_WORKERS = 30

# dependencies
import os
import time
import shutil
import httpx
import tenacity
import polars as pl
from dotenv import load_dotenv
from rich import print
from tqdm import tqdm
from uuid import uuid4
from datetime import date, timedelta, datetime
from typing import List, Dict, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()


def round_to_next_day(date: pl.Expr) -> pl.Expr:
    hour = date.dt.hour()
    minute = date.dt.minute()
    second = date.dt.second()
    year = date.dt.year()
    month = date.dt.month()
    day = date.dt.day()
    condition = ((hour >= 16)) & ((second > 0) | (minute > 0))
    new_day = day + condition.cast(pl.UInt32)
    return pl.datetime(year, month, new_day, 9, 0, 0)


class ScraperError(Exception):
    pass


class RecordContainerFull(Exception):
    pass


class ParseRecordContainer:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.record_counter = 0
        self.author_list = []
        self.content_list = []
        self.date_list = []
        self.source_list = []
        self.summary_list = []
        self.title_list = []
        self.url_list = []

    def add_records(self, records: List[Dict[str, str]]) -> None:
        for cur_record in records:
            self.author_list.append(cur_record["author"])
            self.content_list.append(cur_record["content"])
            date = cur_record["created_at"].rstrip("Z")
            self.date_list.append(datetime.fromisoformat(date))
            self.source_list.append(cur_record["source"])
            self.summary_list.append(cur_record["summary"])
            self.title_list.append(cur_record["headline"])
            self.url_list.append(cur_record["url"])
            self.record_counter += 1
            if self.record_counter == NUM_NEWS_PER_RECORD:
                raise RecordContainerFull

    def pop(self, align_next_date: bool = True) -> Union[pl.DataFrame, None]:
        if self.record_counter == 0:
            return None
        return_df = pl.DataFrame(
            {
                "author": self.author_list,
                "content": self.content_list,
                "datetime": self.date_list,
                "source": self.source_list,
                "summary": self.summary_list,
                "title": self.title_list,
                "url": self.url_list,
            }
        )
        if align_next_date:
            return_df = return_df.with_columns(
                round_to_next_day(return_df["datetime"]).alias("date"),
            )
        else:
            return_df = return_df.with_columns(
                pl.col("datetime").date().alias("date"),
            )
        return return_df.with_columns(pl.lit(self.symbol).alias("equity"))


@retry(stop=stop_after_attempt(MAX_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
def query_one_record(args: Tuple[date, str]) -> None:
    date, symbol = args
    next_date = date + timedelta(days=1)
    request_header = {
        "Apca-Api-Key-Id": os.environ.get("ALPACA_KEY"),
        "Apca-Api-Secret-Key": os.environ.get("ALPACA_KEY_SECRET_KEY"),
    }
    container = ParseRecordContainer(symbol)

    with httpx.Client() as client:
        # first request
        response = client.get(
            END_POINT_TEMPLATE.format(
                start_date=date.strftime("%Y-%m-%d"),
                end_date=next_date.strftime("%Y-%m-%d"),
                symbol=symbol,
            ),
            headers=request_header,
        )
        if response.status_code != 200:
            print("[red]Hit limit[/red]")
            raise ScraperError(response.text)
        result = response.json()
        next_page_token = result["next_page_token"]
        container.add_records(result["news"])

        while next_page_token:
            try:
                response = client.get(
                    END_POINT_TEMPLATE_LINK_PAGE.format(
                        symbol=symbol, page_token=next_page_token
                    ),
                    headers=request_header,
                )
                if response.status_code != 200:
                    raise ScraperError(response.text)
                result = response.json()
                next_page_token = result["next_page_token"]
                container.add_records(result["news"])
            except RecordContainerFull:
                break

    result = container.pop(align_next_date=True)
    if result is not None:
        result.write_parquet(os.path.join("data", "temp", f"{uuid4()}.parquet"))


def main_sync() -> None:
    # load data
    data = pl.read_parquet(os.path.join("data", "03_primary", "price_data.parquet"))
    if os.path.exists(os.path.join("data", "temp")):
        shutil.rmtree(os.path.join("data", "temp"))
    os.mkdir(os.path.join("data", "temp"))

    query_data = (
        data.select([pl.col("est_time").dt.date().alias("date"), pl.col("equity")])
        .unique()
        .to_dict()
    )

    args_list = list(zip(query_data["date"], query_data["equity"]))
    with tqdm(total=len(args_list)) as pbar:
        for i, arg in enumerate(args_list):
            try:
                query_one_record(arg)
            except tenacity.RetryError as e:
                print(f"We caught the error: {e}")
            pbar.update(1)
            if (i + 1) % 3000 == 0:
                time.sleep(90)
    record_dfs = [
        pl.read_parquet(os.path.join("data", "temp", f))
        for f in os.listdir(os.path.join("data", "temp"))
    ]
    df = pl.concat(record_dfs)
    df.write_parquet(os.path.join("data", "03_primary", "news.parquet"))
    print(df.shape)


if __name__ == "__main__":
    main_sync()
