import polars as pl
import numpy as np
from datetime import date
from annotated_types import Gt
from typing import Dict, Annotated, Union
from pydantic import BaseModel


class PriceStructure(BaseModel):
    price: Annotated[float, Gt(0)]


class Portfolio:
    def __init__(self, symbol: str, lookback_window_size: int = 7) -> None:
        self.cur_date = None
        self.symbol = symbol
        self.action_series = {}
        self.market_price = None
        self.day_count = 0
        self.date_series = []
        self.holding_shares = 0
        self.market_price_series = np.array([])
        self.portfolio_share_series = np.array([])
        self.lookback_window_size = lookback_window_size

    def update_market_info(self, new_market_price_info: float, cur_date: date) -> None:
        PriceStructure.model_validate({"price": new_market_price_info})
        self.market_price = new_market_price_info
        self.cur_date = cur_date
        self.date_series.append(cur_date)
        self.day_count += 1
        self.market_price_series = np.append(
            self.market_price_series, new_market_price_info
        )

    def record_action(self, action: Dict[str, int]) -> None:
        self.holding_shares += action["direction"]
        self.action_series[self.cur_date] = action["direction"]

    def get_action_df(self) -> pl.DataFrame:
        temp_dict = {"date": [], "symbol": [], "direction": []}
        for date in self.action_series:
            temp_dict["date"].append(date)
            temp_dict["symbol"].append(self.symbol)
            temp_dict["direction"].append(self.action_series[date])
        return pl.DataFrame(temp_dict)

    def update_portfolio_series(self) -> None:
        self.portfolio_share_series = np.append(
            self.portfolio_share_series, self.holding_shares
        )

    def get_feedback_response(self) -> Union[Dict[str, Union[int, date]], None]:
        if self.day_count <= self.lookback_window_size:
            return None
        if len(np.diff(self.market_price_series)) != len(
            self.portfolio_share_series[:-1]
        ):
            temp = np.cumsum(
                (
                    np.diff(self.market_price_series)[:-1]
                    * self.portfolio_share_series[:-1]
                )[-self.lookback_window_size :]
            )[-1]
        else:
            temp = np.cumsum(
                (np.diff(self.market_price_series) * self.portfolio_share_series[:-1])[
                    -self.lookback_window_size :
                ]
            )[-1]

        if temp > 0:
            return {
                "feedback": 1,
                "date": self.date_series[-self.lookback_window_size],
            }
        elif temp < 0:
            return {
                "feedback": -1,
                "date": self.date_series[-self.lookback_window_size],
            }
        else:
            return {
                "feedback": 0,
                "date": self.date_series[-self.lookback_window_size],
            }

    def get_moment(self, moment_window: int = 3) -> Union[Dict[str, int], None]:
        if self.day_count <= moment_window:
            return None

        temp = np.cumsum((np.diff(self.market_price_series))[-moment_window:])[-1]

        if temp > 0:
            return {
                "moment": 1,
                "date": self.date_series[-moment_window],
            }

        elif temp < 0:
            return {
                "moment": -1,
                "date": self.date_series[-moment_window],
            }

        else:
            return {
                "moment": 0,
                "date": self.date_series[-moment_window],
            }
