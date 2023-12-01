import polars as pl
import numpy as np
from datetime import date
from annotated_types import Gt
from typing import Dict, List, Annotated
from pydantic import BaseModel
import datetime


class PriceStructure(BaseModel):
    price: Dict[str, Annotated[float, Gt(0)]]


class Portfolio:
    def __init__(self, notational_cash: float, lookback_window_size: int = 7) -> None:
        if notational_cash < 0:
            raise ValueError("Notational cash must be positive")
        self.initial_notational_cash = notational_cash
        self.lookback_window_size = lookback_window_size
        self.basket = {
            "cash": notational_cash,
            "portfolio_value": 0.0,
            "portfolio": {},
            "portfolio_price": 0.0,
        }
        self.cur_date = None
        self.value_series = {}
        self.action_series = {}
        self.market_price = None
        self.day_count = 0
        self.date_series = []
        self.market_price_series = {}
        self.portfolio_share_series = {}

        self.market_date_price_series = {}
        self.future_market_delta = {}

    def update_market_info(
        self, new_market_price_info: Dict[str, float], cur_date: date
    ) -> None:
        PriceStructure.validate({"price": new_market_price_info})
        self.market_price = new_market_price_info
        self.cur_date = cur_date
        self.date_series.append(cur_date)
        self.day_count += 1
        for cur_symbol in new_market_price_info:
            if cur_symbol not in self.market_price_series:
                self.market_price_series[cur_symbol] = np.array(
                    [new_market_price_info[cur_symbol]]
                )
                self.portfolio_share_series[cur_symbol] = np.array([])
            else:
                self.market_price_series[cur_symbol] = np.append(
                    self.market_price_series[cur_symbol],
                    new_market_price_info[cur_symbol],
                )

        # for cur_symbol in new_market_price_info:
        #     self.market_date_price_series[cur_symbol] = {}
        #     self.market_date_price_series[cur_symbol][self.cur_date] = new_market_price_info[cur_symbol]

    def update_portfolio_value(self) -> None:
        self.basket["portfolio_value"] = sum(
            self.market_price[symbol] * shares
            for symbol, shares in self.basket["portfolio"].items()
        )
        self.value_series[self.cur_date] = (
            self.basket["portfolio_value"] + self.basket["cash"]
        )

    def update_portfolio_from_actions(
        self, actions: List[Dict[str, str | int]]
    ) -> None:
        # implement actions
        temp_action_dict = {}
        for cur_action in actions:
            cur_symbol = list(cur_action.keys())[0]
            cur_direction = cur_action[cur_symbol]["direction"]
            temp_action_dict[cur_symbol] = cur_direction
            if cur_symbol not in self.basket["portfolio"].keys():
                self.basket["portfolio"][cur_symbol] = cur_direction
            else:
                self.basket["portfolio"][cur_symbol] += cur_direction
            self.basket["cash"] -= self.market_price[cur_symbol] * cur_direction
        self.action_series[self.cur_date] = temp_action_dict

    def get_portfolio_value_series(self) -> Dict[date, float]:
        return self.value_series

    def get_action_df(self) -> pl.DataFrame:
        temp_dict = {"date": [], "symbol": [], "direction": []}
        for date in self.action_series:
            for symbol in self.action_series[date]:
                temp_dict["date"].append(date)
                temp_dict["symbol"].append(symbol)
                temp_dict["direction"].append(self.action_series[date][symbol])
        return pl.DataFrame(temp_dict)

    def update_portfolio_series(self) -> None:
        for cur_symbol in self.market_price_series:
            if cur_symbol not in self.basket["portfolio"].keys():
                self.portfolio_share_series[cur_symbol] = np.append(
                    self.portfolio_share_series[cur_symbol], 0
                )
            else:
                self.portfolio_share_series[cur_symbol] = np.append(
                    self.portfolio_share_series[cur_symbol],
                    self.basket["portfolio"][cur_symbol],
                )

    def get_feedback_response(self) -> Dict[str, int]:
        if self.day_count <= self.lookback_window_size:
            return None
        feedback = {}
        for cur_symbol in self.market_price:
            if len(np.diff(self.market_price_series[cur_symbol])) != len(
                self.portfolio_share_series[cur_symbol][:-1]
            ):
                temp = np.cumsum(
                    (
                        np.diff(self.market_price_series[cur_symbol])[:-1]
                        * self.portfolio_share_series[cur_symbol][:-1]
                    )[-self.lookback_window_size :]
                )[-1]
            else:
                temp = np.cumsum(
                    (
                        np.diff(self.market_price_series[cur_symbol])
                        * self.portfolio_share_series[cur_symbol][:-1]
                    )[-self.lookback_window_size :]
                )[-1]

            if temp > 0:
                feedback[cur_symbol] = {
                    "feedback": 1,
                    "date": self.date_series[-self.lookback_window_size],
                }
            elif temp < 0:
                feedback[cur_symbol] = {
                    "feedback": -1,
                    "date": self.date_series[-self.lookback_window_size],
                }
            else:
                feedback[cur_symbol] = {
                    "feedback": 0,
                    "date": self.date_series[-self.lookback_window_size],
                }
        return feedback

    # def get_delta_market_truth(self) -> Dict[str, int]:

    # for cur_symbol in self.market_price:
    #     self.future_market_delta[cur_symbol] = {}
    #     for date in self.date_series:
    #         self.future_market_delta[cur_symbol]

    #     = np.diff(self.market_price_series[cur_symbol])

    def get_moment(self, moment_window=3) -> Dict[str, int]:
        if self.day_count <= moment_window:
            return None

        moment = {}
        for cur_symbol in self.market_price:
            temp = np.cumsum(
                (np.diff(self.market_price_series[cur_symbol]))[-moment_window:]
            )[-1]

            if temp > 0:
                moment[cur_symbol] = {
                    "moment": 1,
                    "date": self.date_series[-moment_window],
                }

            elif temp < 0:
                moment[cur_symbol] = {
                    "moment": -1,
                    "date": self.date_series[-moment_window],
                }

            else:
                moment[cur_symbol] = {
                    "moment": 0,
                    "date": self.date_series[-moment_window],
                }

        return moment

    def get_current_position(self) -> Dict[str, int]:
        if self.day_count <= 2:
            return None

        current_position = {}

        for cur_symbol in self.market_price:
            temp = np.cumsum(self.portfolio_share_series[cur_symbol])[-1]

            if temp > 0:
                current_position[cur_symbol] = {
                    "current_position": 1,
                }

            elif temp < 0:
                current_position[cur_symbol] = {
                    "current_position": -1,
                }

            else:
                current_position[cur_symbol] = {"current_position": 0}

        return current_position
