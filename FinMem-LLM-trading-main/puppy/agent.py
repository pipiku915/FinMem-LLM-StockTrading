import os
import time
import shutil
import pickle
import logging
from datetime import date
from abc import ABC, abstractmethod
from typing import Dict, Union, Any, List, Tuple
from .memorydb import BrainDB
from .portfolio import Portfolio
from .environment import market_info_type, terminated_market_info_type
#from .prompts import eco_prompt, EcoObservation, train_reflection, train_trade
from .prompts import train_reflection, train_trade
from .chat import get_chat_end_points
from .environment import record_type
from tqdm import tqdm
import timeout_decorator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler("run.log", "a")
file_handler.setFormatter(logging_formatter)
logger.addHandler(file_handler)


class Agent(ABC):
    @abstractmethod
    def from_config(self, config: Dict[str, Any]) -> "Agent":
        pass

    @abstractmethod
    def train_step(self) -> None:
        pass


# LLM Agent
class LLMAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        character_string: str,
        brain_db: BrainDB,
        top_k: int = 1,
        notational_cash: float = 1000000.0,
        chat_end_point_name: str = "openai",
        chat_end_point_config: Dict[str, Any] = None,
    ):
        if chat_end_point_config is None:
            chat_end_point_config = {"model_name": "gpt-4", "temperature": 0.7}
        # base
        self.agent_name = agent_name
        self.character_string = character_string
        self.top_k = top_k
        self.notational_cash = notational_cash
        self.chat_end_point_name = chat_end_point_name
        self.chat_end_point_config = chat_end_point_config
        self.counter = 1
        # brain db
        self.brain = brain_db
        # portfolio class
        self.portfolio = Portfolio(notational_cash=notational_cash)
        # chat end points
        self.chat_end_point = get_chat_end_points(
            end_point_type=chat_end_point_name, chat_config=chat_end_point_config
        )
        self.guardrail_endpoint = self.chat_end_point.guardrail_endpoint()
        # economy belief
        # self.eco_belief = None
        # self.past_eco = None
        self.reflection_result_series_dict = {}
        self.access_counter = {}

    @classmethod
    def from_config(cls, agent_config: Dict[str, Any], agent_name: str) -> "LLMAgent":
        return cls(
            agent_name=agent_name,
            character_string=agent_config[agent_name]["general"]["character_string"],
            brain_db=BrainDB.from_config(agent_config, "agent_1"),
            top_k=agent_config[agent_name]["general"].get("top_k", 10),  ###otherwise change to 5
            notational_cash=agent_config[agent_name]["general"].get(
                "notational_cash", 1000000.0
            ),
        ) 

    # def _handle_eco_belief(self, eco_info: Dict[str, float]) -> None:
    #     EcoObservation.validate(eco_info)
    #     if (
    #         self.eco_belief is not None
    #         and self.past_eco != eco_info
    #         or self.eco_belief is None
    #     ):
    #         self.eco_belief = self.chat_end_point(eco_prompt(eco_info))[0]
    #         self.past_eco = eco_info

    def _handling_filings(
        self, cur_date: date, filing_q: Dict[str, str], filing_k: Dict[str, str]
    ) -> None:
        if filing_q != {}:
            for key, val in filing_q.items():
                self.brain.add_memory_mid(symbol=key, date=cur_date, text=val)
        if filing_k != {}:
            for key, val in filing_k.items():
                self.brain.add_memory_long(
                    symbol=key,
                    date=cur_date,
                    text=val,
                )

    def _handling_news(self, cur_date: date, news: Dict[str, List[str]]):
        if news != {}:
            for key, val in news.items():
                self.brain.add_memory_short(symbol=key, date=cur_date, text=val)

    def _reflection_on_record(
        self, cur_date: date, cur_record: Dict[str, record_type], future_date: date
    ) -> Dict[str, Dict[str, Any]]:
        if not cur_record:
            logger.info(f"No record\n")
            return {}
        ret_result = {}
        for cur_symbol in tqdm(cur_record):
            logger.info(f"Symbol: {cur_symbol}\n")
            cur_short_queried, cur_short_memory_id = self.brain.query_short(
                query_text=self.character_string, symbol=cur_symbol, top_k=self.top_k
            )
            for cur_id, cur_memory in zip(cur_short_memory_id, cur_short_queried):
                logger.info(f"Top-k Short: {cur_id}: {cur_memory}\n")
            cur_mid_queried, cur_mid_memory_id = self.brain.query_mid(
                query_text=self.character_string, symbol=cur_symbol, top_k=self.top_k
            )
            for cur_id, cur_memory in zip(cur_mid_memory_id, cur_mid_queried):
                logger.info(f"Top-k Mid: {cur_id}: {cur_memory}\n")
            cur_long_queried, cur_long_memory_id = self.brain.query_long(
                query_text=self.character_string, symbol=cur_symbol, top_k=self.top_k
            )
            for cur_id, cur_memory in zip(cur_long_memory_id, cur_long_queried):
                logger.info(f"Top-k Long: {cur_id}: {cur_memory}\n")
            (
                cur_reflection_queried,
                cur_reflection_memory_id,
            ) = self.brain.query_reflection(
                query_text=self.character_string, symbol=cur_symbol, top_k=self.top_k
            )
            for cur_id, cur_memory in zip(
                cur_reflection_memory_id, cur_reflection_queried
            ):
                logger.info(f"Top-k Reflection: {cur_id}: {cur_memory}\n")

            

            reflection_result = train_reflection(
                cur_date=cur_date,
                endpoint_func=self.guardrail_endpoint,
                #eco_info=self.eco_belief,
                short_memory=cur_short_queried,
                short_memory_id=cur_short_memory_id,
                mid_memory=cur_mid_queried,
                mid_memory_id=cur_mid_memory_id,
                long_memory=cur_long_queried,
                long_memory_id=cur_long_memory_id,
                reflection_memory=cur_reflection_queried,
                reflection_memory_id=cur_reflection_memory_id,
                future_date = future_date,
                future_record= cur_record[cur_symbol],
                symbol=cur_symbol,
            )

            if (reflection_result is not {}) and (
                "summary_reason" in reflection_result
            ):
                self.brain.add_memory_reflection(
                    symbol=cur_symbol,
                    date=cur_date,
                    text=reflection_result["summary_reason"],
                )
            else:
                logger.info(
                    f"No reflection result for symbol {cur_symbol}, not converged\n"
                )
            ret_result[cur_symbol] = reflection_result
            # ugly but useful
            time.sleep(0.5)
            # ugly but useful
        return ret_result


    def _trade(self, cur_date:date, cur_price: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        ret_result = {}
        for cur_symbol in tqdm(cur_price):
            logger.info(f"Symbol: {cur_symbol}\n")
            # short
            cur_short_queried, cur_short_memory_id = self.brain.query_short(
                query_text=self.character_string, symbol=cur_symbol, top_k=self.top_k
            )
            for cur_id, cur_memory in zip(cur_short_memory_id, cur_short_queried):
                logger.info(f"Top-k Short: {cur_id}: {cur_memory}\n")
            # mid
            cur_mid_queried, cur_mid_memory_id = self.brain.query_mid(
                query_text=self.character_string, symbol=cur_symbol, top_k=self.top_k
            )
            for cur_id, cur_memory in zip(cur_mid_memory_id, cur_mid_queried):
                logger.info(f"Top-k Mid: {cur_id}: {cur_memory}\n")
            # long
            cur_long_queried, cur_long_memory_id = self.brain.query_long(
                query_text=self.character_string, symbol=cur_symbol, top_k=self.top_k
            )
            for cur_id, cur_memory in zip(cur_long_memory_id, cur_long_queried):
                logger.info(f"Top-k Long: {cur_id}: {cur_memory}\n")
            # reflection
            (
                cur_reflection_queried,
                cur_reflection_memory_id,
            ) = self.brain.query_reflection(
                query_text=self.character_string, symbol=cur_symbol, top_k=self.top_k
            )
            for cur_id, cur_memory in zip(
                cur_reflection_memory_id, cur_reflection_queried
            ):
                logger.info(f"Top-k Reflection: {cur_id}: {cur_memory}\n")
                
            
            cur_moment_dict = self.portfolio.get_moment(moment_window = 2)
            if cur_moment_dict is not None:
                cur_moment = cur_moment_dict[cur_symbol]["moment"]
                
            else:
                cur_moment = None
            
            cur_position_dict = self.portfolio.get_current_position()
            
            if cur_position_dict is not None:
                cur_position = cur_position_dict[cur_symbol]["current_position"]
            
            else: 
                cur_position = None

            try: 
                trade_result = train_trade(
                    cur_date=cur_date,
                    endpoint_func=self.guardrail_endpoint,
                    short_memory=cur_short_queried,
                    short_memory_id=cur_short_memory_id,
                    mid_memory=cur_mid_queried,
                    mid_memory_id=cur_mid_memory_id,
                    long_memory=cur_long_queried,
                    long_memory_id=cur_long_memory_id,
                    reflection_memory=cur_reflection_queried,
                    reflection_memory_id=cur_reflection_memory_id,
                    symbol=cur_symbol,
                    moment = cur_moment,
                    position = cur_position,
                )
                if (trade_result is not {}) and (
                    "summary_reason" in trade_result
                ):
                    self.brain.add_memory_reflection(
                        symbol=cur_symbol,
                        date=cur_date,
                        text=trade_result["summary_reason"],
                    )
                else:
                    logger.info(
                        f"No reflection result for symbol {cur_symbol}, not converged\n"
                    )
                ret_result[cur_symbol] = trade_result
            except TimeoutError:
                logger.info(f"timeouterror")
                

        return ret_result


    
    @timeout_decorator.timeout(300, timeout_exception=TimeoutError)
    def train_step(
        self, market_info: Union[market_info_type, terminated_market_info_type]
    ) -> Tuple[
        Dict[date, Dict[str, float]], Dict[str, float], Dict[date, Dict[str, Any]]
    ]:
        cur_date = market_info[0]
        future_date = market_info[1]
        cur_price = market_info[2]
        # cur_eco = market_info[2]
        cur_filing_k = market_info[3]
        cur_filing_q = market_info[4]
        cur_news = market_info[5]
        cur_record = market_info[6]

        # 1. make it aware the eco environment
        #self._handle_eco_belief(cur_eco)
        # 2. handling filings
        self._handling_filings(cur_date, cur_filing_q, cur_filing_k)
        # 3. handling news
        self._handling_news(cur_date, cur_news)
        # 4. update the price to portfolio
        self.portfolio.update_market_info(
            new_market_price_info=cur_price,
            cur_date=cur_date,
        )
        # 5. reflection on the record
        if cur_date == date(2021, 9, 2):
            pass
        reflection_result_cur_date = self._reflection_on_record(
            cur_date=cur_date, cur_record=cur_record, future_date =future_date
        )
        self.reflection_result_series_dict[cur_date] = reflection_result_cur_date
        for cur_symbol in reflection_result_cur_date:
            logger.info(
                f"{cur_symbol}-Day {cur_date}\nreflection summary: {reflection_result_cur_date[cur_symbol].get('summary_reason')}\n\n"
            )

        if cur_record[cur_symbol] > 0:
            cur_direction = 1
        else:
            cur_direction = -1
        cur_actions = [
            {
                cur_symbol: {
                    "direction": cur_direction,
                    "quantity": 1,
                }
            }
            for cur_symbol in cur_record
        ]
        self.portfolio.update_portfolio_from_actions(actions=cur_actions)
        self.portfolio.update_portfolio_value()
        self.portfolio.update_portfolio_series()
        # 7. update the access counter if need to
        if feedback := self.portfolio.get_feedback_response():
            for cur_symbol in feedback:
                if feedback[cur_symbol]["feedback"] != 0:
                    # update short memory if it is not none
                    cur_date = feedback[cur_symbol]["date"]
                    if cur_symbol in self.reflection_result_series_dict[cur_date]:
                        cur_memory = self.reflection_result_series_dict[cur_date][
                            cur_symbol
                        ]
                        if "short_memory_index" in cur_memory:
                            cur_ids = []
                            for i in cur_memory["short_memory_index"]:
                                cur_id = i["memory_index"]
                                if cur_id not in cur_ids:
                                    cur_ids.append(cur_id)
                            # cur_ids = cur_memory["short_memory_index"][0]["memory_index"]
                            self.brain.update_access_count_with_feed_back(
                                symbol=cur_symbol,
                                ids=cur_ids,
                                feedback=feedback[cur_symbol]["feedback"],
                            )
                        # update middle memory if it is not none
                        if "middle_memory_index" in cur_memory:
                            cur_ids = []
                            for i in cur_memory["middle_memory_index"]:
                                cur_id = i["memory_index"]
                                if cur_id not in cur_ids:
                                    cur_ids.append(cur_id)
                            # cur_ids = cur_memory["middle_memory_index"][0]["memory_index"]
                            self.brain.update_access_count_with_feed_back(
                                symbol=cur_symbol,
                                ids=cur_ids,
                                feedback=feedback[cur_symbol]["feedback"],
                            )
                        # update long memory if it is not none
                        if "long_memory_index" in cur_memory:
                            cur_ids = []
                            for i in cur_memory["long_memory_index"]:
                                cur_id = i["memory_index"]
                                if cur_id not in cur_ids:
                                    cur_ids.append(cur_id)
                            # cur_ids = cur_memory["long_memory_index"][0]["memory_index"]
                            self.brain.update_access_count_with_feed_back(
                                symbol=cur_symbol,
                                ids=cur_ids,
                                feedback=feedback[cur_symbol]["feedback"],
                            )
                        # update reflection memory if it is not none
                        if "reflection_memory_index" in cur_memory:
                            cur_ids = []
                            for i in cur_memory["reflection_memory_index"]:
                                cur_id = i["memory_index"]
                                if cur_id not in cur_ids:
                                    cur_ids.append(cur_id)
                            # 8
                            self.brain.update_access_count_with_feed_back(
                                symbol=cur_symbol,
                                ids=cur_ids,
                                feedback=feedback[cur_symbol]["feedback"],
                            )
        # 8. brain step
        self.brain.step()

    
    @timeout_decorator.timeout(300, timeout_exception=TimeoutError)
    def test_step(
        self, market_info: Union[market_info_type, terminated_market_info_type]
    ):
        cur_date = market_info[0]
        future_date = market_info[1]
        cur_price = market_info[2]
        # cur_eco = market_info[2]
        cur_filing_k = market_info[3]
        cur_filing_q = market_info[4]
        cur_news = market_info[5]
        # cur_record = market_info[6]

        # 1. make it aware the eco environment
        #self._handle_eco_belief(cur_eco)
        # 2. handling filings
        self._handling_filings(cur_date, cur_filing_q, cur_filing_k)
        # 3. handling news
        self._handling_news(cur_date, cur_news)
        # 4. update the price to portfolio
        self.portfolio.update_market_info(
            new_market_price_info=cur_price,
            cur_date=cur_date,
        )
        # 5. trade
        trade_result = self._trade(cur_date, cur_price)
        self.reflection_result_series_dict[cur_date] = trade_result
        for cur_symbol in trade_result:
            if len(trade_result[cur_symbol])!= 0:
                logger.info(
                            f"!!trading decision: {trade_result[cur_symbol]['investment_decision']} !! {cur_symbol}-Day {cur_date}\ninvestment reason: {trade_result[cur_symbol].get('summary_reason')}\n\n"
                        )
            else:
                logger.info(f"no decision")


        for i_symbol in trade_result:
            if len(trade_result[i_symbol]) != 0:
                if trade_result[i_symbol]["investment_decision"] == "buy":
                    trade_action = [{i_symbol: {"direction":1}}]
                elif trade_result[i_symbol]["investment_decision"] == "hold": 
                    trade_action = [{i_symbol: {"direction":0}}]
                else:
                    trade_action = [{i_symbol: {"direction":-1}}]
            
            else:
                trade_action = [{i_symbol: {"direction":0}}]
        
        
        self.portfolio.update_portfolio_from_actions(actions=trade_action)
        self.portfolio.update_portfolio_value()
        self.portfolio.update_portfolio_series()
        
        
        # 7. update the access counter if need to
        if feedback := self.portfolio.get_feedback_response():
            for cur_symbol in feedback:
                if feedback[cur_symbol]["feedback"] != 0:
                    # update short memory if it is not none
                    cur_date = feedback[cur_symbol]["date"]
                    if cur_symbol in self.reflection_result_series_dict[cur_date]:
                        cur_memory = self.reflection_result_series_dict[cur_date][
                            cur_symbol
                        ]
                        # if cur_memory != None:
                        if "short_memory_index" in cur_memory:
                            cur_ids = []
                            for i in cur_memory["short_memory_index"]:
                                cur_id = i["memory_index"]
                                if cur_id not in cur_ids:
                                    cur_ids.append(cur_id)
                            # cur_ids = cur_memory["short_memory_index"][0]["memory_index"]
                            self.brain.update_access_count_with_feed_back(
                                symbol=cur_symbol,
                                ids=cur_ids,
                                feedback=feedback[cur_symbol]["feedback"],
                            )
                        # update middle memory if it is not none
                        if "middle_memory_index" in cur_memory:
                            cur_ids = []
                            for i in cur_memory["middle_memory_index"]:
                                cur_id = i["memory_index"]
                                if cur_id not in cur_ids:
                                    cur_ids.append(cur_id)
                            # cur_ids = cur_memory["middle_memory_index"][0]["memory_index"]
                            self.brain.update_access_count_with_feed_back(
                                symbol=cur_symbol,
                                ids=cur_ids,
                                feedback=feedback[cur_symbol]["feedback"],
                            )
                        # update long memory if it is not none
                        if "long_memory_index" in cur_memory:
                            cur_ids = []
                            for i in cur_memory["long_memory_index"]:
                                cur_id = i["memory_index"]
                                if cur_id not in cur_ids:
                                    cur_ids.append(cur_id)
                            # cur_ids = cur_memory["long_memory_index"][0]["memory_index"]
                            self.brain.update_access_count_with_feed_back(
                                symbol=cur_symbol,
                                ids=cur_ids,
                                feedback=feedback[cur_symbol]["feedback"],
                            )
                        # update reflection memory if it is not none
                        if "reflection_memory_index" in cur_memory:
                            cur_ids = []
                            for i in cur_memory["reflection_memory_index"]:
                                cur_id = i["memory_index"]
                                if cur_id not in cur_ids:
                                    cur_ids.append(cur_id)
                            self.brain.update_access_count_with_feed_back(
                                symbol=cur_symbol,
                                ids=cur_ids,
                                feedback=feedback[cur_symbol]["feedback"],
                            )

        # # 9. brain step
        self.brain.step()


    def save_checkpoint(self, path: str, force: bool = False) -> None:
        path = os.path.join(path, self.agent_name)
        if os.path.exists(path):
            if force:
                shutil.rmtree(path)
            else:
                raise FileExistsError(f"Path {path} already exists")
        os.mkdir(path)
        os.mkdir(os.path.join(path, "brain"))
        state_dict = {
            "agent_name": self.agent_name,
            "character_string": self.character_string,
            "top_k": self.top_k,
            "notational_cash": self.notational_cash,
            "counter": self.counter,
            "chat_end_point_name": self.chat_end_point_name,
            "chat_end_point_config": self.chat_end_point_config,
            "portfolio": self.portfolio,
            "chat_end_point": self.chat_end_point,
            #"eco_belief": self.eco_belief,
            #"past_eco": self.past_eco,
            "reflection_result_series_dict": self.reflection_result_series_dict,  #
            "access_counter": self.access_counter,
        }
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)
        self.brain.save_checkpoint(os.path.join(path, "brain"), force=force)

    @classmethod
    def load_checkpoint(cls, path: str) -> "LLMAgent":
        # load state dict
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
        # load brain
        brain = BrainDB.load_checkpoint(os.path.join(path, "brain"))
        class_obj = cls(
            agent_name=state_dict["agent_name"],
            character_string=state_dict["character_string"],
            brain_db=brain,
            top_k=state_dict["top_k"],
            notational_cash=state_dict["notational_cash"],
            chat_end_point_name=state_dict["chat_end_point_name"],
            chat_end_point_config=state_dict["chat_end_point_config"],
        )
        class_obj.chat_end_point = state_dict["chat_end_point"]
        class_obj.portfolio = state_dict["portfolio"]
        class_obj.reflection_result_series_dict = state_dict[
            "reflection_result_series_dict"
        ]
        class_obj.access_counter = state_dict["access_counter"]
        class_obj.counter = state_dict["counter"]
        return class_obj
