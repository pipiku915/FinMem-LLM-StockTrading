import os
import shutil
import pickle
import logging
from datetime import date
from .run_type import RunMode
from .memorydb import BrainDB
from .portfolio import Portfolio
from abc import ABC, abstractmethod
from .chat import ChatOpenAICompatible
from .environment import market_info_type
from typing import Dict, Union, Any, List
from .reflection import trading_reflection
from transformers import AutoTokenizer


class TextTruncator:
    def __init__(self, tokenization_model_name):
        self.tokenization_model_name = tokenization_model_name
        self.token = os.environ.get("HF_TOKEN", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenization_model_name, auth_token=self.token
        )

    def _tokenize_cnt_texts(self, input_text):
        # Tokenize the text
        encoded_input = self.tokenizer(input_text)
        # Count the number of tokens
        num_tokens = len(encoded_input["input_ids"])
        return encoded_input, num_tokens

    def process_list_of_texts(self, list_of_texts, max_total_tokens=320):
        if "gpt" in self.tokenization_model_name:
            return list_of_texts

        truncated_list = []
        total_tokens = 0
        for text in list_of_texts:
            encoded_input, num_tokens = self._tokenize_cnt_texts(text)

            if total_tokens + num_tokens <= max_total_tokens:
                truncated_list.append(text)
                total_tokens += num_tokens
            else:
                # Calculate remaining tokens
                remaining_tokens = max_total_tokens - total_tokens
                if remaining_tokens > 0:
                    # Truncate the current text to fit the remaining token count
                    truncated_input_ids = encoded_input["input_ids"][:remaining_tokens]
                    truncated_text = self.tokenizer.decode(
                        truncated_input_ids, skip_special_tokens=True
                    )
                    truncated_list.append(truncated_text)
                    total_tokens += len(
                        truncated_input_ids
                    )  # Update total tokens with truncated token count
                break  # Stop processing further texts

        return truncated_list, total_tokens

    # for single text case
    def truncate_text(self, input_text, max_tokens):
        # Tokenize the text
        encoded_input, num_tokens = self.tokenize_cnt_texts(input_text)

        if len(encoded_input["input_ids"]) <= max_tokens:
            return input_text, len(encoded_input["input_ids"])
        encoded_input["input_ids"] = encoded_input["input_ids"][:max_tokens]
        encoded_input["attention_mask"] = encoded_input["attention_mask"][:max_tokens]
        # Optionally, decode the tokens back to a string
        output_text = self.tokenizer.decode(encoded_input["input_ids"])
        num_tokens = max_tokens
        return output_text, num_tokens


class Agent(ABC):
    @abstractmethod
    def from_config(self, config: Dict[str, Any]) -> "Agent":
        pass

    @abstractmethod
    def step(self) -> None:
        pass


# LLM Agent
class LLMAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        trading_symbol: str,
        character_string: str,
        brain_db: BrainDB,
        chat_config: Dict[str, Any],
        top_k: int = 1,
        look_back_window_size: int = 7,
    ):
        # base
        self.counter = 1
        self.top_k = top_k
        self.agent_name = agent_name
        self.trading_symbol = trading_symbol
        self.character_string = character_string
        self.look_back_window_size = look_back_window_size
        # logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(
            os.path.join(
                "data",
                "04_model_output_log",
                f"{self.trading_symbol}_run.log",
            ),
            mode="a",
        )
        file_handler.setFormatter(logging_formatter)
        self.logger.addHandler(file_handler)
        # brain db
        self.brain = brain_db
        # portfolio class
        self.portfolio = Portfolio(
            symbol=self.trading_symbol, lookback_window_size=self.look_back_window_size
        )
        self.chat_config_save = chat_config.copy()
        chat_config = chat_config.copy()
        end_point = chat_config["end_point"]
        model = chat_config["model"]
        system_message = chat_config["system_message"]# truncator
        self.model_name = chat_config["model"]
        self.max_token_short = chat_config.get("max_token_short", None)
        self.max_token_mid = chat_config.get("max_token_mid", None)
        self.max_token_long = chat_config.get("max_token_long", None)
        self.max_token_reflection = chat_config.get("max_token_reflection", None)
        del chat_config["end_point"]
        del chat_config["model"]
        del chat_config["system_message"]
        if self.max_token_short:
            self.truncator = TextTruncator(
                tokenization_model_name=chat_config["tokenization_model_name"]
            )
        self.guardrail_endpoint = ChatOpenAICompatible(
            end_point=end_point,
            model=model,
            system_message=system_message,
            other_parameters=chat_config,
        ).guardrail_endpoint()
        # records
        self.reflection_result_series_dict = {}
        self.access_counter = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMAgent":
        return cls(
            agent_name=config["general"]["agent_name"],
            trading_symbol=config["general"]["trading_symbol"],
            character_string=config["general"]["character_string"],
            brain_db=BrainDB.from_config(config=config),
            top_k=config["general"].get("top_k", 5),
            chat_config=config["chat"],
            look_back_window_size=config["general"]["look_back_window_size"],
        )

    def _handling_filings(self, cur_date: date, filing_q: str, filing_k: str) -> None:
        if filing_q:
            self.brain.add_memory_mid(
                symbol=self.trading_symbol, date=cur_date, text=filing_q
            )
        if filing_k:
            self.brain.add_memory_long(
                symbol=self.trading_symbol,
                date=cur_date,
                text=filing_k,
            )

    def _handling_news(self, cur_date: date, news: List[str]) -> None:
        if news != {}:
            self.brain.add_memory_short(
                symbol=self.trading_symbol, date=cur_date, text=news
            )
    
    def __query_info_for_reflection(self, run_mode: RunMode):
        # sourcery skip: low-code-quality
        self.logger.info(f"Symbol: {self.trading_symbol}\n")
        cur_short_queried, cur_short_memory_id = self.brain.query_short(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.trading_symbol,
        )
        if self.model_name.startswith("tgi"):
            cur_short_queried_truc, cur_short_num_tokens = (
                self.truncator.process_list_of_texts(
                    cur_short_queried, max_total_tokens=self.max_token_short
                )
            )
            cur_short_memory_id_truc = [
                cur_short_memory_id[k] for k in range(len(cur_short_queried_truc))
            ]
            for cur_id, cur_memory in zip(
                cur_short_memory_id_truc, cur_short_queried_truc
            ):
                self.logger.info(f"Top-k Short: {cur_id}: {cur_memory}\n")
            self.logger.info(f"Total tokens of Short Memory: {cur_short_num_tokens}\n")
        else:
            for cur_id, cur_memory in zip(cur_short_memory_id, cur_short_queried):
                self.logger.info(f"Top-k Short: {cur_id}: {cur_memory}\n")

        cur_mid_queried, cur_mid_memory_id = self.brain.query_mid(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.trading_symbol,
        )
        if self.model_name.startswith("tgi"):
            cur_mid_queried_truc, cur_mid_num_tokens = (
                self.truncator.process_list_of_texts(
                    cur_mid_queried, max_total_tokens=self.max_token_mid
                )
            )
            cur_mid_memory_id_truc = [
                cur_mid_memory_id[k] for k in range(len(cur_mid_queried_truc))
            ]
            for cur_id, cur_memory in zip(cur_mid_memory_id_truc, cur_mid_queried_truc):
                self.logger.info(f"Top-k Mid: {cur_id}: {cur_memory}\n")
            self.logger.info(f"Total tokens of Middle Memory: {cur_mid_num_tokens}\n")
        else:
            for cur_id, cur_memory in zip(cur_mid_memory_id, cur_mid_queried):
                self.logger.info(f"Top-k Mid: {cur_id}: {cur_memory}\n")

        cur_long_queried, cur_long_memory_id = self.brain.query_long(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.trading_symbol,
        )
        if self.model_name.startswith("tgi"):
            cur_long_queried_truc, cur_long_num_tokens = (
                self.truncator.process_list_of_texts(
                    cur_long_queried, max_total_tokens=self.max_token_long
                )
            )
            cur_long_memory_id_truc = [
                cur_long_memory_id[k] for k in range(len(cur_long_queried_truc))
            ]
            for cur_id, cur_memory in zip(
                cur_long_memory_id_truc, cur_long_queried_truc
            ):
                self.logger.info(f"Top-k Long: {cur_id}: {cur_memory}\n")
            self.logger.info(f"Total tokens of Long Memory: {cur_long_num_tokens}\n")
        else:
            for cur_id, cur_memory in zip(cur_long_memory_id, cur_long_queried):
                self.logger.info(f"Top-k Long: {cur_id}: {cur_memory}\n")

        (
            cur_reflection_queried,
            cur_reflection_memory_id,
        ) = self.brain.query_reflection(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.trading_symbol,
        )
        if self.model_name.startswith("tgi"):
            cur_reflection_queried_truc, cur_reflection_num_tokens = (
                self.truncator.process_list_of_texts(
                    cur_reflection_queried, max_total_tokens=self.max_token_reflection
                )
            )
            cur_reflection_memory_id_truc = [
                cur_reflection_memory_id[k]
                for k in range(len(cur_reflection_queried_truc))
            ]
            for cur_id, cur_memory in zip(
                cur_reflection_memory_id_truc, cur_reflection_queried_truc
            ):
                self.logger.info(f"Top-k Reflection: {cur_id}: {cur_memory}\n")
            self.logger.info(
                f"Total tokens of Reflection Memory: {cur_reflection_num_tokens}\n"
            )
        else:
            for cur_id, cur_memory in zip(
                cur_reflection_memory_id, cur_reflection_queried
            ):
                self.logger.info(f"Top-k Reflection: {cur_id}: {cur_memory}\n")

        if self.model_name.startswith("tgi"):
            cur_all_num_tokens = (
                cur_short_num_tokens
                + cur_mid_num_tokens
                + cur_long_num_tokens
                + cur_reflection_num_tokens
            )
            self.logger.info(f"Total tokens of **ALL** Memory: {cur_all_num_tokens}\n")

        # extra config in test
        if run_mode == RunMode.Test:
            cur_moment_ret = self.portfolio.get_moment(moment_window=3)
            cur_moment = (
                cur_moment_ret["moment"] if cur_moment_ret is not None else None
            )

        if run_mode == RunMode.Train:
            if self.model_name.startswith("tgi"):
                return (
                    cur_short_queried_truc,
                    cur_short_memory_id_truc,
                    cur_mid_queried_truc,
                    cur_mid_memory_id_truc,
                    cur_long_queried_truc,
                    cur_long_memory_id_truc,
                    cur_reflection_queried_truc,
                    cur_reflection_memory_id_truc,
                )
            else:
                return (
                    cur_short_queried,
                    cur_short_memory_id,
                    cur_mid_queried,
                    cur_mid_memory_id,
                    cur_long_queried,
                    cur_long_memory_id,
                    cur_reflection_queried,
                    cur_reflection_memory_id,
                )
        elif run_mode == RunMode.Test:
            if self.model_name.startswith("tgi"):
                return (
                    cur_short_queried_truc,
                    cur_short_memory_id_truc,
                    cur_mid_queried_truc,
                    cur_mid_memory_id_truc,
                    cur_long_queried_truc,
                    cur_long_memory_id_truc,
                    cur_reflection_queried_truc,
                    cur_reflection_memory_id_truc,
                    cur_moment,  # type: ignore
                )
            else:
                return (
                    cur_short_queried,
                    cur_short_memory_id,
                    cur_mid_queried,
                    cur_mid_memory_id,
                    cur_long_queried,
                    cur_long_memory_id,
                    cur_reflection_queried,
                    cur_reflection_memory_id,
                    cur_moment,  # type: ignore
                )

    def __reflection_on_record(
        self,
        cur_date: date,
        run_mode: RunMode,
        cur_record: Union[float, None] = None,
    ) -> Dict[str, Any]:
        if (run_mode == RunMode.Train) and (not cur_record):
            self.logger.info("No record\n")
            return {}
        # reflection
        if run_mode == RunMode.Train:
            (
                cur_short_queried,
                cur_short_memory_id,
                cur_mid_queried,
                cur_mid_memory_id,
                cur_long_queried,
                cur_long_memory_id,
                cur_reflection_queried,
                cur_reflection_memory_id,
            ) = self.__query_info_for_reflection(  # type: ignore
                run_mode=run_mode
            )
            reflection_result = trading_reflection(
                cur_date=cur_date,
                symbol=self.trading_symbol,
                run_mode=run_mode,
                endpoint_func=self.guardrail_endpoint,
                short_memory=cur_short_queried,
                short_memory_id=cur_short_memory_id,
                mid_memory=cur_mid_queried,
                mid_memory_id=cur_mid_memory_id,
                long_memory=cur_long_queried,
                long_memory_id=cur_long_memory_id,
                reflection_memory=cur_reflection_queried,
                reflection_memory_id=cur_reflection_memory_id,
                future_record=cur_record,  # type: ignore
                logger=self.logger,
            )
        elif run_mode == RunMode.Test:
            (
                cur_short_queried,
                cur_short_memory_id,
                cur_mid_queried,
                cur_mid_memory_id,
                cur_long_queried,
                cur_long_memory_id,
                cur_reflection_queried,
                cur_reflection_memory_id,
                cur_moment,
            ) = self.__query_info_for_reflection(  # type: ignore
                run_mode=run_mode
            )
            reflection_result = trading_reflection(
                cur_date=cur_date,
                symbol=self.trading_symbol,
                run_mode=run_mode,
                endpoint_func=self.guardrail_endpoint,
                short_memory=cur_short_queried,
                short_memory_id=cur_short_memory_id,
                mid_memory=cur_mid_queried,
                mid_memory_id=cur_mid_memory_id,
                long_memory=cur_long_queried,
                long_memory_id=cur_long_memory_id,
                reflection_memory=cur_reflection_queried,
                reflection_memory_id=cur_reflection_memory_id,
                momentum=cur_moment,
                logger=self.logger,
            )

        if (reflection_result is not {}) and ("summary_reason" in reflection_result):
            self.brain.add_memory_reflection(
                symbol=self.trading_symbol,
                date=cur_date,
                text=reflection_result["summary_reason"],
            )
        else:
            self.logger.info("No reflection result , not converged\n")
        return reflection_result

    def _reflect(
        self,
        cur_date: date,
        run_mode: RunMode,
        cur_record: Union[float, None] = None,
    ) -> None:
        if run_mode == RunMode.Train:
            reflection_result_cur_date = self.__reflection_on_record(
                cur_date=cur_date,
                cur_record=cur_record,
                run_mode=run_mode,
            )
        else:
            reflection_result_cur_date = self.__reflection_on_record(
                cur_date=cur_date, run_mode=run_mode
            )
        self.reflection_result_series_dict[cur_date] = reflection_result_cur_date
        if run_mode == RunMode.Train:
            self.logger.info(
                f"{self.trading_symbol}-Day {cur_date}\nreflection summary: {reflection_result_cur_date.get('summary_reason')}\n\n"
            )
        elif run_mode == RunMode.Test:
            if len(reflection_result_cur_date) != 0:
                self.logger.info(
                    f"!!trading decision: {reflection_result_cur_date['investment_decision']} !! {self.trading_symbol}-Day {cur_date}\ninvestment reason: {reflection_result_cur_date.get('summary_reason')}\n\n"
                )
            else:
                self.logger.info("no decision")

    def _construct_train_actions(self, cur_record: float) -> Dict[str, int]:
        cur_direction = 1 if cur_record > 0 else -1
        return {"direction": cur_direction, "quantity": 1}

    def _portfolio_step(self, cur_action: Dict[str, int]) -> None:
        self.portfolio.record_action(action=cur_action)  # type: ignore
        self.portfolio.update_portfolio_series()

    def __update_short_memory_access_counter(
        self,
        feedback: Dict[str, Union[int, date]],
        cur_memory: Dict[str, Any],
    ) -> None:
        if "short_memory_index" in cur_memory:
            self.__update_access_counter_sub(
                cur_memory=cur_memory,
                layer_index_name="short_memory_index",
                feedback=feedback,
            )

    def __update_mid_memory_access_counter(
        self,
        feedback: Dict[str, Union[int, date]],
        cur_memory: Dict[str, Any],
    ) -> None:
        if "middle_memory_index" in cur_memory:
            self.__update_access_counter_sub(
                cur_memory=cur_memory,
                layer_index_name="middle_memory_index",
                feedback=feedback,
            )

    def __update_long_memory_access_counter(
        self,
        feedback: Dict[str, Union[int, date]],
        cur_memory: Dict[str, Any],
    ) -> None:
        if "long_memory_index" in cur_memory:
            self.__update_access_counter_sub(
                cur_memory=cur_memory,
                layer_index_name="long_memory_index",
                feedback=feedback,
            )

    def __update_reflection_memory_access_counter(
        self,
        feedback: Dict[str, Union[int, date]],
        cur_memory: Dict[str, Any],
    ) -> None:
        if "reflection_memory_index" in cur_memory:
            self.__update_access_counter_sub(
                cur_memory=cur_memory,
                layer_index_name="reflection_memory_index",
                feedback=feedback,
            )

    def __update_access_counter_sub(self, cur_memory, layer_index_name, feedback):
        if cur_memory[layer_index_name] is not None:
            cur_ids = []
            for i in cur_memory[layer_index_name]:
                cur_id = i["memory_index"]
                if cur_id not in cur_ids:
                    cur_ids.append(cur_id)
            self.brain.update_access_count_with_feed_back(
                symbol=self.trading_symbol,
                ids=cur_ids,
                feedback=feedback["feedback"],
            )

    @staticmethod
    def __process_test_action(test_reflection_result: Dict[str, Any]) -> Dict[str, int]:
        if (
            test_reflection_result
            and test_reflection_result["investment_decision"] == "buy"
        ):
            return {"direction": 1}
        elif (
            len(test_reflection_result) != 0
            and test_reflection_result["investment_decision"] == "hold"
            or not test_reflection_result
        ):
            return {"direction": 0}
        else:
            return {"direction": -1}

    def _update_access_counter(self):
        if not (feedback := self.portfolio.get_feedback_response()):
            return
        if feedback["feedback"] != 0:
            # update short memory if it is not none
            cur_date = feedback["date"]
            cur_memory = self.reflection_result_series_dict[cur_date]
            self.__update_short_memory_access_counter(
                feedback=feedback, cur_memory=cur_memory
            )
            self.__update_mid_memory_access_counter(
                feedback=feedback, cur_memory=cur_memory
            )
            self.__update_long_memory_access_counter(
                feedback=feedback, cur_memory=cur_memory
            )
            self.__update_reflection_memory_access_counter(
                feedback=feedback, cur_memory=cur_memory
            )

    def step(
        self,
        market_info: market_info_type,
        run_mode: RunMode,
    ) -> None:
        # mode assertion
        if run_mode not in [RunMode.Train, RunMode.Test]:
            raise ValueError("run_mode should be either Train or Test")
        # market info
        cur_date = market_info[0]
        cur_price = market_info[1]
        cur_filing_k = market_info[2]
        cur_filing_q = market_info[3]
        cur_news = market_info[4]
        cur_record = market_info[5] if run_mode == RunMode.Train else None
        # 1. handling filings
        self._handling_filings(
            cur_date=cur_date, filing_q=cur_filing_q, filing_k=cur_filing_k  # type: ignore
        )
        # 2. handling news
        self._handling_news(cur_date=cur_date, news=cur_news)
        # 3. update the price to portfolio
        self.portfolio.update_market_info(
            new_market_price_info=cur_price,
            cur_date=cur_date,
        )
        self._reflect(
            cur_date=cur_date,
            run_mode=run_mode,
            cur_record=cur_record,
        )
        # 5. construct actions
        if run_mode == RunMode.Train:
            cur_action = self._construct_train_actions(
                cur_record=cur_record  # type: ignore
            )
        elif run_mode == RunMode.Test:
            cur_action = self.__process_test_action(
                test_reflection_result=self.reflection_result_series_dict[cur_date]
            )
        # 6. portfolio step
        self._portfolio_step(cur_action=cur_action)  # type: ignore
        # 7. update the access counter if need to
        self._update_access_counter()
        # 8. brain step
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
            "counter": self.counter,
            "trading_symbol": self.trading_symbol,
            "portfolio": self.portfolio,
            "chat_config": self.chat_config_save,
            "reflection_result_series_dict": self.reflection_result_series_dict,  #
            "access_counter": self.access_counter,
        }
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)
        self.brain.save_checkpoint(path=os.path.join(path, "brain"), force=force)

    @classmethod
    def load_checkpoint(cls, path: str) -> "LLMAgent":
        # load state dict
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
        # load brain
        brain = BrainDB.load_checkpoint(path=os.path.join(path, "brain"))
        class_obj = cls(
            agent_name=state_dict["agent_name"],
            trading_symbol=state_dict["trading_symbol"],
            character_string=state_dict["character_string"],
            brain_db=brain,
            top_k=state_dict["top_k"],
            chat_config=state_dict["chat_config"],
        )
        class_obj.portfolio = state_dict["portfolio"]
        class_obj.reflection_result_series_dict = state_dict[
            "reflection_result_series_dict"
        ]
        class_obj.access_counter = state_dict["access_counter"]
        class_obj.counter = state_dict["counter"]
        return class_obj
