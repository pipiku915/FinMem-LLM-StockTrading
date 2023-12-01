from typing import List, Callable
from pydantic import BaseModel, Field
from guardrails.validators import ValidChoices
import guardrails as gd
import logging
from typing import List, Callable, Dict, Union
import timeout_decorator
from datetime import date


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler("run.log", mode="a")
file_handler.setFormatter(logging_formatter)
logger.addHandler(file_handler)


def _memory_factory(memory_layer: str, id_list: List[int]) -> "BaseModel":
    class Memory(BaseModel):
        memory_index: int = Field(
            ...,
            description=f"Provide the piece of information related most to your investment decisions in the {memory_layer} memory?",
            validators=[ValidChoices(id_list)],
        )

    return Memory


def _train_execution_factory(
    short_id_list: List[int],
    mid_id_list: List[int],
    long_id_list: List[int],
    reflection_id_list: List[int],
):
    LongMem = _memory_factory("long-level", long_id_list)
    MidMem = _memory_factory("mid-level", mid_id_list)
    ShortMem = _memory_factory("short-level", short_id_list)
    ReflectionMem = _memory_factory("reflection-level", reflection_id_list)

    class InvestInfo(BaseModel):
        investment_decision: str = Field(
            ...,
            description="Given the information, please make an investment decision: buy the stock, sell, and hold the stock",
            validators=[ValidChoices(choices=["buy", "sell", "hold"])],
        )

        summary_reason: str = Field(
            ...,
            description="Given the information of text and the summary of the stock price movement. Please explain the reason why you make the investment decision.",
        )

        if short_id_list:
            short_memory_index: List[ShortMem] = Field(
                ...,
                description="The id of the short-term information",
            )

        if mid_id_list:
            middle_memory_index: List[MidMem] = Field(
                ...,
                description="The id of the mid-term information",
            )

        if long_id_list:
            long_memory_index: List[LongMem] = Field(
                ...,
                description="The id of the long-term information",
            )

        if reflection_id_list:
            reflection_memory_index: List[ReflectionMem] = Field(
                ...,
                description="The id of the reflection-term information.",
            )

    return InvestInfo


# @timeout_decorator.timeout(5, timeout_exception=TimeoutError)
def train_trade(
    cur_date: date,
    endpoint_func: Callable[[str], str],
    # eco_info: str,
    symbol: str,
    # feedback: int,
    moment: int,
    position: int,
    short_memory: Union[List[str], None] = None,
    short_memory_id: Union[List[int], None] = None,
    mid_memory: Union[List[str], None] = None,
    mid_memory_id: Union[List[int], None] = None,
    long_memory: Union[List[str], None] = None,
    long_memory_id: Union[List[int], None] = None,
    reflection_memory: Union[List[str], None] = None,
    reflection_memory_id: Union[List[int], None] = None,
):
    # remap the input: fix none and empty list case
    if (short_memory is None) or len(short_memory) == 0:
        short_memory = ["No short-term information.", "No short-term information."]
        short_memory_id = [-1, -1]
    elif len(short_memory) == 1:
        short_memory = [short_memory[0], short_memory[0]]
        short_memory_id = [short_memory_id[0], short_memory_id[0]]
    if (mid_memory is None) or len(mid_memory) == 0:
        mid_memory = ["No mid-term information.", "No mid-term information."]
        mid_memory_id = [-1, -1]
    elif len(mid_memory) == 1:
        mid_memory = [mid_memory[0], mid_memory[0]]
        mid_memory_id = [mid_memory_id[0], mid_memory_id[0]]
    if (long_memory is None) or len(long_memory) == 0:
        long_memory = ["No long-term information.", "No long-term information."]
        long_memory_id = [-1, -1]
    elif len(long_memory) == 1:
        long_memory = [long_memory[0], long_memory[0]]
        long_memory_id = [long_memory_id[0], long_memory_id[0]]
    if (reflection_memory is None) or len(reflection_memory) == 0:
        reflection_memory = [
            "No reflection-term information.",
            "No reflection-term information.",
        ]
        reflection_memory_id = [-1, -1]
    elif len(reflection_memory) == 1:
        reflection_memory = [reflection_memory[0], reflection_memory[0]]
        reflection_memory_id = [reflection_memory_id[0], reflection_memory_id[0]]

    # get output parser format
    response_model = _train_execution_factory(
        short_id_list=short_memory_id,
        mid_id_list=mid_memory_id,
        long_id_list=long_memory_id,
        reflection_id_list=reflection_memory_id,
    )

    # investment info
    investment_info = ""
    # symbol info
    investment_info += f"The ticker of the stock to be analyzed is {symbol} and the current date is {cur_date}"
    # eco info
    # investment_info += f"The current economy trend: {eco_info}\n\n"
    # short term
    if short_memory:
        investment_info += "The short-term information: \n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
        )

        investment_info += """For example, positive news about a company can lift investor sentiment, encouraging more buying activity which in turn can push stock prices higher. 
        Conversely, negative news can dampen investor sentiment, leading to selling pressure and a decrease in stock prices.
        News about competitors can also have a ripple effect on a company’s stock price. 
        For instance, if a competitor announces a groundbreaking new product, other companies in the industry might see their stock prices fall as investors anticipate a loss of market share."""

        investment_info += """The positive score, neutral score and negative score are sentiment score. 
        Sentiment score involves evaluating and interpreting subjective information in text data to understand the sentiments, opinions, or emotions expressed.
        The positive score, neutral score, and negative scores are ratios for proportions of text that fall in each category (so these should all add up to be 1).
        These are the most useful metrics if you want to analyze the context & presentation of how sentiment is conveyed or embedded in rhetoric for a given sentence.
        """

        investment_info += "\n\n"

    # mid term
    if mid_memory:
        investment_info += "The mid-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}.{i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
        )
        investment_info += "\n\n"

    # long term
    if long_memory:
        investment_info += "The long-term information: \n"
        investment_info += "\n".join(
            [f"{i[0]}.{i[1].strip()}" for i in zip(long_memory_id, long_memory)]
        )
        investment_info += "\n\n"

    # reflection term
    if reflection_memory:
        investment_info += "The reflection-term information: \n"
        investment_info += "\n".join(
            [f"{i[0]}.{i[1]}" for i in enumerate(reflection_memory, 1)]
        )
        investment_info += "\n\n"

    # feedback
    # if feedback:
    #     investment_info += '''The information below provides a summary of stock price fluctuations over the previous seven days.
    #     These are important indicators to reflect the influence of text information from the short-term information, the mid-term information, the long-term information, the reflection-term information.

    #     '''

    #     if feedback == 1:
    #         investment_info += "The cumulative return of past 7 days for this stock is positive."

    #     if feedback == -1:
    #         investment_info += "The cumulative return of past 7 days for this stock is negative."

    #     if feedback == 0:
    #         investment_info += "The cumulative return of past 7 days for this stock is zero."
    if moment:
        investment_info += """The information below provides a summary of stock price fluctuations over the previous few days, which is the "Momentum" of a stock.
        It reflects the trend of a stock.
        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
        
        """

        if moment == 1:
            investment_info += (
                "The cumulative return of past 3 days for this stock is positive."
            )

        if moment == -1:
            investment_info += (
                "The cumulative return of past 3 days for this stock is negative."
            )

        if moment == 0:
            investment_info += (
                "The cumulative return of past 3 days for this stock is zero."
            )

    # # position
    # if position:
    #     investment_info += f"the investor holds {position} share of stock{symbol}."

    # Please bear in mind that as a risk-seeking trader, positive cumulative returns have a greater influence on your investment decisions, while negative cumulative returns have a lesser impact.
    # Try to make a decisions(in buy or sell) by utilizing all information as much as possible.
    # prompt -- change the prompt corresponds to different character design.
    # When cumulative return is positive, you are a risk-seeking investor, positive information have a greater influence on your investment decisions, while negative information have a lesser impact. 
    # But when cumulative return is negative, you are a risk-averse investor, negative information have a greater influence on your investment decisions, while positive information have a lesser impact. 
    # When cumulative return is positive, you are a risk-seeking investor. 
    # But when cumulative return is negative, you are a risk-averse investor.  
    # Please bear in mind you can switch between investment tendencies, which are risk-seeking and risk-averse. 
    # When cumulative return is positive, you are a risk-seeking investor, positive information have a greater influence on your investment decisions, while negative information have a lesser impact. 
    # But when cumulative return is negative, you are a risk-averse investor, negative information have a greater influence on your investment decisions, while positive information have a lesser impact. 
    # When it is really hard to make a 'buy'-or-'sell' decision, you could go with hold option.
    # Please bear in mind you can switch between investment tendencies, which are risk-seeking and risk-averse. 
    # When cumulative return is positive, you are a risk-seeking investor.
    ###for character experiments:
    
    _train_prompt = """ Given the information, can you make an investment decision?
    please consider the short-term information, the mid-term information, the long-term information, the reflection-term information.
    please consider the momentum of the historical stock price.
    Please bear in mind you can switch between investment tendencies, which are risk-seeking and risk-averse.
    When cumulative return is positive, you are a risk-seeking investor.
    But when cumulative return is negative, you are a risk-averse investor. 
    please consider how much share of the stock the investor holds now.   
    You should provide exactly one of the following investment decisions: buy or sell.
    When it is really hard to make a 'buy'-or-'sell' decision, you could go with hold option. You also need to provide the id of the information to support your decision.

    ${investment_info}

    ${gr.complete_json_suffix_v2}    
    """

    # guardrails
    guard = gd.Guard.from_pydantic(output_class=response_model, prompt=_train_prompt)
    _, validated_output = guard(
        endpoint_func,
        prompt_params={"investment_info": investment_info},
    )

    # print(guard.state.most_recent_call.tree)

    if (validated_output is None) or (not isinstance(validated_output, dict)):
        logger.info(f"reflection failed for {symbol}")
        return {}

    # delete placeholder information
    if "reflection_memory_index" in validated_output.keys():
        if (validated_output["reflection_memory_index"]) and (
            validated_output["reflection_memory_index"][0]["memory_index"] == -1
        ):
            del validated_output["reflection_memory_index"]

    if "long_memory_index" in validated_output.keys():
        if (validated_output["long_memory_index"]) and (
            validated_output["long_memory_index"][0]["memory_index"] == -1
        ):
            del validated_output["long_memory_index"]

    if "middle_memory_index" in validated_output.keys():
        if (validated_output["middle_memory_index"]) and (
            validated_output["middle_memory_index"][0]["memory_index"] == -1
        ):
            del validated_output["middle_memory_index"]

    if "short_memory_index" in validated_output.keys():
        if (validated_output["short_memory_index"]) and (
            validated_output["short_memory_index"][0]["memory_index"] == -1
        ):
            del validated_output["short_memory_index"]
    return validated_output
