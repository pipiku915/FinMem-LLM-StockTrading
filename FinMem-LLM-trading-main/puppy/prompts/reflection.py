import guardrails as gd
import logging
from typing import List, Callable, Dict, Union
from pydantic import BaseModel, Field
from guardrails.validators import ValidChoices
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
            description=f"Provide the piece of information  related the most to the investment decisions from mainstream sources such as the investment suggestions major fund firms such as ARK, Two Sigma, Bridgewater Associates, etc. in the {memory_layer} memory?",
            validators=[ValidChoices(id_list, on_fail="reask")],
        )

    return Memory


# train
def _train_reflection_factory(
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
        if reflection_id_list:
            reflection_memory_index: List[ReflectionMem] = Field(
                ...,
                description="The id of the reflection-term information.",
            )
        if long_id_list:
            long_memory_index: List[LongMem] = Field(
                ...,
                description="The id of the long-term information.",
            )

        if mid_id_list:
            middle_memory_index: List[MidMem] = Field(
                ...,
                description="The id of the mid-term information.",
            )

        if short_id_list:
            short_memory_index: List[ShortMem] = Field(
                ...,
                description="The id of the short-term information.",
            )

        summary_reason: str = Field(
            ...,
            description="Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you? ",
        )

    return InvestInfo


def train_reflection(
    cur_date: date,
    endpoint_func: Callable[[str], str],
    # eco_info: str,
    symbol: str,
    future_date: date,
    future_record: Dict[str, float | str],
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

    # pydantic reflection model
    response_model = _train_reflection_factory(
        short_id_list=short_memory_id,
        mid_id_list=mid_memory_id,
        long_id_list=long_memory_id,
        reflection_id_list=reflection_memory_id,
    )

    # investment info
    investment_info = f"The current date is {cur_date}. "
    # ark record
    investment_info += f"""Here are the observed financial market facts: for {symbol}, the price difference between the next trading day and the current trading day is: {future_record}\n\n"""

    # eco info
    # investment_info += f"The current economy trend: {eco_info}\n\n"
    # short term
    if short_memory:
        investment_info += "The short-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
        )
        investment_info += "\n\n"
    # mid term
    if mid_memory:
        investment_info += "The mid-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
        )
        investment_info += "\n\n"
    # long term
    if long_memory:
        investment_info += "The long-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
        )
        investment_info += "\n\n"
    # reflection term
    if reflection_memory:
        investment_info += "The reflection-term information:\n"
        investment_info += "\n".join(
            [f"{i[0]}. {i[1]}" for i in enumerate(reflection_memory, 1)]
        )
        investment_info += "\n\n"

    # prompt
    _train_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this?
    Your should provide a summary information and the id of the information to support your summary.

    ${investment_info}

    ${gr.complete_json_suffix_v2}
    """

    # guardrails
    guard = gd.Guard.from_pydantic(
        output_class=response_model, prompt=_train_prompt, num_reasks=2
    )
    _, validated_output = guard(
        endpoint_func,
        prompt_params={"investment_info": investment_info},
    )

    if (validated_output is None) or (not isinstance(validated_output, dict)):
        logger.info(f"reflection failed for {symbol}")
        return {}

    # delete placeholder information
    if (validated_output["reflection_memory_index"]) and (
        validated_output["reflection_memory_index"][0]["memory_index"] == -1
    ):
        del validated_output["reflection_memory_index"]
    if (validated_output["long_memory_index"]) and (
        validated_output["long_memory_index"][0]["memory_index"] == -1
    ):
        del validated_output["long_memory_index"]
    if (validated_output["middle_memory_index"]) and (
        validated_output["middle_memory_index"][0]["memory_index"] == -1
    ):
        del validated_output["middle_memory_index"]
    if (validated_output["short_memory_index"]) and (
        validated_output["short_memory_index"][0]["memory_index"] == -1
    ):
        del validated_output["short_memory_index"]
    return validated_output
