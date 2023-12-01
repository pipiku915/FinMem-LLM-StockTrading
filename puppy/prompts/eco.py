import math
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    SystemMessage,
    HumanMessagePromptTemplate,
)
from langchain.schema.messages import BaseMessage
from typing import Dict, List
from pydantic import BaseModel, validator


class EcoObservation(BaseModel):
    t10yff: float
    permit: float
    m2sl: float
    ismnmdi: float
    icsa: float
    cci: float
    awhaeman: float

    @validator("*", pre=True, always=True)
    def check_nan(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError("NaN is not allowed.")
        return v


def eco_prompt(eco_obs: Dict[str, float]) -> List[BaseMessage]:
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a macro economist in Two Sigma, you need to take into all the economic indicators and make a prediction on the economy and give investment advice."
            ),
            HumanMessagePromptTemplate.from_template(
                "• AWHAEMAN (Average Weekly Hours): Currently at {awhaeman_val}. Indicates the overall workload in the economy."
            ),
            HumanMessagePromptTemplate.from_template(
                "• T10YFF (Treasury Spread): Currently at {t10yff_val}. Reflects the yield curve and economic sentiment."
            ),
            HumanMessagePromptTemplate.from_template(
                "• PERMIT (Building Permits): Currently at {permit_val}. Indicates construction activity level."
            ),
            HumanMessagePromptTemplate.from_template(
                "• M2SL (Money Supply): Currently at {m2sl_val}. Provides insight into market liquidity."
            ),
            HumanMessagePromptTemplate.from_template(
                "• ISMNMDI (Supplier Deliveries Index): Currently at {ismnmdi_val}. Measures supply chain efficiency."
            ),
            HumanMessagePromptTemplate.from_template(
                "• ICSA (Initial Unemployment Claims): Currently at {icsa_val}. Shows the unemployment level."
            ),
            HumanMessagePromptTemplate.from_template(
                "• CCI (Consumer Confidence): Currently at {cci_val}. Reflects consumer sentiment."
            ),
            SystemMessage(
                content="Please give specific investment advice based on the above indicators. don't output the indicators."
            ),
        ]
    )
    return template.format_messages(
        awhaeman_val=eco_obs["awhaeman"],
        t10yff_val=eco_obs["t10yff"],
        permit_val=eco_obs["permit"],
        m2sl_val=eco_obs["m2sl"],
        ismnmdi_val=eco_obs["ismnmdi"],
        icsa_val=eco_obs["icsa"],
        cci_val=eco_obs["cci"],
    )
