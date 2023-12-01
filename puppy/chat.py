import os
import openai
import asyncio
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema.messages import BaseMessage
from abc import ABC, abstractclassmethod
from typing import List, Union, Dict, Any, Callable

openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatBase(ABC):
    @abstractclassmethod
    def __call__(
        self, messages: Union[List[BaseMessage], List[List[BaseMessage]]]
    ) -> List[str]:
        pass


class ChatOpenAIEndPoint(ChatBase):
    def __init__(
        self,
        openai_api_key: Union[None, str],
        model_name: str = "gpt-4",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.chat_endpoint = ChatOpenAI(
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model_name=model_name,
            temperature=temperature,
        )

    def __call__(
        self, messages: Union[List[BaseMessage], List[List[BaseMessage]]]
    ) -> List[str]:
        if isinstance(messages[0], BaseMessage):
            messages = [messages]
        results = asyncio.run(self.chat_endpoint.agenerate(messages))
        return [i.generations[0][0].text for i in results.flatten()]

    def guardrail_endpoint(
        self, system_message: str = "You are a helpful assistant."
    ) -> Callable[[str], str]:
        model_name = self.model_name
        temperature = self.temperature

        def end_point(input: str, **kwargs) -> str:
            input_str = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{input}"},
            ]
            return openai.ChatCompletion.create(
                model=model_name, messages=input_str, temperature=temperature
            )["choices"][0]["message"]["content"]

        return end_point


def get_chat_end_points(end_point_type: str, chat_config: Dict[str, Any]) -> ChatBase:
    match end_point_type:
        case "openai":
            return ChatOpenAIEndPoint(
                chat_config.get("openai_api_key"),
                chat_config["model_name"],
                chat_config["temperature"],
            )
        case _:
            raise NotImplementedError(
                f"Chat end point type {end_point_type} is not implemented."
            )
