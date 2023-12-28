import os
import openai
import requests
from abc import ABC, abstractmethod
from typing import Callable, Union, List, Dict, Any

openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatBase(ABC):
    @abstractmethod  # type: ignore
    def __call__(self) -> None:
        pass

    @abstractmethod  # type: ignore
    def guardrail_endpoint(self) -> None:
        pass


class ChatTogetherEndpoint(ChatBase):
    def __init__(
        self,
        api_key: Union[str, None] = None,
        model: str = "togethercomputer/llama-2-70b-chat",
        max_tokens: int = 1000,
        stop: Union[List[str], None] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        request_timeout: int = 600,
    ) -> None:
        # stop
        if stop is None:
            self.stop = ["<human>"]
        # params
        self.api_key = os.getenv("TOGETHER_API_KEY") if api_key is None else api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.request_timeout = request_timeout
        # api related
        self.end_point = "https://api.together.xyz/inference"
        # transaction
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def parse_response(self, response: requests.Response) -> str:
        return response.json()["output"]["choices"][0]["text"]

    def __call__(self, text: str, **kwargs) -> str:
        transaction_payload = {
            "model": self.model,
            "prompt": f"<human>: {text}\n<bot>:",
            "stop": [
                "<human>",
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

        response = requests.post(
            self.end_point,
            json=transaction_payload,
            headers=self.headers,
            timeout=self.request_timeout,
        )
        response.raise_for_status()

        return self.parse_response(response)

    def guardrail_endpoint(self) -> Callable[[str], str]:
        return self


class ChatOpenAIEndPoint:
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature

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
            return openai.ChatCompletion.create(  # type: ignore
                model=model_name, messages=input_str, temperature=temperature
            )["choices"][
                0
            ][  # type: ignore
                "message"
            ][  # type: ignore
                "content"
            ]  # type: ignore

        return end_point


def get_chat_end_points(
    end_point_type: str, chat_config: Dict[str, Any]
) -> Union[ChatOpenAIEndPoint, ChatTogetherEndpoint]:
    match end_point_type:
        case "openai":
            return ChatOpenAIEndPoint(
                chat_config["model_name"],
                chat_config["temperature"],
            )
        case "together":
            return ChatTogetherEndpoint(
                chat_config.get("api_key"),
                chat_config["model"],
                chat_config["max_tokens"],
                chat_config.get("stop"),
                chat_config["temperature"],
                chat_config["top_p"],
                chat_config["top_k"],
                chat_config["repetition_penalty"],
                chat_config["request_timeout"],
            )
        case _:
            raise NotImplementedError(
                f"Chat end point type {end_point_type} is not implemented."
            )
