import os
import httpx
import requests
import json
from abc import ABC
from typing import Callable, Union, Dict, Any, Union


def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt


class LongerThanContextError(Exception):
    pass


class ChatOpenAICompatible(ABC):
    def __init__(
        self,
        end_point: str,
        model="tgi",
        system_message: str = "You are a helpful assistant.",
        other_parameters: Union[Dict[str, Any], None] = None,
    ):
        api_key = os.environ.get("OPENAI_API_KEY", "-")
        self.end_point = end_point
        self.model = model
        self.system_message = system_message
        if self.model.startswith("tgi"):
            self.headers = {"Content-Type": "application/json"}
        else:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            self.other_parameters = {} if other_parameters is None else other_parameters

    def parse_response(self, response: httpx.Response) -> str:
        if self.model.startswith("gpt"):
            response_out = response.json()
            return response_out["choices"][0]["message"]["content"]
        elif self.model.startswith("tgi"):
            response_out = response.json()
            return response_out["generated_text"]
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")

    def guardrail_endpoint(self) -> Callable:
        def end_point(input: str, **kwargs) -> str:
            input_str = [
                {"role": "system", "content": f"{self.system_message}"},
                {"role": "user", "content": f"{input}"},
            ]

            if self.model.startswith("tgi"):
                llama_input_str = build_llama2_prompt(input_str)
                # print(llama_input_str)

                payload = {
                    "inputs": llama_input_str,
                    "parameters": {
                        "do_sample": True,
                        "top_p": 0.6,
                        "temperature": 0.8,
                        "top_k": 50,
                        "max_new_tokens": 512,
                        "repetition_penalty": 1.03,
                        "stop": ["</s>"],
                    },
                }

                # payload = json.dumps(payload)
                response = httpx.post(
                    self.end_point, headers=self.headers, json=payload, timeout=600.0  # type: ignore
                )
            else:
                payload = {
                    "model": self.model,  # or another model like "gpt-4.0-turbo"
                    "messages": input_str,
                }
                payload.update(self.other_parameters)
                payload = json.dumps(payload)

                response = httpx.post(
                    self.end_point, headers=self.headers, data=payload, timeout=600.0  # type: ignore
                )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if (response.status_code == 422) and (
                    "must have less than" in response.text
                ):
                    raise LongerThanContextError
                else:
                    raise e

            return self.parse_response(response)

        return end_point
