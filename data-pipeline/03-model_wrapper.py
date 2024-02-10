from abc import ABC, abstractmethod
import openai
from langchain_together import Together
from tenacity import retry, stop_after_attempt, wait_fixed

MAX_ATTEMPTS = 5
WAIT_TIME = 10

class Model_Wrapper(ABC):
    @retry(stop=stop_after_attempt(MAX_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
    def summarize(self, text, summary_token_size = 200):
        return self._summarize(text, summary_token_size)
    
    @abstractmethod
    def _summarize(self, text, summary_token_size):
        pass
    
class Chatgpt(Model_Wrapper):
    def __init__(self, key, model_name):
        self.__key = key
        openai.api_key = self.__key
        self.model_name = model_name
    
    def _summarize(self, text, summary_token_size):
        prompt = f"Summarize the following news within {summary_token_size} tokens:\n{text}\nSummary:"
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        summary = response['choices'][0]['message']['content']
        return summary
    
class Together(Model_Wrapper):
    def __init__(self, key, model_name):
        self.__key = key
        self.model_name = model_name
        
        self.llm = Together(
            model=self.model_name,
            temperature=0.7,
            max_tokens=200,
            top_k=1,
            together_api_key=self.__key,
        )
        
    def _summarize(self, text, summary_token_size):
        prompt = f"Summarize the following news within {summary_token_size} tokens:\n{text}\nSummary:"
        return self.llm.invoke(prompt)
    
class Dummy(Model_Wrapper):
    '''
    For test only
    '''
    import random
    import time
    def __init__(self, *args, **kwargs) -> None:
        print("Initializing a dummy model!")
    
    def _summarize(self, text, summary_token_size):
        self.time.sleep(self.random.randint(1, 5))
        if self.random.random() < 0.1:
            print("attempt", summary_token_size)
            raise
        else:
            return text[:summary_token_size]
    
class Model_Factory:
    registered_model_class = ("chatgpt", 'together', 'dummy')
    @classmethod
    def create_model(cls, model_class:str, key:str = None, model_name:str = None, *args, **kwargs)->(Chatgpt | Together):
        assert model_class in cls.registered_model_class, f"Invalid model class name: choose one from {cls.registered_model_class}"
        match model_class:
            case "chatgpt":
                return Chatgpt(key, model_name)
            case "together":
                return Together(key, model_name)
            case "dummy":
                return Dummy()
            case _:
                raise

    

    
    