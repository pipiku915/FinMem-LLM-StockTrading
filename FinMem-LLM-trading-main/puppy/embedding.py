import os
import asyncio
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class EmbeddingFunction(ABC):
    @abstractmethod
    def __call__(self, text: Union[List[str], str]) -> np.array:
        """
        Abstract method for performing embedding on a list of text.

        Args:
            text (List[str]): A list of text to be embedded.

        Returns:
            np.array: The embedding of the input text.

        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Abstract method for getting the dimension of the embedding.

        Returns:
            int: The dimension of the embedding.

        """
        pass


class OpenAILongerThanContextEmb(EmbeddingFunction):
    """
    Embedding function with openai as embedding backend.
    If the input is larger than the context size, the input is split into chunks of size `chunk_size` and embedded separately.
    The final embedding is the average of the embeddings of the chunks.
    Details see: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    """

    def __init__(
        self,
        openai_api_key: Union[str, None] = None,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 5000,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the Embedding object.

        Args:
            openai_api_key (str): The API key for OpenAI.
            embedding_model (str, optional): The model to use for embedding. Defaults to "text-embedding-ada-002".
            chunk_size (int, optional): The maximum number of token to send to openai embedding model at one time. Defaults to 5000.
            verbose (bool, optional): Whether to show progress bar during embedding. Defaults to False.

        Returns:
            None
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.emb_model = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            chunk_size=chunk_size,
            show_progress_bar=verbose,
        )

    async def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        Asynchronously performs embedding on a list of text.

        This method calls the `aembed_documents` method of the `emb_model` object to embed the input text.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            List[List[float]]: The embeddings of the input text as a list of lists of floats.

        """
        if isinstance(text, str):
            text = [text]
        return await self.emb_model.aembed_documents(texts=text, chunk_size=None)

    def __call__(self, text: Union[List[str], str]) -> np.array:
        """
        Performs embedding on a list of text.

        This method calls the `_emb` method to asynchronously embed the input text using the `emb_model` object.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            np.array: The embedding of the input text as a NumPy array.

        """
        return np.array(asyncio.run(self._emb(text))).astype("float32")

    def get_embedding_dimension(self):
        """
        Returns the dimension of the embedding.

        This method checks the value of `self.emb_model.model` and returns the corresponding embedding dimension. If the model is not implemented, a `NotImplementedError` is raised.

        Args:
            self: The instance of the class.

        Returns:
            int: The dimension of the embedding.

        Raises:
            NotImplementedError: Raised when the embedding dimension for the specified model is not implemented.

        """
        match self.emb_model.model:
            case "text-embedding-ada-002":
                return 1536
            case _:
                raise NotImplementedError(
                    f"Embedding dimension for model {self.emb_model.model} not implemented"
                )


class OpenAISummarizationEmb(EmbeddingFunction):
    # TODO: implement the summarization embedding function
    # TODO: see https://python.langchain.com/docs/use_cases/summarization
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, text: List[str]) -> np.array:
        pass

    def get_embedding_dimension(self) -> int:
        pass


def get_embedding_func(type: str, config: Dict[str, Any]) -> EmbeddingFunction:
    match type:
        case "open-ai-longer-than-context":
            return OpenAILongerThanContextEmb(**config)
        case "open-ai-summarization":
            raise NotImplementedError("OpenAI summarization embedding not implemented")
        case _:
            raise NotImplementedError(f"Embedding function type {type} not implemented")
