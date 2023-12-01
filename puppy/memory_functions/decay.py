import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


# decay functions
class DecayFunctions(ABC):
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def __call__(
        self, recency_score: float, important_score: float, term: str
    ) -> Tuple[float, float]:
        pass


def get_decay_func(
    type: str, memory_layer: str, **kwargs: Dict[str, Any]
) -> DecayFunctions:
    match type:
        case "exponential":
            match memory_layer:
                case _:
                    return ExponentialDecay(**kwargs)
        case _:
            raise ValueError("Invalid decay function type")


class ExponentialDecay(DecayFunctions):
    def __init__(
        self,
        recency_factor: float = 10.0,
        importance_factor: float = 0.988,
    ):
        self.recency_factor = recency_factor
        self.importance_factor = importance_factor

    def __call__(
        self, important_score: float, delta: float
    ) -> Tuple[float, float, float]:
        delta += 1
        new_recency_score = np.exp(-(delta / self.recency_factor))
        new_important_score = important_score * self.importance_factor

        return new_recency_score, new_important_score, delta
