import numpy as np
from typing import Tuple


class ExponentialDecay:
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
