import numpy as np
from abc import ABC, abstractmethod


class ImportanceScoreInitialization(ABC):
    @abstractmethod
    def __call__(self) -> float:
        pass


def get_importance_score_initialization_func(
    type: str, memory_layer: str
) -> ImportanceScoreInitialization:
    match type:
        case "sample":
            match memory_layer:
                case "short":
                    return I_SampleInitialization_Short()
                case "mid":
                    return I_SampleInitialization_Mid()
                case "long":
                    return I_SampleInitialization_Long()
                case "reflection":
                    return I_SampleInitialization_Long()
                case _:
                    raise ValueError("Invalid memory layer type")
        case _:
            raise ValueError("Invalid importance score initialization type")


class I_SampleInitialization_Short(ImportanceScoreInitialization):
    def __call__(self) -> float:
        probabilities = [0.5, 0.45, 0.05]
        scores = [50.0, 70.0, 90.0]
        return np.random.choice(scores, p=probabilities)


class I_SampleInitialization_Mid(ImportanceScoreInitialization):
    def __call__(self) -> float:
        probabilities = [0.05, 0.8, 0.15]
        scores = [40.0, 60.0, 80.0]
        return np.random.choice(scores, p=probabilities)


class I_SampleInitialization_Long(ImportanceScoreInitialization):
    def __call__(self) -> float:
        probabilities = [0.05, 0.15, 0.8]
        scores = [40.0, 60.0, 80.0]
        return np.random.choice(scores, p=probabilities)
