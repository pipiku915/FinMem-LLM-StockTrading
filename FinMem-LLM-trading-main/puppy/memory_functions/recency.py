from abc import ABC, abstractmethod


class RecencyScoreInitialization(ABC):
    @abstractmethod
    def __call__(self) -> float:
        pass


def get_recency_score_initialization_func(
    type: str, memory_layer: str
) -> RecencyScoreInitialization:
    match type:
        case "constant":
            match memory_layer:
                case _:
                    return R_ConstantInitialization()
        case _:
            raise ValueError("Invalid recency score initialization type")


class R_ConstantInitialization(RecencyScoreInitialization):
    def __call__(self) -> float:
        return 1.0
