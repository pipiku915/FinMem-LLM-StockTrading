from abc import ABC, abstractmethod


# access count importance score change
class ImportanceScoreChangeAccessCounter(ABC):
    @abstractmethod
    def __call__(self, access_counter: int, importance_score: float) -> float:
        pass


def get_access_counter_change_func(
    type: str, memory_layer: str
) -> ImportanceScoreChangeAccessCounter:
    match type:
        case "linear":
            match memory_layer:
                case _:
                    return LinearImportanceScoreChange()
        case _:
            raise ValueError("Invalid access counter change function type")


class LinearImportanceScoreChange(ImportanceScoreChangeAccessCounter):
    def __call__(self, access_counter: int, importance_score: float) -> float:
        importance_score += access_counter * 5
        return importance_score
