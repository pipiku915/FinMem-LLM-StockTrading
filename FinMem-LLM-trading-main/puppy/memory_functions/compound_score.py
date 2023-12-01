from abc import ABC, abstractmethod


class CompoundScoreCalculation(ABC):
    @abstractmethod
    def recency_and_importance_score(
        self, recency_score: float, importance_score: float
    ) -> float:
        pass

    @abstractmethod
    def merge_score(
        self, similarity_score: float, recency_and_importance: float
    ) -> float:
        pass


def get_compound_score_calculation_func(
    type: str, memory_layer: str
) -> CompoundScoreCalculation:
    match type:
        case "linear":
            match memory_layer:
                case _:
                    return LinearCompoundScore()
        case _:
            raise ValueError("Invalid compound score calculation type")


class LinearCompoundScore(CompoundScoreCalculation):
    def recency_and_importance_score(
        self, recency_score: float, importance_score: float
    ) -> float:
        importance_score = min(importance_score, 100)
        return recency_score + importance_score / 100

    def merge_score(
        self, similarity_score: float, recency_and_importance: float
    ) -> float:
        return similarity_score + recency_and_importance

# importance changed
