class LinearCompoundScore:
    def recency_and_importance_score(
        self, recency_score: float, importance_score: float
    ) -> float:
        importance_score = min(importance_score, 100)
        return recency_score + importance_score / 100

    def merge_score(
        self, similarity_score: float, recency_and_importance: float
    ) -> float:
        return similarity_score + recency_and_importance
