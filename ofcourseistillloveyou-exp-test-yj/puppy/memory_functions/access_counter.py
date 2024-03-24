class LinearImportanceScoreChange:
    def __call__(self, access_counter: int, importance_score: float) -> float:
        importance_score += access_counter * 5
        return importance_score
