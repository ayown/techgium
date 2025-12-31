import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


class AnomalyScorer:
    def __init__(self, model_path: str):
        self.model: IsolationForest = joblib.load(model_path)

    def score(self, embedding: np.ndarray) -> float:
        score = -self.model.score_samples(embedding.reshape(1, -1))[0]
        return float(score)


class RiskNormalizer:
    def __init__(self, s_min: float, s_max: float):
        self.s_min = s_min
        self.s_max = s_max

    def normalize(self, score: float) -> float:
        return 100 * (score - self.s_min) / (self.s_max - self.s_min + 1e-6)
        