import numpy as np
from models.base import BaseRiskEngine
from ml.inference.embed import TimeSeriesEmbedder
from ml.risk.anomaly_scorer import AnomalyScorer, RiskNormalizer


class CardioRiskEngine(BaseRiskEngine):
    def __init__(
        self,
        hr_spo2_timeseries: np.ndarray,
        age: int,
        signal_quality: float = 1.0,
    ):
        super().__init__(system_name="cardiovascular")

        self.timeseries = hr_spo2_timeseries
        self.age = age
        self.signal_quality = max(0.0, min(signal_quality, 1.0))

        self.embedder = TimeSeriesEmbedder("ml/train/encoder.pt")
        self.scorer = AnomalyScorer("ml/risk/anomaly_model.joblib")

        s_min, s_max = np.load("ml/risk/risk_bounds.npy")
        self.normalizer = RiskNormalizer(s_min, s_max)

        self.flags = []
        self.explanations = []

    def run(self):
        embedding = self.embedder.embed(self.timeseries).squeeze()

        raw_score = self.scorer.score(embedding)
        ml_risk = self.normalizer.normalize(raw_score)

        if ml_risk > 85:
            self.flags.append("abnormal_cardiovascular_pattern")
            self.explanations.append(
                "Detected deviation from learned cardiovascular physiological patterns."
            )

        if self.age >= 65:
            ml_risk *= 1.1
            self.explanations.append(
                "Age-adjusted cardiovascular risk."
            )

        if self.signal_quality < 0.6:
            ml_risk *= 0.8
            self.explanations.append(
                "Reduced confidence due to suboptimal signal quality."
            )

        ml_risk = min(100, max(0, ml_risk))

        return {
            "system": self.system,
            "risk_score": round(ml_risk, 2),
            "risk_level": self.classify(ml_risk),
            "confidence": round(self.signal_quality, 2),
            "flags": self.flags,
            "explanation": " ".join(self.explanations),
        }

    def classify(self, score: float):
        if score >= 70:
            return "RED"
        elif score >= 35:
            return "YELLOW"
        return "GREEN"
        