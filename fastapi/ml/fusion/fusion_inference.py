import torch
import numpy as np
import os
from ml.fusion.attention_fusion import AttentionFusion


class FusionEngine:
	def __init__(self):
		self.model = AttentionFusion()
		model_path = os.path.join(os.path.dirname(__file__), "fusion_model.pt")
		self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
		self.model.eval()

	def run(self, embeddings):
		emb = np.array(embeddings, dtype=np.float32)

		with torch.no_grad():
			out = self.model(torch.tensor(emb))
			risk = torch.sigmoid(out["systemic_risk"])

		return {
			"systemic_risk": float(risk.item() * 100),
			"attention_weights": out["system_weights"].tolist()
		}
		