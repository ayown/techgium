import torch
import torch.nn as nn
import numpy as np


class AttentionFusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_systems: int = 2,
    ):
        super().__init__()
        
        self.num_systems = num_systems

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        risk_in_dim = embedding_dim * 3  # fused + |diff| + elementwise product
        self.risk_head = nn.Sequential(
            nn.Linear(risk_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings: torch.Tensor):
        """
        embeddings: (num_systems, embedding_dim)
        """

        if embeddings.dim() != 2:
            raise ValueError(f"Expected embeddings with shape (num_systems, embedding_dim). Got {tuple(embeddings.shape)}")

        embeddings = embeddings.float()

        # Global scaling for stability while keeping cross-system scale differences.
        raw = embeddings
        raw_mean = raw.mean()
        raw_std = raw.std().clamp_min(1e-6)
        embeddings = (raw - raw_mean) / raw_std

        attn_scores = []
        for i in range(self.num_systems):
            score = self.attention(embeddings[i])
            attn_scores.append(score)

        attn_scores = torch.stack(attn_scores).squeeze()
        weights = torch.softmax(attn_scores, dim=0)

        fused_embedding = torch.sum(weights.unsqueeze(1) * embeddings, dim=0)

        # Pairwise features on globally scaled raw embeddings so distances remain informative.
        e0 = embeddings[0]
        e1 = embeddings[1]
        diff = torch.abs(e0 - e1)
        prod = e0 * e1
        raw_fused = torch.sum(weights.unsqueeze(1) * embeddings, dim=0)
        risk_features = torch.cat([raw_fused, diff, prod], dim=0)

        risk = self.risk_head(risk_features)

        return {
            "system_weights": weights.detach().cpu().numpy(),
            "fused_embedding": fused_embedding,
            "systemic_risk": risk.squeeze()
        }
        