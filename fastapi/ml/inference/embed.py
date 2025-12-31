import torch
from ml.models.ts_autoencoder import Encoder


class TimeSeriesEmbedder:
    def __init__(self, model_path="encoder.pt", latent_dim=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = Encoder(latent_dim).to(self.device)
        self.encoder.load_state_dict(torch.load(model_path))
        self.encoder.eval()

    def embed(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.encoder(x).cpu().numpy()
            