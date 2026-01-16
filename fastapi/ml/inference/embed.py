import torch
import os
from ml.models.ts_autoencoder import Encoder


class TimeSeriesEmbedder:
    def __init__(self, model_path=None, latent_dim=32):
        # Use absolute path if not provided
        if model_path is None:
            script_dir = os.path.dirname(__file__)
            model_path = os.path.join(script_dir, '..', 'train', 'encoder.pt')
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = Encoder(latent_dim).to(self.device)
        self.encoder.load_state_dict(torch.load(model_path, map_location=self.device))
        self.encoder.eval()

    def embed(self, x):
        with torch.no_grad():
            if torch.is_tensor(x):
                x_tensor = x.detach().clone()
            else:
                x_tensor = torch.as_tensor(x)

            x_tensor = x_tensor.to(device=self.device, dtype=torch.float32).unsqueeze(0)
            return self.encoder(x_tensor).cpu().numpy()
            