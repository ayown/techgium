import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, latent_dim)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.fc(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=32, seq_len=256):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, 32 * 4)  # Start with length 4
        self.net = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 5, stride=2, padding=2, output_padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose1d(16, 16, 5, stride=2, padding=2, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, 5, stride=2, padding=2, output_padding=1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose1d(8, 2, 5, stride=2, padding=2, output_padding=1),    # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose1d(2, 2, 5, stride=2, padding=2, output_padding=1),    # 128 -> 256
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 32, 4)  # Reshape to [batch, 32, 4]
        x = self.net(h)
        return x[:, :, :self.seq_len]  # Crop to exact length


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32, seq_len=256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z