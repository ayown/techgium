import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=32, input_channels=4, use_lstm=True):
        """
        Enhanced encoder with optional LSTM layers for temporal modeling
        
        Args:
            latent_dim: Size of latent embedding (default: 32)
            input_channels: Number of input channels - 2 for HR+SpO2, 4 with time features (default: 4)
            use_lstm: Whether to use LSTM layers for temporal dependencies (default: True)
        """
        super().__init__()
        self.use_lstm = use_lstm
        self.input_channels = input_channels
        
        # Convolutional feature extraction
        self.conv1 = nn.Conv1d(input_channels, 16, 5, stride=2, padding=2)  # 256 → 128
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, 5, stride=2, padding=2)  # 128 → 64
        self.relu2 = nn.ReLU()
        
        if use_lstm:
            # LSTM for temporal modeling
            # Input: (batch, seq_len=64, features=32)
            # Output: (batch, seq_len=64, hidden=64)
            self.lstm = nn.LSTM(
                input_size=32,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.5  # Prevent overfitting on small datasets
            )
            self.fc = nn.Linear(64, latent_dim)  # LSTM output → latent
        else:
            # Direct FC without LSTM (original architecture)
            self.fc = nn.Linear(32 * 64, latent_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, timesteps)
        Returns:
            latent: Latent embedding (batch, latent_dim)
        """
        # Conv layers
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)  # Shape: (batch, 32, 64)
        
        if self.use_lstm:
            # Permute for LSTM: (batch, 32, 64) → (batch, 64, 32)
            h = h.permute(0, 2, 1)  # (batch, seq_len, features)
            
            # LSTM forward pass
            lstm_out, (h_n, c_n) = self.lstm(h)  # lstm_out: (batch, 64, 64)
            
            # Take last timestep output
            h = lstm_out[:, -1, :]  # Shape: (batch, 64)
            
            # Project to latent space
            latent = self.fc(h)
        else:
            # Flatten and project (original)
            h = h.view(h.size(0), -1)  # (batch, 32*64)
            latent = self.fc(h)
        
        return latent
        

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, seq_len=256, output_channels=4, use_lstm=True):
        """
        Enhanced decoder with LSTM and support for multi-channel output
        
        Args:
            latent_dim: Size of latent embedding
            seq_len: Target sequence length (default: 256)
            output_channels: Number of output channels (default: 4 to match encoder)
            use_lstm: Whether to use LSTM in decoder (default: True)
        """
        super().__init__()
        self.seq_len = seq_len
        self.use_lstm = use_lstm
        self.output_channels = output_channels
        
        if use_lstm:
            # Expand latent to LSTM initial state
            self.fc_expand = nn.Linear(latent_dim, 64)
            
            # LSTM decoder
            self.lstm = nn.LSTM(
                input_size=64,
                hidden_size=32,
                num_layers=2,
                batch_first=True,
                dropout=0.5
            )
            
            # Project LSTM output to conv features
            self.fc_to_conv = nn.Linear(64, 32 * 4)  # 64 timesteps → 4 for conv
        else:
            self.fc_expand = nn.Linear(latent_dim, 32 * 4)
        
        # Transposed convolutions for upsampling
        self.net = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 5, stride=2, padding=2, output_padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose1d(16, 16, 5, stride=2, padding=2, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, 5, stride=2, padding=2, output_padding=1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose1d(8, output_channels, 5, stride=2, padding=2, output_padding=1),    # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose1d(output_channels, output_channels, 5, stride=2, padding=2, output_padding=1),    # 128 -> 256
        )

    def forward(self, z):
        """
        Args:
            z: Latent embedding (batch, latent_dim)
        Returns:
            x_hat: Reconstructed sequence (batch, output_channels, seq_len)
        """
        if self.use_lstm:
            # Expand latent
            h = self.fc_expand(z)  # (batch, 64)
            
            # Repeat for sequence length (create input for LSTM)
            h = h.unsqueeze(1).repeat(1, 64, 1)  # (batch, 64, 64)
            
            # LSTM decode
            lstm_out, _ = self.lstm(h)  # (batch, 64, 32)
            
            # Take mean over sequence for conv input
            h = lstm_out.mean(dim=1)  # (batch, 32)
            
            # Project to conv dimensions
            h = self.fc_to_conv(h).view(-1, 32, 4)
        else:
            h = self.fc_expand(z).view(-1, 32, 4)
        
        # Transposed convolutions
        x = self.net(h)
        
        return x[:, :, :self.seq_len]  # Crop to exact length


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32, seq_len=256, input_channels=4, output_channels=4, use_lstm=True):
        """
        Complete autoencoder with LSTM-enhanced temporal modeling
        
        Args:
            latent_dim: Size of latent bottleneck (default: 32)
            seq_len: Sequence length (default: 256)
            input_channels: Number of input channels (default: 4 for HR+SpO2+sin+cos)
            output_channels: Number of output channels (default: 4, matches input)
            use_lstm: Whether to use LSTM layers (default: True)
        """
        super().__init__()
        self.encoder = Encoder(latent_dim, input_channels, use_lstm)
        self.decoder = Decoder(latent_dim, seq_len, output_channels, use_lstm)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, timesteps)
        Returns:
            x_hat: Reconstruction (batch, channels, timesteps)
            z: Latent embedding (batch, latent_dim)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z