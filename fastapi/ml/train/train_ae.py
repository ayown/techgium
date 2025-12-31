import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from torch.utils.data import DataLoader
from ml.data.ts_dataset import TimeSeriesDataset
from ml.data.synthetic_ts import generate_hr_spo2_sequence
from ml.models.ts_autoencoder import AutoEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Generate sequences
sequences = []
for _ in range(120):
    seq = generate_hr_spo2_sequence(anomaly=False)
    sequences.append(seq)

dataset = TimeSeriesDataset(sequences)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Debug: check shapes on first batch
print("\n=== Shape Debugging ===")
sample_batch = next(iter(loader)).to(device)
print(f"Raw batch shape: {sample_batch.shape}")

if sample_batch.dim() == 4 and sample_batch.shape[1] == 1:
    sample_batch = sample_batch.squeeze(1)
    print(f"After squeeze: {sample_batch.shape}")

sample_batch = sample_batch.permute(0, 2, 1)
print(f"After permute: {sample_batch.shape}")

sample_recon, sample_latent = model(sample_batch)
print(f"Reconstruction shape: {sample_recon.shape}")
print(f"Latent shape: {sample_latent.shape}")
print("=== Shape check passed! ===\n")

# Training loop
print("Starting training...")
for epoch in range(25):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        
        # Remove extra channel dimension if present
        if batch.dim() == 4 and batch.shape[1] == 1:
            batch = batch.squeeze(1)
        
        # Permute to [batch, features, timesteps]
        batch = batch.permute(0, 2, 1)
        
        # Forward pass
        recon, _ = model(batch)
        
        # Compute loss
        loss = loss_fn(recon, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/25, Loss: {avg_loss:.4f}")

# Save model
print("\nSaving encoder weights...")
torch.save(model.encoder.state_dict(), "encoder.pt")
print("Training complete! Model saved to encoder.pt")