import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from torch.utils.data import DataLoader
from ml.data.ts_dataset import TimeSeriesDataset
from ml.data.synthetic_ts import generate_hr_spo2_sequence
from ml.models.ts_autoencoder import AutoEncoder

# Check GPU availability with detailed info
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Device Count: {torch.cuda.device_count()}")
else:
    device = "cpu"
    print(f"⚠️  GPU NOT available - using CPU")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   If you have GPU, install PyTorch with CUDA:")
    print(f"   pip install torch --index-url https://download.pytorch.org/whl/cu121")

print(f"\nUsing device: {device}\n")

# Generate sequences (256 length to match model)
sequences = []
for _ in range(120):
    seq = generate_hr_spo2_sequence(length=256, anomaly=False)
    sequences.append(seq)

dataset = TimeSeriesDataset(sequences)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Debug: check shapes on first batch
print("=== Shape Debugging ===")
sample_batch = next(iter(loader)).to(device)
print(f"Raw batch shape: {sample_batch.shape}")
print(f"Raw batch dtype: {sample_batch.dtype}")

if sample_batch.dim() == 4 and sample_batch.shape[1] == 1:
    sample_batch = sample_batch.squeeze(1)
    print(f"After squeeze: {sample_batch.shape}")

sample_batch = sample_batch.permute(0, 2, 1)
print(f"After permute: {sample_batch.shape}")

# FIX: Convert to float32 (model expects float32, not float64/double)
sample_batch = sample_batch.float()
print(f"After .float(): {sample_batch.dtype}")

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
        
        # FIX: Ensure float32 type (not float64)
        batch = batch.float()
        
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

# Save models
print("\nSaving model weights...")
torch.save(model.encoder.state_dict(), "encoder.pt")
print("✓ encoder.pt saved")

# IMPROVED: Also save complete model
torch.save(model.state_dict(), "autoencoder_full.pt")
print("✓ autoencoder_full.pt saved")

print("\n✅ Training complete!")