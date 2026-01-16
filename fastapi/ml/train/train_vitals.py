"""
Pre-training script for LSTM-enhanced autoencoder on Human Vital Signs dataset
Implements patient-level validation and early stopping
"""

import torch
import torch.optim as optim
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.models.ts_autoencoder import AutoEncoder
from ml.data.vitals_loader import VitalsDataLoader

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print(f"⚠️  Using CPU")

print(f"Device: {device}\n")

# Hyperparameters
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 32
LATENT_DIM = 32
WINDOW_SIZE = 256
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001

# Paths
script_dir = os.path.dirname(__file__)
fastapi_dir = os.path.join(script_dir, '..', '..')
save_dir = os.path.join(fastapi_dir, 'ml', 'train')
os.makedirs(save_dir, exist_ok=True)

print("=" * 80)
print("PRE-TRAINING AUTOENCODER ON HUMAN VITAL SIGNS DATASET")
print("=" * 80)

# Load vitals dataset with patient-level splits
print("\n📂 Loading dataset...")
loader = VitalsDataLoader(window_size=WINDOW_SIZE, stride=128, test_size=0.15, val_size=0.15)
loader.load_data()
loader.create_patient_splits()

# Get train and validation sets (with time features)
X_train, y_train, meta_train = loader.get_dataset('train', include_time_features=True, normalize=False)
X_val, y_val, meta_val = loader.get_dataset('val', include_time_features=True, normalize=False)

print(f"\n✅ Dataset loaded:")
print(f"   Train: {X_train.shape} ({meta_train['n_patients']} patients)")
print(f"   Val:   {X_val.shape} ({meta_val['n_patients']} patients)")
print(f"   Input channels: {X_train.shape[1]} (HR, SpO2, sin(hour), cos(hour))")
print(f"   Sequence length: {X_train.shape[2]}")

# Create DataLoaders
train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).long()
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val).float(),
    torch.from_numpy(y_val).long()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n🔧 DataLoaders created:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches:   {len(val_loader)}")

# Initialize model with LSTM
print(f"\n🏗️  Initializing LSTM-enhanced AutoEncoder...")
model = AutoEncoder(
    latent_dim=LATENT_DIM,
    seq_len=WINDOW_SIZE,
    input_channels=4,  # HR + SpO2 + sin(hour) + cos(hour)
    output_channels=4,
    use_lstm=True
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

# Early stopping setup
best_val_loss = float('inf')
patience_counter = 0
best_epoch = 0

print(f"\n🚀 Starting training...")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LR}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print("=" * 80)

# Training loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        
        # Forward pass
        recon, latent = model(batch_x)
        loss = loss_fn(recon, batch_x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            recon, latent = model(batch_x)
            loss = loss_fn(recon, batch_x)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Print progress
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    # Early stopping check
    if avg_val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_epoch = epoch
        
        # Save best model
        encoder_path = os.path.join(save_dir, 'encoder_vitals.pt')
        decoder_path = os.path.join(save_dir, 'decoder_vitals.pt')
        full_path = os.path.join(save_dir, 'autoencoder_vitals.pt')
        
        torch.save(model.encoder.state_dict(), encoder_path)
        torch.save(model.decoder.state_dict(), decoder_path)
        torch.save(model.state_dict(), full_path)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'latent_dim': LATENT_DIM,
            'input_channels': 4,
            'use_lstm': True,
            'train_patients': meta_train['patient_ids'],
            'val_patients': meta_val['patient_ids']
        }
        torch.save(metadata, os.path.join(save_dir, 'vitals_metadata.pt'))
        
        if epoch % 5 == 0:
            print(f"   ✅ Saved checkpoint (val_loss improved by {best_val_loss - avg_val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏸️  Early stopping triggered at epoch {epoch}")
            print(f"   Best epoch: {best_epoch} with val_loss: {best_val_loss:.6f}")
            break

print("\n" + "=" * 80)
print("✅ PRE-TRAINING COMPLETED")
print("=" * 80)
print(f"\n📊 Final Results:")
print(f"   Best epoch: {best_epoch}")
print(f"   Best validation loss: {best_val_loss:.6f}")
print(f"\n💾 Saved models:")
print(f"   {os.path.join(save_dir, 'encoder_vitals.pt')}")
print(f"   {os.path.join(save_dir, 'decoder_vitals.pt')}")
print(f"   {os.path.join(save_dir, 'autoencoder_vitals.pt')}")
print(f"   {os.path.join(save_dir, 'vitals_metadata.pt')}")
print(f"\n🎯 Next step: Run finetune_iot.py for transfer learning on IoT dataset")
