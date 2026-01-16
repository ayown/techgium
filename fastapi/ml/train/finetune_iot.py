"""
Fine-tuning script for LSTM-enhanced autoencoder on Healthcare IoT dataset
Transfer learning with frozen Conv layers
"""

import torch
import torch.optim as optim
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.models.ts_autoencoder import AutoEncoder
from ml.data.iot_loader import IoTDataLoader

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print(f"⚠️  Using CPU")

print(f"Device: {device}\n")

# Hyperparameters
EPOCHS = 50
LR = 1e-4  # Lower learning rate for fine-tuning
BATCH_SIZE = 16  # Smaller batch size due to small dataset
LATENT_DIM = 32
WINDOW_SIZE = 256

# Paths
script_dir = os.path.dirname(__file__)
fastapi_dir = os.path.join(script_dir, '..', '..')
save_dir = os.path.join(fastapi_dir, 'ml', 'train')
pretrained_path = os.path.join(save_dir, 'autoencoder_vitals.pt')

print("=" * 80)
print("FINE-TUNING AUTOENCODER ON HEALTHCARE IoT DATASET")
print("=" * 80)

# Check if pre-trained model exists
if not os.path.exists(pretrained_path):
    print(f"\n❌ ERROR: Pre-trained model not found at {pretrained_path}")
    print(f"   Please run train_vitals.py first to pre-train the model")
    sys.exit(1)

# Load IoT dataset
print("\n📂 Loading IoT dataset...")
loader = IoTDataLoader(window_size=WINDOW_SIZE, test_size=0.3)
loader.load_data()
loader.create_patient_splits()

# Get train and test sets
X_train, y_train, meta_train = loader.get_dataset('train', include_time_features=True, normalize=False)
X_test, y_test, meta_test = loader.get_dataset('test', include_time_features=True, normalize=False)

print(f"\n✅ Dataset loaded:")
print(f"   Train: {X_train.shape} ({len(meta_train['patient_ids'])} patients)")
print(f"   Test:  {X_test.shape} ({len(meta_test['patient_ids'])} patients)")

# Handle class imbalance with weighted sampling
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]

print(f"\n⚖️  Handling class imbalance:")
print(f"   Unhealthy (0): {class_counts[0]} samples, weight: {class_weights[0]:.2f}")
print(f"   Healthy (1): {class_counts[1]} samples, weight: {class_weights[1]:.2f}")

# Create weighted sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Create DataLoaders
train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).long()
)
test_dataset = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test).long()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n🔧 DataLoaders created:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Test batches:  {len(test_loader)}")

# Initialize model and load pre-trained weights
print(f"\n🏗️  Loading pre-trained model...")
model = AutoEncoder(
    latent_dim=LATENT_DIM,
    seq_len=WINDOW_SIZE,
    input_channels=4,
    output_channels=4,
    use_lstm=True
).to(device)

# Load pre-trained weights
model.load_state_dict(torch.load(pretrained_path, map_location=device))
print(f"   ✅ Loaded weights from {pretrained_path}")

# Freeze convolutional layers
print(f"\n❄️  Freezing convolutional layers...")
model.encoder.conv1.requires_grad_(False)
model.encoder.conv2.requires_grad_(False)
model.decoder.net.requires_grad_(False)

print(f"   ✅ Conv layers frozen")

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
print(f"   Frozen: {total_params - trainable_params:,} ({(total_params-trainable_params)/total_params*100:.1f}%)")

# Optimizer and loss
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
loss_fn = torch.nn.MSELoss()

best_test_loss = float('inf')
best_epoch = 0

print(f"\n🚀 Starting fine-tuning...")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LR}")
print(f"   Batch size: {BATCH_SIZE}")
print("=" * 80)

# Fine-tuning loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
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
    
    # Test phase
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            recon, latent = model(batch_x)
            loss = loss_fn(recon, batch_x)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    
    # Print progress
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
    
    # Save best model
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_epoch = epoch
        
        encoder_path = os.path.join(save_dir, 'encoder_finetuned.pt')
        decoder_path = os.path.join(save_dir, 'decoder_finetuned.pt')
        full_path = os.path.join(save_dir, 'autoencoder_finetuned.pt')
        
        torch.save(model.encoder.state_dict(), encoder_path)
        torch.save(model.decoder.state_dict(), decoder_path)
        torch.save(model.state_dict(), full_path)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'latent_dim': LATENT_DIM,
            'input_channels': 4,
            'use_lstm': True,
            'pretrained_from': 'vitals',
            'train_patients': meta_train['patient_ids'],
            'test_patients': meta_test['patient_ids']
        }
        torch.save(metadata, os.path.join(save_dir, 'iot_metadata.pt'))
        
        if epoch % 5 == 0:
            print(f"   ✅ Saved checkpoint (test_loss: {avg_test_loss:.6f})")

print("\n" + "=" * 80)
print("✅ FINE-TUNING COMPLETED")
print("=" * 80)
print(f"\n📊 Final Results:")
print(f"   Best epoch: {best_epoch}")
print(f"   Best test loss: {best_test_loss:.6f}")
print(f"\n💾 Saved models:")
print(f"   {os.path.join(save_dir, 'encoder_finetuned.pt')}")
print(f"   {os.path.join(save_dir, 'decoder_finetuned.pt')}")
print(f"   {os.path.join(save_dir, 'autoencoder_finetuned.pt')}")
print(f"   {os.path.join(save_dir, 'iot_metadata.pt')}")
print(f"\n🎯 Next step: Update retrain_anomaly.py to use new embeddings")
