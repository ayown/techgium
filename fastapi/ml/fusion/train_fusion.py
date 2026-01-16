import torch
import torch.optim as optim
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path so 'ml' module can be found
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.fusion.attention_fusion import AttentionFusion

EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 32

# Get the directory where this script is located
script_dir = os.path.dirname(__file__)
fastapi_dir = os.path.join(script_dir, '..', '..')

# Use absolute paths for data files
cardio_path = os.path.join(fastapi_dir, "ml", "validation", "cardio_embeddings.npy")
resp_path = os.path.join(fastapi_dir, "ml", "validation", "resp_embeddings.npy")

# synthetic embeddings (from notebook or generator)
cardio = np.load(cardio_path)
resp = np.load(resp_path)

# Build a simple self-supervised dataset:
# - matched (cardio[i], resp[i])  -> label 0 (low risk)
# - mismatched (cardio[i], resp[j]) -> label 1 (high risk)
cardio = np.asarray(cardio, dtype=np.float32)
resp = np.asarray(resp, dtype=np.float32)

if cardio.ndim != 2 or resp.ndim != 2:
    raise ValueError(f"Expected cardio/resp arrays with shape (N, D). Got cardio={cardio.shape}, resp={resp.shape}")
if cardio.shape != resp.shape:
    raise ValueError(f"cardio and resp must have same shape. Got cardio={cardio.shape}, resp={resp.shape}")
    
n_samples, emb_dim = cardio.shape

pos = np.stack([cardio, resp], axis=1)  # (N, 2, D)

rng = np.random.default_rng(42)
perm = rng.permutation(n_samples)
if np.all(perm == np.arange(n_samples)) and n_samples > 1:
    perm = np.roll(perm, 1)

neg = np.stack([cardio, resp[perm]], axis=1)  # (N, 2, D)

X = np.concatenate([pos, neg], axis=0)  # (2N, 2, D)
y = np.concatenate([
    np.zeros((n_samples,), dtype=np.float32),
    np.ones((n_samples,), dtype=np.float32)
], axis=0)  # (2N,)

dataset = TensorDataset(
    torch.from_numpy(X),
    torch.from_numpy(y)
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AttentionFusion().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for emb_batch, y_batch in loader:
        emb_batch = emb_batch.to(device)  # (B, 2, D)
        y_batch = y_batch.to(device)      # (B,)

        # AttentionFusion currently operates on one sample: (2, D)
        risks = []
        for i in range(emb_batch.shape[0]):
            out = model(emb_batch[i])
            risks.append(out["systemic_risk"])
        risk_batch = torch.stack(risks).float()  # (B,)

        loss = loss_fn(risk_batch, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    if epoch % 25 == 0:
        print(f"Epoch {epoch}, Loss {total_loss/len(loader):.4f}")

# Save model with absolute path
model_save_path = os.path.join(fastapi_dir, "ml", "fusion", "fusion_model.pt")
torch.save(model.state_dict(), model_save_path)
print(f"Saved model to: {model_save_path}")