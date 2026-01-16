"""
Regenerate embeddings using the newly retrained encoder
This fixes the pipeline incompatibility issue
"""

import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.data.synthetic_ts import generate_hr_spo2_sequence
from ml.inference.embed import TimeSeriesEmbedder

print("=" * 80)
print("REGENERATING EMBEDDINGS WITH NEW ENCODER")
print("=" * 80)

embedder = TimeSeriesEmbedder()

# Generate same number of sequences as before
NORMAL_SAMPLES = 120
ANOMALY_SAMPLES = 60

print(f"\nGenerating {NORMAL_SAMPLES} normal sequences...")
normal_sequences = [generate_hr_spo2_sequence(256, anomaly=False) for _ in range(NORMAL_SAMPLES)]

print(f"Generating {ANOMALY_SAMPLES} anomaly sequences...")
anomaly_sequences = [generate_hr_spo2_sequence(256, anomaly=True) for _ in range(ANOMALY_SAMPLES)]

# Convert to embeddings
def embed_batch(sequences):
    embeddings = []
    for seq in sequences:
        seq_tensor = torch.tensor(seq.T, dtype=torch.float32)
        emb = embedder.embed(seq_tensor)
        embeddings.append(emb.squeeze())
    return np.array(embeddings)

print("\n📊 Embedding normal sequences...")
normal_emb = embed_batch(normal_sequences)

print("📊 Embedding anomaly sequences...")
anomaly_emb = embed_batch(anomaly_sequences)

print(f"\n✅ Embeddings generated:")
print(f"   Normal: {normal_emb.shape} - Mean={normal_emb.mean():.3f}, Std={normal_emb.std():.3f}")
print(f"   Anomaly: {anomaly_emb.shape} - Mean={anomaly_emb.mean():.3f}, Std={anomaly_emb.std():.3f}")

# Calculate separation
distance = np.linalg.norm(normal_emb.mean(axis=0) - anomaly_emb.mean(axis=0))
print(f"   Centroid separation: {distance:.3f}")

# Save new embeddings
print("\n💾 Saving embeddings...")

# Always save into this folder (avoid depending on CWD)
out_dir = os.path.dirname(__file__)

# Create a *correlated but not identical* second-system embedding set for fusion training.
# Scale noise to the embedding distribution so the signal isn't numerically negligible.
rng = np.random.default_rng(42)

# Use both normal and anomaly embeddings to increase diversity for the fusion self-supervised task.
cardio_emb = np.vstack([normal_emb, anomaly_emb]).astype(np.float32)

# Feature-wise noise scaled to feature std (relative noise ~= 0.3 like the Tier-3 synthetic evaluator).
feat_std = cardio_emb.std(axis=0, keepdims=True)
noise = rng.normal(0.0, 1.0, size=cardio_emb.shape).astype(np.float32) * (0.3 * feat_std)
resp_emb = cardio_emb + noise

np.save(os.path.join(out_dir, "normal_embeddings.npy"), normal_emb)
np.save(os.path.join(out_dir, "cardio_embeddings.npy"), cardio_emb)  # For fusion training
np.save(os.path.join(out_dir, "resp_embeddings.npy"), resp_emb)      # For fusion training

# Also save anomaly embeddings for later use
np.save(os.path.join(out_dir, "anomaly_embeddings.npy"), anomaly_emb)

print("   ✓ normal_embeddings.npy")
print("   ✓ cardio_embeddings.npy")
print("   ✓ resp_embeddings.npy")
print("   ✓ anomaly_embeddings.npy")

print("\n" + "=" * 80)
print("✅ EMBEDDINGS REGENERATED SUCCESSFULLY")
print("=" * 80)
print("\nNext step: Run ml/risk/retrain_anomaly.py")
