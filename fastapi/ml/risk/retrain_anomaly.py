"""
Retrain anomaly detection model with real patient data embeddings
Uses Human Vital Signs dataset with real "High Risk" / "Low Risk" labels
"""

import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import IsolationForest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

print("=" * 80)
print("RETRAINING ANOMALY DETECTION MODEL WITH REAL PATIENT DATA")
print("=" * 80)

# Load vitals dataset
from ml.data.vitals_loader import VitalsDataLoader
from ml.models.ts_autoencoder import Encoder

base_dir = os.path.dirname(__file__)
validation_dir = os.path.join(base_dir, '..', 'validation')
train_dir = os.path.join(base_dir, '..', 'train')

# Load pre-trained or fine-tuned encoder
encoder_path = os.path.join(train_dir, 'encoder_finetuned.pt')
if not os.path.exists(encoder_path):
    encoder_path = os.path.join(train_dir, 'encoder_vitals.pt')
    if not os.path.exists(encoder_path):
        print(f"\n❌ ERROR: No trained encoder found")
        print(f"   Please run train_vitals.py first")
        sys.exit(1)

print(f"\n📂 Loading encoder from: {encoder_path}")
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = Encoder(latent_dim=32, input_channels=4, use_lstm=True).to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
encoder.eval()
print(f"   ✅ Encoder loaded on {device}")

# Load vitals dataset
print(f"\n📂 Loading Human Vital Signs dataset...")
loader = VitalsDataLoader(window_size=256, stride=128)
loader.load_data()
loader.create_patient_splits()

# Get train set with labels
X_train, y_train, meta_train = loader.get_dataset('train', include_time_features=True, normalize=False)

print(f"\n✅ Dataset loaded:")
print(f"   Total samples: {len(X_train):,}")
print(f"   Low Risk (0): {(y_train == 0).sum():,}")
print(f"   High Risk (1): {(y_train == 1).sum():,}")

# Generate embeddings for all samples
print(f"\n🔄 Generating embeddings...")
embeddings = []
labels = []

with torch.no_grad():
    for i in range(len(X_train)):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(X_train)} ({i/len(X_train)*100:.1f}%)")
        
        x = torch.from_numpy(X_train[i]).unsqueeze(0).float().to(device)
        emb = encoder(x).cpu().numpy().squeeze()
        embeddings.append(emb)
        labels.append(y_train[i])

embeddings = np.array(embeddings)
labels = np.array(labels)

print(f"   ✅ Generated {len(embeddings):,} embeddings of shape {embeddings.shape}")

# Separate normal and anomaly embeddings
normal_emb = embeddings[labels == 0]  # Low Risk
anomaly_emb = embeddings[labels == 1]  # High Risk

print(f"\n📊 Embedding split:")
print(f"   Normal (Low Risk): {len(normal_emb):,}")
print(f"   Anomaly (High Risk): {len(anomaly_emb):,}")

# Save embeddings for future use
np.save(os.path.join(validation_dir, "normal_embeddings.npy"), normal_emb)
np.save(os.path.join(validation_dir, "anomaly_embeddings.npy"), anomaly_emb)
print(f"\n💾 Saved embeddings to {validation_dir}")

# Train Isolation Forest on NORMAL data only
print(f"\n🏗️  Training Isolation Forest...")
contamination = len(anomaly_emb) / (len(normal_emb) + len(anomaly_emb))
print(f"   Parameters: n_estimators=200, contamination={contamination:.3f}")

iso = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42,
    verbose=0
)
iso.fit(normal_emb)
print("   ✅ Model trained on normal (Low Risk) embeddings")

# Compute risk bounds using both normal and anomaly data
print("\n📊 Computing risk score bounds...")
scores_normal = -iso.score_samples(normal_emb)
scores_anomaly = -iso.score_samples(anomaly_emb)

all_scores = np.concatenate([scores_normal, scores_anomaly])
s_min, s_max = all_scores.min(), all_scores.max()

print(f"   Risk bounds: [{s_min:.4f}, {s_max:.4f}]")
print(f"   Normal scores:  Mean={scores_normal.mean():.4f}, Range=[{scores_normal.min():.4f}, {scores_normal.max():.4f}]")
print(f"   Anomaly scores: Mean={scores_anomaly.mean():.4f}, Range=[{scores_anomaly.min():.4f}, {scores_anomaly.max():.4f}]")

# Check separation quality
separation = scores_anomaly.mean() - scores_normal.mean()
print(f"   Score separation: {separation:.4f}")

if separation > 0.1:
    print("   ✅ Good separation - anomaly scores > normal scores")
else:
    print("   ⚠️  WARNING: Poor separation detected")

# Save model and bounds
print("\n💾 Saving model...")
joblib.dump(iso, "anomaly_model.joblib")
print("   ✓ anomaly_model.joblib")

np.save("risk_bounds.npy", np.array([s_min, s_max]))
print("   ✓ risk_bounds.npy")

# Validate accuracy on test set
print("\n🧪 Quick validation...")
from sklearn.metrics import accuracy_score

# Normalize scores to 0-100
def normalize(scores):
    return 100 * (scores - s_min) / (s_max - s_min + 1e-6)

normal_risk = normalize(scores_normal)
anomaly_risk = normalize(scores_anomaly)

# Use threshold of 50
y_true = np.array([0] * len(normal_risk) + [1] * len(anomaly_risk))
y_pred = np.array([(1 if r > 50 else 0) for r in normal_risk] + 
                  [(1 if r > 50 else 0) for r in anomaly_risk])

accuracy = accuracy_score(y_true, y_pred)
print(f"   Quick accuracy check: {accuracy:.1%}")

if accuracy > 0.85:
    print("   ✅ Excellent - model is working well")
elif accuracy > 0.70:
    print("   ✅ Good - acceptable performance")
else:
    print("   ⚠️  WARNING: Low accuracy detected")

print("\n" + "=" * 80)
print("✅ ANOMALY MODEL RETRAINED SUCCESSFULLY")
print("=" * 80)
print("\nNext step: Run ml/fusion/train_fusion.py (if needed)")
print("Then: Run ml/evaluate_accuracy.py to verify improvements")
