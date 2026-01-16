"""
Train tabular classifier on Human Vital Signs dataset
High-accuracy risk prediction without time-series complexity
"""

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.models.tabular_classifier import VitalsRiskClassifier

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{'✅ GPU' if device == 'cuda' else '⚠️  CPU'}: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU mode'}\n")

print("=" * 80)
print("TRAINING TABULAR RISK CLASSIFIER ON HUMAN VITAL SIGNS")
print("=" * 80)

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'human_vital_signs_dataset_2024.csv')
print(f"\n📂 Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"   ✓ Loaded {len(df):,} patient records")

# Feature selection
feature_cols = [
    'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age',
    'Weight (kg)', 'Height (m)', 'Derived_HRV', 'Derived_Pulse_Pressure',
    'Derived_BMI', 'Derived_MAP'
]

# Add gender encoding
df['Gender_Encoded'] = (df['Gender'] == 'Male').astype(int)
feature_cols.append('Gender_Encoded')

print(f"\n📊 Features ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col}")

# Prepare data
X = df[feature_cols].values.astype(np.float32)
y = (df['Risk Category'] == 'High Risk').astype(int).values

print(f"\n📈 Dataset statistics:")
print(f"   Total samples: {len(X):,}")
print(f"   Low Risk:  {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"   High Risk: {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")

# Patient-level split (by Patient ID)
unique_patients = df['Patient ID'].unique()
train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.15, random_state=42
)
train_patients, val_patients = train_test_split(
    train_patients, test_size=0.15 / 0.85, random_state=42
)

# Create train/val/test masks
train_mask = df['Patient ID'].isin(train_patients)
val_mask = df['Patient ID'].isin(val_patients)
test_mask = df['Patient ID'].isin(test_patients)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"\n🔧 Patient-level splits:")
print(f"   Train: {len(X_train):,} samples ({len(train_patients):,} patients)")
print(f"   Val:   {len(X_val):,} samples ({len(val_patients):,} patients)")
print(f"   Test:  {len(X_test):,} samples ({len(test_patients):,} patients)")

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"   ✓ Features normalized (z-score)")

# Save scaler for inference
import joblib
scaler_path = os.path.join(os.path.dirname(__file__), 'vitals_scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"   ✓ Saved scaler to {scaler_path}")

# Create DataLoaders
BATCH_SIZE = 256
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float())
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).float())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model
model = VitalsRiskClassifier(
    input_dim=len(feature_cols),
    hidden_dims=[128, 64, 32],
    dropout=0.3
).to(device)

print(f"\n🏗️  Model architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,}")

# Training setup
EPOCHS = 100
LR = 1e-3
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

best_val_auc = 0.0
patience = 15
patience_counter = 0

print(f"\n🚀 Training configuration:")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LR}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Early stopping patience: {patience}")
print("=" * 80)

# Training loop
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        pred = model(batch_x).squeeze()
        loss = criterion(pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x).squeeze()
            loss = criterion(pred, batch_y)
            val_loss += loss.item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_auc = roc_auc_score(all_labels, all_preds)
    pred_binary = (all_preds > 0.5).astype(int)
    val_acc = accuracy_score(all_labels, pred_binary)
    
    # Print progress
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        
        save_path = os.path.join(os.path.dirname(__file__), 'vitals_classifier_best.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_auc': val_auc,
            'val_acc': val_acc,
            'feature_cols': feature_cols
        }, save_path)
        
        if epoch % 5 == 0:
            print(f"   ✅ Saved best model (AUC: {val_auc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏸️  Early stopping at epoch {epoch}")
            break

# Load best model for final evaluation
checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'vitals_classifier_best.pt'))
model.load_state_dict(checkpoint['model_state_dict'])

# Final test evaluation
print("\n" + "=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).squeeze()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
pred_binary = (all_preds > 0.5).astype(int)

# Calculate metrics
test_acc = accuracy_score(all_labels, pred_binary)
test_precision = precision_score(all_labels, pred_binary)
test_recall = recall_score(all_labels, pred_binary)
test_f1 = f1_score(all_labels, pred_binary)
test_auc = roc_auc_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, pred_binary)

print(f"\n📊 Test Set Performance:")
print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   ROC-AUC:   {test_auc:.4f}")

print(f"\n📊 Confusion Matrix:")
print(f"                Predicted Low  Predicted High")
print(f"   Actual Low   {cm[0,0]:8d}      {cm[0,1]:8d}")
print(f"   Actual High  {cm[1,0]:8d}      {cm[1,1]:8d}")

# Quality assessment
print(f"\n✅ Model Quality Assessment:")
if test_auc > 0.95:
    print(f"   ✅ EXCELLENT - Very high discrimination (AUC > 0.95)")
elif test_auc > 0.90:
    print(f"   ✅ VERY GOOD - High discrimination (AUC > 0.90)")
elif test_auc > 0.85:
    print(f"   ✅ GOOD - Acceptable performance (AUC > 0.85)")
else:
    print(f"   ⚠️  FAIR - Consider more features or tuning")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETED")
print("=" * 80)
print(f"\n💾 Saved files:")
print(f"   Model: {save_path}")
print(f"   Scaler: {scaler_path}")
print(f"\n🎯 Ready for deployment in production inference pipeline!")
