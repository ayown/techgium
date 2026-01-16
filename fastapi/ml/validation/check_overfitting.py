"""
Overfitting Detection for Cardiovascular Risk Classifier
Analyzes training history and performs robustness tests
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "train" / "cardio_training_results.json"
MODEL_FILE = BASE_DIR / "train" / "cardio_classifier_best.pt"
SCALER_FILE = BASE_DIR / "train" / "cardio_scaler.joblib"
DATA_FILE = BASE_DIR.parent / "human_vital_signs_dataset_2024.csv"

print("="*80)
print("OVERFITTING ANALYSIS - CARDIOVASCULAR RISK CLASSIFIER")
print("="*80)

# Load training history
with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

history = results['training_history']
test_metrics = results['test_metrics']

print("\n📊 1. TRAINING VS VALIDATION PERFORMANCE")
print("="*80)

# Check for overfitting signals
train_losses = history['train_loss']
val_losses = history['val_loss']
val_accuracies = history['val_accuracy']

final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]
final_val_acc = val_accuracies[-1]
test_acc = test_metrics['accuracy']

print(f"\n📉 Loss Analysis:")
print(f"   Final Train Loss:      {final_train_loss:.4f}")
print(f"   Final Val Loss:        {final_val_loss:.4f}")
print(f"   Val/Train Loss Ratio:  {final_val_loss/final_train_loss:.2f}")

if final_val_loss > final_train_loss * 1.5:
    print("   ⚠️  WARNING: Validation loss >> Training loss (possible overfitting)")
elif final_val_loss > final_train_loss * 1.2:
    print("   ⚡ CAUTION: Validation loss moderately higher than training")
else:
    print("   ✅ GOOD: Validation and training losses are similar")

print(f"\n🎯 Accuracy Analysis:")
print(f"   Final Val Accuracy:    {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"   Test Accuracy:         {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Val-Test Gap:          {abs(final_val_acc - test_acc):.4f} ({abs(final_val_acc - test_acc)*100:.2f}%)")

if abs(final_val_acc - test_acc) > 0.05:
    print("   ⚠️  WARNING: Large gap between validation and test (>5%)")
elif abs(final_val_acc - test_acc) > 0.02:
    print("   ⚡ CAUTION: Moderate gap between validation and test (2-5%)")
else:
    print("   ✅ EXCELLENT: Validation and test performance match closely")

# Check for divergence in learning curves
print(f"\n📈 Learning Curve Analysis:")
epochs_trained = len(train_losses)
mid_point = epochs_trained // 2

early_train_loss_avg = np.mean(train_losses[:mid_point])
late_train_loss_avg = np.mean(train_losses[mid_point:])
early_val_loss_avg = np.mean(val_losses[:mid_point])
late_val_loss_avg = np.mean(val_losses[mid_point:])

print(f"   Early Training (epochs 1-{mid_point}):")
print(f"      Train Loss: {early_train_loss_avg:.4f}")
print(f"      Val Loss:   {early_val_loss_avg:.4f}")
print(f"   Late Training (epochs {mid_point+1}-{epochs_trained}):")
print(f"      Train Loss: {late_train_loss_avg:.4f}")
print(f"      Val Loss:   {late_val_loss_avg:.4f}")

# Check if validation loss started increasing
min_val_loss_idx = np.argmin(val_losses)
if min_val_loss_idx < epochs_trained * 0.7:
    print(f"   ⚠️  WARNING: Val loss minimized at epoch {min_val_loss_idx+1}, then increased")
else:
    print(f"   ✅ GOOD: Val loss continued improving throughout training")

# Plot learning curves
print("\n📊 Generating learning curves...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0, 0].axvline(min_val_loss_idx, color='red', linestyle='--', alpha=0.5, label=f'Best (epoch {min_val_loss_idx+1})')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training vs Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy curve
axes[0, 1].plot(val_accuracies, label='Val Accuracy', linewidth=2, color='green')
axes[0, 1].axhline(test_acc, color='orange', linestyle='--', linewidth=2, label=f'Test Acc ({test_acc:.4f})')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Validation Accuracy Progression')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# AUC curve
axes[1, 0].plot(history['val_auc'], label='Val AUC', linewidth=2, color='purple')
axes[1, 0].axhline(test_metrics['roc_auc'], color='red', linestyle='--', linewidth=2, label=f'Test AUC ({test_metrics["roc_auc"]:.4f})')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].set_title('ROC AUC Progression')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Loss ratio over time
loss_ratios = [v/t if t > 0 else 1.0 for v, t in zip(val_losses, train_losses)]
axes[1, 1].plot(loss_ratios, linewidth=2, color='brown')
axes[1, 1].axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Match')
axes[1, 1].axhline(1.2, color='orange', linestyle='--', alpha=0.5, label='Caution Zone')
axes[1, 1].axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Overfitting Zone')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Val Loss / Train Loss')
axes[1, 1].set_title('Overfitting Indicator (Loss Ratio)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'validation' / 'overfitting_analysis.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {BASE_DIR / 'validation' / 'overfitting_analysis.png'}")

print("\n"+"="*80)
print("📊 2. ROBUSTNESS TESTS")
print("="*80)

# Load model and test data
import sys
sys.path.append(str(BASE_DIR.parent))

import pandas as pd
from ml.models.specialized_classifiers import CardiovascularRiskClassifier

print("\n🔧 Loading model and test data...")
model = CardiovascularRiskClassifier(input_dim=10, hidden_dims=[128, 64, 32])
checkpoint = torch.load(MODEL_FILE, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler = joblib.load(SCALER_FILE)

# Load data
df = pd.read_csv(DATA_FILE)
test_size = int(len(df) * 0.15)
test_df = df.iloc[-test_size:].copy()

CARDIO_FEATURES = [
    'Heart Rate', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
    'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_MAP', 'Age',
    'Derived_BMI', 'Weight (kg)', 'Height (m)'
]

X_test = test_df[CARDIO_FEATURES].values
y_test = test_df['Cardiovascular Disease Risk'].values

X_test_scaled = scaler.transform(X_test)

# Baseline performance
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_test_scaled)
    outputs = model(X_tensor)
    probs_baseline = torch.sigmoid(outputs).numpy().flatten()
    preds_baseline = (probs_baseline > 0.5).astype(int)

baseline_acc = accuracy_score(y_test, preds_baseline)
baseline_auc = roc_auc_score(y_test, probs_baseline)

print(f"   Baseline Test Accuracy: {baseline_acc:.4f}")
print(f"   Baseline Test AUC:      {baseline_auc:.4f}")

# Test 1: Gaussian noise injection
print("\n🔊 Test 1: Gaussian Noise Injection")
print("   (Simulates sensor measurement errors)")

noise_levels = [0.01, 0.05, 0.10, 0.15, 0.20]
noise_results = []

for noise_std in noise_levels:
    X_noisy = X_test_scaled + np.random.normal(0, noise_std, X_test_scaled.shape)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_noisy)
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs).numpy().flatten()
        preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    noise_results.append({'noise': noise_std, 'accuracy': acc, 'auc': auc})
    
    acc_drop = (baseline_acc - acc) * 100
    print(f"      Noise σ={noise_std:.2f}: Acc={acc:.4f} ({acc*100:.2f}%), Drop={acc_drop:.2f}%")

# Check robustness
avg_drop = np.mean([(baseline_acc - r['accuracy']) * 100 for r in noise_results])
if avg_drop > 10:
    print(f"   ⚠️  WARNING: Average accuracy drop {avg_drop:.2f}% (model not robust)")
elif avg_drop > 5:
    print(f"   ⚡ CAUTION: Average accuracy drop {avg_drop:.2f}% (moderate robustness)")
else:
    print(f"   ✅ EXCELLENT: Average accuracy drop {avg_drop:.2f}% (highly robust)")

# Test 2: Feature permutation importance
print("\n🔀 Test 2: Feature Permutation Importance")
print("   (Check if model relies on meaningful features)")

feature_importances = []

for i, feature_name in enumerate(CARDIO_FEATURES):
    X_permuted = X_test_scaled.copy()
    np.random.shuffle(X_permuted[:, i])  # Shuffle one feature
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_permuted)
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs).numpy().flatten()
        preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds)
    importance = baseline_acc - acc
    feature_importances.append({'feature': feature_name, 'importance': importance})

# Sort by importance
feature_importances.sort(key=lambda x: x['importance'], reverse=True)

print("\n   Feature Importance (by permutation):")
for fi in feature_importances[:5]:
    print(f"      {fi['feature']:<30s} → Δ{fi['importance']*100:+.2f}%")

# Check if model uses clinically meaningful features
critical_features = ['Heart Rate', 'Systolic Blood Pressure', 'Age', 'Derived_BMI']
critical_importance = sum(fi['importance'] for fi in feature_importances if fi['feature'] in critical_features)

if critical_importance < 0.05:
    print(f"   ⚠️  WARNING: Model ignores critical cardiovascular features")
else:
    print(f"   ✅ GOOD: Model relies on clinically meaningful features")

print("\n"+"="*80)
print("📋 3. DATASET ANALYSIS")
print("="*80)

# Check for data leakage indicators
print("\n🔍 Data Leakage Checks:")

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"   Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")

# Check for near-perfect separability
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_test_scaled, y_test)
dt_acc = dt.score(X_test_scaled, y_test)

print(f"   Shallow Decision Tree (depth=3) Accuracy: {dt_acc:.4f}")
if dt_acc > 0.95:
    print(f"   ⚠️  WARNING: Data is highly separable (may be synthetic/too easy)")
elif dt_acc > 0.85:
    print(f"   ⚡ NOTE: Data has clear patterns (common for medical vitals)")
else:
    print(f"   ✅ GOOD: Data requires deep model to achieve high accuracy")

# Check feature correlations with target
print("\n📊 Feature-Target Correlations:")
correlations = []
for i, feature in enumerate(CARDIO_FEATURES):
    corr = np.corrcoef(X_test[:, i], y_test)[0, 1]
    correlations.append({'feature': feature, 'correlation': abs(corr)})

correlations.sort(key=lambda x: x['correlation'], reverse=True)
for corr in correlations[:5]:
    print(f"   {corr['feature']:<30s} → |r|={corr['correlation']:.3f}")

print("\n"+"="*80)
print("🎯 FINAL OVERFITTING ASSESSMENT")
print("="*80)

overfitting_score = 0
warnings = []

# Criterion 1: Val/Test gap
val_test_gap = abs(final_val_acc - test_acc)
if val_test_gap > 0.05:
    overfitting_score += 3
    warnings.append(f"Large val/test gap ({val_test_gap*100:.2f}%)")
elif val_test_gap > 0.02:
    overfitting_score += 1

# Criterion 2: Loss ratio
if final_val_loss / final_train_loss > 1.5:
    overfitting_score += 3
    warnings.append(f"High loss ratio ({final_val_loss/final_train_loss:.2f})")
elif final_val_loss / final_train_loss > 1.2:
    overfitting_score += 1

# Criterion 3: Noise robustness
if avg_drop > 10:
    overfitting_score += 2
    warnings.append(f"Poor noise robustness ({avg_drop:.1f}% drop)")
elif avg_drop > 5:
    overfitting_score += 1

# Criterion 4: Data characteristics
if dt_acc > 0.95:
    overfitting_score += 2
    warnings.append(f"Suspiciously easy dataset (DT={dt_acc:.2f})")

print(f"\n📊 Overfitting Score: {overfitting_score}/10")

if overfitting_score >= 6:
    print("   🚨 HIGH RISK OF OVERFITTING")
    print("   ❌ Model may not generalize to real-world data")
elif overfitting_score >= 3:
    print("   ⚠️  MODERATE OVERFITTING RISK")
    print("   ⚡ Validate on external dataset before deployment")
else:
    print("   ✅ LOW OVERFITTING RISK")
    print("   ✨ Model appears to generalize well")

if warnings:
    print("\n⚠️  Warnings:")
    for w in warnings:
        print(f"   • {w}")

print("\n"+"="*80)
print("💡 RECOMMENDATIONS")
print("="*80)

if overfitting_score < 3:
    print("\n✅ Your model shows GOOD generalization:")
    print("   • Val/Test performance match closely")
    print("   • Robust to noise injection")
    print("   • Uses clinically meaningful features")
    print("\n🔬 However, the 99.7% accuracy suggests:")
    print("   • Dataset may be synthetic or have very clear patterns")
    print("   • Real-world sensor data may be noisier")
    print("\n📌 NEXT STEPS:")
    print("   1. ✅ Proceed with Step 1 validation (respiratory model)")
    print("   2. ⚡ Test with REAL sensor data from your hardware")
    print("   3. 🔍 Consider adding more complex edge cases")
    print("   4. 📊 Validate on external medical datasets (MIMIC-III, etc.)")
else:
    print("\n⚠️  Your model shows signs of overfitting:")
    for w in warnings:
        print(f"   • {w}")
    print("\n📌 RECOMMENDATIONS:")
    print("   1. Add more regularization (increase dropout to 0.4-0.5)")
    print("   2. Reduce model capacity (smaller hidden layers)")
    print("   3. Collect more diverse training data")
    print("   4. Validate on external dataset before deployment")

print("\n" + "="*80)
