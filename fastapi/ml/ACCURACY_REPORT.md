# ML Pipeline Accuracy Assessment Report

## Executive Summary

**Date**: January 7, 2026  
**Overall Status**: ⚠️ **NEEDS IMPROVEMENT** (1 of 3 tiers meets production requirements)

---

## Component Performance

### ✅ **Tier 2: Anomaly Detection - EXCELLENT**

**Metrics:**

- Accuracy: **95.5%** ✅
- ROC-AUC: **1.000** ✅ (Perfect discrimination)
- Precision: **91.7%**
- Recall: **100.0%** (Zero false negatives)
- F1-Score: **95.7%**

**Status:** ✅ **PRODUCTION READY**

**Analysis:**

- Perfect anomaly detection with zero false negatives (critical for healthcare)
- 91% of normal samples correctly identified
- 9% false positive rate (9 normal samples flagged as anomalies)
- Risk score distribution shows excellent separation:
  - Normal: Mean=26.9 (low risk)
  - Anomaly: Mean=97.2 (high risk)

**Note:** Minor version mismatch (trained on scikit-learn 1.7.1, running on 1.8.0) - recommend retraining to eliminate warnings.

---

### ⚠️ **Tier 1: Autoencoder - NEEDS IMPROVEMENT**

**Metrics:**

- Separation Ratio: **1.19x** ❌ (Target: >1.5x)
- Normal MSE: 7174.27
- Anomaly MSE: 8538.32

**Status:** ❌ **REQUIRES RETRAINING**

**Issues Identified:**

1. **Insufficient separation** between normal and anomaly reconstruction errors
2. Decoder appears undertrained (high baseline MSE for both classes)
3. Embedding variance for normal data is effectively zero (0.00) - indicates potential mode collapse

**Root Causes:**

- Decoder weights not properly trained (only encoder weights saved in encoder.pt)
- Limited training data diversity (120 synthetic samples)
- No validation set used during training

**Impact on Pipeline:**
Despite poor reconstruction, the encoder still produces usable embeddings for downstream tasks (anomaly detection works well).

---

### ❌ **Tier 3: Attention Fusion - POOR PERFORMANCE**

**Metrics:**

- Accuracy: **52.5%** ❌ (barely better than random)
- ROC-AUC: **0.543** ❌ (Target: >0.8)
- Precision: **52.3%**
- Recall: **57.5%**
- F1-Score: **54.8%**

**Status:** ❌ **CRITICAL - REQUIRES COMPLETE RETRAINING**

**Issues Identified:**

1. **Model performs at near-random level** (50% baseline for binary classification)
2. Attention weights are balanced (Cardio 49.6%, Resp 50.4%) but this doesn't help performance
3. Cannot distinguish matched vs mismatched physiological system pairs

**Root Causes:**

1. **Training data quality**: Both cardio and respiratory embeddings are duplicates of the same normal embeddings
   ```python
   # From validate_embeddings.ipynb:
   cardio_emb = normal_emb
   resp_emb = normal_emb.copy()  # IDENTICAL DATA
   ```
2. **No real paired system data** - model never learned meaningful cross-system correlations
3. **Self-supervised task too simple** - synthetic matched/mismatched pairs don't reflect real physiology

**Critical Finding:**
The fusion model was trained on **fake paired data** (identical embeddings labeled as different systems), so it never learned real multi-system integration.

---

## Detailed Recommendations

### Priority 1: Fix Tier 3 (Fusion Model) 🔴 CRITICAL

**Action Items:**

1. **Collect Real Paired Data**

   - Record actual cardiovascular + respiratory time series from same patients simultaneously
   - Generate separate embeddings from truly different physiological systems
   - Minimum dataset: 500+ paired samples with ground truth labels

2. **Redesign Training Strategy**

   ```python
   # Instead of:
   cardio_emb = normal_emb
   resp_emb = normal_emb.copy()

   # Use:
   cardio_emb = embed(cardiovascular_time_series)  # HR, SpO2
   resp_emb = embed(respiratory_time_series)       # RR, breath patterns
   ```

3. **Add Validation Set**

   - Split data: 70% train, 15% validation, 15% test
   - Monitor validation loss during training
   - Early stopping to prevent overfitting

4. **Consider Alternative Architectures**
   - Cross-attention mechanism instead of simple weighted sum
   - Transformer-based fusion
   - Multi-task learning (predict individual + systemic risk)

**Expected Improvement:** From 52.5% → 80%+ accuracy

---

### Priority 2: Retrain Tier 1 (Autoencoder) 🟡 MEDIUM

**Action Items:**

1. **Train Complete Autoencoder**

   - Save full model weights (encoder + decoder)
   - Current issue: Only encoder.pt exists, decoder is randomly initialized

2. **Increase Training Data**

   - Generate 1000+ synthetic sequences (currently only 120)
   - Add more variation in anomaly patterns
   - Include edge cases (tachycardia, bradycardia, hypoxemia)

3. **Add Regularization**

   ```python
   # In training loop:
   loss = reconstruction_loss + beta * kl_divergence(latent_dist, prior)
   ```

4. **Monitor Reconstruction Quality**
   - Track MSE per epoch
   - Visualize reconstructed signals
   - Target separation ratio > 1.5x

**Expected Improvement:** Separation ratio from 1.19x → 2.0x+

---

### Priority 3: Refresh Tier 2 (Anomaly Detection) 🟢 LOW

**Action Items:**

1. **Retrain with Current Scikit-Learn Version**

   ```bash
   python ml/validation/validate_embeddings.ipynb  # Re-run cells to regenerate model
   ```

2. **Minor Hyperparameter Tuning**

   - Test contamination values: [0.10, 0.15, 0.20]
   - Increase n_estimators: 200 → 300
   - Cross-validate to find optimal settings

3. **Reduce False Positives**
   - Current: 9% of normal samples flagged as anomalies
   - Adjust decision threshold from 50 → 60 for stricter detection
   - Balance false positive rate vs recall based on clinical requirements

**Expected Improvement:** Maintain 95%+ accuracy, reduce false positives from 9% → 5%

---

## Data Collection Strategy

### What You Need:

1. **Cardiovascular System**

   - Heart rate time series (60s @ 4 Hz = 240 samples)
   - SpO2 levels (same timestamps)
   - Labels: normal, tachycardia, arrhythmia, etc.

2. **Respiratory System**

   - Respiratory rate (same patient, same time)
   - Breathing pattern features
   - Cough/wheeze indicators
   - Labels: normal, dyspnea, hyperventilation, etc.

3. **Ground Truth Labels**
   - Matched pairs: Healthy patients (both systems normal)
   - Mismatched pairs: One system abnormal while other is normal
   - High risk pairs: Both systems showing distress

### Minimum Sample Requirements:

- **Matched (low risk)**: 300 samples
- **Mismatched (medium risk)**: 200 samples
- **Both abnormal (high risk)**: 200 samples
- **Total**: 700+ paired recordings

---

## Quick Wins (Can Implement Today)

### 1. Fix Autoencoder Training Script

Add decoder weight saving to [train_ae.py](train/train_ae.py):

```python
# Save complete model instead of just encoder
torch.save(model.state_dict(), "autoencoder.pt")  # Full model
torch.save(model.encoder.state_dict(), "encoder.pt")  # Encoder only
```

### 2. Add Data Augmentation

Create more diverse synthetic data:

```python
def augment_sequence(seq):
    # Add noise
    noisy = seq + np.random.randn(*seq.shape) * 0.5
    # Time warping
    warped = scipy.interpolate.interp1d(...)
    # Amplitude scaling
    scaled = seq * np.random.uniform(0.9, 1.1)
    return [noisy, warped, scaled]
```

### 3. Create Train/Val/Test Splits

Modify [train_fusion.py](fusion/train_fusion.py):

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

---

## Production Readiness Checklist

### Current Status:

| Component            | Accuracy  | Status       | Ready?  |
| -------------------- | --------- | ------------ | ------- |
| Autoencoder          | 1.19x sep | ❌ Poor      | No      |
| Anomaly Detection    | 95.5%     | ✅ Excellent | **Yes** |
| Fusion Model         | 52.5%     | ❌ Critical  | No      |
| **Overall Pipeline** | -         | ⚠️ Partial   | **No**  |

### What Works NOW:

✅ Individual risk assessment (cardiovascular OR respiratory)  
✅ Embedding generation from time series  
✅ Anomaly detection with high accuracy  
✅ Risk score normalization (0-100 scale)

### What Needs Work:

❌ Multi-system fusion (unusable in current state)  
❌ Autoencoder reconstruction quality  
❌ Cross-system correlation learning

---

## Recommended Timeline

### Week 1: Data Collection

- [ ] Set up data collection infrastructure
- [ ] Record 100+ real paired samples
- [ ] Label data with ground truth

### Week 2: Retrain Tier 3

- [ ] Prepare paired dataset
- [ ] Modify fusion training script
- [ ] Train with proper validation
- [ ] Target: 80%+ accuracy

### Week 3: Improve Tier 1

- [ ] Expand synthetic dataset
- [ ] Train complete autoencoder
- [ ] Validate reconstruction quality
- [ ] Target: 1.5x+ separation

### Week 4: Integration & Testing

- [ ] Refresh Tier 2 models
- [ ] End-to-end pipeline testing
- [ ] Performance benchmarking
- [ ] Deploy to staging

---

## Alternative Approach (If Data Collection Is Blocked)

If you cannot collect real paired physiological data:

### Option A: Remove Fusion Layer

- Use individual risk scores only (Tier 2 already works great)
- Aggregate with simple rules: `max(cardio_risk, resp_risk)`
- Deploy Tier 1+2 pipeline without fusion

### Option B: Use Domain Adaptation

- Find public datasets (MIMIC-III, PhysioNet)
- Transfer learn from ECG/PPG/respiratory datasets
- Fine-tune on your synthetic data

### Option C: Simulat Better Paired Data

- Use physiological models (cardiovascular-respiratory coupling)
- Generate correlated signals based on medical literature
- Create realistic matched/mismatched patterns

---

## Conclusion

**Bottom Line:**

- Your **anomaly detection is excellent** (95.5% accuracy, perfect ROC-AUC)
- Your **fusion model is broken** (52.5% accuracy = random guessing)
- Your **autoencoder is marginal** (works for embeddings but poor reconstruction)

**Immediate Action:**

1. **Deploy Tier 2 only** (individual risk assessment) - this works well
2. **Collect real paired data** to fix Tier 3
3. **Retrain autoencoder** for better representation learning

**Risk Mitigation:**

- Current system is safe for **single-system risk assessment**
- Do NOT use fusion scores in production (unreliable)
- Clearly communicate limitations to stakeholders

---

**Files Created:**

- [ml/evaluate_accuracy.py](ml/evaluate_accuracy.py) - Accuracy evaluation script
- [ml/ACCURACY_REPORT.md](ml/ACCURACY_REPORT.md) - This report
