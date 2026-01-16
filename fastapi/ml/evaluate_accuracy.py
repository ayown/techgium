"""
Model Accuracy Evaluation Script
Evaluates the performance of all 3 ML pipeline tiers with metrics:
1. Autoencoder reconstruction quality
2. Anomaly detection accuracy (ROC-AUC, precision, recall)
3. Fusion model classification performance
"""

import os
import sys
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.models.ts_autoencoder import AutoEncoder
from ml.data.synthetic_ts import generate_hr_spo2_sequence
from ml.inference.embed import TimeSeriesEmbedder
from ml.risk.anomaly_scorer import AnomalyScorer, RiskNormalizer
from ml.fusion.attention_fusion import AttentionFusion


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def evaluate_autoencoder():
    """Evaluate autoencoder reconstruction quality"""
    print_header("TIER 1: AUTOENCODER RECONSTRUCTION QUALITY")
    
    base_dir = os.path.dirname(__file__)
    encoder_path = os.path.join(base_dir, 'train', 'encoder.pt')
    autoencoder_path = os.path.join(base_dir, 'train', 'autoencoder_full.pt')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoEncoder(latent_dim=32, seq_len=256).to(device)

    # Reconstruction quality requires a trained decoder.
    # Prefer loading the full autoencoder checkpoint; fall back to encoder-only when missing.
    if os.path.exists(autoencoder_path):
        state = torch.load(autoencoder_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(f"⚠️  Full AE checkpoint not found at: {autoencoder_path}")
        print("    Falling back to encoder-only weights; Tier 1 reconstruction separation will be unreliable.")
        encoder_state = torch.load(encoder_path, map_location=device)
        model.encoder.load_state_dict(encoder_state)
    model.eval()
    
    # Generate test sequences
    print("Generating 50 normal and 50 anomaly test sequences...")
    normal_sequences = [generate_hr_spo2_sequence(256, anomaly=False) for _ in range(50)]
    anomaly_sequences = [generate_hr_spo2_sequence(256, anomaly=True) for _ in range(50)]
    
    def compute_reconstruction_error(sequences):
        errors = []
        with torch.no_grad():
            for seq in sequences:
                # seq is (T, 2), convert to (2, T)
                x = torch.tensor(seq.T, dtype=torch.float32).unsqueeze(0).to(device)
                x_hat, _ = model(x)
                
                # Mean squared error
                mse = ((x - x_hat) ** 2).mean().item()
                errors.append(mse)
        return np.array(errors)
    
    normal_errors = compute_reconstruction_error(normal_sequences)
    anomaly_errors = compute_reconstruction_error(anomaly_sequences)
    
    print(f"\n📊 Reconstruction MSE Results:")
    print(f"   Normal sequences:  Mean={normal_errors.mean():.4f}, Std={normal_errors.std():.4f}")
    print(f"   Anomaly sequences: Mean={anomaly_errors.mean():.4f}, Std={anomaly_errors.std():.4f}")
    
    # Compute separation quality
    separation_ratio = anomaly_errors.mean() / (normal_errors.mean() + 1e-6)
    print(f"\n✅ Separation Ratio (Anomaly/Normal): {separation_ratio:.2f}x")
    
    if separation_ratio > 1.5:
        print("   ✅ EXCELLENT - Clear separation between normal and anomaly")
    elif separation_ratio > 1.2:
        print("   ⚠️  GOOD - Moderate separation")
    else:
        print("   ❌ POOR - Insufficient separation")
    
    # Compute latent space quality
    embedder = TimeSeriesEmbedder()
    normal_emb = np.array([embedder.embed(torch.tensor(s.T, dtype=torch.float32)).flatten() 
                           for s in normal_sequences])
    anomaly_emb = np.array([embedder.embed(torch.tensor(s.T, dtype=torch.float32)).flatten() 
                            for s in anomaly_sequences])
    
    # Measure embedding variance
    normal_variance = np.var(normal_emb, axis=0).mean()
    anomaly_variance = np.var(anomaly_emb, axis=0).mean()
    
    print(f"\n📊 Embedding Space Variance:")
    print(f"   Normal: {normal_variance:.2f}")
    print(f"   Anomaly: {anomaly_variance:.2f}")
    
    return {
        'normal_mse': normal_errors.mean(),
        'anomaly_mse': anomaly_errors.mean(),
        'separation_ratio': separation_ratio
    }


def evaluate_anomaly_detection():
    """Evaluate anomaly detection accuracy"""
    print_header("TIER 2: ANOMALY DETECTION PERFORMANCE")
    
    base_dir = os.path.dirname(__file__)
    anomaly_path = os.path.join(base_dir, 'risk', 'anomaly_model.joblib')
    risk_bounds_path = os.path.join(base_dir, 'risk', 'risk_bounds.npy')
    
    # Generate test data
    print("Generating 100 normal and 100 anomaly test sequences...")
    normal_sequences = [generate_hr_spo2_sequence(256, anomaly=False) for _ in range(100)]
    anomaly_sequences = [generate_hr_spo2_sequence(256, anomaly=True) for _ in range(100)]
    
    # Generate embeddings
    embedder = TimeSeriesEmbedder()
    
    normal_emb = np.array([embedder.embed(torch.tensor(s.T, dtype=torch.float32)).flatten() 
                           for s in normal_sequences])
    anomaly_emb = np.array([embedder.embed(torch.tensor(s.T, dtype=torch.float32)).flatten() 
                            for s in anomaly_sequences])
    
    X = np.vstack([normal_emb, anomaly_emb])
    y_true = np.array([0] * len(normal_emb) + [1] * len(anomaly_emb))
    
    # Load anomaly scorer
    scorer = AnomalyScorer(anomaly_path)
    risk_bounds = np.load(risk_bounds_path)
    normalizer = RiskNormalizer(risk_bounds[0], risk_bounds[1])
    
    # Compute risk scores
    risk_scores = np.array([normalizer.normalize(scorer.score(emb)) for emb in X])
    
    # Use threshold of 50 for binary classification
    y_pred = (risk_scores > 50).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # ROC-AUC using continuous scores
    roc_auc = roc_auc_score(y_true, risk_scores)
    
    print(f"\n📊 Binary Classification Metrics (Threshold=50):")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1-Score:  {f1:.3f}")
    print(f"   ROC-AUC:   {roc_auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"                  Predicted Normal  Predicted Anomaly")
    print(f"   Actual Normal      {cm[0,0]:4d}              {cm[0,1]:4d}")
    print(f"   Actual Anomaly     {cm[1,0]:4d}              {cm[1,1]:4d}")
    
    # Quality assessment
    print(f"\n✅ Model Quality Assessment:")
    if accuracy >= 0.9 and roc_auc >= 0.95:
        print("   ✅ EXCELLENT - High accuracy and discrimination")
    elif accuracy >= 0.8 and roc_auc >= 0.85:
        print("   ✅ GOOD - Acceptable performance for production")
    elif accuracy >= 0.7:
        print("   ⚠️  FAIR - May need retraining or tuning")
    else:
        print("   ❌ POOR - Requires retraining")
    
    # Score distribution analysis
    normal_scores = risk_scores[:len(normal_emb)]
    anomaly_scores = risk_scores[len(normal_emb):]
    
    print(f"\n📊 Risk Score Distribution:")
    print(f"   Normal:  Mean={normal_scores.mean():.1f}, Std={normal_scores.std():.1f}, Range=[{normal_scores.min():.1f}, {normal_scores.max():.1f}]")
    print(f"   Anomaly: Mean={anomaly_scores.mean():.1f}, Std={anomaly_scores.std():.1f}, Range=[{anomaly_scores.min():.1f}, {anomaly_scores.max():.1f}]")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def evaluate_fusion_model():
    """Evaluate attention fusion model"""
    print_header("TIER 3: ATTENTION FUSION PERFORMANCE")
    
    base_dir = os.path.dirname(__file__)
    fusion_path = os.path.join(base_dir, 'fusion', 'fusion_model.pt')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionFusion().to(device)
    model.load_state_dict(torch.load(fusion_path, map_location=device))
    model.eval()
    
    print("Generating 200 matched and 200 mismatched pairs...")

    # Prefer evaluating on the same embedding distribution used for fusion training.
    cardio_path = os.path.join(base_dir, 'validation', 'cardio_embeddings.npy')
    resp_path = os.path.join(base_dir, 'validation', 'resp_embeddings.npy')

    n_samples = 200
    rng = np.random.default_rng(42)

    if os.path.exists(cardio_path) and os.path.exists(resp_path):
        cardio_all = np.load(cardio_path).astype(np.float32)
        resp_all = np.load(resp_path).astype(np.float32)

        if cardio_all.ndim != 2 or resp_all.ndim != 2 or cardio_all.shape != resp_all.shape:
            raise ValueError(f"Expected cardio/resp arrays with shape (N, D) and same shape. Got cardio={cardio_all.shape}, resp={resp_all.shape}")

        n_total = cardio_all.shape[0]
        idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
        cardio_matched = cardio_all[idx]
        resp_matched = resp_all[idx]

        perm = rng.permutation(idx)
        if np.all(perm == idx) and len(idx) > 1:
            perm = np.roll(perm, 1)
        cardio_mismatched = cardio_all[idx]
        resp_mismatched = resp_all[perm]

        cardio = np.vstack([cardio_matched, cardio_mismatched])
        resp = np.vstack([resp_matched, resp_mismatched])
        y_true = np.array([0] * len(cardio_matched) + [1] * len(cardio_mismatched))
    else:
        # Fallback: purely synthetic paired embeddings
        cardio_matched = rng.normal(0.0, 1.0, size=(n_samples, 32)).astype(np.float32)
        resp_matched = cardio_matched + rng.normal(0.0, 0.3, size=(n_samples, 32)).astype(np.float32)

        cardio_mismatched = rng.normal(0.0, 1.0, size=(n_samples, 32)).astype(np.float32)
        resp_mismatched = rng.normal(0.0, 1.0, size=(n_samples, 32)).astype(np.float32)

        cardio = np.vstack([cardio_matched, cardio_mismatched])
        resp = np.vstack([resp_matched, resp_mismatched])
        y_true = np.array([0] * n_samples + [1] * n_samples)  # 0=matched (low risk), 1=mismatched (high risk)
    
    # Predict with fusion model
    systemic_risks = []
    attention_weights_list = []
    
    with torch.no_grad():
        for i in range(len(cardio)):
            emb = np.stack([cardio[i], resp[i]], axis=0)
            emb_tensor = torch.from_numpy(emb).to(device)
            
            output = model(emb_tensor)
            risk = torch.sigmoid(output['systemic_risk'])
            systemic_risks.append(risk.cpu().item())
            attention_weights_list.append(output['system_weights'])
    
    systemic_risks = np.array(systemic_risks)
    
    # Binary classification (threshold 0.5)
    y_pred = (systemic_risks > 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, systemic_risks)
    
    print(f"\n📊 Fusion Classification Metrics (Threshold=0.5):")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1-Score:  {f1:.3f}")
    print(f"   ROC-AUC:   {roc_auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"                  Predicted Low Risk  Predicted High Risk")
    print(f"   Actual Low      {cm[0,0]:4d}                 {cm[0,1]:4d}")
    print(f"   Actual High     {cm[1,0]:4d}                 {cm[1,1]:4d}")
    
    # Attention weights analysis
    attention_weights_array = np.array(attention_weights_list)
    cardio_weights = attention_weights_array[:, 0]
    resp_weights = attention_weights_array[:, 1]
    
    print(f"\n📊 Attention Weights Distribution:")
    print(f"   Cardio:      Mean={cardio_weights.mean():.3f}, Std={cardio_weights.std():.3f}")
    print(f"   Respiratory: Mean={resp_weights.mean():.3f}, Std={resp_weights.std():.3f}")
    
    # Check if attention is balanced or biased
    if abs(cardio_weights.mean() - 0.5) < 0.1:
        print("   ✅ Balanced attention between systems")
    else:
        dominant = "Cardio" if cardio_weights.mean() > 0.5 else "Respiratory"
        print(f"   ⚠️  Attention biased towards {dominant}")
    
    # Quality assessment
    print(f"\n✅ Fusion Model Quality Assessment:")
    if accuracy >= 0.85 and roc_auc >= 0.9:
        print("   ✅ EXCELLENT - Strong multi-system integration")
    elif accuracy >= 0.75 and roc_auc >= 0.8:
        print("   ✅ GOOD - Acceptable fusion performance")
    elif accuracy >= 0.65:
        print("   ⚠️  FAIR - May need more training data")
    else:
        print("   ❌ POOR - Requires retraining")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'attention_balance': abs(cardio_weights.mean() - 0.5)
    }


def main():
    print("\n" + "🎯" * 40)
    print("  ML PIPELINE ACCURACY EVALUATION")
    print("🎯" * 40)
    
    # Evaluate each tier
    autoencoder_metrics = evaluate_autoencoder()
    anomaly_metrics = evaluate_anomaly_detection()
    fusion_metrics = evaluate_fusion_model()
    
    # Overall summary
    print_header("OVERALL PERFORMANCE SUMMARY")
    
    print("\n🔹 Tier 1: Autoencoder")
    print(f"   Separation Ratio: {autoencoder_metrics['separation_ratio']:.2f}x")
    print(f"   Status: {'✅ PASS' if autoencoder_metrics['separation_ratio'] > 1.2 else '❌ FAIL'}")
    
    print("\n🔹 Tier 2: Anomaly Detection")
    print(f"   Accuracy: {anomaly_metrics['accuracy']:.1%}")
    print(f"   ROC-AUC:  {anomaly_metrics['roc_auc']:.3f}")
    print(f"   F1-Score: {anomaly_metrics['f1']:.3f}")
    print(f"   Status: {'✅ PASS' if anomaly_metrics['accuracy'] >= 0.8 and anomaly_metrics['roc_auc'] >= 0.85 else '❌ FAIL'}")
    
    print("\n🔹 Tier 3: Attention Fusion")
    print(f"   Accuracy: {fusion_metrics['accuracy']:.1%}")
    print(f"   ROC-AUC:  {fusion_metrics['roc_auc']:.3f}")
    print(f"   F1-Score: {fusion_metrics['f1']:.3f}")
    print(f"   Status: {'✅ PASS' if fusion_metrics['accuracy'] >= 0.75 and fusion_metrics['roc_auc'] >= 0.8 else '❌ FAIL'}")
    
    # Overall verdict
    all_pass = (
        autoencoder_metrics['separation_ratio'] > 1.2 and
        anomaly_metrics['accuracy'] >= 0.8 and
        anomaly_metrics['roc_auc'] >= 0.85 and
        fusion_metrics['accuracy'] >= 0.75 and
        fusion_metrics['roc_auc'] >= 0.8
    )
    
    print("\n" + "=" * 80)
    if all_pass:
        print("🎉 OVERALL VERDICT: ALL MODELS MEET ACCURACY REQUIREMENTS")
        print("✅ Pipeline is ready for production deployment")
    else:
        print("⚠️  OVERALL VERDICT: SOME MODELS NEED IMPROVEMENT")
        print("📝 Recommendations:")
        
        if autoencoder_metrics['separation_ratio'] <= 1.2:
            print("   - Retrain autoencoder with more diverse data")
        if anomaly_metrics['accuracy'] < 0.8 or anomaly_metrics['roc_auc'] < 0.85:
            print("   - Tune anomaly detection hyperparameters (contamination, n_estimators)")
        if fusion_metrics['accuracy'] < 0.75 or fusion_metrics['roc_auc'] < 0.8:
            print("   - Collect more paired system data for fusion training")
    print("=" * 80)


if __name__ == "__main__":
    main()
