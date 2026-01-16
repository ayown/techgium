"""
ML Pipeline Validation Script
Tests all 3 tiers of the health risk assessment pipeline:
1. Time Series Autoencoder (encoder.pt)
2. Anomaly Detection (anomaly_model.joblib)
3. Attention Fusion (fusion_model.pt)
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.models.ts_autoencoder import Encoder, AutoEncoder
from ml.fusion.attention_fusion import AttentionFusion
from ml.risk.anomaly_scorer import AnomalyScorer, RiskNormalizer
from ml.inference.embed import TimeSeriesEmbedder


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_model_loading():
    """Test 1: Verify all model files can be loaded"""
    print_section("TEST 1: Model Loading")
    
    # Get base directory
    base_dir = os.path.dirname(__file__)
    
    # Model paths
    encoder_path = os.path.join(base_dir, 'train', 'encoder.pt')
    fusion_path = os.path.join(base_dir, 'fusion', 'fusion_model.pt')
    anomaly_path = os.path.join(base_dir, 'risk', 'anomaly_model.joblib')
    risk_bounds_path = os.path.join(base_dir, 'risk', 'risk_bounds.npy')
    
    results = {}
    
    # Test encoder
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = Encoder(latent_dim=32).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder.eval()
        print(f"✅ Encoder loaded successfully from {encoder_path}")
        print(f"   Device: {device}")
        results['encoder'] = True
    except Exception as e:
        print(f"❌ Encoder loading failed: {e}")
        results['encoder'] = False
    
    # Test fusion model
    try:
        fusion = AttentionFusion().to(device)
        fusion.load_state_dict(torch.load(fusion_path, map_location=device))
        fusion.eval()
        print(f"✅ Fusion model loaded successfully from {fusion_path}")
        results['fusion'] = True
    except Exception as e:
        print(f"❌ Fusion model loading failed: {e}")
        results['fusion'] = False
    
    # Test anomaly model
    try:
        import joblib
        anomaly_model = joblib.load(anomaly_path)
        print(f"✅ Anomaly model loaded successfully from {anomaly_path}")
        print(f"   Model type: {type(anomaly_model).__name__}")
        results['anomaly'] = True
    except Exception as e:
        print(f"❌ Anomaly model loading failed: {e}")
        results['anomaly'] = False
    
    # Test risk bounds
    try:
        risk_bounds = np.load(risk_bounds_path)
        print(f"✅ Risk bounds loaded successfully from {risk_bounds_path}")
        print(f"   Bounds: {risk_bounds}")
        results['risk_bounds'] = True
    except Exception as e:
        print(f"❌ Risk bounds loading failed: {e}")
        results['risk_bounds'] = False
    
    return results


def test_embedder():
    """Test 2: TimeSeriesEmbedder with synthetic data"""
    print_section("TEST 2: Time Series Embedding")
    
    try:
        embedder = TimeSeriesEmbedder()
        
        # Generate synthetic time series (HR + SpO2)
        # Shape: (2, T) where T is time steps
        T = 256
        hr_signal = 70 + 10 * np.sin(np.linspace(0, 4 * np.pi, T)) + np.random.randn(T) * 2
        spo2_signal = 96 + 2 * np.cos(np.linspace(0, 3 * np.pi, T)) + np.random.randn(T) * 1
        
        time_series = np.stack([hr_signal, spo2_signal], axis=0).astype(np.float32)
        print(f"Input shape: {time_series.shape}")
        print(f"HR range: [{hr_signal.min():.1f}, {hr_signal.max():.1f}]")
        print(f"SpO2 range: [{spo2_signal.min():.1f}, {spo2_signal.max():.1f}]")
        
        # Generate embedding
        embedding = embedder.embed(time_series)
        
        print(f"✅ Embedding generated successfully")
        print(f"   Output shape: {embedding.shape}")
        print(f"   Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        print(f"   Embedding mean: {embedding.mean():.3f}, std: {embedding.std():.3f}")
        
        return embedding
    except Exception as e:
        print(f"❌ Embedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_anomaly_scorer(embedding):
    """Test 3: Anomaly scoring"""
    print_section("TEST 3: Anomaly Detection & Risk Scoring")
    
    if embedding is None:
        print("⚠️  Skipping (no embedding from previous test)")
        return None
    
    try:
        base_dir = os.path.dirname(__file__)
        anomaly_path = os.path.join(base_dir, 'risk', 'anomaly_model.joblib')
        risk_bounds_path = os.path.join(base_dir, 'risk', 'risk_bounds.npy')
        
        # Load scorer and normalizer
        scorer = AnomalyScorer(anomaly_path)
        risk_bounds = np.load(risk_bounds_path)
        normalizer = RiskNormalizer(risk_bounds[0], risk_bounds[1])
        
        # Score the embedding
        raw_score = scorer.score(embedding)
        normalized_score = normalizer.normalize(raw_score)
        
        print(f"✅ Anomaly scoring successful")
        print(f"   Raw anomaly score: {raw_score:.4f}")
        print(f"   Normalized risk score (0-100): {normalized_score:.2f}")
        print(f"   Risk bounds: [{risk_bounds[0]:.4f}, {risk_bounds[1]:.4f}]")
        
        return normalized_score
    except Exception as e:
        print(f"❌ Anomaly scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_fusion():
    """Test 4: Multi-system attention fusion"""
    print_section("TEST 4: Attention Fusion (Multi-System)")
    
    try:
        base_dir = os.path.dirname(__file__)
        fusion_path = os.path.join(base_dir, 'fusion', 'fusion_model.pt')
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fusion = AttentionFusion().to(device)
        fusion.load_state_dict(torch.load(fusion_path, map_location=device))
        fusion.eval()
        
        # Create synthetic embeddings for cardio and respiratory systems
        cardio_emb = np.random.randn(32).astype(np.float32)
        resp_emb = np.random.randn(32).astype(np.float32)
        
        # Stack into (num_systems, embedding_dim)
        multi_system_emb = np.stack([cardio_emb, resp_emb], axis=0)
        
        print(f"Input: 2 system embeddings, shape {multi_system_emb.shape}")
        
        # Run fusion
        with torch.no_grad():
            emb_tensor = torch.from_numpy(multi_system_emb).to(device)
            output = fusion(emb_tensor)
        
        weights = output['system_weights']
        systemic_risk = output['systemic_risk'].cpu().item()
        
        print(f"✅ Fusion successful")
        print(f"   System attention weights: {weights}")
        print(f"   Weights sum: {weights.sum():.4f} (should be ~1.0)")
        print(f"   Systemic risk score: {systemic_risk:.4f}")
        
        return output
    except Exception as e:
        print(f"❌ Fusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_end_to_end():
    """Test 5: Complete pipeline with 2 systems"""
    print_section("TEST 5: End-to-End Pipeline (Cardio + Respiratory)")
    
    try:
        # Initialize components
        embedder = TimeSeriesEmbedder()
        
        base_dir = os.path.dirname(__file__)
        anomaly_path = os.path.join(base_dir, 'risk', 'anomaly_model.joblib')
        risk_bounds_path = os.path.join(base_dir, 'risk', 'risk_bounds.npy')
        fusion_path = os.path.join(base_dir, 'fusion', 'fusion_model.pt')
        
        scorer = AnomalyScorer(anomaly_path)
        risk_bounds = np.load(risk_bounds_path)
        normalizer = RiskNormalizer(risk_bounds[0], risk_bounds[1])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fusion = AttentionFusion().to(device)
        fusion.load_state_dict(torch.load(fusion_path, map_location=device))
        fusion.eval()
        
        # Generate synthetic data for cardio system
        T = 256
        hr = 75 + 8 * np.sin(np.linspace(0, 4 * np.pi, T)) + np.random.randn(T) * 3
        spo2 = 97 + 1.5 * np.cos(np.linspace(0, 3 * np.pi, T)) + np.random.randn(T) * 0.5
        cardio_ts = np.stack([hr, spo2], axis=0).astype(np.float32)
        
        # Generate synthetic data for respiratory system
        rr = 16 + 2 * np.sin(np.linspace(0, 2 * np.pi, T)) + np.random.randn(T) * 1
        spo2_2 = 95 + 3 * np.cos(np.linspace(0, 2 * np.pi, T)) + np.random.randn(T) * 1
        resp_ts = np.stack([rr, spo2_2], axis=0).astype(np.float32)
        
        print("Step 1: Generate embeddings")
        cardio_emb = embedder.embed(cardio_ts)
        resp_emb = embedder.embed(resp_ts)
        print(f"  Cardio embedding: {cardio_emb.shape}")
        print(f"  Respiratory embedding: {resp_emb.shape}")
        
        print("\nStep 2: Calculate individual risk scores")
        cardio_score = normalizer.normalize(scorer.score(cardio_emb))
        resp_score = normalizer.normalize(scorer.score(resp_emb))
        print(f"  Cardio risk: {cardio_score:.2f}")
        print(f"  Respiratory risk: {resp_score:.2f}")
        
        print("\nStep 3: Fuse embeddings")
        multi_emb = np.stack([cardio_emb.flatten(), resp_emb.flatten()], axis=0)
        with torch.no_grad():
            emb_tensor = torch.from_numpy(multi_emb).to(device)
            fusion_output = fusion(emb_tensor)
        
        weights = fusion_output['system_weights']
        systemic_risk = fusion_output['systemic_risk'].cpu().item()
        
        print(f"  Attention weights: Cardio={weights[0]:.3f}, Respiratory={weights[1]:.3f}")
        print(f"  Systemic risk: {systemic_risk:.4f}")
        
        print("\n" + "=" * 70)
        print("✅ END-TO-END PIPELINE SUCCESSFUL")
        print("=" * 70)
        print(f"Final Health Risk Index:")
        print(f"  - Cardiovascular Risk: {cardio_score:.2f}/100")
        print(f"  - Respiratory Risk: {resp_score:.2f}/100")
        print(f"  - Systemic Risk (Fusion): {systemic_risk * 100:.2f}/100")
        print(f"  - System Importance: Cardio {weights[0]*100:.1f}%, Resp {weights[1]*100:.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "🔬" * 35)
    print("  ML PIPELINE VALIDATION TEST SUITE")
    print("🔬" * 35)
    
    # Test 1: Model loading
    model_results = test_model_loading()
    
    # Test 2: Embedder
    embedding = test_embedder()
    
    # Test 3: Anomaly scorer
    risk_score = test_anomaly_scorer(embedding)
    
    # Test 4: Fusion
    fusion_output = test_fusion()
    
    # Test 5: End-to-end
    e2e_success = test_end_to_end()
    
    # Final summary
    print_section("VALIDATION SUMMARY")
    print(f"✅ Encoder Loading: {'PASS' if model_results.get('encoder') else 'FAIL'}")
    print(f"✅ Fusion Model Loading: {'PASS' if model_results.get('fusion') else 'FAIL'}")
    print(f"✅ Anomaly Model Loading: {'PASS' if model_results.get('anomaly') else 'FAIL'}")
    print(f"✅ Risk Bounds Loading: {'PASS' if model_results.get('risk_bounds') else 'FAIL'}")
    print(f"✅ Time Series Embedding: {'PASS' if embedding is not None else 'FAIL'}")
    print(f"✅ Anomaly Scoring: {'PASS' if risk_score is not None else 'FAIL'}")
    print(f"✅ Attention Fusion: {'PASS' if fusion_output is not None else 'FAIL'}")
    print(f"✅ End-to-End Pipeline: {'PASS' if e2e_success else 'FAIL'}")
    
    all_pass = all(model_results.values()) and embedding is not None and \
               risk_score is not None and fusion_output is not None and e2e_success
    
    print("\n" + "=" * 70)
    if all_pass:
        print("🎉 ALL TESTS PASSED - ML PIPELINE IS FULLY FUNCTIONAL")
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    print("=" * 70)


if __name__ == "__main__":
    main()
