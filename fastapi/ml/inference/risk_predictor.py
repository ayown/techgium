"""
Production-ready inference pipeline for IoT sensors → risk prediction
Combines tabular classifier with real-time sensor data processing
"""

import torch
import numpy as np
import joblib
import os
from typing import Dict, Tuple


class HealthRiskPredictor:
    """
    High-accuracy risk prediction for health chamber walkthrough
    Processes real-time vital signs and IoT sensor data
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize predictor with trained models
        
        Args:
            model_dir: Directory containing trained models (default: ml/train/)
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'train')
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tabular classifier
        from ml.models.tabular_classifier import VitalsRiskClassifier
        
        classifier_path = os.path.join(model_dir, 'vitals_classifier_best.pt')
        scaler_path = os.path.join(model_dir, 'vitals_scaler.joblib')
        
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Model not found: {classifier_path}\nRun: python -m ml.train.train_tabular_classifier")
        
        # Load model
        checkpoint = torch.load(classifier_path, map_location=self.device)
        self.feature_cols = checkpoint['feature_cols']
        
        self.classifier = VitalsRiskClassifier(
            input_dim=len(self.feature_cols),
            hidden_dims=[128, 64, 32],
            dropout=0.3
        ).to(self.device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        print(f"✅ Loaded model (Val AUC: {checkpoint['val_auc']:.4f})")
    
    def predict(self, patient_data: Dict) -> Dict:
        """
        Predict health risk from patient vitals
        
        Args:
            patient_data: Dictionary with vital signs
                {
                    'Heart Rate': 75.0,
                    'Oxygen Saturation': 98.0,
                    'Systolic Blood Pressure': 120.0,
                    'Diastolic Blood Pressure': 80.0,
                    'Age': 45,
                    'Gender': 'Male',
                    ... (all 14 features)
                }
        
        Returns:
            {
                'risk_score': 45.2,  # 0-100
                'risk_level': 'YELLOW',  # GREEN/YELLOW/RED
                'risk_category': 'Low Risk',  # Low Risk / High Risk
                'confidence': 0.92  # Model confidence
            }
        """
        # Extract features
        features = self._extract_features(patient_data)
        
        # Normalize
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        with torch.no_grad():
            features_tensor = torch.from_numpy(features_scaled).float().to(self.device)
            risk_prob = self.classifier(features_tensor).cpu().numpy().squeeze()
        
        risk_score = float(risk_prob * 100)
        
        # Determine risk level
        if risk_score < 30:
            risk_level = 'GREEN'
            risk_category = 'Low Risk'
        elif risk_score < 60:
            risk_level = 'YELLOW'
            risk_category = 'Moderate Risk'
        else:
            risk_level = 'RED'
            risk_category = 'High Risk'
        
        # Confidence (distance from decision boundary at 0.5)
        confidence = abs(risk_prob - 0.5) * 2  # Map [0, 0.5] → [0, 1]
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'risk_category': risk_category,
            'confidence': round(float(confidence), 3),
            'risk_probability': round(float(risk_prob), 4)
        }
    
    def predict_batch(self, patient_data_list):
        """Predict for multiple patients at once"""
        results = []
        for patient_data in patient_data_list:
            results.append(self.predict(patient_data))
        return results
    
    def _extract_features(self, patient_data: Dict) -> np.ndarray:
        """Extract feature vector from patient data"""
        feature_values = []
        
        for col in self.feature_cols:
            if col == 'Gender_Encoded':
                value = 1.0 if patient_data.get('Gender') == 'Male' else 0.0
            else:
                value = float(patient_data.get(col, 0))
                
                # Handle missing derived features
                if col == 'Derived_Pulse_Pressure' and value == 0:
                    value = patient_data.get('Systolic Blood Pressure', 120) - \
                            patient_data.get('Diastolic Blood Pressure', 80)
                elif col == 'Derived_BMI' and value == 0:
                    weight = patient_data.get('Weight (kg)', 70)
                    height = patient_data.get('Height (m)', 1.7)
                    value = weight / (height ** 2) if height > 0 else 22.0
                elif col == 'Derived_MAP' and value == 0:
                    sys_bp = patient_data.get('Systolic Blood Pressure', 120)
                    dia_bp = patient_data.get('Diastolic Blood Pressure', 80)
                    value = (sys_bp + 2 * dia_bp) / 3
                elif col == 'Derived_HRV' and value == 0:
                    value = 0.1  # Default HRV
            
            feature_values.append(value)
        
        return np.array(feature_values, dtype=np.float32)
    
    def get_feature_importance(self, patient_data: Dict, baseline_data: Dict = None):
        """
        Calculate which features contribute most to the risk score
        Useful for explaining predictions to clinicians
        
        Args:
            patient_data: Patient vitals
            baseline_data: Optional baseline for comparison (default: population average)
        
        Returns:
            List of (feature_name, importance_score) tuples
        """
        if baseline_data is None:
            # Use population averages as baseline
            baseline_data = {
                'Heart Rate': 75,
                'Respiratory Rate': 16,
                'Body Temperature': 37.0,
                'Oxygen Saturation': 98,
                'Systolic Blood Pressure': 120,
                'Diastolic Blood Pressure': 80,
                'Age': 45,
                'Weight (kg)': 70,
                'Height (m)': 1.7,
                'Gender': 'Male',
                'Derived_HRV': 0.1,
                'Derived_Pulse_Pressure': 40,
                'Derived_BMI': 24.2,
                'Derived_MAP': 93.3
            }
        
        # Get baseline prediction
        baseline_result = self.predict(baseline_data)
        baseline_score = baseline_result['risk_score']
        
        # Get actual prediction
        actual_result = self.predict(patient_data)
        actual_score = actual_result['risk_score']
        
        # Calculate feature importance by ablation
        importances = []
        for feature in self.feature_cols:
            if feature == 'Gender_Encoded':
                continue
            
            # Create modified data with this feature at baseline
            modified_data = patient_data.copy()
            modified_data[feature] = baseline_data.get(feature, 0)
            
            # Get prediction with modified feature
            modified_result = self.predict(modified_data)
            modified_score = modified_result['risk_score']
            
            # Importance = how much score changes when feature is at baseline
            importance = abs(actual_score - modified_score)
            importances.append((feature, importance))
        
        # Sort by importance
        importances.sort(key=lambda x: x[1], reverse=True)
        
        return importances


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("HEALTH RISK PREDICTOR - INFERENCE EXAMPLE")
    print("=" * 80)
    
    # Initialize predictor
    predictor = HealthRiskPredictor()
    
    # Example patient data from IoT sensors
    patient = {
        'Heart Rate': 92,  # Elevated
        'Respiratory Rate': 18,
        'Body Temperature': 37.2,
        'Oxygen Saturation': 96,  # Slightly low
        'Systolic Blood Pressure': 145,  # High
        'Diastolic Blood Pressure': 92,  # High
        'Age': 65,  # Older patient
        'Weight (kg)': 85,
        'Height (m)': 1.75,
        'Gender': 'Male',
        'Derived_HRV': 0.06,  # Low variability
        'Derived_Pulse_Pressure': 53,
        'Derived_BMI': 27.8,  # Overweight
        'Derived_MAP': 109.7  # High
    }
    
    # Predict risk
    result = predictor.predict(patient)
    
    print(f"\n📊 Patient Assessment:")
    print(f"   Risk Score: {result['risk_score']}/100")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Category: {result['risk_category']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    
    # Feature importance
    print(f"\n🔍 Key Risk Factors:")
    importances = predictor.get_feature_importance(patient)
    for i, (feature, importance) in enumerate(importances[:5], 1):
        print(f"   {i}. {feature}: +{importance:.1f} points")
    
    print("\n✅ Inference complete!")
