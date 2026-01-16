"""
FastAPI endpoints for health chamber walkthrough diagnosis
High-accuracy IoT sensor → risk assessment pipeline
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from ml.inference.risk_predictor import HealthRiskPredictor
import os

router = APIRouter(prefix="/api/health", tags=["Health Assessment"])

# Initialize predictor (loads once at startup)
try:
    predictor = HealthRiskPredictor()
    PREDICTOR_READY = True
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    print("   Run: python -m ml.train.train_tabular_classifier")
    PREDICTOR_READY = False


# Request/Response Models
class VitalSigns(BaseModel):
    """Patient vital signs from IoT sensors"""
    heart_rate: float = Field(..., ge=30, le=200, description="Heart rate in BPM")
    respiratory_rate: float = Field(..., ge=8, le=40, description="Breaths per minute")
    body_temperature: float = Field(..., ge=35.0, le=42.0, description="Temperature in °C")
    oxygen_saturation: float = Field(..., ge=70, le=100, description="SpO2 percentage")
    systolic_bp: float = Field(..., ge=70, le=250, description="Systolic blood pressure")
    diastolic_bp: float = Field(..., ge=40, le=150, description="Diastolic blood pressure")
    
    # Patient demographics
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(Male|Female)$")
    weight_kg: float = Field(..., ge=20, le=300)
    height_m: float = Field(..., ge=0.5, le=2.5)
    
    # Optional derived metrics (auto-calculated if missing)
    hrv: Optional[float] = Field(None, ge=0, le=1, description="Heart rate variability")
    
    class Config:
        json_schema_extra = {
            "example": {
                "heart_rate": 75,
                "respiratory_rate": 16,
                "body_temperature": 37.0,
                "oxygen_saturation": 98,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "age": 45,
                "gender": "Male",
                "weight_kg": 70,
                "height_m": 1.75,
                "hrv": 0.1
            }
        }


class RiskAssessment(BaseModel):
    """Risk assessment result"""
    risk_score: float = Field(..., description="Risk score 0-100")
    risk_level: str = Field(..., description="GREEN/YELLOW/RED")
    risk_category: str = Field(..., description="Low Risk / Moderate Risk / High Risk")
    confidence: float = Field(..., description="Model confidence 0-1")
    risk_probability: float = Field(..., description="Raw probability 0-1")
    recommendations: List[str] = Field(..., description="Clinical recommendations")
    key_risk_factors: List[Dict[str, float]] = Field(..., description="Top contributing factors")


# Endpoints
@router.post("/assess", response_model=RiskAssessment)
async def assess_health_risk(vitals: VitalSigns):
    """
    **Health Chamber Walkthrough Diagnosis**
    
    Analyzes patient vital signs from IoT sensors and returns comprehensive risk assessment.
    
    - **Input**: Real-time vital signs from sensors
    - **Output**: Risk score, level, and clinical recommendations
    - **Model**: 17-feature tabular classifier (trained on 200k patients)
    - **Accuracy**: >95% on validation set
    """
    if not PREDICTOR_READY:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run: python -m ml.train.train_tabular_classifier"
        )
    
    # Convert to predictor format
    patient_data = {
        'Heart Rate': vitals.heart_rate,
        'Respiratory Rate': vitals.respiratory_rate,
        'Body Temperature': vitals.body_temperature,
        'Oxygen Saturation': vitals.oxygen_saturation,
        'Systolic Blood Pressure': vitals.systolic_bp,
        'Diastolic Blood Pressure': vitals.diastolic_bp,
        'Age': vitals.age,
        'Gender': vitals.gender,
        'Weight (kg)': vitals.weight_kg,
        'Height (m)': vitals.height_m,
        'Derived_HRV': vitals.hrv if vitals.hrv else 0.1,  # Auto-calculate later
        'Derived_Pulse_Pressure': vitals.systolic_bp - vitals.diastolic_bp,
        'Derived_BMI': vitals.weight_kg / (vitals.height_m ** 2),
        'Derived_MAP': (vitals.systolic_bp + 2 * vitals.diastolic_bp) / 3
    }
    
    # Predict
    result = predictor.predict(patient_data)
    
    # Get feature importance
    importances = predictor.get_feature_importance(patient_data)
    key_factors = [
        {"feature": feat, "impact": round(imp, 2)}
        for feat, imp in importances[:5]
    ]
    
    # Generate recommendations
    recommendations = _generate_recommendations(result, vitals)
    
    return RiskAssessment(
        risk_score=result['risk_score'],
        risk_level=result['risk_level'],
        risk_category=result['risk_category'],
        confidence=result['confidence'],
        risk_probability=result['risk_probability'],
        recommendations=recommendations,
        key_risk_factors=key_factors
    )


@router.post("/assess/batch", response_model=List[RiskAssessment])
async def assess_health_risk_batch(vitals_list: List[VitalSigns]):
    """
    **Batch Health Assessment**
    
    Assess multiple patients at once (useful for clinic screening)
    """
    if not PREDICTOR_READY:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run: python -m ml.train.train_tabular_classifier"
        )
    
    results = []
    for vitals in vitals_list:
        # Reuse single assessment logic
        result = await assess_health_risk(vitals)
        results.append(result)
    
    return results


@router.get("/model/info")
async def get_model_info():
    """
    **Model Information**
    
    Returns metadata about the trained risk prediction model
    """
    if not PREDICTOR_READY:
        return {
            "status": "not_ready",
            "message": "Model not trained",
            "training_command": "python -m ml.train.train_tabular_classifier"
        }
    
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'ml', 'train', 'vitals_classifier_best.pt'
    )
    
    import torch
    checkpoint = torch.load(model_path, map_location='cpu')
    
    return {
        "status": "ready",
        "model_type": "VitalsRiskClassifier",
        "architecture": "17 features → MLP (128→64→32) → 1 output",
        "training_samples": checkpoint.get('total_samples', 'unknown'),
        "validation_auc": round(checkpoint['val_auc'], 4),
        "test_accuracy": round(checkpoint.get('test_accuracy', 0.0), 4),
        "features": checkpoint['feature_cols'],
        "device": "CUDA" if torch.cuda.is_available() else "CPU"
    }


@router.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy" if PREDICTOR_READY else "degraded",
        "service": "Health Chamber Walkthrough Diagnosis",
        "model_ready": PREDICTOR_READY
    }


# Helper Functions
def _generate_recommendations(result: Dict, vitals: VitalSigns) -> List[str]:
    """Generate clinical recommendations based on risk assessment"""
    recommendations = []
    
    # Risk-level recommendations
    if result['risk_level'] == 'RED':
        recommendations.append("⚠️ URGENT: Immediate medical attention recommended")
        recommendations.append("Consider emergency department evaluation")
    elif result['risk_level'] == 'YELLOW':
        recommendations.append("⚠️ CAUTION: Schedule appointment with healthcare provider within 48 hours")
        recommendations.append("Monitor vitals closely")
    else:
        recommendations.append("✅ Current vitals within normal range")
        recommendations.append("Continue regular health monitoring")
    
    # Vital-specific recommendations
    if vitals.heart_rate > 100:
        recommendations.append("🫀 Elevated heart rate detected - avoid stimulants, ensure adequate rest")
    elif vitals.heart_rate < 60:
        recommendations.append("🫀 Low heart rate detected - monitor for dizziness or fatigue")
    
    if vitals.oxygen_saturation < 95:
        recommendations.append("🫁 Low oxygen saturation - ensure good ventilation, consider oxygen therapy")
    
    if vitals.systolic_bp > 140 or vitals.diastolic_bp > 90:
        recommendations.append("🩸 Elevated blood pressure - reduce sodium intake, manage stress")
    
    if vitals.body_temperature > 37.5:
        recommendations.append("🌡️ Elevated temperature - stay hydrated, monitor for infection")
    
    # BMI recommendation
    bmi = vitals.weight_kg / (vitals.height_m ** 2)
    if bmi > 30:
        recommendations.append("⚖️ BMI indicates obesity - consider nutrition counseling and exercise plan")
    elif bmi < 18.5:
        recommendations.append("⚖️ BMI indicates underweight - ensure adequate nutrition")
    
    return recommendations
