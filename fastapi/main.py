from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import importlib.util
import sys
import os

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.aggregator_hri import HealthRiskIndex
from models.engine_cardiovascular import CardioRiskEngine
from models.engine_respiratory import RespiratoryRiskEngine

app = FastAPI(title="Health Screening API", version="1.0.0")

# Pydantic models for request/response
class IoTSensorData(BaseModel):
    age: int
    heart_rate: float
    spo2: float
    respiratory_rate: float
    ecg_rr_intervals: Optional[List[float]] = None
    nasal_airflow_variability: Optional[float] = None
    cough_present: bool = False
    signal_quality: float = 1.0

class HealthScreeningResponse(BaseModel):
    hri_score: int
    hri_level: str
    confidence: float
    flags: List[str]
    summary: str
    engine_results: List[Dict]

@app.get("/")
async def root():
    return {"message": "Health Screening API", "status": "active"}

@app.post("/screen", response_model=HealthScreeningResponse)
async def health_screening(data: IoTSensorData):
    """Main endpoint for IoT device health screening"""
    try:
        # Run cardiovascular engine
        cardio_engine = CardioRiskEngine(
            age=data.age,
            heart_rate=data.heart_rate,
            spo2=data.spo2,
            ecg_rr_intervals=data.ecg_rr_intervals,
            signal_quality=data.signal_quality
        )
        cardio_result = cardio_engine.run()
        cardio_result["system"] = "cardiovascular"
        
        # Run respiratory engine
        respiratory_engine = RespiratoryRiskEngine(
            age=data.age,
            respiratory_rate=data.respiratory_rate,
            spo2=data.spo2,
            nasal_airflow_variability=data.nasal_airflow_variability,
            cough_present=data.cough_present,
            signal_quality=data.signal_quality
        )
        respiratory_result = respiratory_engine.run()
        
        # Aggregate results
        engine_results = [cardio_result, respiratory_result]
        hri = HealthRiskIndex(engine_results)
        final_result = hri.run()
        
        return HealthScreeningResponse(
            **final_result,
            engine_results=engine_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")

@app.post("/screen/cardiovascular")
async def cardiovascular_screening(data: IoTSensorData):
    """Endpoint for cardiovascular-only screening"""
    try:
        engine = CardioRiskEngine(
            age=data.age,
            heart_rate=data.heart_rate,
            spo2=data.spo2,
            ecg_rr_intervals=data.ecg_rr_intervals,
            signal_quality=data.signal_quality
        )
        return engine.run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cardiovascular screening failed: {str(e)}")

@app.post("/screen/respiratory")
async def respiratory_screening(data: IoTSensorData):
    """Endpoint for respiratory-only screening"""
    try:
        engine = RespiratoryRiskEngine(
            age=data.age,
            respiratory_rate=data.respiratory_rate,
            spo2=data.spo2,
            nasal_airflow_variability=data.nasal_airflow_variability,
            cough_present=data.cough_present,
            signal_quality=data.signal_quality
        )
        return engine.run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Respiratory screening failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)