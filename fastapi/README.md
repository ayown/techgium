# Health Screening FastAPI Backend

## Overview
This FastAPI backend integrates your health screening engines to process IoT sensor data and provide real-time health risk assessments. The system aggregates multiple physiological measurements into a comprehensive Health Risk Index (HRI).

## Architecture

### Core Components
- **FastAPI Application** (`main.py`): REST API endpoints for health screening
- **Risk Engines**: Modular assessment engines for different physiological systems
- **HRI Aggregator**: Combines individual engine results into overall risk score
- **Pydantic Models**: Data validation and serialization

### Supported Engines
1. **Cardiovascular Engine**: Heart rate, SpO2, ECG variability analysis
2. **Respiratory Engine**: Breathing rate, oxygen saturation, airflow patterns
3. **Extensible**: Ready for neuro-functional, posture, and dermatology engines

## API Endpoints

### 1. Main Health Screening
```
POST /screen
```
**Purpose**: Complete health assessment using all available engines
**Input**: IoT sensor data (heart rate, SpO2, respiratory rate, etc.)
**Output**: Aggregated HRI score with risk level and recommendations

### 2. Cardiovascular Screening
```
POST /screen/cardiovascular
```
**Purpose**: Focused cardiovascular risk assessment
**Input**: Heart rate, SpO2, ECG data, signal quality
**Output**: Cardiovascular-specific risk analysis

### 3. Respiratory Screening
```
POST /screen/respiratory
```
**Purpose**: Respiratory system risk assessment
**Input**: Breathing rate, SpO2, airflow variability, cough status
**Output**: Respiratory-specific risk analysis

### 4. Health Check
```
GET /
```
**Purpose**: API status verification
**Output**: Service status and version info

## Data Flow

1. **IoT Device** → Collects physiological measurements
2. **API Endpoint** → Receives and validates sensor data
3. **Risk Engines** → Process measurements independently
4. **HRI Aggregator** → Combines results with medical weighting
5. **Response** → Returns risk assessment with actionable insights

## Risk Classification

### Risk Levels
- **GREEN**: No significant risk detected (score < 35)
- **YELLOW**: Moderate risk, monitoring advised (score 35-69)
- **RED**: High risk, clinical evaluation recommended (score ≥ 70)

### Confidence Scoring
- Based on signal quality and data completeness
- Range: 0.3 to 1.0
- Accounts for missing sensors or poor signal conditions

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python main.py
```
Server runs on `http://localhost:8000`

### 3. Test the API
```bash
python test_api.py
```

## IoT Integration

### Expected Sensor Data Format
```json
{
  "age": 52,
  "heart_rate": 110.0,
  "spo2": 91.0,
  "respiratory_rate": 24.0,
  "ecg_rr_intervals": [0.82, 0.80, 0.85, 0.78, 0.81],
  "nasal_airflow_variability": 0.4,
  "cough_present": false,
  "signal_quality": 0.85
}
```

### Response Format
```json
{
  "hri_score": 45,
  "hri_level": "YELLOW",
  "confidence": 0.85,
  "flags": ["tachycardia", "borderline_oxygenation"],
  "summary": "Some physiological parameters are outside normal ranges. Monitoring and follow-up are advised.",
  "engine_results": [...]
}
```

## Medical Weighting System

The HRI aggregator uses evidence-based weights:
- **Cardiovascular**: 30% (highest priority)
- **Respiratory**: 25%
- **Neuro-functional**: 15%
- **Skeletal/Posture**: 15%
- **Dermatology**: 15%

## Security & Compliance

### Data Handling
- No persistent storage of health data
- Stateless processing for privacy
- Input validation prevents malformed data
- Error handling protects against system exposure

### Medical Disclaimers
- System provides screening insights, not diagnosis
- Clinical evaluation recommended for high-risk results
- Signal quality affects confidence scoring

## Extending the System

### Adding New Engines
1. Create engine class inheriting from `BaseRiskEngine`
2. Implement required `run()` method
3. Add endpoint in `main.py`
4. Update HRI weights in `aggregator_hri.py`

### Example New Engine Integration
```python
# In main.py
@app.post("/screen/neurological")
async def neuro_screening(data: IoTSensorData):
    engine = NeuroRiskEngine(...)
    return engine.run()
```

## Production Considerations

### Performance
- Lightweight processing suitable for real-time screening
- Stateless design enables horizontal scaling
- Minimal dependencies reduce deployment complexity

### Monitoring
- Add logging for screening requests and results
- Monitor API response times and error rates
- Track confidence scores for data quality assessment

### Deployment
- Use Docker for containerized deployment
- Configure reverse proxy (nginx) for production
- Implement rate limiting for IoT device management
- Add authentication for device registration

## Files Created

1. `main.py` - FastAPI application with health screening endpoints
2. `requirements.txt` - Python dependencies
3. `models/aggregator_hri.py` - HRI aggregation logic
4. `models/engine_cardiovascular.py` - Cardiovascular risk engine
5. `models/engine_respiratory.py` - Respiratory risk engine
6. `test_api.py` - API testing script
7. `README.md` - This documentation

## Next Steps

1. **Test the API**: Run `python main.py` and `python test_api.py`
2. **Add Remaining Engines**: Integrate neuro-functional, posture, and dermatology engines
3. **IoT Integration**: Connect your actual sensor devices
4. **Database Layer**: Add optional data persistence for analytics
5. **Authentication**: Implement device/user authentication
6. **Monitoring**: Add logging and health monitoring
7. **Documentation**: Generate OpenAPI docs at `/docs` endpoint