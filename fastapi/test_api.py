import requests
import json

# Test data simulating IoT sensor readings
test_data = {
    "age": 52,
    "heart_rate": 110,
    "spo2": 91,
    "respiratory_rate": 24,
    "ecg_rr_intervals": [0.82, 0.80, 0.85, 0.78, 0.81],
    "nasal_airflow_variability": 0.4,
    "cough_present": False,
    "signal_quality": 0.85
}

def test_api():
    base_url = "http://localhost:8000"
    
    # Test main screening endpoint
    response = requests.post(f"{base_url}/screen", json=test_data)
    print("Full Health Screening:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*50 + "\n")
    
    # Test cardiovascular only
    response = requests.post(f"{base_url}/screen/cardiovascular", json=test_data)
    print("Cardiovascular Screening:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*50 + "\n")
    
    # Test respiratory only
    response = requests.post(f"{base_url}/screen/respiratory", json=test_data)
    print("Respiratory Screening:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api()