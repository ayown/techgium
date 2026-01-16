"""
Tabular Feature-Based Risk Classifier
High-accuracy model for real patient vital signs (non-time-series)
"""

import torch
import torch.nn as nn
import numpy as np


class VitalsRiskClassifier(nn.Module):
    """
    Deep neural network for vital signs risk prediction
    Input: 17 clinical features (HR, SpO2, BP, HRV, BMI, Age, etc.)
    Output: Risk probability [0-1]
    """
    
    def __init__(self, input_dim=17, hidden_dims=[128, 64, 32], dropout=0.3):
        """
        Args:
            input_dim: Number of input features (default: 17 for vitals dataset)
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) tensor of features
        Returns:
            risk_prob: (batch, 1) risk probability
        """
        return self.net(x)
    
    def predict_risk_score(self, x):
        """
        Predict risk score [0-100]
        
        Args:
            x: (batch, input_dim) or (input_dim,) numpy array or tensor
        Returns:
            risk_score: Risk score 0-100
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            prob = self.forward(x)
            return (prob * 100).cpu().numpy().squeeze()


class IoTSensorRiskNet(nn.Module):
    """
    Multi-sensor fusion network for IoT healthcare data
    Handles Temperature, HR, BP, Battery as separate streams
    """
    
    def __init__(self, sensor_features=4, fusion_dim=64, dropout=0.3):
        """
        Args:
            sensor_features: Features per sensor type
            fusion_dim: Dimension for sensor fusion
            dropout: Dropout probability
        """
        super().__init__()
        
        # Individual sensor encoders
        self.temp_encoder = nn.Sequential(
            nn.Linear(sensor_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.hr_encoder = nn.Sequential(
            nn.Linear(sensor_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.bp_encoder = nn.Sequential(
            nn.Linear(sensor_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.battery_encoder = nn.Sequential(
            nn.Linear(sensor_features, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        fusion_input_dim = 32 + 32 + 32 + 16  # 112
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, temp, hr, bp, battery):
        """
        Args:
            temp: (batch, sensor_features) temperature sensor data
            hr: (batch, sensor_features) heart rate sensor data
            bp: (batch, sensor_features) blood pressure sensor data
            battery: (batch, sensor_features) battery/device health data
        Returns:
            risk_prob: (batch, 1) health risk probability
        """
        temp_emb = self.temp_encoder(temp)
        hr_emb = self.hr_encoder(hr)
        bp_emb = self.bp_encoder(bp)
        battery_emb = self.battery_encoder(battery)
        
        # Concatenate all sensor embeddings
        fused = torch.cat([temp_emb, hr_emb, bp_emb, battery_emb], dim=1)
        
        # Predict risk
        risk = self.fusion(fused)
        return risk


class HybridHealthRiskModel(nn.Module):
    """
    Combines tabular vitals classifier with IoT sensor network
    For comprehensive health risk assessment
    """
    
    def __init__(self, vitals_dim=17, iot_enabled=True):
        super().__init__()
        
        self.iot_enabled = iot_enabled
        
        # Vitals classifier
        self.vitals_net = VitalsRiskClassifier(input_dim=vitals_dim)
        
        # IoT sensor network (if available)
        if iot_enabled:
            self.iot_net = IoTSensorRiskNet()
            # Fusion of both models
            self.final_fusion = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
    
    def forward(self, vitals, iot_sensors=None):
        """
        Args:
            vitals: (batch, 17) vital signs features
            iot_sensors: dict with keys ['temp', 'hr', 'bp', 'battery'] (optional)
        Returns:
            risk: Combined risk score
        """
        vitals_risk = self.vitals_net(vitals)
        
        if self.iot_enabled and iot_sensors is not None:
            iot_risk = self.iot_net(
                iot_sensors['temp'],
                iot_sensors['hr'],
                iot_sensors['bp'],
                iot_sensors['battery']
            )
            # Weighted fusion
            combined = torch.cat([vitals_risk, iot_risk], dim=1)
            final_risk = self.final_fusion(combined)
            return final_risk, vitals_risk, iot_risk
        else:
            return vitals_risk, vitals_risk, None


def create_feature_vector(patient_data):
    """
    Extract feature vector from patient vital signs data
    
    Args:
        patient_data: dict with keys matching vitals dataset columns
    
    Returns:
        features: numpy array of shape (17,)
    """
    feature_names = [
        'Heart Rate',
        'Respiratory Rate', 
        'Body Temperature',
        'Oxygen Saturation',
        'Systolic Blood Pressure',
        'Diastolic Blood Pressure',
        'Age',
        'Weight (kg)',
        'Height (m)',
        'Derived_HRV',
        'Derived_Pulse_Pressure',
        'Derived_BMI',
        'Derived_MAP'
    ]
    
    # Add gender encoding
    gender = 1.0 if patient_data.get('Gender') == 'Male' else 0.0
    
    features = []
    for name in feature_names:
        features.append(float(patient_data.get(name, 0)))
    
    features.append(gender)
    
    # Add derived features if not present
    if 'Derived_HRV' not in patient_data:
        features[-4] = 0.1  # Default HRV
    if 'Derived_Pulse_Pressure' not in patient_data:
        features[-3] = patient_data.get('Systolic Blood Pressure', 120) - \
                       patient_data.get('Diastolic Blood Pressure', 80)
    if 'Derived_BMI' not in patient_data:
        weight = patient_data.get('Weight (kg)', 70)
        height = patient_data.get('Height (m)', 1.7)
        features[-2] = weight / (height ** 2)
    if 'Derived_MAP' not in patient_data:
        sys_bp = patient_data.get('Systolic Blood Pressure', 120)
        dia_bp = patient_data.get('Diastolic Blood Pressure', 80)
        features[-1] = (sys_bp + 2 * dia_bp) / 3
    
    return np.array(features, dtype=np.float32)
