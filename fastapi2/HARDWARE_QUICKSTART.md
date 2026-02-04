# Hardware Bridge Quick Start

This guide explains how to run the health screening system with hardware sensors.

## ⚠️ Important: Camera Connection

**The camera/webcam connects directly to your LAPTOP, not to the ESP32!**

- Use your laptop's **built-in webcam** OR
- Plug an external webcam (Logitech C270) into your **laptop's USB port**

The ESP32 only handles: thermal camera, mmWave radar, and pulse oximeter.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR LAPTOP                              │
│  ┌────────────────┐      ┌─────────────────────────────┐   │
│  │ Webcam/Camera  │─────►│        bridge.py            │   │
│  │ (USB/built-in) │      │  • CameraCapture (OpenCV)   │   │
│  └────────────────┘      │  • ESP32Reader (serial)     │   │
│                          │  • DataFusion → FastAPI     │   │
│  ┌────────────────┐      └─────────────────────────────┘   │
│  │ ESP32 DevKit   │──────────────┘                         │
│  │ (USB serial)   │    Serial @ 115200 baud                │
│  └────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘

ESP32 handles ONLY:
  • MLX90640 Thermal Camera (I2C)
  • mmWave Radar (UART)  
  • MAX30102 Pulse Oximeter (I2C)
```

## Prerequisites

```bash
pip install pyserial
```

## Files Created

| File | Purpose |
|------|---------|
| `bridge.py` | Python integration script (ESP32 → API) |
| `firmware/esp32_health_bridge.ino` | ESP32 Arduino firmware |
| `HARDWARE.md` | Complete hardware design documentation |

```
ESP32 Sensors ──► Serial JSON ──► bridge.py ──► HTTP POST ──► FastAPI ──► Reports
      │                                │
      │                                ▼
mmWave Radar                    Logitech C270
MLX90640 Thermal                (via OpenCV)
MAX30102 Pulse Ox
```

## Expected Output

```
============================================================
  HEALTH SCREENING HARDWARE BRIDGE
============================================================

📷 Phase 1: Face capture (10s)
Please look directly at the camera...
Captured 300 face frames

🚶 Phase 2: Body capture (10s)
Please walk naturally or stand for posture analysis...
Extracted 250 pose frames

ESP32 data received: ['radar', 'thermal', 'pulse_ox']

📊 Processing biomarkers and sending to API...
Screening request: 4 systems
  - cardiovascular: 4 biomarkers
  - pulmonary: 3 biomarkers
  - skin: 3 biomarkers
  - cns: 2 biomarkers

✅ Screening completed!
Screening ID: SCR-A1B2C3D4
Overall Risk: low (25.3)
Patient report: reports/patient_SCR-A1B2C3D4.pdf
Doctor report: reports/doctor_SCR-A1B2C3D4.pdf
```
