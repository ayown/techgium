# Hardware Architecture Design for Non-Contact Physiological Measurement System

> **Target:** Hackathon prototype system for walk-through health screening chamber  
> **Budget:** 20-30k INR (~$250-350 USD)  
> **Goal:** 70-90% accuracy match with software algorithms

## User Review Required

> [!IMPORTANT]
> **Sensor Selection Decision:** The design uses mmWave radar (60 GHz) as the primary respiratory sensor. This replaces RIS (Radar Impedance Spectroscopy) which is unavailable. Please confirm this substitution is acceptable.

> [!IMPORTANT]
> **Thermal Camera Option:** MLX90640 (32×24 pixels, ~₹3,500) vs AMG8833 (8×8 pixels, ~₹1,800). MLX90640 recommended for better thermal mapping. Confirm preference.

---

## Proposed Design

### System Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WALK-THROUGH HEALTH SCREENING CHAMBER                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────────────┐                   ┌──────────────────────────────┐   │
│   │   OPTICAL SENSORS    │                   │      RF/MOTION SENSORS       │   │
│   ├──────────────────────┤                   ├──────────────────────────────┤   │
│   │ ┌─────────────────┐  │                   │  ┌─────────────────────────┐ │   │
│   │ │ Logitech C270   │──┼───USB─────────────┼──│   60 GHz mmWave Radar  │ │   │
│   │ │ 1080p/30fps     │  │        │          │  │   (IWR6843 / AWR1642)  │ │   │
│   │ │ rPPG, Pose, Eye │  │        │          │  │   Respiration, Motion  │ │   │
│   │ └─────────────────┘  │        │          │  └───────────┬────────────┘ │   │
│   │ ┌─────────────────┐  │        │          │              │UART/SPI      │   │
│   │ │ MLX90640 Thermal│──┼───I2C──┼──┐       │              │              │   │
│   │ │ 32×24 IR Array  │  │        │  │       └──────────────┼──────────────┘   │
│   │ │ Skin Temp Map   │  │        │  │                      │                  │
│   │ └─────────────────┘  │        │  │                      │                  │
│   └──────────────────────┘        │  │                      │                  │
│                                   │  │                      │                  │
│   ┌──────────────────────┐        │  │                      │                  │
│   │  CONTACT VALIDATION  │        │  │                      │                  │
│   ├──────────────────────┤        │  │                      │                  │
│   │ ┌─────────────────┐  │        │  │                      │                  │
│   │ │ Pulse Oximeter  │──┼───I2C──┼──┤                      │                  │
│   │ │ MAX30102        │  │        │  │                      │                  │
│   │ │ SpO₂ + HR       │  │        │  │                      │                  │
│   │ └─────────────────┘  │        │  │                      │                  │
│   │ ┌─────────────────┐  │        │  │                      │                  │
│   │ │ Glucometer      │──┼───UART─┼──┼──┐                   │                  │
│   │ │ (External BLE)  │  │        │  │  │                   │                  │
│   │ └─────────────────┘  │        │  │  │                   │                  │
│   └──────────────────────┘        │  │  │                   │                  │
│                                   ▼  ▼  ▼                   ▼                  │
│                          ┌───────────────────────────────────┐                 │
│                          │         ESP32-WROOM-32            │                 │
│                          │       (Primary Controller)        │                 │
│                          │                                   │                 │
│                          │  GPIO34 ─ ADC (Spare Analog)      │                 │
│                          │  GPIO21 ─ SDA (I2C Bus)           │                 │
│                          │  GPIO22 ─ SCL (I2C Bus)           │                 │
│                          │  GPIO16 ─ UART2 RX (Radar)        │                 │
│                          │  GPIO17 ─ UART2 TX (Radar)        │                 │
│                          │  GPIO01 ─ UART0 TX (Host/Debug)   │                 │
│                          │  GPIO03 ─ UART0 RX (Host/Debug)   │                 │
│                          │  GPIO04 ─ LED Status              │                 │
│                          │  GPIO05 ─ LED Activity            │                 │
│                          │  GPIO18 ─ Test Point 1            │                 │
│                          │  GPIO19 ─ Test Point 2            │                 │
│                          └─────────────┬─────────────────────┘                 │
│                                        │ USB/UART                              │
│                                        ▼                                        │
│                          ┌───────────────────────────────────┐                 │
│                          │           USB HUB                 │                 │
│                          │        (4-Port Powered)           │                 │
│                          └─────────────┬─────────────────────┘                 │
│                                        │                                        │
│                                        ▼                                        │
│            ┌─────────────────────────────────────────────────────────┐         │
│            │                    HOST PROCESSOR                        │         │
│            │              (Raspberry Pi 4B / Intel NUC)               │         │
│            │                                                          │         │
│            │  ┌─────────────────────────────────────────────────┐    │         │
│            │  │                   SOFTWARE                       │    │         │
│            │  │  FastAPI Server ─────► Biomarker Extraction      │    │         │
│            │  │  MediaPipe ──────────► Pose/Face Detection       │    │         │
│            │  │  OpenCV ─────────────► Video Processing          │    │         │
│            │  │  DL Models ──────────► BP/Stress/rPPG Inference  │    │         │
│            │  │  LLM Agents ─────────► Report Generation         │    │         │
│            │  └─────────────────────────────────────────────────┘    │         │
│            └─────────────────────────────────────────────────────────┘         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. Power Supply Design

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           POWER DISTRIBUTION NETWORK                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   AC MAINS                                                                       │
│   220V/50Hz ──► [Wall Adapter 12V/5A] ──► DC 12V RAIL (Main Power)              │
│                      │                                                           │
│                      ▼                                                           │
│   ┌──────────────────────────────────────────────────────────────┐              │
│   │                    12V Rail (Main Power)                      │              │
│   │                    Max Current: 5A                            │              │
│   └──────────────────────┬─────────────────┬─────────────────────┘              │
│                          │                 │                                     │
│                          ▼                 ▼                                     │
│   ┌─────────────────────────────┐  ┌─────────────────────────────┐              │
│   │ Buck Converter              │  │ Buck Converter              │              │
│   │ LM2596 (12V→5V/3A)          │  │ LM2596 (12V→3.3V/3A)        │              │
│   │                             │  │                             │              │
│   │ Output: 5V ± 50mV ripple    │  │ Output: 3.3V ± 30mV ripple  │              │
│   │ Efficiency: ~85%            │  │ Efficiency: ~80%            │              │
│   └─────────────┬───────────────┘  └─────────────┬───────────────┘              │
│                 │                                │                               │
│                 ▼                                ▼                               │
│   ┌─────────────────────────────┐  ┌─────────────────────────────┐              │
│   │      5V Rail (USB Level)    │  │     3.3V Rail (Logic)       │              │
│   │  ┌────────────────────────┐ │  │  ┌────────────────────────┐ │              │
│   │  │ Raspberry Pi 4B   3.0A │ │  │  │ ESP32 Module     500mA │ │              │
│   │  │ USB Hub           500mA│ │  │  │ MLX90640         80mA  │ │              │
│   │  │ Logitech C270     500mA│ │  │  │ MAX30102         50mA  │ │              │
│   │  │ ESP32 (USB)       500mA│ │  │  │ mmWave Radar*   depends│ │              │
│   │  └────────────────────────┘ │  │  └────────────────────────┘ │              │
│   │  Total: ~4.5A               │  │  Total: ~700mA + radar      │              │
│   └─────────────────────────────┘  └─────────────────────────────┘              │
│                                                                                  │
│   * mmWave radar typically needs 5V/1A - check specific module datasheet        │
│                                                                                  │
│   DECOUPLING:                                                                    │
│   • 10μF bulk cap at each regulator output                                      │
│   • 100nF ceramic at each IC VCC pin                                            │
│   • 10μF at ESP32 3.3V input                                                    │
│   • Keep power traces wide (≥40 mil for 5V, ≥30 mil for 3.3V)                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Power Budget Estimation:**

| Component | Voltage | Current (typ) | Current (max) | Power |
|-----------|---------|---------------|---------------|-------|
| Raspberry Pi 4B | 5V | 2.5A | 3.0A | 15W |
| ESP32-WROOM-32 | 3.3V | 80mA | 500mA | 1.65W |
| Logitech C270 | 5V (USB) | 300mA | 500mA | 2.5W |
| MLX90640 | 3.3V | 18mA | 80mA | 0.26W |
| MAX30102 | 3.3V | 20mA | 50mA | 0.17W |
| mmWave Radar | 5V | 700mA | 1A | 5W |
| USB Hub | 5V | 100mA | 500mA | 2.5W |
| LEDs/Misc | 3.3V | 50mA | 100mA | 0.33W |
| **Total** | - | - | - | **~27W** |

> [!TIP]
> Use a 12V/5A (60W) power supply to allow 50% headroom for safety.

---

### 2. Circuit Schematic - Sensor Connections

#### 2.1 ESP32 Core Circuit

```
                                ESP32-WROOM-32D
                          ┌──────────────────────────┐
                    EN ───┤1   EN               GND  ├─── GND
              GPIO36/VP ───┤2   VP               3V3  ├─── 3.3V        
              GPIO39/VN ───┤3   VN          GPIO23   ├─── NC
              GPIO34/A6 ───┤4   34          GPIO22   ├─── I2C SCL ──►
              GPIO35/A7 ───┤5   35          GPIO01/TX├─── UART0 TX (Debug/Host)
              GPIO32/A4 ───┤6   32          GPIO03/RX├─── UART0 RX (Debug/Host)
              GPIO33/A5 ───┤7   33          GPIO21   ├─── I2C SDA ──►
              GPIO25/A18───┤8   25          GPIO19   ├─── Test Point 2
              GPIO26/A19───┤9   26          GPIO18   ├─── Test Point 1
              GPIO27/A17───┤10  27          GPIO05   ├─── LED Activity (Green)
              GPIO14/A16───┤11  14          GPIO17/TX├─── UART2 TX (Radar) ──►
              GPIO12/A15───┤12  12          GPIO16/RX├─── UART2 RX (Radar) ◄──
                    GND ───┤13  GND         GPIO04   ├─── LED Status (Blue)
              GPIO13/A14───┤14  13          GPIO00   ├─── Boot (10k pull-up)
              SD2/GPIO09───┤15  SD2         GPIO02   ├─── NC (Boot strapping)
              SD3/GPIO10───┤16  SD3         GPIO15/A13├── Test Point 3
              CMD/GPIO11───┤17  CMD          SD1/GPIO8├── NC
              CLK/GPIO06───┤18  CLK          SD0/GPIO7├── NC
              SD0/GPIO07───┤19  SD0               GND ├─── GND
                          └──────────────────────────┘

Power Filtering:
    3.3V ───┬─── 10μF ─┬─── ESP32 3V3
            │          │
            └─ 100nF ──┘
                       │
                      GND

Boot Circuit:
    GPIO0 ──┬── 10kΩ ── 3.3V
            │
            └── [SW_BOOT] ── GND

Reset Circuit:
    EN ─────┬── 10kΩ ── 3.3V
            │
            └── 10μF ── GND
            │
            └── [SW_RST] ── GND
```

#### 2.2 I2C Bus (Thermal Camera + Pulse Oximeter)

```
                         I2C BUS SCHEMATIC
            ┌────────────────────────────────────────────────────┐
            │                                                    │
            │     SDA (GPIO21)                SCL (GPIO22)       │
            │        │                           │               │
            │        ├── 4.7kΩ ── 3.3V           ├── 4.7kΩ ── 3.3V
            │        │                           │               │
            │        │     ┌─────────────────────┘               │
            │        │     │                                     │
            │    ┌───▼─────▼───┐                                 │
            │    │   MLX90640   │  Addr: 0x33                    │
            │    │  32×24 IR    │                                │
            │    │  Thermal Cam │                                │
            │    └──────┬───────┘                                │
            │           │ VDD=3.3V, GND                          │
            │           │                                        │
            │    ┌──────▼───────┐                                │
            │    │   MAX30102   │  Addr: 0x57                    │
            │    │ Pulse/SpO2   │                                │
            │    │              │                                │
            │    └──────┬───────┘                                │
            │           │ VDD=3.3V, GND                          │
            │           │                                        │
            │    ┌──────▼───────┐                                │
            │    │   OPTIONAL   │                                │
            │    │  SSD1306     │  Addr: 0x3C (OLED Display)     │
            │    │  128×64 OLED │                                │
            │    └──────────────┘                                │
            │                                                    │
            └────────────────────────────────────────────────────┘

MLX90640 Pinout (SMD breakout):
    VDD ─── 3.3V (with 10μF + 100nF)
    GND ─── GND
    SDA ─── GPIO21
    SCL ─── GPIO22

MAX30102 Pinout:
    VIN ─── 3.3V (with 10μF + 100nF)  
    GND ─── GND
    SDA ─── GPIO21
    SCL ─── GPIO22
    INT ─── GPIO35 (optional interrupt)
```

#### 2.3 mmWave Radar Interface (UART)

```
                     60 GHz RADAR MODULE CONNECTION
            ┌────────────────────────────────────────────────────┐
            │                                                    │
            │    IWR6843 / AWR1642 EVM (TI Radar Module)         │
            │    ┌─────────────────────────────────────┐         │
            │    │                                     │         │
            │    │  VCC ─────────────── 5V Rail        │         │
            │    │  GND ─────────────── GND            │         │
            │    │                                     │         │
            │    │  UART TX ─────┬───── Level Shift ───┼─── GPIO16 (ESP32 RX)
            │    │               │     (5V → 3.3V)     │         │
            │    │  UART RX ────┬┼───── Level Shift ───┼─── GPIO17 (ESP32 TX)
            │    │              ││     (3.3V → 5V)     │         │
            │    │              ││                     │         │
            │    │  RESET ──────┼┼── GPIO27 (Optional) │         │
            │    │              ││                     │         │
            │    └──────────────┼┼─────────────────────┘         │
            │                   ││                               │
            │                   ▼▼                               │
            │    ┌─────────────────────────────────────┐         │
            │    │  BSS138 Level Shifter Module        │         │
            │    │  ┌─────────────────────────────────┐│         │
            │    │  │ LV ─── 3.3V    HV ─── 5V        ││         │
            │    │  │ GND ─── GND   GND ─── GND       ││         │
            │    │  │ L1 ──── ◄───► ── H1             ││         │
            │    │  │ L2 ──── ◄───► ── H2             ││         │
            │    │  └─────────────────────────────────┘│         │
            │    └─────────────────────────────────────┘         │
            │                                                    │
            │    UART Configuration:                             │
            │      Baud Rate: 921600 (typical for TI radar)      │
            │      Data Bits: 8                                  │
            │      Stop Bits: 1                                  │
            │      Parity: None                                  │
            │                                                    │
            └────────────────────────────────────────────────────┘
```

#### 2.4 USB Hub and Camera Connection

```
                USB TOPOLOGY
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │    Raspberry Pi 4B                                       │
    │    ┌─────────────────┐                                   │
    │    │  USB 3.0 Port ──┼──► USB Hub (Powered 4-Port)       │
    │    │                 │    ┌──────────────────────────┐   │
    │    │                 │    │  Port 1 ─── Logitech C270│   │
    │    │                 │    │  Port 2 ─── ESP32 (USB)  │   │
    │    │                 │    │  Port 3 ─── Spare        │   │
    │    │                 │    │  Port 4 ─── Spare        │   │
    │    │                 │    └──────────────────────────┘   │
    │    │                 │                                   │
    │    │  GPIO Header ───┼──► (Optional: direct I2C to      │
    │    │                 │     thermal camera if needed)     │
    │    │                 │                                   │
    │    │  USB-C Power ───┼──► 5V/3A Power Supply             │
    │    └─────────────────┘                                   │
    │                                                          │
    │    Camera Placement Notes:                               │
    │    • Logitech C270 mounted at chest height (~1.2m)       │
    │    • 1-2 meter distance from subject                     │
    │    • Diffused LED lighting (avoid direct glare)          │
    │    • 30fps capture for rPPG (sufficient for BPM 40-200)  │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

---

### 3. Pin Assignment Table

| ESP32 Pin | GPIO | Function | Interface | Voltage | Connected To |
|-----------|------|----------|-----------|---------|--------------|
| 3V3 | - | Power | Power | 3.3V | All 3.3V devices |
| GND | - | Ground | Power | 0V | All grounds |
| GPIO21 | 21 | I2C SDA | I2C | 3.3V | MLX90640, MAX30102, OLED |
| GPIO22 | 22 | I2C SCL | I2C | 3.3V | MLX90640, MAX30102, OLED |
| GPIO16 | 16 | UART2 RX | UART | 3.3V | mmWave Radar TX (via level shifter) |
| GPIO17 | 17 | UART2 TX | UART | 3.3V | mmWave Radar RX (via level shifter) |
| GPIO01 | 01 | UART0 TX | UART | 3.3V | Host Processor / Debug |
| GPIO03 | 03 | UART0 RX | UART | 3.3V | Host Processor / Debug |
| GPIO04 | 04 | LED Status | GPIO | 3.3V | Blue LED (status) |
| GPIO05 | 05 | LED Activity | GPIO | 3.3V | Green LED (data activity) |
| GPIO18 | 18 | Test Point 1 | Debug | 3.3V | TP1 Header |
| GPIO19 | 19 | Test Point 2 | Debug | 3.3V | TP2 Header |
| GPIO15 | 15 | Test Point 3 | Debug | 3.3V | TP3 Header |
| GPIO35 | 35 | Interrupt | GPIO In | 3.3V | MAX30102 INT (optional) |
| GPIO27 | 27 | Radar Reset | GPIO Out | 3.3V | mmWave Reset (optional) |
| GPIO00 | 00 | Boot Mode | Boot | 3.3V | Boot Button + 10k pull-up |
| EN | - | Enable/Reset | Reset | 3.3V | Reset Button + RC filter |

---

### 4. Communication Protocol Summary

| Interface | Protocol | Speed | Devices | Data Flow |
|-----------|----------|-------|---------|-----------|
| I2C (GPIO21/22) | I2C @ 400kHz | 400 kbps | MLX90640, MAX30102, OLED | Bidirectional |
| UART2 (GPIO16/17) | UART | 921600 baud | mmWave Radar | Radar → ESP32 (processed frames) |
| UART0 (GPIO01/03) | UART | 115200 baud | Host Processor | ESP32 ↔ Host (JSON data) |
| USB | USB 2.0 | 480 Mbps | Logitech C270 | Camera → Host (video frames) |
| USB | USB 2.0 | 12 Mbps | ESP32 DevKit | Host ↔ ESP32 (optional) |

**Data Format (ESP32 → Host via UART):**

> [!IMPORTANT]
> The ESP32 must send data in the **exact format expected by the FastAPI `/api/v1/screening` endpoint**. See format below.

**Raw Sensor Data (ESP32 → Python Serial Reader):**
```json
{
  "timestamp": 1707050232,
  "radar": {
    "respiration_rate": 15.2,
    "breathing_depth": 0.73,
    "micro_motion": 0.12
  },
  "thermal": {
    "skin_temp_avg": 36.4,
    "skin_temp_max": 37.1,
    "thermal_asymmetry": 0.3,
    "thermal_map": [[36.1, 36.2, ...], ...]
  },
  "pulse_ox": {
    "spo2": 98,
    "heart_rate": 72
  }
}
```

**Python Bridge (Transforms ESP32 data → API ScreeningRequest):**

A Python script on the Raspberry Pi must transform the raw sensor data into the API format:

```python
# bridge.py - ESP32 to API bridge
import serial
import json
import requests

API_URL = "http://localhost:8000/api/v1/screening"

def transform_to_api_format(esp32_data: dict) -> dict:
    """Transform ESP32 sensor data to API ScreeningRequest format."""
    return {
        "patient_id": "WALK_THROUGH_PATIENT",
        "include_validation": True,
        "systems": [
            {
                "system": "cardiovascular",
                "biomarkers": [
                    {"name": "heart_rate", "value": esp32_data["pulse_ox"]["heart_rate"], 
                     "unit": "bpm", "normal_range": [60, 100]},
                    {"name": "spo2", "value": esp32_data["pulse_ox"]["spo2"], 
                     "unit": "%", "normal_range": [95, 100]},
                    {"name": "chest_micro_motion", "value": esp32_data["radar"]["micro_motion"],
                     "unit": "normalized_amplitude", "normal_range": [0.001, 0.01]}
                ]
            },
            {
                "system": "respiratory",  # Maps to pulmonary
                "biomarkers": [
                    {"name": "respiration_rate", "value": esp32_data["radar"]["respiration_rate"],
                     "unit": "breaths/min", "normal_range": [12, 20]},
                    {"name": "breathing_depth", "value": esp32_data["radar"]["breathing_depth"],
                     "unit": "normalized", "normal_range": [0.5, 1.0]}
                ]
            },
            {
                "system": "skin",
                "biomarkers": [
                    {"name": "skin_temperature", "value": esp32_data["thermal"]["skin_temp_avg"],
                     "unit": "celsius", "normal_range": [35.5, 37.5]},
                    {"name": "thermal_asymmetry", "value": esp32_data["thermal"]["thermal_asymmetry"],
                     "unit": "delta_celsius", "normal_range": [0, 0.5]}
                ]
            }
        ]
    }

# Main loop
ser = serial.Serial('/dev/ttyUSB0', 115200)
while True:
    line = ser.readline().decode().strip()
    if line:
        esp32_data = json.loads(line)
        api_payload = transform_to_api_format(esp32_data)
        response = requests.post(API_URL, json=api_payload)
        print(f"Screening ID: {response.json()['screening_id']}")
```

**Final API Request Format (what the API receives):**
```json
{
  "patient_id": "WALK_THROUGH_PATIENT",
  "include_validation": true,
  "systems": [
    {
      "system": "cardiovascular",
      "biomarkers": [
        {"name": "heart_rate", "value": 72, "unit": "bpm", "normal_range": [60, 100]},
        {"name": "spo2", "value": 98, "unit": "%", "normal_range": [95, 100]},
        {"name": "chest_micro_motion", "value": 0.12, "unit": "normalized_amplitude"}
      ]
    },
    {
      "system": "pulmonary",
      "biomarkers": [
        {"name": "respiration_rate", "value": 15.2, "unit": "breaths/min"},
        {"name": "breathing_depth", "value": 0.73, "unit": "normalized"}
      ]
    },
    {
      "system": "skin",
      "biomarkers": [
        {"name": "skin_temperature", "value": 36.4, "unit": "celsius"},
        {"name": "thermal_asymmetry", "value": 0.3, "unit": "delta_celsius"}
      ]
    }
  ]
}
```

---

### 5. Bill of Materials (BOM)

| # | Component | Part Number | Qty | Unit Price (INR) | Total (INR) | Source |
|---|-----------|-------------|-----|------------------|-------------|--------|
| 1 | ESP32-WROOM-32D DevKit | ESP32-DevKitC-32D | 1 | ₹450 | ₹450 | Robu.in |
| 2 | Raspberry Pi 4B 4GB | RPI4-MODBP-4GB | 1 | ₹5,500 | ₹5,500 | Robu.in |
| 3 | Logitech C270 HD Webcam | C270 | 1 | ₹1,500 | ₹1,500 | Amazon.in |
| 4 | MLX90640 Thermal Camera | MLX90640ESF-BAB | 1 | ₹3,500 | ₹3,500 | Robu.in |
| 5 | MAX30102 Pulse Oximeter | MAX30102 Module | 1 | ₹250 | ₹250 | Robu.in |
| 6 | 60GHz mmWave Radar | IWR6843ISK-ODS | 1 | ₹8,000 | ₹8,000 | TI Store* |
| 7 | LM2596 Buck Converter | LM2596-ADJ | 2 | ₹100 | ₹200 | Robu.in |
| 8 | BSS138 Level Shifter | BSS138 Module | 2 | ₹50 | ₹100 | Robu.in |
| 9 | USB Hub 4-Port Powered | Generic USB2.0 Hub | 1 | ₹400 | ₹400 | Amazon.in |
| 10 | 12V/5A Power Adapter | DC Adapter | 1 | ₹350 | ₹350 | Robu.in |
| 11 | SSD1306 OLED 128×64 | 0.96" I2C OLED | 1 | ₹180 | ₹180 | Robu.in |
| 12 | LEDs (Blue, Green) | 5mm LED | 4 | ₹5 | ₹20 | Local |
| 13 | Resistors (10k, 4.7k, 330Ω) | Assorted | 20 | ₹2 | ₹40 | Local |
| 14 | Capacitors (10μF, 100nF) | Ceramic/Electrolytic | 20 | ₹3 | ₹60 | Local |
| 15 | Headers (Male/Female) | 2.54mm Pitch | 4 | ₹20 | ₹80 | Local |
| 16 | Jumper Wires | Dupont Wires | 40 | ₹2 | ₹80 | Local |
| 17 | Prototype PCB | 7×9cm Perf Board | 2 | ₹50 | ₹100 | Local |
| 18 | USB Cables | Type-A to Micro | 3 | ₹80 | ₹240 | Amazon.in |
| 19 | Enclosure/Mount | 3D Printed | 1 | ₹500 | ₹500 | Custom |
| | | | | **Total** | **₹21,550** | |

> [!NOTE]
> *TI mmWave radar modules may require IndiaMART or direct TI order. Alternative: LD2410 radar module (~₹800) for simplified respiration detection with reduced accuracy.

**Budget-Conscious Alternative (₹15,000):**
- Replace IWR6843 with LD2410 (₹800) - saves ₹7,200
- Use basic USB webcam - saves ₹500
- Total: ~₹14,350

---

### 6. PCB Footprint Recommendations

| Component | Package | Footprint | Notes |
|-----------|---------|-----------|-------|
| ESP32-WROOM-32D | Module | 25.5×18mm | 38-pin castellated |
| MLX90640 | QFP-24 | 7×7mm | IR-transparent window required |
| MAX30102 | Module | 14×11mm | Finger clip mount |
| LM2596 | TO-263 | 10×15mm | Use adequate copper pour |
| BSS138 | SOT-23 | 3×3mm | 4-channel module preferred |
| USB Connector | Type-B | Standard | Use through-hole for durability |
| LED | 5mm THT | 5mm | 330Ω series resistor |

---

### 7. Mechanical Mounting Considerations

```
                    WALK-THROUGH CHAMBER LAYOUT (Top View)
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │                        ENTRY                                │
    │                          │                                  │
    │    ┌─────────────────────▼─────────────────────────────┐   │
    │    │                                                    │   │
    │    │        [THERMAL CAM]                               │   │
    │    │        Height: 1.2m                                │   │
    │    │        Angle: Slightly downward                    │   │
    │    │                                                    │   │
    │    │    ────────────── WALK PATH ────────────────       │   │
    │    │                      ↑                             │   │
    │    │                      │                             │   │
    │    │              [LOGITECH C270]                       │   │
    │    │              Height: 1.2m (face) / 0.8m (body)     │   │
    │    │              Distance: 1.5-2.0m from subject       │   │
    │    │                                                    │   │
    │    │    [mmWAVE RADAR]                                  │   │
    │    │    Height: 1.0m (chest level)                      │   │
    │    │    Distance: 0.5-1.5m optimal range                │   │
    │    │                                                    │   │
    │    │    ────────────── WALK PATH ────────────────       │   │
    │    │                                                    │   │
    │    │    [CONTROL STATION]                               │   │
    │    │    ESP32 + RPi + Power + Display                   │   │
    │    │    Side-mounted, accessible for maintenance        │   │
    │    │                                                    │   │
    │    └────────────────────────────────────────────────────┘   │
    │                          │                                  │
    │                        EXIT                                 │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Lighting Requirements:
• Diffused LED panels (3000-4000K, CRI>90)
• Avoid direct reflections on subject's face
• 200-500 lux at face level for optimal rPPG
• Consistent lighting reduces motion artifacts
```

**Sensor Mounting Specifications:**

| Sensor | Height | Distance | Angle | Notes |
|--------|--------|----------|-------|-------|
| Logitech C270 (Face) | 1.2m | 0.5-1.0m | Eye level | Phase 1: Close-up face |
| Logitech C270 (Body) | 0.8-1.0m | 2.0-3.0m | Full body | Phase 2: Gait analysis |
| MLX90640 | 1.2m | 0.5-1.0m | Face-level | Forehead/cheek thermal |
| mmWave Radar | 1.0m | 0.5-1.5m | Chest-level | Perpendicular to chest |
| MAX30102 | Side station | N/A | Finger clip | Calibration station |

---

### 8. EMI Shielding and Cable Guidelines

> [!WARNING]
> 60 GHz radar is extremely sensitive to EMI and reflections. Proper shielding is critical.

**Shielding Guidelines:**

1. **mmWave Radar:**
   - Use shielded twisted pair for UART lines
   - Keep radar module away from power cables (>10cm)
   - Shield radar enclosure with metal mesh (aperture <λ/10 = 0.5mm)
   - Grounded metal plate behind radar to reduce multipath

2. **Camera:**
   - Use shielded USB cables with ferrite cores
   - Keep USB cables away from radar and power
   - Max USB cable length: 3m (use active extension if longer)

3. **I2C Bus:**
   - Keep I2C traces short (<30cm)
   - Use proper pull-up resistors (4.7kΩ for 400kHz)
   - Avoid running parallel to high-speed signals

4. **Power:**
   - Separate analog and digital grounds (star topology)
   - Add ferrite beads on power input
   - Use ground plane on PCB

---

### 9. Design Notes

#### 9.1 Component Selection Rationale

| Component | Why Chosen | Alternatives |
|-----------|------------|--------------|
| ESP32-WROOM-32 | WiFi/BT, dual-core, multiple UART, low cost | Arduino Mega (no WiFi), STM32 (complex) |
| Raspberry Pi 4B | Runs MediaPipe/OpenCV, Python ecosystem | Intel NUC (expensive), Jetson Nano (overkill) |
| Logitech C270 | 720p/30fps, good low-light, USB, cheap | Logitech C920 (overkill), Intel RealSense (expensive) |
| MLX90640 | 32×24 resolution, I2C, radiometric | AMG8833 (8×8, too low), FLIR Lepton (expensive) |
| MAX30102 | Proven accuracy, I2C, low power | MAX30100 (older), pulse sensor (unreliable) |
| IWR6843 | TI's 60GHz vital signs radar, SDK available | AWR1642 (similar), LD2410 (simpler but less data) |

#### 9.2 Hardware Limitations & Mitigations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Camera motion blur | rPPG artifacts | Fixed mount, software stabilization |
| Ambient light variation | rPPG SNR | Controlled LED lighting |
| Thermal drift | Temperature offset | Calibration every 30 min |
| Radar multipath | False respiratory rate | Shielding, signal processing |
| ESP32 UART limited buffers | Data loss | DMA transfer, flow control |
| I2C clock stretching | Timing issues | Proper timeouts in firmware |

#### 9.3 Safety Considerations

- **Electrical:** Use fused power input, low-voltage DC only
- **EMI:** Radar complies with FCC Part 15, CE marked modules only
- **Optical:** Camera uses visible light only, no IR laser
- **Thermal:** Passive heat dissipation, no active cooling needed (<40°C ambient)

---

### 10. Verification Plan

#### Automated Tests

| Test | Method | Pass Criteria |
|------|--------|---------------|
| Power rail voltages | Multimeter at test points | 5V ±0.25V, 3.3V ±0.1V |
| I2C device scan | `i2cdetect -y 1` | Detect 0x33, 0x57, 0x3C |
| ESP32 UART echo | Loopback test | 100% byte accuracy |
| Camera detection | `lsusb` / OpenCV | Device ID visible |
| Radar data stream | Serial monitor | JSON frames at 20Hz |

#### Manual Verification

1. **Camera Test:**
   - Run `python camera_test.py`
   - Verify face detection and pose landmarks
   - Check frame rate: should be 25-30 fps

2. **Thermal Camera Test:**
   - Point at known temperature reference (body temp ~36.5°C)
   - Verify ±1°C accuracy after calibration

3. **Radar Test:**
   - Stand 1m from radar, breathe normally
   - Verify respiration rate matches manual count (±1 BPM)

4. **Pulse Oximeter Test:**
   - Compare SpO₂ with clinical reference device
   - Verify ±2% accuracy

#### Data Synchronization Checklist

- [ ] All sensor streams tagged with Unix timestamp
- [ ] ESP32 NTP sync at boot (via WiFi)
- [ ] Maximum timestamp drift: <50ms over 60s session
- [ ] JSON packet includes `timestamp` field

---

### 11. Software Integration Notes

The hardware interfaces with the existing FastAPI software via:

1. **Logitech C270 → OpenCV → MediaPipe**
   - Provides: [face_frames](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/camera_test.py#556-656), [pose_sequence](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/camera_test.py#452-492), [face_landmarks](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/camera_test.py#494-554)
   - Used by: [CardiovascularExtractor](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/app/core/extraction/cardiovascular.py#19-625), `CNSExtractor`, `EyeExtractor`, `SkeletalExtractor`

2. **mlX90640 → ESP32 → UART → Python serial**
   - Provides: `thermal_data` (32×24 array)
   - Used by: `SkinExtractor` (skin temperature mapping)

3. **mmWave Radar → ESP32 → UART → Python serial**
   - Provides: `radar_data` (respiration rate, breathing depth, micro-motion)
   - Used by: [CardiovascularExtractor](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/app/core/extraction/cardiovascular.py#19-625) (chest_micro_motion), respiratory proxy

4. **MAX30102 → ESP32 → UART → Python serial**
   - Provides: `vital_signs` (SpO₂, heart_rate)
   - Used by: [CardiovascularExtractor](file:///c:/Users/KOUSTAV%20BERA/OneDrive/Desktop/chiranjeevi/fastapi2/app/core/extraction/cardiovascular.py#19-625) (ground truth calibration)

**ESP32 Firmware Loop:**
```
while(true) {
    read_mlx90640_frame();
    read_max30102_values();
    process_radar_data();
    
    json_packet = format_json(timestamp, thermal, pulse, radar);
    serial_send(json_packet);
    
    delay(50);  // 20 Hz output
}
```

---

## Summary

This design provides a **modular, hackathon-ready hardware architecture** that:

- ✅ Uses readily available components (~₹21,550 total)
- ✅ Interfaces all required sensors with ESP32 + Raspberry Pi
- ✅ Provides clear schematics, pin assignments, and BOM
- ✅ Includes power supply design with proper decoupling
- ✅ Supports the existing FastAPI software extractors
- ✅ Prioritizes workability over sophistication

The mmWave radar replaces RIS for respiratory and micro-motion sensing, which should achieve 70-90% of the accuracy target for the hackathon demonstration.
