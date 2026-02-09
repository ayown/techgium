# 🏥 Walkthrough Health Screening Chamber

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-FFA000?style=for-the-badge&logo=google)
![Gemini](https://img.shields.io/badge/AI-Gemini%201.5%20Flash-4285F4?style=for-the-badge&logo=google)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **The future of diagnostics is passive, continuous, and zero-touch.**

The **Walkthrough Health Screening Chamber** is a multimodal, non-invasive diagnostic platform designed to assess comprehensive health metrics simply by having a user walk through a sensor-equipped corridor. By fusing Computer Vision, mmWave Radar, and Thermal Imaging, we abstract away the complexity of medical checkups into a seamless 30-second experience.

---

## 📑 Table of Contents

- [Abstract & Philosophy](#-abstract--philosophy)
- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Roadmap](#-roadmap)
- [Disclaimer](#-disclaimer)

---

## 🧘 Abstract & Philosophy

### Why "Abstract"?
In software engineering, abstraction hides complex implementation details behind a simple interface. We apply this principle to healthcare.

Traditional diagnostics are **imperative**: "Sit down, roll up your sleeve, wear this cuff, breathe into this tube."
Our approach is **declarative**: "Walk from point A to point B."

The system relies on **Passive Sensing** and **Proxy Biomarkers**. Instead of invasive probes, we measure external signals—micro-vibrations of the chest, thermal asymmetries, gait kinematics—and use validated physical models to infer internal physiological states. We don't just measure; we *interpret* utilizing a multi-agent AI framework to validate findings against medical knowledge bases.

---

## 📉 The Problem

1.  **Friction**: Routine health checkups are time-consuming, invasive, and often unpleasant.
2.  **Snapshot Bias**: Clinical visits provide only a snapshot of health, often influenced by "white coat syndrome" (stress-induced spikes in vitals).
3.  **Accessibility**: High-quality diagnostics require specialized personnel and equipment, limiting access in remote or resource-constrained areas.

---

## 💡 Our Solution

A fully automated, sensor-rich chamber that performs a **9-system physiological review** in under 30 seconds.

*   **Zero Contact**: No cuffs, no electrodes, no wearables.
*   **Privacy First**: All processing can be done on-edge.
*   **AI-Augmented**: Deterministic physics-based extraction grounded by Probabilistic LLM interpretation.
*   **Holistic**: simultaneous analysis of multiple body systems.

---

## 🌟 Key Features

We monitor **9 Physiological Systems** using a fusion of RGB-D, Thermal, and Radar data:

### 🧠 1. Central Nervous System (CNS)
*   **Gait Analysis**: detects ataxia and parkinsonian gait patterns.
*   **Tremor Detection**: Sub-pixel motion analysis for resting and intention tremors.
*   **Posture Entropy**: Quantifies spinal stability and sway.

### ❤️ 2. Cardiovascular Health
*   **rPPG & Radar Fusion**: Contactless heart rate and HRV monitoring.
*   **Chest Micro-Motion**: Ballistocardiography via high-res visuals.
*   **Blood Pressure Proxies**: Pulse Wave Velocity (PWV) estimation.

### 🫁 3. Pulmonary Function
*   **Breathing Pattern**: Thoracic excursion depth and symmetry.
*   **Respiration Rate**: Radar-based chest wall vibration analysis.
*   **Audiology**: (Planned) Analysis of wheezing/crackles.

### 💧 4. Renal (Kidney) Function
*   **Fluid Retention**: Bioimpedance analysis (via Radar RIS) to detect edema.
*   **Orbital Puffiness**: Facial landmark analysis for fluid overload signs.

### 🍎 5. Gastrointestinal
*   **Abdominal Motility**: Micro-tremor analysis of the abdominal wall.
*   **Visceral Motion**: Frequency analysis of peristaltic rhythms.

### 🦴 6. Skeletal System
*   **Joint Range of Motion (ROM)**: Dynamic angle tracking during gait.
*   **Spinal Alignment**: Real-time scoliosis screening surrogates.

### 🌡️ 7. Skin & Integumentary
*   **Thermal Imaging**: Vascular perfusion and inflammation detection.
*   **Lesion Scanning**: Anomaly detection for potential dermatological issues.
*   **Jaundice Detection**: Scleral and skin colorimetric analysis.

### 👁️ 8. Ocular (Eyes)
*   **Oculometrics**: Blink rate variability (dopamine/fatigue proxy).
*   **Saccadic Latency**: Reaction time analysis.

### 🧬 9. Reproductive / Autonomic
*   **Stress Response**: Sympathovagal balance via HRV.
*   **Thermoregulation**: Core vs. peripheral temperature gradients.

---

## 🏗 System Architecture

The system utilizes a **Split-USB Topology** for modularity and safety.

```mermaid
graph TD
    subgraph Hardware Layer
        A[RGB-D Camera] -->|Video Feed| Hub[USB Hub]
        B[Seeed mmWave Radar] -->|Serial JSON| Hub
        C[Thermal Camera] -->|ESP32 Bridge| Hub
    end

    subgraph "Edge Compute (Laptop)"
        Hub -->|USB 3.0| Bridge[Bridge.py]
        
        Bridge -->|Frames| CV[Computer Vision Eng.]
        Bridge -->|Signal| SP[Signal Proc. Eng.]
        
        CV --> Fusion[Data Fusion Layer]
        SP --> Fusion
    end

    subgraph "Intelligence Layer"
        Fusion --> Risk[Risk Engine (Deterministic)]
        Risk --> Agent[AI Agents (Probabilistic)]
        Agent --> LLM[Gemini 1.5 Flash]
    end

    subgraph "Output Layer"
        LLM --> API[FastAPI Server]
        API --> PDF[Patient/Doctor Reports]
    end
```

### The "Trust Envelope"
We implement a strictly gated architecture:
1.  **Signal Quality**: Physics-based checks (SNR, exposure, dropout).
2.  **Biomarker Plausibility**: Physiologically impossible values are rejected.
3.  **Cross-System Consistency**: e.g., "If HR is >180, Gait cannot be Steady".
4.  **AI Interpretation**: Only validated data reaches the LLM for explanation.

---

## 💻 Technology Stack

*   **Core Backend**: Python 3.10+, FastAPI, Pydantic, Uvicorn.
*   **Computer Vision**: OpenCV (Headless), Google MediaPipe (Pose/Face/Hand).
*   **Signal Processing**: NumPy, SciPy (FFT, Digital Filters).
*   **Hardware Interface**: PySerial (UART), Bleak (Bluetooth Low Energy).
*   **Generative AI**: Google Generative AI (Gemini 1.5), HuggingFace Inference Client.
*   **Reporting**: WeasyPrint (HTML-to-PDF), Jinja2 Templating.
*   **Testing**: Pytest, Unittest options.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10 or higher
*   Webcam (Required)
*   Seeed MR60BHA2 Radar (Optional, for full features)
*   ESP32 + MLX90640 Thermal Cam (Optional)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/walkthrough-chamber.git
    cd walkthrough-chamber
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Environment Variables**
    Create a `.env` file in the root directory:
    ```ini
    GOOGLE_API_KEY=your_gemini_api_key
    HF_TOKEN=your_huggingface_token
    ```

### Running the System

1.  **Start the API Server**
    ```bash
    uvicorn app.main:app --reload
    ```

2.  **Run the Hardware Bridge** (in a separate terminal)
    ```bash
    # For full hardware Mode
    python bridge.py --camera 0 --radar-port COM3 --thermal-port COM4

    # For Simulation/Test Mode (No hardware needed)
    python bridge.py --simulate
    ```

3.  **Access the Dashboard**
    Open `http://localhost:8000/docs` to interact with the API Swagger UI.

---

## 🛣 Roadmap

- [x] **Phase 1: Alpha**: Core CV and Signal extraction algorithms.
- [x] **Phase 2: Integration**: Split-USB hardware bridge and sensor fusion.
- [x] **Phase 3: Intelligence**: Multi-agent validation and Gemini integration.
- [ ] **Phase 4: Beta**: Real-time feedback UI and mobile app connector.
- [ ] **Phase 5: Clinical**: Validation studies against gold-standard vitals.

---

## ⚠️ Disclaimer

> **Investigational Device**: This software is for research and educational purposes only. It is **not** a confirmed medical diagnostic tool. The "diagnoses" and "risks" calculated are screening estimates and should not replace professional medical advice.

---

### 📩 Contact & Support

For questions, issues, or collaboration, please open a GitHub Issue or contact the maintainers.

Designed with ❤️ for the Future of Health.
