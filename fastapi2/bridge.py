"""
Hardware Bridge - ESP32 to FastAPI Integration

This module provides seamless integration between hardware sensors (ESP32, camera, thermal)
and the FastAPI health screening pipeline.

Components:
- ESP32 serial reader (radar, thermal, pulse ox data)
- Camera capture (rPPG, pose, face analysis)
- Data fusion and synchronization
- API client for screening endpoint

Usage:
    python bridge.py [--port COM3] [--camera 0] [--api-url http://localhost:8000]
"""

import asyncio
import json
import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import argparse

import numpy as np
import cv2
import requests

# Optional serial import (may not be available on all systems)
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. Run: pip install pyserial")

# Optional MediaPipe import
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Camera features limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class BridgeConfig:
    """Configuration for the hardware bridge."""
    # Serial (ESP32)
    serial_port: str = "COM3"  # Windows: COM3, Linux: /dev/ttyUSB0
    serial_baud: int = 115200
    
    # Camera
    camera_index: int = 0
    camera_fps: int = 30
    face_capture_seconds: int = 10
    body_capture_seconds: int = 10
    
    # API
    api_url: str = "http://localhost:8000"
    api_timeout: int = 30
    
    # Timing
    sensor_poll_interval: float = 0.05  # 20 Hz
    data_fusion_interval: float = 1.0   # 1 Hz API calls


# ==============================================================================
# ESP32 SERIAL READER
# ==============================================================================

class ESP32Reader:
    """Reads sensor data from ESP32 via serial port."""
    
    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.serial_conn: Optional[serial.Serial] = None
        self.running = False
        self.data_queue: queue.Queue = queue.Queue(maxsize=100)
        self.last_data: Dict[str, Any] = {}
        
    def connect(self) -> bool:
        """Connect to ESP32 serial port."""
        if not SERIAL_AVAILABLE:
            logger.error("pyserial not installed")
            return False
            
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=1.0
            )
            logger.info(f"Connected to ESP32 on {self.port}")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32."""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Disconnected from ESP32")
    
    def read_loop(self):
        """Background thread to read serial data."""
        self.running = True
        while self.running and self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line:
                    data = json.loads(line)
                    data['received_at'] = time.time()
                    self.last_data = data
                    
                    # Add to queue (drop oldest if full)
                    try:
                        self.data_queue.put_nowait(data)
                    except queue.Full:
                        self.data_queue.get()
                        self.data_queue.put_nowait(data)
                        
            except json.JSONDecodeError as e:
                logger.debug(f"Invalid JSON from ESP32: {e}")
            except Exception as e:
                logger.error(f"Serial read error: {e}")
                time.sleep(0.1)
    
    def start_reading(self):
        """Start background reading thread."""
        thread = threading.Thread(target=self.read_loop, daemon=True)
        thread.start()
        return thread
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get most recent sensor data."""
        return self.last_data if self.last_data else None


# ==============================================================================
# CAMERA CAPTURE (rPPG + Pose)
# ==============================================================================

class CameraCapture:
    """Captures face and body frames for biomarker extraction."""
    
    def __init__(self, camera_index: int = 0, fps: int = 30):
        self.camera_index = camera_index
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        
        # MediaPipe components
        self.face_detector = None
        self.pose_landmarker = None
        self.face_landmarker = None
        
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe models."""
        if not MEDIAPIPE_AVAILABLE:
            return
            
        try:
            # Face detection for rPPG
            base_options = mp_python.BaseOptions(
                model_asset_path='face_detection_short_range.task'
            )
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE
            )
            self.face_detector = mp_vision.FaceDetector.create_from_options(options)
        except Exception as e:
            logger.warning(f"Face detector init failed: {e}")
        
        try:
            # Pose landmarker
            base_options = mp_python.BaseOptions(
                model_asset_path='pose_landmarker.task'
            )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE
            )
            self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            logger.warning(f"Pose landmarker init failed: {e}")
    
    def open(self) -> bool:
        """Open camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info(f"Camera {self.camera_index} opened")
        return True
    
    def close(self):
        """Close camera."""
        if self.cap:
            self.cap.release()
            logger.info("Camera closed")
    
    def capture_frames(self, duration_seconds: int) -> Tuple[List[np.ndarray], List[float]]:
        """Capture frames for specified duration."""
        frames = []
        timestamps = []
        target_frames = duration_seconds * self.fps
        
        start_time = time.time()
        while len(frames) < target_frames:
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
                timestamps.append(time.time() - start_time)
            else:
                break
                
            # Maintain frame rate
            elapsed = time.time() - start_time
            expected = len(frames) / self.fps
            if elapsed < expected:
                time.sleep(expected - elapsed)
        
        return frames, timestamps
    
    def extract_face_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Extract cropped face regions for rPPG."""
        face_frames = []
        
        if not self.face_detector:
            # Fallback: use center region
            for frame in frames:
                h, w = frame.shape[:2]
                face_crop = frame[h//4:3*h//4, w//4:3*w//4]
                face_frames.append(face_crop)
            return face_frames
        
        for frame in frames:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = self.face_detector.detect(mp_image)
                
                if result.detections:
                    bbox = result.detections[0].bounding_box
                    x, y = bbox.origin_x, bbox.origin_y
                    w, h = bbox.width, bbox.height
                    
                    # Add padding
                    pad = int(max(w, h) * 0.2)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    y2 = min(frame.shape[0], y + h + pad)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    face_frames.append(face_crop)
                else:
                    # Fallback to center
                    fh, fw = frame.shape[:2]
                    face_frames.append(frame[fh//4:3*fh//4, fw//4:3*fw//4])
            except Exception as e:
                logger.debug(f"Face detection error: {e}")
                fh, fw = frame.shape[:2]
                face_frames.append(frame[fh//4:3*fh//4, fw//4:3*fw//4])
        
        return face_frames
    
    def extract_pose_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Extract pose landmarks for gait/posture analysis."""
        pose_sequence = []
        
        if not self.pose_landmarker:
            return pose_sequence
        
        for frame in frames:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = self.pose_landmarker.detect(mp_image)
                
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    landmark_array = np.array([
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in landmarks
                    ])
                    pose_sequence.append(landmark_array)
            except Exception as e:
                logger.debug(f"Pose detection error: {e}")
        
        return pose_sequence


# ==============================================================================
# DATA FUSION & API CLIENT
# ==============================================================================

class DataFusion:
    """Fuses sensor data and sends to API."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.screening_endpoint = f"{api_url}/api/v1/screening"
        self.report_endpoint = f"{api_url}/api/v1/reports/generate"
    
    def transform_esp32_data(self, esp32_data: Dict[str, Any]) -> List[Dict]:
        """Transform ESP32 sensor data to API biomarker format."""
        systems = []
        
        # Cardiovascular biomarkers from pulse oximeter
        if "pulse_ox" in esp32_data:
            pulse = esp32_data["pulse_ox"]
            cv_biomarkers = []
            
            if "heart_rate" in pulse:
                cv_biomarkers.append({
                    "name": "heart_rate",
                    "value": float(pulse["heart_rate"]),
                    "unit": "bpm",
                    "normal_range": [60, 100]
                })
            if "spo2" in pulse:
                cv_biomarkers.append({
                    "name": "spo2",
                    "value": float(pulse["spo2"]),
                    "unit": "%",
                    "normal_range": [95, 100]
                })
            
            if cv_biomarkers:
                systems.append({
                    "system": "cardiovascular",
                    "biomarkers": cv_biomarkers
                })
        
        # Respiratory biomarkers from radar
        if "radar" in esp32_data:
            radar = esp32_data["radar"]
            resp_biomarkers = []
            
            if "respiration_rate" in radar:
                resp_biomarkers.append({
                    "name": "respiration_rate",
                    "value": float(radar["respiration_rate"]),
                    "unit": "breaths/min",
                    "normal_range": [12, 20]
                })
            if "breathing_depth" in radar:
                resp_biomarkers.append({
                    "name": "breathing_depth",
                    "value": float(radar["breathing_depth"]),
                    "unit": "normalized",
                    "normal_range": [0.5, 1.0]
                })
            if "micro_motion" in radar:
                resp_biomarkers.append({
                    "name": "chest_micro_motion",
                    "value": float(radar["micro_motion"]),
                    "unit": "normalized_amplitude",
                    "normal_range": [0.001, 0.01]
                })
            
            if resp_biomarkers:
                systems.append({
                    "system": "pulmonary",
                    "biomarkers": resp_biomarkers
                })
        
        # Skin biomarkers from thermal camera
        if "thermal" in esp32_data:
            thermal = esp32_data["thermal"]
            skin_biomarkers = []
            
            if "skin_temp_avg" in thermal:
                skin_biomarkers.append({
                    "name": "skin_temperature",
                    "value": float(thermal["skin_temp_avg"]),
                    "unit": "celsius",
                    "normal_range": [35.5, 37.5]
                })
            if "thermal_asymmetry" in thermal:
                skin_biomarkers.append({
                    "name": "thermal_asymmetry",
                    "value": float(thermal["thermal_asymmetry"]),
                    "unit": "delta_celsius",
                    "normal_range": [0, 0.5]
                })
            if "skin_temp_max" in thermal:
                skin_biomarkers.append({
                    "name": "skin_temperature_max",
                    "value": float(thermal["skin_temp_max"]),
                    "unit": "celsius",
                    "normal_range": [36.0, 38.0]
                })
            
            if skin_biomarkers:
                systems.append({
                    "system": "skin",
                    "biomarkers": skin_biomarkers
                })
        
        return systems
    
    def extract_rppg_biomarkers(
        self, 
        face_frames: List[np.ndarray], 
        fps: float
    ) -> List[Dict]:
        """Extract cardiovascular biomarkers from face frames using rPPG."""
        if len(face_frames) < fps * 5:  # Need at least 5 seconds
            logger.warning("Insufficient face frames for rPPG")
            return []
        
        try:
            # Extract green channel signal
            green_signal = []
            for frame in face_frames:
                if len(frame.shape) == 3:
                    h, w = frame.shape[:2]
                    roi = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8), 1]
                    green_signal.append(np.mean(roi))
                else:
                    green_signal.append(np.mean(frame))
            
            green_signal = np.array(green_signal)
            
            # Detrend and filter
            from scipy import signal as scipy_signal
            
            # Detrend
            detrended = scipy_signal.detrend(green_signal)
            
            # Bandpass filter (0.8-3 Hz for heart rate)
            nyquist = fps / 2
            low = 0.8 / nyquist
            high = min(3.0 / nyquist, 0.99)
            sos = scipy_signal.butter(4, [low, high], btype='band', output='sos')
            filtered = scipy_signal.sosfilt(sos, detrended)
            
            # FFT for heart rate
            n = len(filtered)
            freqs = np.fft.fftfreq(n, d=1/fps)
            fft_vals = np.abs(np.fft.fft(filtered))
            
            # Find peak in cardiac range
            cardiac_mask = (freqs > 0.8) & (freqs < 3.0)
            if np.any(cardiac_mask):
                cardiac_freqs = freqs[cardiac_mask]
                cardiac_power = fft_vals[cardiac_mask]
                peak_idx = np.argmax(cardiac_power)
                peak_freq = cardiac_freqs[peak_idx]
                hr = float(abs(peak_freq) * 60)
                hr = np.clip(hr, 45, 180)
            else:
                hr = 72.0
            
            # Compute HRV (simplified)
            peaks, _ = scipy_signal.find_peaks(filtered, distance=int(fps * 0.4))
            if len(peaks) >= 3:
                rr_intervals = np.diff(peaks) / fps * 1000  # ms
                valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
                if len(valid_rr) >= 2:
                    hrv = float(np.sqrt(np.mean(np.diff(valid_rr)**2)))
                    hrv = np.clip(hrv, 10, 150)
                else:
                    hrv = 40.0
            else:
                hrv = 40.0
            
            return [
                {"name": "heart_rate", "value": hr, "unit": "bpm", "normal_range": [60, 100]},
                {"name": "hrv_rmssd", "value": hrv, "unit": "ms", "normal_range": [20, 80]}
            ]
            
        except Exception as e:
            logger.error(f"rPPG extraction failed: {e}")
            return []
    
    def extract_motion_biomarkers(
        self, 
        pose_sequence: List[np.ndarray]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract CNS and skeletal biomarkers from pose data."""
        cns_biomarkers = []
        skeletal_biomarkers = []
        
        if len(pose_sequence) < 30:
            return cns_biomarkers, skeletal_biomarkers
        
        try:
            pose_array = np.array(pose_sequence)
            
            # Gait analysis (using hip landmarks 23, 24)
            if pose_array.shape[1] >= 25:
                left_hip = pose_array[:, 23, :2]
                right_hip = pose_array[:, 24, :2]
                hip_center = (left_hip + right_hip) / 2
                
                # Gait variability (vertical movement std)
                vertical_motion = hip_center[:, 1]
                gait_var = float(np.std(np.diff(vertical_motion)))
                cns_biomarkers.append({
                    "name": "gait_variability",
                    "value": gait_var,
                    "unit": "normalized",
                    "normal_range": [0.001, 0.02]
                })
            
            # Posture analysis (shoulder alignment)
            if pose_array.shape[1] >= 13:
                left_shoulder = pose_array[:, 11, :2]
                right_shoulder = pose_array[:, 12, :2]
                
                # Shoulder tilt
                shoulder_diff = np.mean(np.abs(left_shoulder[:, 1] - right_shoulder[:, 1]))
                skeletal_biomarkers.append({
                    "name": "shoulder_asymmetry",
                    "value": float(shoulder_diff),
                    "unit": "normalized",
                    "normal_range": [0, 0.05]
                })
            
            # Tremor analysis (hand movement)
            if pose_array.shape[1] >= 22:
                left_wrist = pose_array[:, 15, :2]
                right_wrist = pose_array[:, 16, :2]
                
                # High-frequency motion as tremor proxy
                wrist_motion = np.diff(left_wrist, axis=0)
                tremor_power = float(np.std(wrist_motion))
                cns_biomarkers.append({
                    "name": "tremor_proxy",
                    "value": tremor_power,
                    "unit": "normalized",
                    "normal_range": [0, 0.01]
                })
            
        except Exception as e:
            logger.error(f"Motion biomarker extraction failed: {e}")
        
        return cns_biomarkers, skeletal_biomarkers
    
    def build_screening_request(
        self,
        patient_id: str,
        esp32_data: Optional[Dict] = None,
        face_frames: Optional[List[np.ndarray]] = None,
        pose_sequence: Optional[List[np.ndarray]] = None,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """Build complete screening request from all data sources."""
        systems = []
        
        # Add ESP32 sensor data
        if esp32_data:
            systems.extend(self.transform_esp32_data(esp32_data))
        
        # Add rPPG biomarkers
        if face_frames and len(face_frames) > 0:
            rppg_biomarkers = self.extract_rppg_biomarkers(face_frames, fps)
            if rppg_biomarkers:
                # Merge with existing cardiovascular or create new
                cv_system = next((s for s in systems if s["system"] == "cardiovascular"), None)
                if cv_system:
                    cv_system["biomarkers"].extend(rppg_biomarkers)
                else:
                    systems.append({
                        "system": "cardiovascular",
                        "biomarkers": rppg_biomarkers
                    })
        
        # Add motion biomarkers
        if pose_sequence and len(pose_sequence) > 0:
            cns_biomarkers, skeletal_biomarkers = self.extract_motion_biomarkers(pose_sequence)
            
            if cns_biomarkers:
                systems.append({
                    "system": "cns",
                    "biomarkers": cns_biomarkers
                })
            if skeletal_biomarkers:
                systems.append({
                    "system": "skeletal",
                    "biomarkers": skeletal_biomarkers
                })
        
        return {
            "patient_id": patient_id,
            "include_validation": True,
            "systems": systems
        }
    
    def send_screening(self, request: Dict[str, Any]) -> Optional[Dict]:
        """Send screening request to API."""
        try:
            response = requests.post(
                self.screening_endpoint,
                json=request,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Screening completed: {result.get('screening_id')}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def generate_report(self, screening_id: str, report_type: str = "patient") -> Optional[Dict]:
        """Generate report for completed screening."""
        try:
            response = requests.post(
                self.report_endpoint,
                json={"screening_id": screening_id, "report_type": report_type},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Report generation failed: {e}")
            return None


# ==============================================================================
# MAIN BRIDGE CONTROLLER
# ==============================================================================

class HardwareBridge:
    """Main controller for hardware-software integration."""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.esp32_reader: Optional[ESP32Reader] = None
        self.camera: Optional[CameraCapture] = None
        self.data_fusion = DataFusion(config.api_url)
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize all hardware connections."""
        success = True
        
        # Initialize ESP32
        if SERIAL_AVAILABLE:
            self.esp32_reader = ESP32Reader(
                port=self.config.serial_port,
                baud=self.config.serial_baud
            )
            if not self.esp32_reader.connect():
                logger.warning("ESP32 not connected - continuing without sensors")
                self.esp32_reader = None
        
        # Initialize camera
        self.camera = CameraCapture(
            camera_index=self.config.camera_index,
            fps=self.config.camera_fps
        )
        if not self.camera.open():
            logger.warning("Camera not available - continuing without camera")
            self.camera = None
        
        return success
    
    def run_single_screening(self, patient_id: str = "WALKTHROUGH_PATIENT") -> Optional[Dict]:
        """Run a complete screening session."""
        logger.info("=" * 60)
        logger.info("Starting health screening session")
        logger.info("=" * 60)
        
        esp32_data = None
        face_frames = []
        body_frames = []
        pose_sequence = []
        
        # Start ESP32 data collection
        if self.esp32_reader:
            logger.info("Starting sensor data collection...")
            self.esp32_reader.start_reading()
        
        # Phase 1: Face capture for rPPG
        if self.camera:
            logger.info(f"\n📷 Phase 1: Face capture ({self.config.face_capture_seconds}s)")
            logger.info("Please look directly at the camera...")
            
            face_frames, _ = self.camera.capture_frames(self.config.face_capture_seconds)
            face_frames = self.camera.extract_face_frames(face_frames)
            logger.info(f"Captured {len(face_frames)} face frames")
        
        # Phase 2: Body capture for gait/posture
        if self.camera:
            logger.info(f"\n🚶 Phase 2: Body capture ({self.config.body_capture_seconds}s)")
            logger.info("Please walk naturally or stand for posture analysis...")
            
            body_frames, _ = self.camera.capture_frames(self.config.body_capture_seconds)
            pose_sequence = self.camera.extract_pose_sequence(body_frames)
            logger.info(f"Extracted {len(pose_sequence)} pose frames")
        
        # Get ESP32 sensor data
        if self.esp32_reader:
            esp32_data = self.esp32_reader.get_latest_data()
            if esp32_data:
                logger.info(f"ESP32 data received: {list(esp32_data.keys())}")
            else:
                logger.warning("No ESP32 data received")
        
        # Build and send screening request
        logger.info("\n📊 Processing biomarkers and sending to API...")
        
        request = self.data_fusion.build_screening_request(
            patient_id=patient_id,
            esp32_data=esp32_data,
            face_frames=face_frames if face_frames else None,
            pose_sequence=pose_sequence if pose_sequence else None,
            fps=self.config.camera_fps
        )
        
        logger.info(f"Screening request: {len(request['systems'])} systems")
        for sys in request['systems']:
            logger.info(f"  - {sys['system']}: {len(sys['biomarkers'])} biomarkers")
        
        result = self.data_fusion.send_screening(request)
        
        if result:
            logger.info("\n✅ Screening completed!")
            logger.info(f"Screening ID: {result.get('screening_id')}")
            logger.info(f"Overall Risk: {result.get('overall_risk_level')} ({result.get('overall_risk_score')})")
            
            # Generate reports
            screening_id = result.get('screening_id')
            if screening_id:
                patient_report = self.data_fusion.generate_report(screening_id, "patient")
                if patient_report:
                    logger.info(f"Patient report: {patient_report.get('pdf_path')}")
                
                doctor_report = self.data_fusion.generate_report(screening_id, "doctor")
                if doctor_report:
                    logger.info(f"Doctor report: {doctor_report.get('pdf_path')}")
        else:
            logger.error("❌ Screening failed")
        
        return result
    
    def cleanup(self):
        """Cleanup all resources."""
        if self.esp32_reader:
            self.esp32_reader.disconnect()
        if self.camera:
            self.camera.close()
        logger.info("Hardware bridge cleanup complete")


# ==============================================================================
# SIMULATED ESP32 DATA (for testing without hardware)
# ==============================================================================

def generate_simulated_esp32_data() -> Dict[str, Any]:
    """Generate realistic simulated sensor data for testing."""
    return {
        "timestamp": int(time.time()),
        "radar": {
            "respiration_rate": round(np.random.uniform(12, 18), 1),
            "breathing_depth": round(np.random.uniform(0.5, 0.9), 2),
            "micro_motion": round(np.random.uniform(0.002, 0.008), 4)
        },
        "thermal": {
            "skin_temp_avg": round(np.random.uniform(36.0, 37.0), 1),
            "skin_temp_max": round(np.random.uniform(36.5, 37.5), 1),
            "thermal_asymmetry": round(np.random.uniform(0.1, 0.4), 2)
        },
        "pulse_ox": {
            "spo2": int(np.random.uniform(96, 99)),
            "heart_rate": int(np.random.uniform(65, 85))
        }
    }


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    """Main entry point for hardware bridge."""
    parser = argparse.ArgumentParser(description="Hardware Bridge for Health Screening")
    parser.add_argument("--port", default="COM3", help="ESP32 serial port")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--simulate", action="store_true", help="Use simulated sensor data")
    parser.add_argument("--patient-id", default="WALKTHROUGH_001", help="Patient ID")
    args = parser.parse_args()
    
    config = BridgeConfig(
        serial_port=args.port,
        camera_index=args.camera,
        api_url=args.api_url
    )
    
    bridge = HardwareBridge(config)
    
    try:
        print("\n" + "=" * 60)
        print("  HEALTH SCREENING HARDWARE BRIDGE")
        print("=" * 60)
        
        if args.simulate:
            print("\n⚠️  SIMULATION MODE - Using generated sensor data")
            # Create mock ESP32 data
            simulated_data = generate_simulated_esp32_data()
            print(f"Simulated data: {json.dumps(simulated_data, indent=2)}")
            
            # Send directly
            fusion = DataFusion(config.api_url)
            request = fusion.build_screening_request(
                patient_id=args.patient_id,
                esp32_data=simulated_data
            )
            result = fusion.send_screening(request)
            
            if result:
                print(f"\n✅ Screening ID: {result.get('screening_id')}")
                print(f"Risk Level: {result.get('overall_risk_level')}")
        else:
            bridge.initialize()
            bridge.run_single_screening(patient_id=args.patient_id)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    main()
