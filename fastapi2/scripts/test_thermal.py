import serial
import json
import time
import sys

# CONFIGURATION
PORT = "COM6"
BAUD = 115200
TIMEOUT = 1

def run_test():
    print(f"--- ESP32 Thermal Camera Test on {PORT} ---")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        print(f"Successfully connected to {PORT}")
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {PORT}. {e}")
        sys.exit(1)

    print("Listening for data... (Press Ctrl+C to stop)")
    
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            # Try to parse as JSON
            try:
                data = json.loads(line)
                
                # Check for status or error messages
                if "status" in data:
                    print(f"[{time.strftime('%H:%M:%S')}] STATUS: {data['status']} (FPS: {data.get('fps', 'N/A')})")
                elif "error" in data:
                    print(f"[{time.strftime('%H:%M:%S')}] ERROR: {data['error']}")
                elif "thermal" in data:
                    thermal = data["thermal"]
                    core = thermal.get("core_regions", {})
                    stability = thermal.get("stability_metrics", {})
                    symmetry = thermal.get("symmetry", {})
                    gradients = thermal.get("gradients", {})

                    print(f"\n[{time.strftime('%H:%M:%S')}] Thermal Data:")
                    print(f"  - Canthus Mean: {core.get('canthus_mean', 'N/A')}°C")
                    print(f"  - Neck Mean:    {core.get('neck_mean', 'N/A')}°C")
                    print(f"  - Stability:    {stability.get('canthus_range', 'N/A')} (Range)")
                    print(f"  - Symmetry:     {symmetry.get('cheek_asymmetry', 'N/A')} (Cheek Asymmetry)")
                    print(f"  - Gradient:     {gradients.get('forehead_nose_gradient', 'N/A')} (Forehead-Nose)")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Raw JSON: {data}")

            except json.JSONDecodeError:
                # If not JSON, just print the raw line if it's not empty
                print(f"[{time.strftime('%H:%M:%S')}] RAW: {line}")

    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    run_test()
