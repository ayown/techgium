/*
 * ESP32 Health Screening Firmware
 * 
 * Interfaces with:
 * - MLX90640 Thermal Camera (I2C)
 * - MAX30102 Pulse Oximeter (I2C)
 * - 60GHz mmWave Radar (UART2)
 * 
 * Sends JSON data to host via UART0 at 115200 baud
 * 
 * Hardware Connections:
 *   GPIO21 - I2C SDA
 *   GPIO22 - I2C SCL  
 *   GPIO16 - UART2 RX (Radar TX)
 *   GPIO17 - UART2 TX (Radar RX)
 *   GPIO04 - Status LED
 *   GPIO05 - Activity LED
 */

#include <Wire.h>
#include <ArduinoJson.h>

// =============================================================================
// PIN DEFINITIONS
// =============================================================================
#define I2C_SDA 21
#define I2C_SCL 22
#define UART2_RX 16
#define UART2_TX 17
#define LED_STATUS 4
#define LED_ACTIVITY 5

// I2C Addresses
#define MLX90640_ADDR 0x33
#define MAX30102_ADDR 0x57

// =============================================================================
// CONFIGURATION
// =============================================================================
#define SERIAL_BAUD 115200
#define RADAR_BAUD 921600
#define SAMPLE_INTERVAL_MS 50  // 20 Hz output

// =============================================================================
// GLOBAL VARIABLES
// =============================================================================
float skinTempAvg = 0.0;
float skinTempMax = 0.0;
float thermalAsymmetry = 0.0;

int heartRate = 0;
int spO2 = 0;

float respirationRate = 0.0;
float breathingDepth = 0.0;
float microMotion = 0.0;

unsigned long lastSampleTime = 0;

// =============================================================================
// INITIALIZATION
// =============================================================================
void setup() {
    // Initialize serial ports
    Serial.begin(SERIAL_BAUD);
    Serial2.begin(RADAR_BAUD, SERIAL_8N1, UART2_RX, UART2_TX);
    
    // Initialize I2C
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);  // 400 kHz I2C
    
    // Initialize LEDs
    pinMode(LED_STATUS, OUTPUT);
    pinMode(LED_ACTIVITY, OUTPUT);
    digitalWrite(LED_STATUS, HIGH);  // Status on
    digitalWrite(LED_ACTIVITY, LOW);
    
    // Initialize sensors
    initMLX90640();
    initMAX30102();
    
    Serial.println("{\"status\": \"ESP32 Health Bridge Ready\"}");
}

// =============================================================================
// SENSOR INITIALIZATION
// =============================================================================
void initMLX90640() {
    // Check if MLX90640 is present
    Wire.beginTransmission(MLX90640_ADDR);
    if (Wire.endTransmission() == 0) {
        Serial.println("{\"sensor\": \"MLX90640\", \"status\": \"connected\"}");
    } else {
        Serial.println("{\"sensor\": \"MLX90640\", \"status\": \"not found\"}");
    }
}

void initMAX30102() {
    // Check if MAX30102 is present
    Wire.beginTransmission(MAX30102_ADDR);
    if (Wire.endTransmission() == 0) {
        Serial.println("{\"sensor\": \"MAX30102\", \"status\": \"connected\"}");
        // Initialize MAX30102 (simplified)
        writeRegister(MAX30102_ADDR, 0x09, 0x40);  // Mode config
        writeRegister(MAX30102_ADDR, 0x0A, 0x27);  // SpO2 config
    } else {
        Serial.println("{\"sensor\": \"MAX30102\", \"status\": \"not found\"}");
    }
}

void writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
    Wire.beginTransmission(addr);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

uint8_t readRegister(uint8_t addr, uint8_t reg) {
    Wire.beginTransmission(addr);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom(addr, (uint8_t)1);
    return Wire.read();
}

// =============================================================================
// SENSOR READING FUNCTIONS
// =============================================================================
void readMLX90640() {
    // Simplified thermal reading (full implementation would read 768 pixels)
    // For hackathon: simulate reasonable thermal values
    
    // In production: use Adafruit_MLX90640 library
    // mlx.getFrame(frame);
    
    // Simulated values based on typical face thermal profile
    skinTempAvg = 36.0 + random(0, 100) / 100.0;  // 36.0 - 37.0
    skinTempMax = skinTempAvg + 0.5 + random(0, 50) / 100.0;  // 0.5 - 1.0 above avg
    thermalAsymmetry = random(10, 50) / 100.0;  // 0.1 - 0.5
}

void readMAX30102() {
    // Simplified pulse ox reading
    // In production: use SparkFun_MAX3010x library with proper algorithm
    
    // Read FIFO data (simplified)
    // uint32_t irValue = particleSensor.getIR();
    // uint32_t redValue = particleSensor.getRed();
    
    // Simulated physiological values
    heartRate = 60 + random(0, 30);  // 60-90 bpm
    spO2 = 95 + random(0, 4);  // 95-99%
}

void readRadar() {
    // Read and parse mmWave radar data from UART2
    // TI radar typically sends processed vital signs data
    
    String radarData = "";
    while (Serial2.available()) {
        char c = Serial2.read();
        radarData += c;
    }
    
    // Parse radar output (format depends on radar firmware)
    // For hackathon: simulate reasonable respiratory values
    
    respirationRate = 12.0 + random(0, 80) / 10.0;  // 12-20 breaths/min
    breathingDepth = 0.5 + random(0, 40) / 100.0;   // 0.5-0.9 normalized
    microMotion = random(20, 80) / 10000.0;         // 0.002-0.008
}

// =============================================================================
// MAIN LOOP
// =============================================================================
void loop() {
    unsigned long currentTime = millis();
    
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = currentTime;
        
        // Toggle activity LED
        digitalWrite(LED_ACTIVITY, !digitalRead(LED_ACTIVITY));
        
        // Read all sensors
        readMLX90640();
        readMAX30102();
        readRadar();
        
        // Build JSON output
        sendJsonData();
    }
}

void sendJsonData() {
    StaticJsonDocument<512> doc;
    
    // Timestamp (seconds since boot)
    doc["timestamp"] = millis() / 1000;
    
    // Radar data
    JsonObject radar = doc.createNestedObject("radar");
    radar["respiration_rate"] = respirationRate;
    radar["breathing_depth"] = breathingDepth;
    radar["micro_motion"] = microMotion;
    
    // Thermal data
    JsonObject thermal = doc.createNestedObject("thermal");
    thermal["skin_temp_avg"] = skinTempAvg;
    thermal["skin_temp_max"] = skinTempMax;
    thermal["thermal_asymmetry"] = thermalAsymmetry;
    
    // Pulse oximeter data
    JsonObject pulseOx = doc.createNestedObject("pulse_ox");
    pulseOx["heart_rate"] = heartRate;
    pulseOx["spo2"] = spO2;
    
    // Send as single line JSON
    serializeJson(doc, Serial);
    Serial.println();
}
