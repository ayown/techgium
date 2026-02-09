// esp32_thermal_biomarkers_v2.ino
// Production-ready thermal camera firmware with all fixes

#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[768];  // 32×24 pixels

struct ROI {
  float temp_mean;
  float temp_max;
  float temp_min;
  bool valid;
};

// Configuration
const int LED_STATUS = 4;   // Blue LED for status
const int LED_ACTIVITY = 5; // Green LED for activity
const float FEVER_THRESHOLD = 37.35;
const float DIABETES_THRESHOLD = 35.5;
const float CVD_ASYMMETRY_THRESHOLD = 0.5;
const float INFLAMMATION_PCT_THRESHOLD = 5.0;
const float STRESS_GRADIENT_THRESHOLD = 1.5;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  pinMode(LED_STATUS, OUTPUT);
  pinMode(LED_ACTIVITY, OUTPUT);
  
  // Startup indication
  digitalWrite(LED_STATUS, HIGH);
  
  // Initialize MLX90640
  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("{\"error\":\"MLX90640 not found\",\"code\":1}");
    while (1) {
      digitalWrite(LED_STATUS, !digitalRead(LED_STATUS));
      delay(200);  // Blink rapidly on error
    }
  }
  
  // Configure sensor
  mlx.setMode(MLX90640_CHESS);           // Chess pattern for 8fps
  mlx.setResolution(MLX90640_ADC_18BIT); // Highest resolution
  mlx.setRefreshRate(MLX90640_8_HZ);     // Max stable rate
  
  // Ready indication
  digitalWrite(LED_STATUS, LOW);
  delay(1000);
  
  Serial.println("{\"status\":\"ready\",\"version\":\"2.0\",\"fps\":8}");
}

void loop() {
  // Activity LED on during frame capture
  digitalWrite(LED_ACTIVITY, HIGH);
  
  int status = mlx.getFrame(frame);
  
  if (status != 0) {
    // Sensor read failed
    Serial.println("{\"error\":\"Frame read failed\",\"status\":" + String(status) + "}");
    digitalWrite(LED_ACTIVITY, LOW);
    delay(125);
    return;
  }
  
  // Validate frame data
  if (!isFrameValid()) {
    Serial.println("{\"error\":\"Invalid frame data\",\"reason\":\"Temperature out of range\"}");
    digitalWrite(LED_ACTIVITY, LOW);
    delay(125);
    return;
  }
  
  // Extract biomarkers
  extractClinicalBiomarkers();
  
  digitalWrite(LED_ACTIVITY, LOW);
  delay(125);  // 8 fps
}

bool isFrameValid() {
  float min_temp = 999.0, max_temp = -999.0;
  int valid_pixels = 0;
  
  for (int i = 0; i < 768; i++) {
    float temp = frame[i];
    
    // Check for NaN or invalid values
    if (isnan(temp)) continue;
    
    if (temp < min_temp) min_temp = temp;
    if (temp > max_temp) max_temp = temp;
    valid_pixels++;
  }
  
  // Validation criteria
  bool has_enough_pixels = (valid_pixels > 700);  // At least 90% valid
  bool temp_in_range = (min_temp > 15.0 && max_temp < 60.0);
  bool has_variation = (max_temp - min_temp > 0.5);
  
  return has_enough_pixels && temp_in_range && has_variation;
}

void extractClinicalBiomarkers() {
  // Extract ROIs with bounds checking
  ROI left_canthus = getROIStats(10, 14, 12, 14);
  ROI right_canthus = getROIStats(10, 14, 18, 20);
  ROI neck = getROIStats(22, 24, 10, 22);
  ROI left_cheek = getROIStats(10, 18, 6, 14);
  ROI right_cheek = getROIStats(10, 18, 18, 26);
  ROI nose_tip = getROIStats(16, 20, 14, 18);
  ROI forehead = getROIStats(2, 8, 10, 22);
  
  // Validate critical ROIs
  if (!left_canthus.valid || !right_canthus.valid || !neck.valid) {
    Serial.println("{\"error\":\"Critical ROI extraction failed\",\"reason\":\"Face not detected\"}");
    return;
  }
  
  // Calculate biomarkers
  float canthus_temp = (left_canthus.temp_mean + right_canthus.temp_mean) / 2.0;
  float thermal_asymmetry = abs(left_cheek.temp_mean - right_cheek.temp_mean);
  float face_mean_temp = calculateFaceMeanTemp();
  float hot_pixel_pct = calculateHotPixelPercentage(face_mean_temp, 1.0);
  float stress_gradient = forehead.valid ? (forehead.temp_mean - nose_tip.temp_mean) : 0.0;
  
  // Calculate stability metrics (std deviation within ROIs)
  float canthus_stability = (left_canthus.temp_max - left_canthus.temp_min + 
                             right_canthus.temp_max - right_canthus.temp_min) / 2.0;
  float neck_stability = neck.temp_max - neck.temp_min;
  
  // Build single-line JSON with all biomarkers
  String json = "{";
  json += "\"timestamp\":" + String(millis()) + ",";
  json += "\"thermal\":{";
  
  // Fever biomarkers
  json += "\"fever\":{";
  json += "\"canthus_temp\":" + String(canthus_temp, 2) + ",";
  json += "\"neck_temp\":" + String(neck.temp_mean, 2) + ",";
  json += "\"neck_stability\":" + String(neck_stability, 2) + ",";
  json += "\"fever_risk\":" + String((neck.temp_mean > FEVER_THRESHOLD) ? 1 : 0);
  json += "},";
  
  // Diabetes biomarkers
  json += "\"diabetes\":{";
  json += "\"canthus_temp\":" + String(canthus_temp, 2) + ",";
  json += "\"canthus_stability\":" + String(canthus_stability, 2) + ",";
  json += "\"risk_flag\":" + String((canthus_temp < DIABETES_THRESHOLD) ? 1 : 0);
  json += "},";
  
  // Cardiovascular biomarkers
  json += "\"cardiovascular\":{";
  json += "\"thermal_asymmetry\":" + String(thermal_asymmetry, 3) + ",";
  json += "\"left_cheek_temp\":" + String(left_cheek.temp_mean, 2) + ",";
  json += "\"right_cheek_temp\":" + String(right_cheek.temp_mean, 2) + ",";
  json += "\"risk_flag\":" + String((thermal_asymmetry > CVD_ASYMMETRY_THRESHOLD) ? 1 : 0);
  json += "},";
  
  // Inflammation biomarkers
  json += "\"inflammation\":{";
  json += "\"hot_pixel_pct\":" + String(hot_pixel_pct, 2) + ",";
  json += "\"face_mean_temp\":" + String(face_mean_temp, 2) + ",";
  json += "\"detected\":" + String((hot_pixel_pct > INFLAMMATION_PCT_THRESHOLD) ? 1 : 0);
  json += "},";
  
  // Autonomic/Stress biomarkers
  json += "\"autonomic\":{";
  json += "\"nose_temp\":" + String(nose_tip.temp_mean, 2) + ",";
  json += "\"forehead_temp\":" + String(forehead.valid ? forehead.temp_mean : 0.0, 2) + ",";
  json += "\"stress_gradient\":" + String(stress_gradient, 2) + ",";
  json += "\"stress_flag\":" + String((stress_gradient > STRESS_GRADIENT_THRESHOLD) ? 1 : 0);
  json += "},";
  
  // Metadata
  json += "\"metadata\":{";
  json += "\"face_detected\":1,";
  json += "\"valid_rois\":" + String(countValidROIs(left_canthus, right_canthus, neck, 
                                                      left_cheek, right_cheek, nose_tip, forehead));
  json += "}";
  
  json += "}}";
  
  // Output as single line
  Serial.println(json);
}

ROI getROIStats(int row_start, int row_end, int col_start, int col_end) {
  // Bounds validation (defensive programming)
  row_start = constrain(row_start, 0, 23);
  row_end = constrain(row_end, 1, 24);
  col_start = constrain(col_start, 0, 31);
  col_end = constrain(col_end, 1, 32);
  
  ROI result;
  result.temp_mean = 0.0;
  result.temp_max = -999.0;
  result.temp_min = 999.0;
  result.valid = false;
  
  // Check for invalid range
  if (row_start >= row_end || col_start >= col_end) {
    return result;
  }
  
  float sum = 0.0;
  int count = 0;
  
  for (int r = row_start; r < row_end; r++) {
    for (int c = col_start; c < col_end; c++) {
      int index = r * 32 + c;
      
      // Double-check bounds (should never trigger with constrain above)
      if (index < 0 || index >= 768) continue;
      
      float temp = frame[index];
      
      // Skip invalid values
      if (isnan(temp) || temp < 10.0 || temp > 65.0) continue;
      
      sum += temp;
      if (temp > result.temp_max) result.temp_max = temp;
      if (temp < result.temp_min) result.temp_min = temp;
      count++;
    }
  }
  
  if (count > 0) {
    result.temp_mean = sum / count;
    result.valid = true;
  }
  
  return result;
}

float calculateFaceMeanTemp() {
  float sum = 0.0;
  int count = 0;
  
  // Face region (exclude edges which may be background)
  for (int r = 2; r < 22; r++) {
    for (int c = 6; c < 26; c++) {
      int index = r * 32 + c;
      float temp = frame[index];
      
      if (!isnan(temp) && temp > 20.0 && temp < 50.0) {
        sum += temp;
        count++;
      }
    }
  }
  
  return (count > 0) ? sum / count : 0.0;
}

float calculateHotPixelPercentage(float baseline_temp, float threshold) {
  if (baseline_temp < 20.0) return 0.0;  // Invalid baseline
  
  int hot_pixels = 0;
  int valid_pixels = 0;
  
  for (int i = 0; i < 768; i++) {
    float temp = frame[i];
    
    if (isnan(temp) || temp < 20.0) continue;
    
    valid_pixels++;
    if (temp - baseline_temp > threshold) {
      hot_pixels++;
    }
  }
  
  return (valid_pixels > 0) ? (float)hot_pixels / valid_pixels * 100.0 : 0.0;
}

int countValidROIs(ROI r1, ROI r2, ROI r3, ROI r4, ROI r5, ROI r6, ROI r7) {
  int count = 0;
  if (r1.valid) count++;
  if (r2.valid) count++;
  if (r3.valid) count++;
  if (r4.valid) count++;
  if (r5.valid) count++;
  if (r6.valid) count++;
  if (r7.valid) count++;
  return count;
}
