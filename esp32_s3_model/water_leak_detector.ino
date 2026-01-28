/*
 * ESP32-S3 Water Leak Detection - TensorFlow Lite Micro Inference
 * ================================================================
 * 
 * This Arduino sketch runs the water leak detection model on ESP32-S3
 * using TensorFlow Lite for Microcontrollers.
 * 
 * Hardware Requirements:
 * - ESP32-S3 (512KB SRAM, 8MB PSRAM recommended)
 * - Flow sensor (pulse output)
 * - Optional: Turbidity sensor (analog output)
 * 
 * Model Details:
 * - Input: 10-minute window (10 timesteps × 5 features)
 * - Features: flow_normalized, turbidity, flow_duration, hour, is_weekend
 * - Output: Reconstructed window (anomaly if reconstruction error > threshold)
 * - Model size: 56KB (int8 quantized)
 * - Parameters: 4,437
 */

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include the model
#include "water_leak_model.h"

// Model configuration
#define WINDOW_SIZE 10
#define N_FEATURES 5
#define THRESHOLD 346882878.9  // From model_metadata.json

// TensorFlow Lite globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena size (adjust if needed)
  constexpr int kTensorArenaSize = 80 * 1024;  // 80KB
  uint8_t tensor_arena[kTensorArenaSize];
}

// Circular buffer for sliding window
float window_buffer[WINDOW_SIZE][N_FEATURES];
int buffer_index = 0;
bool buffer_full = false;

// Flow sensor variables
volatile unsigned long flow_pulse_count = 0;
unsigned long last_flow_time = 0;
const int FLOW_SENSOR_PIN = 4;  // GPIO4
const float FLOW_CALIBRATION = 7.5;  // Pulses per liter (adjust for your sensor)

// Turbidity sensor (optional)
const int TURBIDITY_PIN = 34;  // ADC pin
const bool USE_TURBIDITY = true;

void IRAM_ATTR flowPulseCounter() {
  flow_pulse_count++;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  Serial.println("\n==============================================");
  Serial.println("ESP32-S3 Water Leak Detection System");
  Serial.println("==============================================");
  
  // Initialize flow sensor
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseCounter, FALLING);
  
  // Initialize turbidity sensor
  if (USE_TURBIDITY) {
    pinMode(TURBIDITY_PIN, INPUT);
  }
  
  // Set up TensorFlow Lite Micro
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  // Load model
  model = tflite::GetModel(water_leak_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema version %d not supported (expected %d)\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  Serial.println("✓ Model loaded successfully");
  
  // Build interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("✗ AllocateTensors() failed");
    return;
  }
  Serial.println("✓ Tensors allocated");
  
  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // Print model info
  Serial.printf("\nModel Information:\n");
  Serial.printf("  Input shape: [%d, %d, %d]\n", 
                input->dims->data[0], input->dims->data[1], input->dims->data[2]);
  Serial.printf("  Output shape: [%d, %d, %d]\n",
                output->dims->data[0], output->dims->data[1], output->dims->data[2]);
  Serial.printf("  Model size: %d bytes\n", water_leak_model_len);
  Serial.printf("  Threshold: %.2f\n", THRESHOLD);
  
  Serial.println("\n==============================================");
  Serial.println("System ready. Monitoring water flow...");
  Serial.println("==============================================\n");
}

void loop() {
  unsigned long current_time = millis();
  
  // Read sensors every minute
  if (current_time - last_flow_time >= 60000) {  // 60 seconds
    
    // Calculate flow rate (L/min)
    float flow_rate = (flow_pulse_count / FLOW_CALIBRATION);
    flow_pulse_count = 0;
    last_flow_time = current_time;
    
    // Normalize flow rate (0-15 L/min range)
    float flow_normalized = flow_rate / 15.0;
    if (flow_normalized > 1.0) flow_normalized = 1.0;
    
    // Read turbidity (0-3 NTU range, normalized)
    float turbidity = 0.5;  // Default if no sensor
    if (USE_TURBIDITY) {
      int turbidity_raw = analogRead(TURBIDITY_PIN);
      turbidity = (turbidity_raw / 4095.0) * 3.0;  // Scale to 0-3 NTU
    }
    
    // Calculate flow duration (cumulative seconds of flow today)
    static unsigned long daily_flow_seconds = 0;
    if (flow_rate > 0.1) {
      daily_flow_seconds += 60;  // Add 1 minute
    }
    
    // Reset at midnight (simplified - use RTC for accurate time)
    struct tm timeinfo;
    if (getLocalTime(&timeinfo)) {
      if (timeinfo.tm_hour == 0 && timeinfo.tm_min == 0) {
        daily_flow_seconds = 0;
      }
    }
    
    // Get current hour and weekend flag
    int hour = timeinfo.tm_hour;
    int is_weekend = (timeinfo.tm_wday == 0 || timeinfo.tm_wday == 6) ? 1 : 0;
    
    // Add to circular buffer
    window_buffer[buffer_index][0] = flow_normalized;
    window_buffer[buffer_index][1] = turbidity;
    window_buffer[buffer_index][2] = daily_flow_seconds;
    window_buffer[buffer_index][3] = hour;
    window_buffer[buffer_index][4] = is_weekend;
    
    buffer_index = (buffer_index + 1) % WINDOW_SIZE;
    if (buffer_index == 0) buffer_full = true;
    
    // Print sensor readings
    Serial.printf("[%02d:%02d] Flow: %.2f L/min | Turbidity: %.2f NTU | Duration: %lu s\n",
                  hour, timeinfo.tm_min, flow_rate, turbidity, daily_flow_seconds);
    
    // Run inference if buffer is full
    if (buffer_full) {
      runInference();
    } else {
      Serial.printf("  Filling buffer... (%d/%d)\n", 
                    (buffer_index == 0 ? WINDOW_SIZE : buffer_index), WINDOW_SIZE);
    }
  }
}

void runInference() {
  // Copy window buffer to input tensor (handle circular buffer)
  for (int t = 0; t < WINDOW_SIZE; t++) {
    int idx = (buffer_index + t) % WINDOW_SIZE;
    for (int f = 0; f < N_FEATURES; f++) {
      // Quantize float to int8 (model uses int8 input)
      float value = window_buffer[idx][f];
      int8_t quantized = (int8_t)((value - input->params.zero_point) * input->params.scale);
      input->data.int8[t * N_FEATURES + f] = quantized;
    }
  }
  
  // Run inference
  unsigned long inference_start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long inference_time = micros() - inference_start;
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("✗ Inference failed!");
    return;
  }
  
  // Calculate reconstruction error (MSE)
  float total_error = 0.0;
  for (int t = 0; t < WINDOW_SIZE; t++) {
    for (int f = 0; f < N_FEATURES; f++) {
      int idx = (buffer_index + t) % WINDOW_SIZE;
      
      // Dequantize output
      int8_t output_quantized = output->data.int8[t * N_FEATURES + f];
      float output_value = (output_quantized - output->params.zero_point) / output->params.scale;
      
      float input_value = window_buffer[idx][f];
      float error = (input_value - output_value) * (input_value - output_value);
      total_error += error;
    }
  }
  
  float mse = total_error / (WINDOW_SIZE * N_FEATURES);
  bool is_anomaly = (mse > THRESHOLD);
  
  // Print results
  Serial.printf("  Inference: %.2f ms | MSE: %.2f | Status: %s\n",
                inference_time / 1000.0, mse, 
                is_anomaly ? "⚠️ LEAK DETECTED!" : "✓ Normal");
  
  if (is_anomaly) {
    // Trigger alert (LED, buzzer, notification, etc.)
    Serial.println("  >>> ALERT: Potential water leak detected! <<<");
    // TODO: Add your alert mechanism here
  }
}
