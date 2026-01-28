# ESP32-S3 Water Leak Detection Deployment Guide

## 📦 Generated Files

All files are in the `esp32_s3_model/` directory:

| File | Size | Description |
|------|------|-------------|
| `esp32_autoencoder.h5` | 126 KB | Optimized Keras model (53% smaller than original) |
| `esp32_autoencoder_float32.tflite` | 54 KB | TFLite float32 model |
| **`esp32_autoencoder_int8.tflite`** | **58 KB** | **Quantized int8 model (RECOMMENDED)** |
| `water_leak_model.h` | ~200 KB | C array header file for ESP32 |
| `water_leak_detector.ino` | - | Arduino sketch with inference code |
| `model_metadata.json` | <1 KB | Model configuration and performance |
| `tflite_to_c_array.py` | - | Converter script |

## 🎯 Model Performance

- **Parameters**: 4,437 (vs 16,549 original = 73% reduction)
- **Model Size**: 58 KB int8 (vs 271 KB original = 79% smaller)
- **Accuracy**: 91.4%
- **Architecture**: LSTM 16→8→8→16 (reduced from 32→16→16→32)

## 🔧 Hardware Requirements

### Minimum
- **ESP32-S3** with 512KB SRAM
- **Flow Sensor** (pulse output, e.g., YF-S201)
- **Power Supply** (5V, 1A minimum)

### Recommended
- **ESP32-S3** with 8MB PSRAM (for smoother operation)
- **Turbidity Sensor** (analog output, optional but improves accuracy)
- **RTC Module** (DS3231 for accurate time tracking)
- **SD Card Module** (for data logging)

## 📋 Arduino IDE Setup

### 1. Install ESP32 Board Support

```
File → Preferences → Additional Board Manager URLs:
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```

Then: `Tools → Board → Boards Manager → Search "ESP32" → Install`

### 2. Install TensorFlow Lite Library

```
Sketch → Include Library → Manage Libraries
Search: "TensorFlowLite_ESP32"
Install: "TensorFlowLite_ESP32" by Eloquent TinyML
```

### 3. Copy Files to Arduino Project

```
1. Create new sketch folder: water_leak_detector/
2. Copy files:
   - water_leak_detector.ino
   - water_leak_model.h
3. Open water_leak_detector.ino in Arduino IDE
```

### 4. Configure Board Settings

```
Tools → Board → ESP32 Arduino → ESP32S3 Dev Module
Tools → PSRAM → "OPI PSRAM"
Tools → Flash Size → "8MB"
Tools → Partition Scheme → "Huge APP (3MB No OTA/1MB SPIFFS)"
Tools → Upload Speed → "921600"
```

### 5. Upload Sketch

```
1. Connect ESP32-S3 via USB
2. Select correct COM port: Tools → Port
3. Click Upload button
4. Monitor output: Tools → Serial Monitor (115200 baud)
```

## 🔌 Hardware Connections

### Flow Sensor (YF-S201)
```
Flow Sensor VCC  →  ESP32 5V
Flow Sensor GND  →  ESP32 GND
Flow Sensor SIG  →  ESP32 GPIO4 (configurable in code)
```

### Turbidity Sensor (Optional)
```
Turbidity VCC  →  ESP32 3.3V
Turbidity GND  →  ESP32 GND
Turbidity OUT  →  ESP32 GPIO34 (ADC pin)
```

### LED Indicator (Optional)
```
LED Anode (+)  →  ESP32 GPIO2 (through 220Ω resistor)
LED Cathode (-) →  ESP32 GND
```

## ⚙️ Configuration

Edit these values in `water_leak_detector.ino`:

```cpp
// Pin assignments
const int FLOW_SENSOR_PIN = 4;      // Flow sensor pulse pin
const int TURBIDITY_PIN = 34;       // Turbidity sensor analog pin
const int LED_PIN = 2;               // Alert LED pin

// Flow sensor calibration
const float FLOW_CALIBRATION = 7.5;  // Pulses per liter (adjust for your sensor)

// Model settings
#define THRESHOLD 346882878.9        // Anomaly detection threshold
#define WINDOW_SIZE 10               // 10-minute sliding window
#define N_FEATURES 5                 // Number of input features

// Optional features
const bool USE_TURBIDITY = true;     // Set false if no turbidity sensor
```

## 📊 How It Works

### 1. Data Collection (Every Minute)
- Reads flow sensor pulses → calculates flow rate (L/min)
- Reads turbidity sensor (if available)
- Tracks cumulative flow duration
- Gets current time (hour, weekend flag)

### 2. Sliding Window Buffer
- Maintains 10-minute window of sensor readings
- Updates buffer every minute (circular buffer)
- Waits for buffer to fill before starting inference

### 3. Inference Process
```
Input: [10 timesteps × 5 features]
  ↓
LSTM Autoencoder (int8 quantized)
  ↓
Output: Reconstructed [10 timesteps × 5 features]
  ↓
Calculate MSE (Mean Squared Error)
  ↓
Compare with threshold → Detect anomaly
```

### 4. Leak Detection
- **Normal**: MSE < threshold → No alert
- **Leak**: MSE > threshold → Trigger alert (LED, serial, notification)

## 🧪 Testing

### Test 1: Normal Operation
```
Expected output:
[08:15] Flow: 2.34 L/min | Turbidity: 0.65 NTU | Duration: 1200 s
  Inference: 45.23 ms | MSE: 12345678.90 | Status: ✓ Normal
```

### Test 2: Simulated Leak
```
1. Turn on faucet continuously for 10+ minutes
2. Expected output:
[08:25] Flow: 5.67 L/min | Turbidity: 1.23 NTU | Duration: 1800 s
  Inference: 45.67 ms | MSE: 456789012.34 | Status: ⚠️ LEAK DETECTED!
  >>> ALERT: Potential water leak detected! <<<
```

## 📈 Performance Metrics

- **Inference Time**: ~45-50 ms per window
- **Memory Usage**: ~80 KB tensor arena + 56 KB model = 136 KB total
- **Power Consumption**: ~150-200 mA @ 5V (active inference)
- **Sampling Rate**: 1 sample/minute (configurable)

## 🐛 Troubleshooting

### Model fails to load
```
Error: "Model schema version X not supported"
Solution: Update TensorFlowLite_ESP32 library to latest version
```

### AllocateTensors() failed
```
Error: "AllocateTensors() failed"
Solution: Increase kTensorArenaSize in code (currently 80KB)
```

### Inference too slow
```
Problem: Inference takes >100ms
Solution: 
  1. Enable PSRAM in board settings
  2. Reduce WINDOW_SIZE to 5 (retrain model)
  3. Use ESP32-S3 with higher clock speed
```

### False positives
```
Problem: Too many leak alerts
Solution: Increase THRESHOLD value (retrain model with different percentile)
```

### Flow sensor not working
```
Problem: Flow rate always 0
Solution:
  1. Check wiring (VCC, GND, SIG)
  2. Verify FLOW_CALIBRATION constant
  3. Test sensor with multimeter (should pulse when water flows)
```

## 🔄 Retraining the Model

To retrain with different parameters:

```bash
# 1. Edit esp32_optimize.py
# Change LSTM layer sizes, window size, or features

# 2. Run training in Docker
docker-compose up --build

# 3. Convert new model to C array
python esp32_s3_model/tflite_to_c_array.py \
  esp32_s3_model/esp32_autoencoder_int8.tflite \
  esp32_s3_model/water_leak_model.h

# 4. Update threshold in Arduino code
# Copy threshold from model_metadata.json
```

## 📝 Next Steps

1. **Add WiFi/MQTT**: Send alerts to cloud/mobile app
2. **Data Logging**: Store readings to SD card
3. **Web Dashboard**: Real-time monitoring via web interface
4. **Multiple Sensors**: Support multiple flow sensors
5. **OTA Updates**: Over-the-air firmware updates

## 📚 References

- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-S3 Datasheet](https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf)
- [YF-S201 Flow Sensor](https://www.adafruit.com/product/828)

## 📄 License

This is a proof-of-concept for educational purposes.

---

**Generated**: January 2026  
**Model Version**: 1.0  
**ESP32-S3 Optimized**: ✓
