# Live Water Detection Simulation Dashboard

Real-time web-based dashboard for monitoring water flow and detecting leaks using AI models with live data streaming.

## Features

- **Real-Time Data Streaming**: Live sensor data at 1 sample/second via WebSocket
- **Moving Graphs**: Scrolling charts showing last 60 seconds of data
- **Dual Model Detection**: Autoencoder + Isolation Forest predictions
- **Live Alerts**: Instant notifications when leaks are detected
- **Interactive Controls**: Start, pause, and stop simulation
- **Premium UI**: Dark mode with glassmorphism effects and smooth animations

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Navigate to live_simulation folder
cd live_simulation

# Build and start the container
docker-compose up --build

# Or using Docker directly
docker build -t water-detection-live .
docker run -p 5000:5000 -v ${PWD}/..:/app/parent water-detection-live
```

### Option 2: Local Python Environment

#### 1. Install Dependencies

```bash
# Navigate to live_simulation folder
cd live_simulation

# Install required packages
pip install -r requirements_live.txt
```

#### 2. Train Models (if not already done)

The live simulation requires pre-trained models. If you haven't trained them yet:

```bash
# Go back to project root
cd ..

# Run the main training pipeline
python main.py
```

This will generate:
- `autoencoder_model.h5` - LSTM Autoencoder model
- `isolation_forest_model.pkl` - Isolation Forest model
- `scaler.pkl` - Feature scaler

#### 3. Start the Dashboard

```bash
# Navigate to live_simulation folder
cd live_simulation

# Start the WebSocket server
python websocket_server.py
```

#### 4. Open in Browser

Navigate to: **http://localhost:5000**

## Usage

1. **Start Simulation**: Click the "Start Simulation" button
2. **Monitor**: Watch real-time graphs update with live sensor data
3. **Detect Leaks**: System automatically detects anomalies and shows alerts
4. **Control**: Use Pause/Resume/Stop buttons to control the simulation

## Architecture

```
live_simulation/
├── config.json                 # Configuration settings
├── live_data_streamer.py      # Real-time data generation
├── live_inference.py          # Model inference engine
├── websocket_server.py        # WebSocket server
├── requirements_live.txt      # Dependencies
├── static/
│   ├── index.html            # Dashboard UI
│   ├── style.css             # Styling
│   └── app.js                # Frontend logic
└── README.md                 # This file
```

## Configuration

Edit `config.json` to customize:

- `sampling_rate_hz`: Data generation frequency (default: 1 Hz)
- `window_size`: Autoencoder window size (default: 10)
- `websocket_port`: Server port (default: 5000)
- `leak_injection_probability`: Chance of leak per sample (default: 0.02)
- `graph_window_seconds`: Chart display window (default: 60)

## WebSocket Events

### Client → Server

- `start_simulation`: Start data streaming
- `stop_simulation`: Stop data streaming
- `pause_simulation`: Pause streaming
- `resume_simulation`: Resume streaming

### Server → Client

- `sensor_data`: Real-time sensor readings and predictions
- `leak_alert`: Leak detection notification
- `simulation_status`: Simulation state changes
- `connection_response`: Initial connection confirmation

## Data Format

### Sensor Data Packet

```json
{
  "timestamp": "2026-02-06T14:42:30.123456",
  "sensor_data": {
    "flow_rate": 3.45,
    "turbidity": 1.23,
    "hour": 14,
    "is_weekend": 0
  },
  "predictions": {
    "autoencoder": {
      "prediction": 0,
      "reconstruction_error": 0.000234
    },
    "isolation_forest": {
      "prediction": 0,
      "anomaly_score": -0.123
    },
    "ensemble": 0,
    "confidence": 0.0
  },
  "ground_truth": 0,
  "leak_active": false
}
```

## Dashboard Components

### Live Metrics
- **Flow Rate**: Current water flow in L/min
- **Turbidity**: Water quality in NTU
- **Autoencoder**: Prediction and reconstruction error
- **Isolation Forest**: Prediction and anomaly score

### Charts
1. **Flow Rate Chart**: Real-time scrolling graph
2. **Turbidity Chart**: Real-time scrolling graph
3. **Anomaly Timeline**: Scatter plot showing normal vs leak points

### Controls
- **Start**: Begin simulation
- **Pause/Resume**: Pause/resume data streaming
- **Stop**: Stop and reset simulation

## Troubleshooting

### Models Not Found

If you see warnings about missing models:

```bash
# Train models first
cd ..
python main.py
```

### Port Already in Use

Change the port in `config.json`:

```json
{
  "websocket_port": 5001
}
```

### WebSocket Connection Failed

1. Check firewall settings
2. Ensure server is running
3. Try accessing via `http://127.0.0.1:5000` instead

## Performance

- **CPU Usage**: ~10-20% (single core)
- **Memory**: ~200-300 MB
- **Network**: ~1-2 KB/s per client
- **Latency**: <50ms for data updates

## Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Edge 90+
- ✅ Safari 14+

## Technical Stack

**Backend:**
- Flask 2.3.3
- Flask-SocketIO 5.3.4
- Python-SocketIO 5.9.0
- Eventlet 0.33.3

**Frontend:**
- Chart.js 4.4.0
- Socket.IO Client 4.5.4
- Vanilla JavaScript (ES6+)

**Models:**
- TensorFlow/Keras (Autoencoder)
- scikit-learn (Isolation Forest)

## License

Educational/demonstration project for intelligent building management systems.

## Support

For issues or questions, refer to the main project README.md in the parent directory.
