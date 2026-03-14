# Apartment Simulator Backend - Documentation Index

**Project:** Water Leak Detection System - Apartment Building Simulator  
**Module:** `apartment_simulator/backend`  
**Version:** 1.0  
**Last Updated:** March 2026

---

## Overview

The apartment simulator backend provides real-time water flow simulation for a 50-unit apartment building with integrated hybrid anomaly detection. This documentation folder contains detailed reports on each backend module.

### Quick Navigation

| Document | Module | Purpose |
|----------|--------|---------|
| [01_init.md](01_init.md) | `__init__.py` | Package initialization |
| [02_live_simulator.md](02_live_simulator.md) | `live_simulator.py` | Flow simulation (individual & building level) |
| [03_model.md](03_model.md) | `model.py` | Hybrid anomaly detection (CUSUM + IF) |
| [04_server.md](04_server.md) | `server.py` | FastAPI + Socket.IO web server |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND DASHBOARD                         │
│              (browser: index.html + app.js)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↕ WebSocket (Socket.IO)
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              FASTAPI + SOCKET.IO SERVER                       │
│                   (server.py)                                 │
│  ├─ GET / → Serves index.html                              │
│  ├─ Event Handlers (start, pause, stop, inject_leak, etc)  │
│  └─ Background simulation_loop()                           │
└──────────────┬──────────────────────────┬────────────────────┘
               │                          │
               ↓                          ↓
    ┌─────────────────────┐    ┌──────────────────────┐
    │  FLOW GENERATOR     │    │ ANOMALY DETECTOR     │
    │ (live_simulator.py) │    │    (model.py)        │
    │                     │    │                      │
    │ • 50 Apartments     │    │ • CUSUM (L2)         │
    │ • Realistic Events  │    │ • IF Model (L3)      │
    │ • Daily Volumes     │    │ • Fusion Layer       │
    │ • Leak Injection    │    │ • Persistence Guard  │
    └─────────────────────┘    └──────────────────────┘
               │                          │
               └──────────┬───────────────┘
                          ↓
                   Result JSON
                   (emitted to frontend)
```

---

## Module Descriptions

### 1. `__init__.py` — Package Initialization

**File:** [01_init.md](01_init.md)

**Purpose:** Marks the backend directory as a Python package.

**Key Topics:**
- Package marker
- Import mechanisms
- Enhancement possibilities (convenience imports)
- Best practices

**Complexity:** ⭐ (Minimal)

**Key Takeaway:** Currently a minimal initialization file; can be extended to provide module-level convenience imports.

---

### 2. `live_simulator.py` — Water Flow Simulation

**File:** [02_live_simulator.md](02_live_simulator.md)

**Purpose:** Implements realistic minute-by-minute water flow simulation based on India-specific usage patterns.

**Classes:**
- `LiveWaterFlowGenerator` — Single apartment simulation
- `LiveApartmentBuildingDataGenerator` — 50-apartment aggregation

**Key Methods:**
- `next()` — Generate next minute's flow
- `inject_leak()` — Introduce synthetic leak
- `_generate_new_day()` — Create daily schedule
- `_render_day()` — Convert events to time-series

**Key Topics:**
- Appliance event generation (Poisson + Lognormal distributions)
- Event scheduling with realistic timing
- Flow shape curves (trapezoid, pulsed, step)
- Leak injection mechanisms
- Independent seeding for 50 apartments
- Water volume constraints

**Complexity:** ⭐⭐⭐ (Moderate)

**Key Takeaway:** Generates highly realistic water flow by: (1) Creating appliance events based on India-specific priors, (2) Rendering events with realistic flow profiles, (3) Ensuring daily volumes stay within realistic bounds.

---

### 3. `model.py` — Anomaly Detection

**File:** [03_model.md](03_model.md)

**Purpose:** Hybrid anomaly detector combining statistical change detection (CUSUM) with machine learning (Isolation Forest).

**Class:**
- `HybridWaterAnomalyDetector` — Main detector class

**Key Methods:**
- `update()` — Process 20-minute window
- `_run_cusum()` — Statistical change detection
- `_extract_features()` — Feature engineering (7 features)

**Detection Levels:**
- **Level 2 (CUSUM):** Sustained small leaks
- **Level 3 (IF):** Point anomalies
- **Fusion Layer:** Weighted combination with bypasses
- **Persistence Filter:** False-positive guard

**Key Topics:**
- CUSUM algorithm (partial resets, accumulation)
- Feature extraction (mnf, inter_mean, flow_trend, baseline_elev, etc.)
- Isolation Forest integration
- Weighted fusion (35% CUSUM + 65% IF)
- Persistence window filtering
- Calibration parameters (JSON-based)

**Complexity:** ⭐⭐⭐⭐ (High)

**Key Takeaway:** Two complementary methods (CUSUM for sustained leaks, IF for outliers) fused with independent bypass rules; persistence filter prevents false positives.

---

### 4. `server.py` — Web Server & Real-Time Streaming

**File:** [04_server.md](04_server.md)

**Purpose:** FastAPI web server with WebSocket (Socket.IO) for real-time, bidirectional communication.

**Components:**
- FastAPI application
- Socket.IO event server
- Model & generator initialization
- Main simulation loop
- Event handlers

**Key Event Handlers:**
- `start_simulation` — Resume/start simulation
- `pause_simulation` — Pause (preserve state)
- `stop_simulation` — Stop and reset
- `set_speed` — 1-10x playback speed
- `inject_leak` — Inject synthetic leak (instant or ramp mode)
- `stop_leak` — Stop active leak

**Key Topics:**
- FastAPI routing
- Socket.IO event patterns
- Model loading with fallbacks
- Simulation loop algorithm
- Leak injection modes (instant vs. ramp)
- Real-time data streaming
- Speed control
- Docker integration

**Complexity:** ⭐⭐⭐ (Moderate-High)

**Key Takeaway:** Orchestrates simulation and detection in real-time; streams JSON results via WebSocket; accepts user controls from frontend.

---

## Data Flow

### Per-Minute Cycle (at 1x speed)

```
1. Generator creates flow for 50 apartments
   flow = LiveApartmentBuildingDataGenerator.next()

2. Optional leak injection (server-side)
   if leak_active:
       flow += leak_intensity * progress

3. Flow added to window buffer
   window_buffer.append(flow)

4. Every 20th minute: detector.update()
   result = detector.update(window_buffer)

5. Build response with all telemetry
   response = {flow, anomaly, score, leak_info, timing}

6. Emit to frontend via Socket.IO
   sio.emit('data_update', response)

7. Sleep for 1.0 / simulation_speed seconds
   (determines actual playback speed)
```

### Feature to Frontend

```
Every minute (at any speed):
  ├─ Current flow (L/min)
  ├─ Anomaly status (boolean)
  ├─ Detection scores (CUSUM, IF, fused)
  ├─ Debugging info (flow_trend, baseline_elev)
  ├─ Simulated time (HH:MM format)
  ├─ Leak status (active, intensity, remaining)
  └─ Simulation progress

Frontend uses this to:
  ├─ Update flow chart
  ├─ Highlight anomalies
  ├─ Display score gauges
  ├─ Show leak indicator
  └─ Update simulated clock
```

---

## Quick Reference: Key Classes & Methods

### LiveWaterFlowGenerator

```python
gen = LiveWaterFlowGenerator("priors.json", seed=42)
flow = gen.next()                    # Get next minute
gen.inject_leak(180, 0.4)            # Inject leak
gen.clear_leak()                     # Remove leak
minute = gen.global_minute()         # Current time
```

### LiveApartmentBuildingDataGenerator

```python
bldg = LiveApartmentBuildingDataGenerator("priors.json", num_apartments=50)
flow = bldg.next()                   # Aggregated flow
bldg.inject_leak(180, 0.4)           # Building-level leak
bldg.reset()                         # Reset to start
```

### HybridWaterAnomalyDetector

```python
detector = HybridWaterAnomalyDetector(if_model, if_scaler)
result = detector.update(window)     # Process window
print(result['anomaly'])             # Final alarm
print(result['final_score'])         # Fused [0,1] score
detector.reset()                     # Reset state
```

---

## Configuration Files

### `calibration_building.json`

```json
{
    "window_minutes": 20,
    "cusum_k": 3.0,
    "cusum_h": 8.0,
    "if_threshold": -0.02,
    "if_score_scale": 0.08,
    "baseline_inter_mean_median": 2.391,
    "baseline_inter_mean_std": 1.394,
    "w_cusum": 0.35,
    "w_if": 0.65,
    "decision_threshold": 0.40,
    "persistence_windows": 4
}
```

**Purpose:** All detector parameters loaded at server startup.

### `all_appliances.json`

```json
{
    "appliances": [
        {
            "appliance": "shower",
            "activation": {"events_per_day": {"lambda": 1.2}},
            "timing": {"start_hour": {"p": [0, 0.01, ...]}},
            "duration": {"type": "lognormal", "scale": 300, "shape": 0.5},
            "flow": {"mean_flow": {"scale": 15, "shape": 0.3}},
            ...
        },
        ...
    ]
}
```

**Purpose:** Appliance-level configurations for realistic event generation.

---

## Usage Examples

### Example 1: Simple Simulation

```python
from apartment_simulator.backend.live_simulator import LiveApartmentBuildingDataGenerator

bldg = LiveApartmentBuildingDataGenerator("priors.json", seed=42)

# Simulate 1 day (1440 minutes)
for minute in range(1440):
    flow = bldg.next()
    print(f"{minute}: {flow:.1f} L/min")
```

### Example 2: With Leak Detection

```python
from apartment_simulator.backend.live_simulator import LiveApartmentBuildingDataGenerator
from apartment_simulator.backend.model import HybridWaterAnomalyDetector
import pickle, json

# Load models
with open("isolation_forest_building.pkl", "rb") as f:
    if_model = pickle.load(f)
with open("scaler_building.pkl", "rb") as f:
    if_scaler = pickle.load(f)
with open("calibration_building.json") as f:
    cal = json.load(f)

# Initialize
bldg = LiveApartmentBuildingDataGenerator("priors.json")
detector = HybridWaterAnomalyDetector(if_model, if_scaler, **cal)

# Simulate with detection
window = []
for minute in range(2880):  # 2 days
    flow = bldg.next()
    window.append(flow)
    
    if len(window) == 20:
        result = detector.update(window)
        if result["anomaly"]:
            print(f"🚨 ALARM at minute {minute}: score={result['final_score']:.2f}")
        window = []
```

### Example 3: Real-Time Server (see server.py)

```python
# Start server
python -m apartment_simulator.backend.server

# Access at http://localhost:5000
# Frontend connects to WebSocket for real-time updates
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Memory (static)** | ~50 MB (models + config) |
| **Memory (per-minute)** | <1 KB |
| **Computation per minute** | ~30 µs |
| **Computation per window (20 min)** | ~1.2 ms |
| **Network bandwidth (1x speed)** | ~500 bytes/sec (4 Mbps) |
| **Network bandwidth (10x speed)** | ~5 KB/sec (40 Mbps) |
| **Typical detection latency** | 80-120 min (4 window persistence) |

---

## Troubleshooting

### Leak not detected?
- ✓ Check CUSUM k/h parameters (calibration)
- ✓ Verify IF threshold (should be negative, e.g., -0.02)
- ✓ Check persistence_windows (if > 4, slower response)
- ✓ Ensure leak intensity is sufficient (> 0.3 L/min recommended for small leaks)

### Too many false positives?
- ✓ Increase persistence_windows (e.g., 5 or 6)
- ✓ Increase decision_threshold (e.g., 0.50)
- ✓ Decrease IF weight (w_if), increase CUSUM weight (w_cusum)
- ✓ Verify calibration reflects your building profile

### Simulation running slow?
- ✓ Increase simulation_speed (set_speed event)
- ✓ Check CPU usage (may be saturated)
- ✓ Reduce logging/output (overhead)

### Server won't start?
- ✓ Check port 5000 is available: `netstat -an | grep 5000`
- ✓ Verify all artifact files present: `ls artifacts/`
- ✓ Check Python dependencies: `pip install -r requirements.txt`
- ✓ Check file paths (priors.json, calibration JSON, models)

---

## File Structure

```
apartment_simulator/
├── backend/
│   ├── __init__.py                    ← Package init
│   ├── live_simulator.py              ← Flow generation
│   ├── model.py                       ← Detection
│   ├── server.py                      ← Web server
│   └── __pycache__/
├── artifacts/
│   ├── all_appliances.json
│   ├── calibration_building.json
│   ├── isolation_forest_building.pkl
│   ├── scaler_building.pkl
│   └── ...
├── frontend/
│   ├── index.html
│   └── static/
│       ├── app.js
│       └── style.css
└── preprocessing/
    └── ...
```

---

## Key Concepts

### CUSUM (Cumulative Sum Control Chart)
- Detects sustained deviations from baseline
- Accumulates evidence over time (partial resets)
- Good for small, persistent leaks

### Isolation Forest
- Machine learning outlier detection
- Independent of baseline assumptions
- Good for point anomalies and pattern shifts

### Hybrid Approach
- CUSUM catches leaks CUSUM misses IF and vice versa
- Fusion combines confidence: 35% CUSUM + 65% IF
- Bypasses allow high-confidence signals to override fusion

### Persistence Filter
- Requires N consecutive anomaly windows before alarm
- Primary false-positive guard
- Typical: 4 windows = ~80 minutes buffering

### Inter-Appliance Flow
- Flow from background leaks and minor usage
- Excludes major appliance events (shower, washing machine)
- Used for baseline calculation and CUSUM reference

---

## Extensions & Enhancements

### Possible Future Improvements

1. **Apartment-Level Isolation:** Identify which apartment has leak
   - Add apartment-level detector
   - Compare apartment flows to building average

2. **Leak Severity Estimation:** Predict leak flow rate
   - Use history of leak progression
   - Train regression model on leak characteristics

3. **Predictive Maintenance:** Forecast pipe failures
   - Model pipe age + usage patterns
   - Predict failure probability

4. **Multi-Building System:** Coordinate across buildings
   - Share models and calibration
   - Advanced anomaly correlation

5. **Real Water Data Integration:** Validation
   - Train on real building data
   - Validate detector on known leaks

---

## References

- **Dataset:** [WEUSEDTO-Data](https://github.com/AnnaDiMauro/WEUSEDTO-Data) — Real-world India water usage priors
- **Standards:** MoHUA (Ministry of Housing) and BIS (Bureau of Indian Standards)
- **ML Libraries:** scikit-learn (Isolation Forest), TensorFlow (IF training)

---

## Support & Questions

For detailed information on each module:
- Flow simulation: See [02_live_simulator.md](02_live_simulator.md)
- Detection: See [03_model.md](03_model.md)
- Server/API: See [04_server.md](04_server.md)
- Package init: See [01_init.md](01_init.md)

For code-level questions, refer to docstrings in source files.

---

**Documentation Version:** 1.0  
**Last Updated:** March 2026  
**Project:** Water Leak Detection System

---

**End of Documentation Index**
