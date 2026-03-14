# Backend Module: `server.py`

**File Path:** `apartment_simulator/backend/server.py`  
**Purpose:** Real-time web server for live simulation streaming and anomaly detection control  
**Framework:** FastAPI + Socket.IO  
**Key Components:** Simulation loop, Event handlers, Model initialization

---

## Overview

This module implements a FastAPI web server with WebSocket (Socket.IO) for real-time, bidirectional communication between the frontend dashboard and the backend simulation/detection system. It:

- Generates live water flow data (50-apartment building simulator)
- Detects anomalies using hybrid detector
- Streams results to frontend via WebSocket
- Accepts user controls (start/pause/stop/leak injection)
- Supports variable simulation speed (1-10x)

### Architecture

```
Frontend Dashboard (Web Browser)
    ↓↑ (WebSocket / Socket.IO)
FastAPI Server
    ├─ GET / → Serves index.html
    ├─ Socket.IO Event Handlers
    │  ├─ start_simulation
    │  ├─ pause_simulation
    │  ├─ stop_simulation
    │  ├─ set_speed
    │  ├─ inject_leak
    │  └─ stop_leak
    └─ Background Loop (simulation_loop)
       ├─ Generates flow via generator
       ├─ Detects anomalies via detector
       ├─ Emits data_update events
       └─ Runs at 1-10x speed
```

---

## Imports & Dependencies

```python
import asyncio                    # Async event loop
import json                       # JSON parsing
import pickle                     # Model serialization
from collections import deque    # Roll window buffer
from pathlib import Path          # File paths

import numpy as np                # Numerical computing
import socketio                   # WebSocket server
from fastapi import FastAPI       # Web framework
from fastapi.responses import FileResponse       # Serve files
from fastapi.staticfiles import StaticFiles     # Static assets
import uvicorn                    # ASGI server

from live_simulator import LiveApartmentBuildingDataGenerator
from model import HybridWaterAnomalyDetector
```

---

## Constants & Directory Structure

```python
BASE_DIR = Path(__file__).resolve().parent.parent
# Result: /path/to/apartment_simulator

ARTIFACTS_DIR = BASE_DIR / "artifacts"
# Contains: all_appliances.json, calibration_building.json, models

FRONTEND_DIR = BASE_DIR / "frontend"
# Contains: index.html, static/

MODELS_DIR = Path("/models")
# Pre-trained models location (Docker volumes)

# FastAPI Configuration
WINDOW_MINUTES = 20  # Loaded from calibration JSON

# Running Port
PORT = 5000
```

---

## Socket.IO & FastAPI Setup

### Initialization

```python
sio = socketio.AsyncServer(
    cors_allowed_origins="*",    # Allow all CORS origins
    async_mode="asgi"            # Async mode for FastAPI
)

app = FastAPI()

socket_app = socketio.ASGIApp(sio, app)
# Wraps FastAPI app in Socket.IO ASGI middleware
```

### Static Files Mounting

```python
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR / "static"),
    name="static"
)
```

**Purpose:** Serves `app.js` and `style.css` at `/static/app.js`, `/static/style.css`, etc.

---

## Model Loading Functions

### Function: `load_pickle(path)`

```python
def load_pickle(path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
```

**Use Case:** Load Isolation Forest model and scaler from `.pkl` files.

**Example:**
```python
if_model = load_pickle("artifacts/isolation_forest_building.pkl")
if_scaler = load_pickle("artifacts/scaler_building.pkl")
```

---

### Function: `load_json(path)`

```python
def load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)
```

**Use Case:** Load configuration files (calibration, appliances).

**Example:**
```python
calibration = load_json("artifacts/calibration_building.json")
window_minutes = calibration["window_minutes"]
```

---

### Function: `find_model_path(name, alternates=None)`

```python
def find_model_path(name, alternates=None):
    """Find model in either artifacts or /models directory."""
    # Try artifacts first (local)
    artifact_path = ARTIFACTS_DIR / name
    if artifact_path.exists():
        print(f"Found {name} in artifacts/")
        return artifact_path

    # Try alternates in artifacts
    if alternates:
        for alt_name in alternates:
            alt_path = ARTIFACTS_DIR / alt_name
            if alt_path.exists():
                print(f"Found {alt_name} (alternate) in artifacts/")
                return alt_path

    # Try /models (shared)
    model_path = MODELS_DIR / name
    if model_path.exists():
        print(f"Found {name} in /models/")
        return model_path

    # Try alternates in /models
    if alternates:
        for alt_name in alternates:
            alt_path = MODELS_DIR / alt_name
            if alt_path.exists():
                print(f"Found {alt_name} (alternate) in /models/")
                return alt_path

    raise FileNotFoundError(f"{name} not found in artifacts/ or /models/")
```

**Purpose:** Fallback-aware model discovery.

**Algorithm:**
```
Try paths in order:
1. artifacts/{name}
2. artifacts/{alternates[0]}, artifacts/{alternates[1]}, ...
3. /models/{name}
4. /models/{alternates[0]}, /models/{alternates[1]}, ...

If any exists: return Path
Otherwise: raise FileNotFoundError
```

**Use Case:** Handle multiple naming conventions and deployment scenarios.

**Example:**
```python
if_model = load_pickle(
    find_model_path(
        "isolation_forest_building.pkl",
        alternates=["if_model.pkl", "isolation_forest_model.pkl"]
    )
)
```

---

## Initialization & Startup

### Model Loading Sequence

```python
print("Loading models...")

# Priority: building-scale > household-scale
try:
    if_model = load_pickle(ARTIFACTS_DIR / "isolation_forest_building.pkl")
    print("Using building-scale Isolation Forest model")
except FileNotFoundError:
    print("Building-scale model not found, falling back to household model")
    if_model = load_pickle(find_model_path(
        "if_model.pkl",
        ["isolation_forest_model.pkl"]
    ))

try:
    if_scaler = load_pickle(ARTIFACTS_DIR / "scaler_building.pkl")
    print("Using building-scale scaler")
except FileNotFoundError:
    print("Building-scale scaler not found, falling back to household scaler")
    if_scaler = load_pickle(find_model_path("if_scaler.pkl"))

# Try building calibration first
try:
    cal = load_json(ARTIFACTS_DIR / "calibration_building.json")
    print("Using building-scale calibration")
except FileNotFoundError:
    cal = load_json(ARTIFACTS_DIR / "if_calibration.json")
    print("Using household calibration")

WINDOW_MINUTES = cal["window_minutes"]
print(f"Models loaded. Window size: {WINDOW_MINUTES} minutes")
```

### Generator & Detector Initialization

```python
print("Initializing apartment building data generator...")
generator = LiveApartmentBuildingDataGenerator(
    ARTIFACTS_DIR / "all_appliances.json",
    num_apartments=50,
    seed=42,
)

print("Initializing hybrid anomaly detector...")
detector = HybridWaterAnomalyDetector(
    if_model=if_model,
    if_scaler=if_scaler,
    cusum_k=cal.get("cusum_k", 3.0),
    cusum_h=cal.get("cusum_h", 8.0),
    noise_floor=cal.get("noise_floor", 0.2),
    if_threshold=cal.get("if_threshold", -0.05),
    if_score_scale=cal.get("if_score_scale", 0.1),
    appliance_flow_thresh=cal.get("appliance_flow_thresh", 8.0),
    baseline_inter_mean_median=cal.get("baseline_inter_mean_median", 1.5),
    baseline_inter_mean_std=cal.get("baseline_inter_mean_std", 0.8),
    w2=cal.get("w_cusum", 0.35),
    w3=cal.get("w_if", 0.65),
    decision_threshold=cal.get("decision_threshold", 0.40),
    persistence_windows=cal.get("persistence_windows", 4),
)

print("Setup complete!")
```

---

## Global State

```python
# Simulation Control
simulation_running = False      # Pause/Resume flag
simulation_speed = 1.0          # Speed multiplier (1.0 to 10.0)
sim_minutes = 0                 # Current simulation time (minutes since start)

# Window Buffering
window_buffer: deque = deque(maxlen=WINDOW_MINUTES)  # Rolling 20-min window

# Latest Detection Result (cached)
last_result = {
    "anomaly": False,
    "final_score": 0.0,
    "level2": {"triggered": False, "score": 0.0},
    "level3": {
        "triggered": False,
        "score": 0.0,
        "reconstruction_error": 0.0,
        "flow_trend": 0.0,
        "baseline_elev": 0.0,
    },
}

# Leak Injection State
leak_active = False
leak_intensity = 0.0
leak_end_minute = None
leak_start_minute = None
leak_mode = "instant"           # or "ramp"
leak_ramp_minutes = 5
```

---

## FastAPI Routes

### Route: `GET /`

```python
@app.get("/")
async def serve_index():
    """Serve the main frontend HTML page."""
    return FileResponse(FRONTEND_DIR / "index.html")
```

**Purpose:** Serves the main web interface when user accesses `http://localhost:5000/`.

**Response:** Frontend HTML file with embedded app.js and style.css references.

---

## Socket.IO Event Handlers

Socket.IO handlers define bidirectional event communication between frontend and backend.

### Event Handler: `start_simulation`

```python
@sio.event
async def start_simulation(sid):
    """Start/resume the simulation."""
    global simulation_running
    simulation_running = True
    await sio.emit("simulation_state", {"state": "running"})
```

**Frontend Emits:** `socket.emit('start_simulation')`

**Backend Response:** Broadcasts `{"state": "running"}` to all connected clients

**Effect:** Unpauses the simulation loop (if paused) or starts from current state

**Use Case:** User clicks "Start" or resume button

---

### Event Handler: `pause_simulation`

```python
@sio.event
async def pause_simulation(sid):
    """Pause the simulation."""
    global simulation_running
    simulation_running = False
    await sio.emit("simulation_state", {"state": "paused"})
```

**Frontend Emits:** `socket.emit('pause_simulation')`

**Effect:** Pauses simulation (preserves all state: current_minute, generator state, detector state)

**Difference from `stop_simulation`:**
- ✓ Preserves state
- ✓ Resume continues from paused point
- ✗ Does not reset

---

### Event Handler: `stop_simulation`

```python
@sio.event
async def stop_simulation(sid):
    """Stop and reset the simulation and leak state."""
    global simulation_running, sim_minutes
    global leak_active, leak_end_minute, leak_start_minute

    simulation_running = False
    sim_minutes = 0

    leak_active = False
    leak_end_minute = None
    leak_start_minute = None

    window_buffer.clear()

    generator.reset()
    detector.reset()

    await sio.emit("simulation_state", {"state": "stopped"})
```

**Frontend Emits:** `socket.emit('stop_simulation')`

**Reset Actions:**
1. Pause simulation
2. Reset time to 0
3. Clear any active leak
4. Clear window buffer
5. Reset all generators to day 0
6. Reset detector state (CUSUM, streak counter, history)

**Effect:** Returns to initial state ready for fresh simulation

---

### Event Handler: `set_speed`

```python
@sio.event
async def set_speed(sid, data):
    """Set the simulation speed (1x to 10x)."""
    global simulation_speed

    try:
        simulation_speed = max(1.0, min(float(data), 10.0))
        await sio.emit("speed_update", {"speed": simulation_speed})
    except Exception:
        pass
```

**Frontend Emits:** `socket.emit('set_speed', 5.0)`  (for 5x speed)

**Parameter Validation:**
```python
speed = max(1.0, min(float(data), 10.0))
# Clamps to [1.0, 10.0]
```

**Effect:** Changes simulation playback speed
```
Speed = 1.0   → Real-time (1 minute = 1 second)
Speed = 2.0   → 2x fast (1 minute = 0.5 second)
Speed = 10.0  → 10x fast (1 minute = 0.1 second)
```

**Implementation:** Modifies `asyncio.sleep` duration in simulation loop

---

### Event Handler: `inject_leak`

```python
@sio.event
async def inject_leak(sid, data):
    """Inject a leak with specified intensity, duration, and mode."""
    global leak_active, leak_intensity
    global leak_end_minute, leak_start_minute
    global leak_mode, leak_ramp_minutes

    try:
        intensity = max(0.1, min(float(data.get("intensity", 0.5)), 20.0))
        duration = max(1, int(data.get("duration", 60)))
        mode = data.get("mode", "instant")
        ramp_minutes = max(1, int(data.get("ramp_minutes", 5)))

        leak_active = True
        leak_intensity = intensity
        leak_end_minute = sim_minutes + duration
        leak_start_minute = sim_minutes
        leak_mode = mode if mode in ["instant", "ramp"] else "instant"
        leak_ramp_minutes = ramp_minutes

        await sio.emit("leak_status", {
            "active": True,
            "mode": leak_mode,
            "intensity": leak_intensity
        })

    except Exception:
        pass
```

**Frontend Emits:**
```javascript
socket.emit('inject_leak', {
    "intensity": 0.5,           // L/min
    "duration": 180,            // minutes
    "mode": "instant",          // or "ramp"
    "ramp_minutes": 5           // (only for ramp mode)
})
```

**Parameter Validation:**
```
intensity: [0.1, 20.0] L/min
duration:  [1, ∞) minutes
mode:      "instant" or "ramp" (default: instant)
ramp_minutes: [1, ∞) minutes (for ramp mode)
```

**Modes:**

**1. Instant Mode:**
- Leak starts immediately at full intensity
- Lasts for specified duration
- Then stops

```
Flow
  │     ___________
  │    │           │
  │____|___________|___
  └────────────────────
  Minute: Start  End
```

**2. Ramp Mode:**
- Leak gradually increases from 0 to intensity
- Ramp duration: `ramp_minutes`
- Then stays at full intensity
- Then stops

```
Flow
  │         ___________
  │        /           \
  │______/             \___
  └─────────────────────────
  0      5  90        120+5
        Ramp    Full   End
```

**Example Frontend Usage:**
```javascript
// Small drip (3 hours at 0.3 L/min)
socket.emit('inject_leak', {
    intensity: 0.3,
    duration: 180,
    mode: 'instant'
});

// Ramping leak (reaches 1.0 L/min over 5 min, stays 30 min)
socket.emit('inject_leak', {
    intensity: 1.0,
    duration: 35,
    mode: 'ramp',
    ramp_minutes: 5
});
```

---

### Event Handler: `stop_leak`

```python
@sio.event
async def stop_leak(sid):
    """Stop any active leak in the simulation."""
    global leak_active, leak_end_minute, leak_start_minute

    leak_active = False
    leak_end_minute = None
    leak_start_minute = None

    await sio.emit("leak_status", {"active": False})
```

**Frontend Emits:** `socket.emit('stop_leak')`

**Effect:** Immediately terminates any active leak injection

---

## Main Simulation Loop

### Function: `simulation_loop()`

```python
async def simulation_loop():
    """
    Main background loop for live simulation and anomaly detection.
    Streams flow and detection results to frontend via Socket.IO.
    """

    global simulation_running, simulation_speed, sim_minutes
    global leak_active, leak_intensity
    global leak_end_minute, leak_start_minute
    global leak_mode, leak_ramp_minutes
    global last_result

    while True:

        if simulation_running:

            # ─────── GENERATE FLOW ────────────────────────
            flow = generator.next()

            # ─────── APPLY LEAK ───────────────────────────
            if leak_active:

                if leak_end_minute is not None and sim_minutes >= leak_end_minute:
                    leak_active = False
                    leak_end_minute = None
                    leak_start_minute = None

                else:

                    if leak_mode == "instant":
                        effective_intensity = leak_intensity

                    elif leak_mode == "ramp":
                        elapsed = sim_minutes - leak_start_minute
                        progress = min(1.0, elapsed / max(1, leak_ramp_minutes))
                        effective_intensity = leak_intensity * progress

                    else:
                        effective_intensity = leak_intensity

                    flow += effective_intensity
                    flow = min(flow, generator.MAX_FLOW_LPM)

            # ─────── BUFFER WINDOW ────────────────────────
            window_buffer.append(float(flow))

            if len(window_buffer) == WINDOW_MINUTES:
                last_result = detector.update(list(window_buffer))

            result = dict(last_result)
            result["flow"] = float(flow)

            # ─────── SIMULATED TIME ───────────────────────
            sim_minutes += 1
            result["sim_time"] = f"{(sim_minutes // 60) % 24:02d}:{sim_minutes % 60:02d}"
            result["sim_minutes"] = sim_minutes

            # ─────── LEAK METADATA ────────────────────────
            result["leak_active"] = bool(leak_active)
            result["leak_mode"] = str(leak_mode)
            result["leak_intensity"] = float(leak_intensity if leak_active else 0.0)
            result["leak_remaining"] = (
                max(0, leak_end_minute - sim_minutes)
                if leak_active and leak_end_minute
                else 0
            )

            # ─────── EMIT TO FRONTEND ─────────────────────
            await sio.emit("data_update", result)

            await asyncio.sleep(1.0 / simulation_speed)

        else:
            await asyncio.sleep(0.1)
```

### Algorithm Breakdown

#### Step 1: Generate Flow
```python
flow = generator.next()
# Returns single apartment building's flow for current minute (L/min)
```

#### Step 2: Apply Server-Side Leak
```python
if leak_active:
    if leak_end_minute is not None and sim_minutes >= leak_end_minute:
        # Leak time expired
        leak_active = False
    else:
        # Calculate effective leak flow
        if leak_mode == "instant":
            effective_intensity = leak_intensity
        elif leak_mode == "ramp":
            elapsed = sim_minutes - leak_start_minute
            progress = min(1.0, elapsed / leak_ramp_minutes)
            effective_intensity = leak_intensity * progress
        
        # Apply leak
        flow += effective_intensity
        flow = min(flow, generator.MAX_FLOW_LPM)
```

**Ramp Calculation Example:**
```
leak_start_minute = 100
leak_end_minute = 200
leak_intensity = 1.0 L/min
leak_ramp_minutes = 10

At minute 105:
  elapsed = 105 - 100 = 5
  progress = min(1.0, 5/10) = 0.5
  effective = 1.0 * 0.5 = 0.5 L/min

At minute 110+:
  elapsed = 110 - 100 = 10
  progress = min(1.0, 10/10) = 1.0
  effective = 1.0 * 1.0 = 1.0 L/min (full intensity)

At minute 200:
  leak_active = False (end reached)
```

#### Step 3: Buffer Window
```python
window_buffer.append(float(flow))

if len(window_buffer) == WINDOW_MINUTES:  # 20 minutes
    last_result = detector.update(list(window_buffer))
    # Detector processes 20-min window, updates anomaly status
```

#### Step 4: Build Response
```python
result = dict(last_result)  # Copy detector results
result["flow"] = float(flow)
result["sim_time"] = f"{HH:MM}"
result["sim_minutes"] = sim_minutes

# Add leak information
result["leak_active"] = bool(leak_active)
result["leak_mode"] = str(leak_mode)
result["leak_intensity"] = float(leak_intensity if leak_active else 0.0)
result["leak_remaining"] = (
    max(0, leak_end_minute - sim_minutes)
    if leak_active and leak_end_minute
    else 0
)
```

#### Step 5: Advance Time
```python
sim_minutes += 1
await sio.emit("data_update", result)
await asyncio.sleep(1.0 / simulation_speed)
```

**Sleep Duration Calculation:**
```
simulation_speed = 1.0  → sleep(1.0) = 1 sec per minute
simulation_speed = 2.0  → sleep(0.5) = 0.5 sec per minute (2x faster)
simulation_speed = 10.0 → sleep(0.1) = 0.1 sec per minute (10x faster)
```

### Emitted Data Format

```javascript
{
    // Current flow measurement
    "flow": 42.5,                       // L/min

    // Anomaly detection results
    "anomaly": false,                   // Final alarm decision
    "final_score": 0.28,                // Fused [0, 1] score
    "level2": {
        "triggered": false,             // CUSUM threshold exceeded
        "score": 0.1                    // CUSUM normalized score
    },
    "level3": {
        "triggered": false,             // IF threshold exceeded
        "score": 0.15,                  // IF normalized score
        "reconstruction_error": -0.08,  // Raw IF decision function
        "flow_trend": 0.002,            // Trend slope (L/min/min)
        "baseline_elev": 0.3            // Baseline deviation (σ)
    },

    // Simulation timing
    "sim_time": "08:30",                // HH:MM format (hours % 24)
    "sim_minutes": 510,                 // Total minutes since start

    // Leak information
    "leak_active": false,               // Leak currently injected
    "leak_mode": "instant",             // "instant" or "ramp"
    "leak_intensity": 0.0,              // Current leak flow (L/min)
    "leak_remaining": 0                 // Minutes until leak stops
}
```

### Emission Frequency

```
Speed 1x   → Emit every 1 second (1 minute data per second)
Speed 2x   → Emit every 0.5 seconds (2 minutes data per second)
Speed 10x  → Emit every 0.1 seconds (10 minutes data per second)
```

---

### Startup Event

```python
@app.on_event("startup")
async def startup_event():
    """Start the simulation loop on server startup."""
    asyncio.create_task(simulation_loop())
```

**Purpose:** Automatically starts background simulation loop when server starts

**Effect:** `simulation_loop()` runs in background forever (or until server stops)

---

## Server Execution

### Main Entry Point

```python
if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=5000, log_level="info")
```

**Parameters:**
- `socket_app`: ASGI application (FastAPI + Socket.IO wrapper)
- `host="0.0.0.0"`: Listen on all network interfaces
- `port=5000`: Listen on port 5000
- `log_level="info"`: Show INFO+ level logs

**Starting the Server:**
```bash
python server.py
# Or from parent directory:
python -m apartment_simulator.backend.server

# Outputs:
# Uvicorn running on http://0.0.0.0:5000
# Ctrl+C to quit
```

**Access Points:**
- HTTP: `http://localhost:5000` or `http://0.0.0.0:5000`
- WebSocket: `ws://localhost:5000/socket.io`

---

## Docker Integration

### Running in Docker

```dockerfile
FROM python:3.9

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY apartment_simulator ./apartment_simulator

# Expose port
EXPOSE 5000

# Run server
CMD ["python", "-m", "apartment_simulator.backend.server"]
```

### Docker Compose

```yaml
version: '3'
services:
  apartment-simulator:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./artifacts:/app/artifacts
      - ./models:/models          # Shared model volume
    environment:
      - PYTHONUNBUFFERED=1
    command: python -m apartment_simulator.backend.server
```

---

## Communication Flow Diagram

```
Frontend (Browser)
    ↓
User clicks "Start"
    ↓
Emits: socket.emit('start_simulation')
    ↓
Server receives via @sio.event start_simulation
    ↓
Backend: simulation_running = True
    ↓
Broadcast: sio.emit('simulation_state', {"state": "running"})
    ↓
Frontend receives & updates UI
    ↓
Background Loop (simulation_loop):
    For each minute:
    ├─ Generate flow
    ├─ Detect anomalies (every 20 min)
    ├─ Emit: sio.emit('data_update', {...})
    ├─ Sleep: 1/simulation_speed seconds
    └─ Repeat
    ↓
Frontend receives data_update
    ↓
Update real-time charts & indicators
```

---

## Error Handling

### Model Loading Fallbacks

```python
try:
    if_model = load_pickle(ARTIFACTS_DIR / "isolation_forest_building.pkl")
except FileNotFoundError:
    print("Fallback: Using household model")
    if_model = load_pickle(find_model_path("if_model.pkl"))
```

### Event Handler Exception Handling

```python
@sio.event
async def inject_leak(sid, data):
    try:
        intensity = float(data.get("intensity", 0.5))
        # ... process ...
    except Exception:
        pass  # Silently ignore errors (preserves server stability)
```

**Rationale:** Prevent malformed frontend requests from crashing server

---

## Performance & Optimization

### Memory Usage

```
Generator (50 apts):       ~600 KB
Detector + Models:         ~50 MB (loaded once)
Window Buffer (20 min):     ~1 KB
Per-user state:            ~10 KB
```
**Total: ~50 MB (constant)**

### Computation Time

```
Per minute:
  - generator.next() × 50:   ~15 µs
  - Leak application:        ~1 µs
  - JSON serialization:      ~10 µs
  
Per 20-min window:
  - detector.update():       ~1.2 ms
  
Total per minute: ~30 µs (negligible)
```

### Network Bandwidth

```
Per emitted message:   ~500 bytes
Emission rate (1x):    1 msg/sec = 500 bytes/sec ≈ 4 Mbps
Emission rate (10x):   10 msg/sec = 5000 bytes/sec ≈ 40 Mbps
```

---

## Testing Scenarios

### Test 1: Normal Day Simulation

```python
# Frontend flow:
socket.emit('start_simulation')
# Wait 1440 seconds for full day
# Expect: no anomalies, varying flow patterns
```

### Test 2: Leak Detection

```python
# Frontend flow:
socket.emit('start_simulation')
# Wait 60 seconds
socket.emit('inject_leak', {"intensity": 0.5, "duration": 180, "mode": "instant"})
# Wait 300 seconds
socket.emit('stop_simulation')
# Expect: anomaly detection after ~100 seconds (persistence filter)
```

### Test 3: Speed Control

```python
socket.emit('start_simulation')
socket.emit('set_speed', 10.0)
# Observe: data updates 10x faster
socket.emit('set_speed', 1.0)
# Observe: data updates return to normal speed
```

### Test 4: Pause/Resume

```python
socket.emit('start_simulation')
# Wait 30 seconds
socket.emit('pause_simulation')
# UI shows paused state, sim_minutes stops incrementing
socket.emit('start_simulation')
# sim_minutes continues from paused point
```

---

## Summary

| Component | Purpose | Trigger |
|-----------|---------|---------|
| **GET /** | Serve frontend | HTTP GET http://localhost:5000/ |
| **start_simulation** | Resume simulation | Frontend button |
| **pause_simulation** | Pause simulation | Frontend button |
| **stop_simulation** | Reset simulation | Frontend button |
| **set_speed** | Change playback speed | Speed slider |
| **inject_leak** | Start leak scenario | Leak injection form |
| **stop_leak** | End leak scenario | Stop leak button |
| **simulation_loop** | Main engine | Runs in background forever |
| **data_update** | Stream results | Emit every ~1 second |

---

**End of `server.py` Documentation**
