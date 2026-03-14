# Apartment Building Water Leak Detection Simulator

A real-time, AI-powered water leak detection system for 50-unit apartment buildings. This simulator generates synthetic water flow data, applies hybrid anomaly detection, and visualizes results via an interactive web dashboard.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Architecture](#architecture)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Backend Components](#backend-components)
7. [Frontend Components](#frontend-components)
8. [Data Processing Pipeline](#data-processing-pipeline)
9. [Anomaly Detection (Hybrid Model)](#anomaly-detection-hybrid-model)
10. [Configuration & Calibration](#configuration--calibration)
11. [API Reference](#api-reference)
12. [Performance Metrics](#performance-metrics)

---

## Overview

### Purpose
Simulate a 50-unit apartment building's real-time water consumption and detect water leaks using a hybrid anomaly detection approach combining statistical (CUSUM) and machine learning (Isolation Forest) methods.

### Key Features
- **50 Independent Apartment Generators**: Each apartment has unique appliance usage patterns
- **Building-Aggregate Simulation**: Real-time aggregation of 50 households → building-level data
- **Hybrid Anomaly Detection**: Combines CUSUM (statistical) + Isolation Forest (ML)
- **Real-Time Web Dashboard**: 4 interactive charts with live data streaming
- **Leak Injection Simulation**: Test detection with synthetic leak events
- **Docker Support**: Containerized backend for easy deployment

### Expected Performance
- **Accuracy**: 94.66%
- **Recall**: 71.71% (catches most leaks)
- **Precision**: 26.44% (some false positives accepted for sensitivity)
- **F1 Score**: 38.64%

---

## Project Structure

```
apartment_simulator/
├── README.md                    ← This file
├── BUILD_SUMMARY.md             ← Detailed build summary
├── requirements.txt             ← Python dependencies
├── Dockerfile                   ← Docker configuration
├── docker-compose.yml           ← Docker Compose setup
│
├── backend/                     ← FastAPI server & simulation logic
│   ├── __init__.py
│   ├── server.py                ← FastAPI + Socket.IO main application
│   ├── live_simulator.py        ← Apartment building data generator
│   ├── model.py                 ← Hybrid anomaly detector (CUSUM + IF)
│   └── __pycache__/
│
├── frontend/                    ← Web UI dashboard
│   ├── index.html               ← Main page (Chart.js visualizations)
│   └── static/
│       ├── app.js               ← Socket.IO client & chart management
│       └── style.css            ← Dashboard styling & animations
│
├── preprocessing/               ← Training pipeline & configuration
│   ├── __init__.py
│   ├── main.py                  ← Orchestrates full training pipeline
│   ├── generate_data.py         ← Synthetic data generation (6 months)
│   ├── train_model.py           ← Isolation Forest training
│   ├── README.md                ← ML pipeline documentation
│   ├── data/                    ← Training/test datasets
│   │   ├── water_train_building.csv
│   │   └── water_test_building.csv
│   └── artifacts/               ← Calibration results
│       ├── calibration_building.json
│       └── metrics_building.json
│
└── artifacts/                   ← Runtime configuration & models
    ├── all_appliances.json      ← Appliance library (priors)
    ├── calibration_building.json ← Detection thresholds
    ├── if_calibration.json      ← Isolation Forest thresholds
    └── metrics_building.json    ← Training metrics
```

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                   WEB DASHBOARD (Browser)                   │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐  │
│  │ Flow Chart   │ IF Score     │ CUSUM Score  │ Timeline │  │
│  └──────────────┴──────────────┴──────────────┴──────────┘  │
│           ▲                                        ▲          │
│           │         WebSocket (Socket.IO)         │          │
│           └────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ :8000
┌─────────────────────────────▼─────────────────────────────────┐
│              FastAPI Server (backend/server.py)               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ WebSocket Endpoint + Simulation Loop (60 Hz)            │ │
│  │  • Runs LiveApartmentBuildingDataGenerator               │ │
│  │  • Extracts features every 20 minutes                    │ │
│  │  • Feeds to HybridWaterAnomalyDetector                   │ │
│  │  • Streams results to frontend over Socket.IO            │ │
│  └───────────────────────┬──────────────────────────────────┘ │
│                          │                                     │
│  ┌───────────────────────▼──────────────────────────────────┐ │
│  │      LiveApartmentBuildingDataGenerator                  │ │
│  │  (backend/live_simulator.py)                             │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │ 50 × LiveWaterFlowGenerator (1 per apartment)       │ │ │
│  │  │ + Aggregator → Building-level flow (L/min)         │ │ │
│  │  │ + Leak injection support                           │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └───────────────────────┬──────────────────────────────────┘ │
│                          │                                     │
│  ┌───────────────────────▼──────────────────────────────────┐ │
│  │   HybridWaterAnomalyDetector (backend/model.py)          │ │
│  │  ┌─────────────────┬──────────────────────────────────┐  │ │
│  │  │ Level 2 CUSUM   │  Level 3 Isolation Forest       │  │ │
│  │  │ (Statistical)   │  (ML-based)                     │  │ │
│  │  └─────────────────┴──────────────────────────────────┘  │ │
│  │         ➜ Weighted Fusion (0.4 + 0.6) + Persistence    │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          │                                     │
└──────────────────────────┼─────────────────────────────────────┘
                           │ Saved Artifacts
                           ▼
               Pre-trained Models & Calibration
               (artifacts/ directory)
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- pip or conda
- Docker (optional, for containerized deployment)
- Port 8000 (backend) and 5173+ (frontend dev server)

### Option 1: Local Installation

```bash
# Navigate to apartment_simulator directory
cd apartment_simulator

# Install dependencies
pip install -r requirements.txt

# Run the server
python backend/server.py

# Server runs at: http://localhost:8000
```

### Option 2: Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

### Option 3: Manual Preprocessing & Training

```bash
# Regenerate training data and train models from scratch
cd preprocessing
python main.py

# This runs:
# 1. generate_data.py   → Creates 6 months synthetic training data
# 2. train_model.py     → Trains Isolation Forest on aggregated building data
# 3. Calibration        → Generates thresholds optimized for building scale
```

---

## Usage

### Starting the Server

```bash
python backend/server.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Accessing the Dashboard

Open browser to: **http://localhost:8000**

### Dashboard Controls

| Control | Function |
|---------|----------|
| **Start** | Begin simulation from minute 0 |
| **Pause/Resume** | Halt and resume data flow |
| **Stop** | Reset simulation |
| **Speed (1-10x)** | Adjust simulation speed multiplier |
| **Leak Intensity** | 0.1–20.0 L/min leak magnitude |
| **Leak Duration** | 1–300 minutes |
| **Leak Mode** | `instant` (immediate) or `ramp` (gradual onset) |
| **Inject Leak** | Trigger leak event at current time |

### Example Scenarios

**Scenario 1: Detect a 5 L/min leak**
1. Press Start
2. Set Speed to 5x (simulates faster)
3. Let it run ~5 minutes, then Inject Leak (intensity: 5.0, duration: 60, mode: instant)
4. Observe anomaly detectors react within 20 minutes of feature window

**Scenario 2: Monitor normal operation**
1. Press Start, keep Speed at 1x
2. Let it run for several hours (real-time)
3. Observe periodic appliance usage patterns (morning/evening peaks)
4. No anomalies should trigger

---

## Backend Components

### 1. Server (`backend/server.py`)

**Purpose**: FastAPI + Socket.IO application serving:
- WebSocket real-time data streaming
- Static frontend assets (index.html, CSS, JS)
- Simulation loop orchestration
- Control endpoints (start, pause, stop, leak injection)

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `setup_cors()` | Enable Cross-Origin requests |
| `@sio.on('connect')` | Handle new WebSocket connections |
| `@sio.on('control')` | Process user controls (start/pause/leak injection) |
| `simulation_loop()` | Main async loop: generate data → detect → emit |
| `serve_static(path)` | Serve frontend assets |

**Data Flow**:
1. User clicks "Start" → client emits `control` event
2. Server starts `simulation_loop()` in background
3. Each iteration:
   - Calls `generator.next()` → single minute of aggregate building flow
   - Every 20 minutes: extracts features → feeds to detector
   - Emits JSON to frontend via `data` event
   - Clients render charts in real-time

**Socket.IO Events**:

**Client → Server**:
```json
{
  "type": "start|pause|stop",
  "leak": { "intensity": 5.0, "duration": 60, "mode": "instant" }
}
```

**Server → Client** (per minute):
```json
{
  "sim_minutes": 240,
  "flow": 125.3,
  "cusum_score": 0.45,
  "if_score": -0.023,
  "is_anomaly": false,
  "leak_detected": false
}
```

---

### 2. Live Simulator (`backend/live_simulator.py`)

**Purpose**: Generate realistic building-level water flow data in real-time.

**Key Classes**:

#### `LiveWaterFlowGenerator`
Single apartment household water flow generator.

**Constructor Parameters**:
```python
LiveWaterFlowGenerator(
    priors_path,              # Path to appliance JSON
    daily_min_l=100,          # Min daily volume per apartment
    daily_max_l=160,          # Max daily volume per apartment
    max_flow_lpm=15.0,        # Peak flow per apartment
    noise_sigma=0.03,         # Sensor noise std dev (fraction)
    max_regen_attempts=10,    # Config robustness
    seed=None                 # RNG seed
)
```

**Key Methods**:
- `next()` → Returns next minute's flow (L/min) with noise + leak
- `inject_leak(start_min, end_min, flow_lpm)` → Add synthetic leak
- `global_minute()` → Total minutes simulated

**Realistic Features**:
- Appliance event generation from JSON priors
- Daily volume constraints (100–160 L/day per apartment)
- Temporal clustering (morning/evening peaks)
- Additive sensor noise on non-zero flows

#### `LiveApartmentBuildingDataGenerator`
Aggregates 50 independent apartments into building-level data.

**Constructor**:
```python
LiveApartmentBuildingDataGenerator(
    priors_path,              # Appliance library
    num_apartments=50,        # Number of units
    daily_min_l=None,         # Per-apartment daily min (override)
    daily_max_l=None,         # Per-apartment daily max (override)
    seed=None                 # RNG seed
)
```

**Key Methods**:
- `next()` → Sum of all 50 apartments' current minute flow
- `set_leak_all_apartments(intensity, duration)` → Inject leak at building level
- `get_status()` → Current flow stats

**Building-Scale Data**:
- Typical range: 5,000–8,000 L/day (100–160 L/min peaks)
- 50 independent appliance patterns → complex aggregate signal
- Used for both training and real-time simulation

---

### 3. Hybrid Anomaly Detector (`backend/model.py`)

**Purpose**: Combine statistical (CUSUM) and ML-based (Isolation Forest) detection.

**Key Class**: `HybridWaterAnomalyDetector`

**Constructor Parameters**:
```python
HybridWaterAnomalyDetector(
    if_model,                  # Pre-trained Isolation Forest
    if_scaler,                 # StandardScaler (fit on training data)
    cusum_k=0.5,               # CUSUM threshold (building scale)
    cusum_h=20.0,              # CUSUM trigger threshold
    noise_floor=0.2,           # Minimum flow level
    if_threshold=-0.05,        # IF model decision threshold
    if_score_scale=0.1,        # IF score normalization
    appliance_flow_thresh=8.0, # Max normal flow
    clip_bound=10.0,           # Feature scaling clip
    w2=0.4,                    # CUSUM weight
    w3=0.6,                    # IF weight
    decision_threshold=0.65,   # Combined anomaly threshold
    persistence_windows=2      # Consecutive anomalies required
)
```

**Key Methods**:

1. **`extract_features(flow_window)`**
   - Inputs: 20-minute rolling window of flows
   - Outputs: 5 features
     - `mnf`: Minimum flow in window
     - `inter_mean`: Mean inter-event time
     - `inter_frac`: Fraction of zero-flow minutes
     - `mean_flow`: Average flow
     - `inter_std`: Std dev of inter-event time
   - Used by both CUSUM and Isolation Forest

2. **`step(flow_value, features)`**
   - Core detection logic
   - Inputs: Current minute flow + 5 extracted features
   - Returns: `(anomaly_score, is_anomaly, detector_scores)`

3. **`detect(flow_value, feature_window)`**
   - Wrapper for step()
   - Manages persistence buffer (requires 2+ consecutive anomalies)

### Detection Levels

**Level 2: CUSUM (Statistical Change Detection)**
- Detects persistent low-flow deviations
- Threshold: `cusum_k = 0.5`
- Trigger: `cusum_h = 20.0` (accumulated slack variable)
- Sensitive to sustained deviations below baseline

**Level 3: Isolation Forest (ML Anomaly Detection)**
- Trained on 6 months of normal building aggregate data
- 200 trees, 5% contamination
- Detects statistical outliers in 5D feature space
- Output: Decision function score (-1 to +1 range)
- Normalized to [0, 1] for fusion

**Fusion Logic** (Level 4):
```
final_score = 0.4 × CUSUM_score + 0.6 × IF_score

anomaly detected when:
  1. final_score > decision_threshold (0.65)
  2. AND at least 2 consecutive windows trigger
```

**Persistence Filter**: Requires 2+ consecutive anomalies before flagging
- Reduces false positives
- Trades latency (~20–40 minutes) for confidence

---

## Frontend Components

### 1. HTML Structure (`frontend/index.html`)

**Key Elements**:
- **Background Canvas**: Animated particle effect
- **Connection Indicator**: Real-time server status
- **Control Panel**: Start/pause/stop, speed, leak injection
- **4 Chart Containers**: Chart.js real-time visualizations

### 2. Client Application (`frontend/static/app.js`)

**Socket.IO Integration**:
- Connects to `http://localhost:8000` via Socket.IO
- Listens for `data` events from server
- Emits `control` events for user interactions

**Key State Variables**:
```javascript
let charts = {}              // Chart.js chart instances
let simStart = null          // Simulation start timestamp
let currentSpeed = 1         // Speed multiplier (1-10x)
let simMinutes = 0           // Current simulation minute
const WINDOW_SIM_MINS = 120  // Rolling window (120 minutes displayed)
const MAX_PTS = 600          // Max points per chart
```

**Buffer Management**:
- Maintains circular buffers for: `flow`, `cusum`, `ifscore`, `anomaly`
- Respects rolling window (only last 2 hours displayed)
- Smooth animations via incremental chart updates

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `initCharts()` | Create 4 Chart.js instances |
| `updateChart(chartId, xVal, yVal)` | Add point to chart |
| `socketUpdate(data)` | Process incoming server event |
| `sendControl(action, payload)` | Send user action to server |
| `injectLeak()` | Trigger leak event |

**Event Handlers**:
- `start()` / `pause()` / `stop()` buttons
- `speed` slider (1–10x)
- Leak injection form (intensity, duration, mode)
- Real-time connection status

### 3. Styling (`frontend/static/style.css`)

**Design Features**:
- Dark theme background (particle animation)
- Glassmorphic cards with 3D effect
- Gradient accents (blue/cyan leak detection theme)
- Responsive layout (mobile-friendly)
- Real-time status badges (connected/disconnected)

**Animations**:
- Smooth chart transitions
- Pulsing anomaly indicators
- Connector dot breathing effect

---

## Data Processing Pipeline

### Overview

The preprocessing pipeline generates training data and trains the Isolation Forest model on building-scale data.

**Pipeline Stages**:
1. **Generate Data** (`generate_data.py`)
2. **Train Model** (`train_model.py`)
3. **Calibration** (automated within train_model.py)

### Stage 1: Data Generation (`preprocessing/generate_data.py`)

**Purpose**: Create 6 months of synthetic apartment building water flow data.

**Output Files**:
- `data/water_train_building.csv` — 180 days normal operation
- `data/water_test_building.csv` — 180 days with 8 leak events

**Configuration**:
```python
DAYS = 180              # 6 months
APARTMENTS = 50         # Building size
MINUTES_PER_DAY = 1440  # 24 hours
```

**Function: `generate_building_training_data()`**
- Creates clean (no leaks) training dataset
- 50 apartments × 180 days × 1,440 minutes/day = 129,600 samples
- Output CSV: `timestamp, day, minute_of_day, hour, flow_lpm`

**Function: `inject_leaks_into_data()`**
- Injects 8 realistic leak events into test data
- Leak types:
  - **Stress**: During peak hours (7–8 AM, 7–9 PM)
  - **Seasonal**: Winter months
  - **Night**: Midnight–6 AM
- Leak duration: ~360 minutes each
- Total leak minutes: ~2,888 out of 129,600 test minutes

**Leak Injection Pattern**:
```python
leak_intensity = np.random.uniform(0.5, 5.0)  # L/min leak magnitude
leak_start = random day × 1440 + random hour × 60  # Random start time
leak_end = leak_start + 360  # 6-hour duration
```

### Stage 2: Model Training (`preprocessing/train_model.py`)

**Purpose**: Train Isolation Forest on aggregated building data.

**Key Steps**:

1. **Load Data**:
   - Read training CSV (129,600 normal samples)
   - Read test CSV (129,600 samples with leaks)

2. **Feature Extraction**:
   - 20-minute sliding windows
   - 5-minute stride → ~25k windows from 180 days
   - Features: `mnf, inter_mean, inter_frac, mean_flow, inter_std`
   - Normalization: StandardScaler fit on training features

3. **Model Training**:
   ```python
   iso_forest = IsolationForest(
       n_estimators=200,
       contamination=0.05,  # 5% normal data flagged as anomaly
       random_state=42
   )
   iso_forest.fit(X_train_scaled)
   ```

4. **Cross-Validation**:
   - Evaluate on test set (contains leak events)
   - Compute: Accuracy, Precision, Recall, F1 Score

5. **Threshold Calibration**:
   - Compute 99th percentile of normal scores → `if_threshold`
   - Normalize scores for [0, 1] fusion range

**Output Artifacts**:
- `artifacts/if_calibration.json` — Thresholds
- `artifacts/metrics_building.json` — Performance metrics
- `models/*.keras` — Keras model (if used)

### Stage 3: Calibration

**CUSUM Thresholds (Building Scale)**:
```
cusum_k = 0.5              # 10× household (0.01)
cusum_h = 20.0             # 10× household (2.0)
noise_floor = 0.2 L/min
appliance_flow_thresh = 8.0 L/min
```

Calibrated by running detector on test data and optimizing for:
- High recall (catch most leaks)
- Manageable precision (accept some false positives)

---

## Anomaly Detection (Hybrid Model)

### Detection Architecture

The hybrid detector combines three levels of analysis:

```
┌─────────────────┐
│  Building Flow  │
│   (L/min)       │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Features│      ← 20-min window, 5 features
    └────┬────┘
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
┌─────────────────────┐    ┌──────────────────────┐
│  CUSUM Detector     │    │ Isolation Forest     │
│  (Level 2)          │    │ (Level 3)            │
├─────────────────────┤    ├──────────────────────┤
│ • Low-flow devs     │    │ • Statistical        │
│ • Threshold: 0.5    │    │   anomalies          │
│ • Trigger: 20.0     │    │ • 200 trees, 5%      │
│ • Output: [0, 1]    │    │   contamination      │
│                     │    │ • Output: [-1, +1]   │
└─────────┬───────────┘    └──────────┬───────────┘
          │                           │
          │  Score                    │  Score
          ▼                           ▼
       0.4×CUSUM              +  0.6×IF_normalized
            │                           │
            └────────┬──────────────────┘
                     ▼
          ┌─────────────────────┐
          │ Weighted Fusion     │
          │ final = 0.4+0.6     │
          └────────┬────────────┘
                   │
                   ▼ (> 0.65?)
         ┌─ ─ ─ ─ ─ ─ ─ ─ ┐
        ┌─────────────┐
        │Persistence  │
        │ Buffer      │
        └─────────────┘
             │
        (≥2 consecutive?)
             │
             ▼
    ┌───────────────────┐
    │ Leak Detected ✓   │
    └───────────────────┘
```

### CUSUM Algorithm (Level 2)

**Change Detection**: Detects persistent deviations below expected baseline.

**Parameters**:
- `k` = threshold divided by 2 (sensitivity)
- `h` = decision threshold for slack variable

**Algorithm**:
```
For each minute:
  1. Compute slack variable: S_t = max(0, S_{t-1} + (flow_t - baseline - k))
  2. If S_t > h → CUSUM triggers
  3. Score_cusum = S_t / h  (normalized to [0, 1])
```

**Building-Scale Configuration**:
- `k = 0.5` (10× household baseline)
- `h = 20.0` (10× household threshold)
- Sensitive to sustained low-flow patterns
- Typical leak: triggers within 20–40 minutes

**Weakness**: Slow to react to sudden spikes (time-based detector)
**Strength**: Low false positives on gradual usage changes

### Isolation Forest (Level 3)

**Statistical Anomaly Detection**: Identifies points far from normal distribution.

**How It Works**:
1. Build ensemble of decision trees (200 trees)
2. Each tree randomly selects features and thresholds
3. Anomalies require fewer splits to isolate (short path lengths)
4. Decision function: negative score = anomalous, positive = normal

**Features Used** (5-dimensional):
```
1. mnf (min flow)        — Lowest flow in 20-min window
2. inter_mean            — Average time between appliance events
3. inter_frac            — Fraction of zero-flow minutes
4. mean_flow             — Average flow in window
5. inter_std             — Variability of inter-event times
```

**Model Characteristics**:
- Trained on 6 months of normal building data
- Contamination = 5% (internally calibrated)
- Detects unusual feature combinations
- Scores normalized for fusion: `IF_score_norm = (IF_raw + 1) / 2`

**Advantage**: Catches sudden spikes and unusual patterns
**Limitation**: May trigger on benign high-demand periods

### Fusion Strategy (Level 4)

**Weighted Ensemble**:
```
final_anomaly_score = 0.4 × CUSUM + 0.6 × IF_norm
```

**Rationale**:
- CUSUM = 40%: Focuses on sustained low-flow (leak signatures)
- IF = 60%: Broader statistical anomaly detection

**Decision**: `anomaly = (final_score > 0.65)`

**Persistence Filter**:
- Requires ≥ 2 consecutive windows flagged
- Windows: 20-minute feature extraction windows
- Delay: ~20–40 minutes (trade-off for confidence)
- Reduces false positives from transient usage spikes

---

## Configuration & Calibration

### Runtime Configuration Files

#### 1. `artifacts/all_appliances.json`
Appliance library defining realistic water usage patterns.

**Structure**:
```json
{
  "appliances": {
    "toilet": {
      "duration_sec": [6, 8],
      "flow_lpm": [12, 15],
      "events_per_day": [4, 8]
    },
    "shower": {
      "duration_sec": [600, 1200],
      "flow_lpm": [6, 8],
      "events_per_day": [0, 2]
    },
    ...
  }
}
```

**Components**:
- `duration_sec`: [min, max] event duration in seconds
- `flow_lpm`: [min, max] flow rate during event
- `events_per_day`: [min, max] expected occurrences per day

#### 2. `artifacts/calibration_building.json`
Building-scale detection thresholds.

**Example**:
```json
{
  "cusum_k": 0.5,
  "cusum_h": 20.0,
  "if_threshold": -0.0797,
  "if_score_scale": -0.0189,
  "noise_floor": 0.2,
  "appliance_flow_thresh": 8.0,
  "decision_threshold": 0.65,
  "persistence_windows": 2
}
```

#### 3. `artifacts/if_calibration.json`
Isolated Forest-specific thresholds.

**Example**:
```json
{
  "if_threshold_99pct": -0.0797,
  "if_score_scale_factor": -0.0189,
  "contamination_rate": 0.05
}
```

#### 4. `artifacts/metrics_building.json`
Training performance metrics.

**Example**:
```json
{
  "accuracy": 0.9466,
  "precision": 0.2644,
  "recall": 0.7171,
  "f1_score": 0.3864,
  "true_positives": 436,
  "false_positives": 1213,
  "false_negatives": 172,
  "true_negatives": 24095
}
```

### Tuning Parameters

To adjust detector sensitivity:

| Parameter | Effect | Range | Default |
|-----------|--------|-------|---------|
| `cusum_k` | Lower = more sensitive to low-flow | 0.1–1.0 | 0.5 |
| `cusum_h` | Lower = faster trigger | 5.0–50.0 | 20.0 |
| `if_threshold` | Lower = more anomalies flagged | -0.5 to 0.0 | -0.05 |
| `decision_threshold` | Lower = more detections | 0.5–0.9 | 0.65 |
| `persistence_windows` | Higher = fewer false positives | 1–5 | 2 |

**Recommended Adjustments**:
- **High Sensitivity** (catch all leaks): `decision_threshold=0.55, persistence_windows=1`
- **Balanced** (default): `decision_threshold=0.65, persistence_windows=2`
- **Low Sensitivity** (fewer false positives): `decision_threshold=0.75, persistence_windows=3`

---

## API Reference

### WebSocket Events

#### Server → Client: `data`

**Emitted**: Every simulated minute

```json
{
  "sim_minutes": 240,
  "flow": 125.34,
  "cusum_score": 0.42,
  "if_score": -0.015,
  "is_anomaly": false,
  "leak_detected": false,
  "speed": 1
}
```

**Fields**:
- `sim_minutes` (int): Total simulated minutes since start
- `flow` (float): Building aggregate flow (L/min)
- `cusum_score` (float): CUSUM detector output [0, 1]
- `if_score` (float): Isolation Forest score [0, 1]
- `is_anomaly` (bool): Final anomaly decision
- `leak_detected` (bool): Same as is_anomaly (legacy)
- `speed` (float): Current simulation speed multiplier

#### Client → Server: `control`

**Sent**: When user interacts with controls

```json
{
  "action": "start|pause|stop",
  "speed": 1,
  "leak": {
    "intensity": 5.0,
    "duration": 60,
    "mode": "instant|ramp"
  }
}
```

**Fields**:
- `action` (str): `start` / `pause` / `stop`
- `speed` (float): Speed multiplier (1–10)
- `leak` (dict, optional):
  - `intensity` (float): Leak rate in L/min
  - `duration` (int): Leak duration in minutes
  - `mode` (str): `instant` or `ramp` onset

---

## Performance Metrics

### Training Dataset Performance

**Metrics** (evaluated on 6-month test set with 8 injected leaks):

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.66% |
| **Precision** | 26.44% |
| **Recall** | 71.71% |
| **F1 Score** | 38.64% |

**Confusion Matrix**:
```
           Predicted
           Leak  Normal
Actual Leak  436    172   (True Positives, False Negatives)
       Normal 1213  24095 (False Positives, True Negatives)
```

### Interpretation

- **High Recall (71.7%)**: Catches most actual leaks – good for safety
- **Low Precision (26.4%)**: Many false positives – requires validation step
- **Accuracy (94.7%)**: Overall correct classifications (normal data dominates)

### Recommended Use

1. **Detection Stage**: Use high recall → flag potential leaks
2. **Validation Stage**: Human review or 2nd detector to filter false positives
3. **Alerting Stage**: Notify building maintenance with confidence level

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Server won't start | Port 8000 in use | `lsof -i :8000` to find PID, `kill <PID>` |
| No data in charts | WebSocket disconnected | Check console (F12), restart browser |
| Detector never triggers | Thresholds too high | Lower `decision_threshold` in config |
| Too many false positives | Thresholds too low | Increase `decision_threshold` or `persistence_windows` |
| Slow performance | High simulation speed | Reduce speed slider (1–3x recommended) |
| Models not loading | Missing artifacts directory | Run `preprocessing/main.py` to regenerate |

---

## Key Technologies

- **Backend**: FastAPI, Socket.IO (WebSocket), Uvicorn
- **ML**: scikit-learn (Isolation Forest), NumPy
- **Frontend**: Chart.js (charting), Socket.IO client (WebSocket)
- **Data**: Pandas, NumPy (data processing)
- **Container**: Docker, Docker Compose

---

## References & Further Reading

- **CUSUM Algorithm**: Page, E.S. "Continuous inspection schemes." Biometrika 41.1 (1954): 100–115.
- **Isolation Forest**: Liu, F.T., et al. "Isolation Forest." ICDM, 2008.
- **Building-Scale Leakage**: ISO 16680:2005 "Water meter units – Design, performance and testing"

---

## Contributing & Support

For questions or issues:
1. Check `BUILD_SUMMARY.md` for detailed architecture
2. Review `preprocessing/README.md` for ML pipeline details
3. Enable debug logging in `backend/server.py` (uncomment `logging.basicConfig(level=logging.DEBUG)`)

---

**Last Updated**: 2026-03-14  
**Version**: 1.0 (Building Scale)
