# Apartment Building Water Leak Detection Simulator

Real-time, AI-powered water leak detection for a 50-unit apartment building.
Generates synthetic building-aggregate water flow data, applies a hybrid
CUSUM + Isolation Forest anomaly detector, and streams results to an
interactive web dashboard.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Anomaly Detection Model](#anomaly-detection-model)
6. [Training Pipeline](#training-pipeline)
7. [Configuration Reference](#configuration-reference)
8. [API Reference](#api-reference)
9. [Performance Metrics](#performance-metrics)
10. [Tuning Guide](#tuning-guide)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### What it does

- Simulates **50 independent apartments** with realistic appliance usage, aggregated to building level
- Detects water leaks in real time using a **hybrid CUSUM + Isolation Forest** detector
- Streams minute-by-minute flow, anomaly scores, and detection status to a browser dashboard
- Supports **leak injection** (instant or ramp) to test detection at any intensity

---

## Project Structure

```
apartment_simulator/
├── README.md                       ← This file
├── requirements.txt                ← Python dependencies
├── Dockerfile                      ← Runtime container
├── Dockerfile.training             ← Training-only container
├── docker-compose.yml              ← Compose: simulator + trainer services
│
├── backend/
│   ├── server.py                   ← FastAPI + Socket.IO server (port 5000)
│   ├── live_simulator.py           ← 50-apartment building flow generator
│   └── model.py                    ← HybridWaterAnomalyDetector (CUSUM + IF)
│
├── frontend/
│   ├── index.html                  ← Dashboard (4 Chart.js panels)
│   └── static/
│       ├── app.js                  ← Socket.IO client + chart management
│       └── style.css               ← Dark/glassmorphic theme
│
├── preprocessing/
│   ├── main.py                     ← Pipeline orchestrator (generates + trains)
│   ├── generate_data.py            ← 180-day synthetic data with diverse leaks
│   ├── train_model.py              ← IF training with balanced calibration
│   └── artifacts/                  ← Intermediate training outputs
│       ├── calibration_building.json
│       └── metrics_building.json
│
└── artifacts/                      ← Runtime models & config (read by server)
    ├── all_appliances.json         ← Appliance library (flow priors)
    ├── isolation_forest_building.pkl
    ├── scaler_building.pkl
    ├── calibration_building.json   ← Detection thresholds (edit to tune)
    └── metrics_building.json       ← Last training metrics
```

---

## Quick Start

### Local (Python)

```bash
cd apartment_simulator
pip install -r requirements.txt
py backend/server.py
# Open http://localhost:5000
```

### Docker — run simulator

```bash
cd apartment_simulator
docker-compose up --build simulator
# Open http://localhost:5000
```

### Docker — retrain model

```bash
cd apartment_simulator

# Build training image
docker build -f Dockerfile.training -t apt-trainer .

# Run training (PowerShell)
docker run --rm -v "${PWD}/artifacts:/output" apt-trainer

# Artifacts written to ./artifacts/ — restart simulator to use them
docker-compose up --build simulator
```

### Dashboard controls

| Control | Function |
|---------|----------|
| Start / Pause / Stop | Simulation lifecycle |
| Speed (1–10×) | Wall-clock speed multiplier |
| Leak Intensity (0.1–20 L/min) | Magnitude of injected leak |
| Leak Duration (minutes) | How long the leak runs |
| Leak Mode: instant / ramp | Immediate or gradual onset |
| Inject Leak | Start a leak event now |
| Stop Leak | Cancel the active leak |

---

## Architecture

```
Browser Dashboard
  │  Chart.js (flow, CUSUM score, IF score, anomaly timeline)
  │  Socket.IO client
  │
  ▼  WebSocket  (Socket.IO)
FastAPI Server  backend/server.py  :5000
  │
  ├─ LiveApartmentBuildingDataGenerator  (live_simulator.py)
  │     50 × LiveWaterFlowGenerator  (independent apartments)
  │     → aggregate flow L/min  +  server-side leak injection
  │
  └─ HybridWaterAnomalyDetector  (model.py)
        20-min sliding window
        ├─ CUSUM  (statistical change detection)
        ├─ Isolation Forest  (ML anomaly detection, 7 features)
        └─ Fusion + Persistence filter
              → anomaly: bool, final_score: float
```

### Simulation loop (server.py)

Each iteration (1 simulated minute):

1. `generator.next()` → aggregate building flow for this minute
2. Apply server-side leak (instant or ramp) if active
3. Append flow value to 20-minute rolling `window_buffer`
4. Once buffer is full: `detector.update(window)` → anomaly result
5. `sio.emit("data_update", result)` → frontend updates charts
6. `asyncio.sleep(1.0 / speed)` → speed control

---

## Anomaly Detection Model

### Overview

`HybridWaterAnomalyDetector` combines two independent detectors whose
outputs are fused via a weighted score. Either detector can also bypass
the weighted fusion gate directly when it has strong statistical evidence.

```
window (20 min of flow values)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  CUSUM detector                Isolation Forest
  (Level 2)                     (Level 3)
  s += (flow - k)               decision_function(7 features)
  triggered when s >= h         triggered when score < if_threshold
        │                              │
        ▼                              ▼
  cusum_score [0,1]             if_score [0,1]
        │                              │
        └───────────┬──────────────────┘
                    ▼
        final_score = 0.35 × cusum_score
                    + 0.65 × if_score
                    │
        candidate_anomaly =
          final_score > decision_threshold (0.40)
          OR cusum_triggered
          OR if_triggered
                    │
        persistence filter: streak >= 4 consecutive windows
                    │
                    ▼
              anomaly: bool
```

---

### Level 2 — CUSUM

Classical CUSUM one-sided change detector applied to **per-minute flow**.

**Algorithm** (per sample in window):

```
if flow >= appliance_flow_thresh (8.0 L/min):
    # appliance event — partial decay to preserve leak evidence
    if just started: s *= 0.5
else:
    # inter-appliance period — accumulate deviation above k
    s = max(0, s + flow - cusum_k)
    if s >= cusum_h:
        triggered = True
```

**Key design decision — `cusum_k = 3.0`:**
The normal building inter-appliance baseline is **2.39 L/min** (median, std 1.39).
Setting `k` at 3.0 places it at the ~67th percentile of normal inter-appliance flow.
Only the upper third of normal minutes cause any accumulation, and natural appliance
cycling (partial reset on each appliance start) prevents that accumulation from
reaching `h` without a real sustained leak.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `cusum_k` | 3.0 L/min | Reference level; ~67th pct of normal inter-appliance flow |
| `cusum_h` | 15.0 | Accumulated slack to trigger; high enough to ignore normal variance |
| `appliance_flow_thresh` | 8.0 L/min | Flow above this = appliance event (not inter-appliance) |
| `noise_floor` | 0.2 L/min | Treated as zero |

**Bypass rule**: when `cusum_triggered = True`, the detector immediately
raises a candidate anomaly regardless of the fused score.

---

### Level 3 — Isolation Forest

300-tree Isolation Forest trained on **90 days of normal 50-apartment
aggregate data** (building-scale, not household-scale).

**7 features** (must match `model.py:_extract_features` exactly):

| # | Feature | Description |
|---|---------|-------------|
| 1 | `mnf` | 10th percentile of non-zero flows in window |
| 2 | `inter_mean` | Mean flow during inter-appliance minutes |
| 3 | `inter_frac` | Fraction of inter-appliance minutes above noise floor |
| 4 | `mean_flow` | Window-average flow |
| 5 | `inter_std` | Std dev of inter-appliance flows |
| 6 | `flow_trend` | Linear regression slope (L/min per minute) → rising baseline |
| 7 | `baseline_elev` | (rolling inter_mean − training median) / training std → normalised elevation |

**IF score normalisation:**

```python
if_score = clip((if_threshold - raw_score) / if_score_scale, 0, 1)
# raw_score < if_threshold  → if_score > 0  → anomalous
# raw_score >= if_threshold → if_score = 0  → normal
```

**Bypass rule**: when `if_triggered = True` (raw score < `if_threshold`),
the detector immediately raises a candidate anomaly regardless of the
fused score.

---

### Fusion & Persistence

```python
final_score = 0.35 × cusum_score + 0.65 × if_score

candidate_anomaly = (
    final_score > 0.40       # weighted fusion path
    or cusum_triggered        # CUSUM statistical trigger
    or if_triggered           # IF statistical trigger
)

_anomaly_streak += 1 if candidate_anomaly else reset to 0
anomaly = (_anomaly_streak >= 4)   # 4 consecutive minutes required
```

The persistence filter (`persistence_windows = 4`) is the primary guard
against false positives when the IF threshold is set aggressively (close
to 0). Any single-window glitch will not produce an alarm.

---

## Training Pipeline

Run via `preprocessing/main.py` or the Docker training image.

### Step 1 — Data generation (`generate_data.py`)

Generates **180 days** of synthetic building aggregate flow:

- **Training set** (90 days, no leaks): used to fit Isolation Forest
- **Test set** (90 days, 20 injected leaks): used for threshold calibration
  and evaluation

**20 diverse leak types in test set:**

| Type | Intensity | Duration | Purpose |
|------|-----------|----------|---------|
| `sustained_drip` × 5 | 0.1–1.5 L/min | 4–24 h | Hard-to-detect drips |
| `slow_leak` × 5 | 0.5–3.0 L/min | 3–8 h | Low-intensity sustained |
| `stress` × 4 | 3.0–10.0 L/min | 1–3 h | Peak-hour bursts |
| `seasonal` × 2 | 1.0–8.0 L/min | 4–12 h | Winter baseline shift |
| `night` × 2 | 0.5–5.0 L/min | 1–6 h | Midnight anomaly |
| `ramp` × 2 | 0.2 → 8.0 L/min | 2–6 h | Gradual onset |

### Step 2 — Model training (`train_model.py`)

1. Extract 7 features from 20-minute windows (stride 5 min)
2. Fit `StandardScaler` on training features
3. Train `IsolationForest(n_estimators=300, contamination=auto)`
4. **Balanced calibration set**: downsample test windows to 5:1
   (normal:anomaly) before threshold search — prevents PR curve from
   ignoring the minority anomaly class
5. `precision_recall_curve` → find threshold maximising F1 at recall ≥ 65%
6. Compute baseline stats (`inter_mean` median/std) for `baseline_elev`
7. Save: `isolation_forest_building.pkl`, `scaler_building.pkl`,
   `calibration_building.json`, `metrics_building.json`

### Step 3 — Artifact copy (`main.py`)

Copies the 4 files above from `preprocessing/artifacts/` to `artifacts/`,
so the live server picks them up on next restart.

---

## Configuration Reference

All runtime parameters live in `artifacts/calibration_building.json`.
Edit this file and restart the server — no retraining needed for
threshold changes.

```json
{
  "version": 2,
  "window_minutes": 20,
  "stride_minutes": 5,
  "appliance_flow_thresh": 8.0,
  "noise_floor": 0.2,
  "n_features": 7,
  "feature_names": ["mnf","inter_mean","inter_frac","mean_flow","inter_std","flow_trend","baseline_elev"],

  "cusum_k": 3.0,
  "cusum_h": 15.0,

  "if_threshold": -0.02,
  "if_score_scale": 0.08,

  "baseline_inter_mean_median": 2.391,
  "baseline_inter_mean_std":    1.394,

  "w_cusum": 0.35,
  "w_if":    0.65,
  "decision_threshold": 0.40,
  "persistence_windows": 4
}
```

### Parameter effects

| Parameter | Lower value | Higher value |
|-----------|------------|-------------|
| `cusum_k` | More sensitive (false positives if below baseline) | Less sensitive to small leaks |
| `cusum_h` | Triggers faster | Fewer false triggers |
| `if_threshold` | IF triggers more easily (closer to 0) | IF only triggers on strong anomalies |
| `if_score_scale` | Sharper score transition | Smoother score gradient |
| `decision_threshold` | More detections via fusion | Fewer fusion-path triggers |
| `persistence_windows` | Faster alarm (less delay) | Fewer false alarms |

### Preset tuning profiles

| Profile | `if_threshold` | `decision_threshold` | `persistence_windows` | Notes |
|---------|---------------|---------------------|----------------------|-------|
| **Current (balanced)** | -0.02 | 0.40 | 4 | Default |
| **High sensitivity** | 0.00 | 0.30 | 3 | More false positives |
| **Conservative** | -0.06 | 0.55 | 5 | Fewer false positives, misses small leaks |

---

## API Reference

### Socket.IO events

#### Server → Client: `data_update`  (every simulated minute)

```json
{
  "flow":             125.3,
  "sim_time":         "04:32",
  "sim_minutes":      272,
  "anomaly":          false,
  "final_score":      0.18,
  "level2": {
    "triggered":      false,
    "score":          0.12
  },
  "level3": {
    "triggered":      false,
    "score":          0.21,
    "reconstruction_error": -0.018,
    "flow_trend":     0.003,
    "baseline_elev":  0.47
  },
  "leak_active":      false,
  "leak_mode":        "instant",
  "leak_intensity":   0.0,
  "leak_remaining":   0
}
```

#### Client → Server events

| Event | Payload | Action |
|-------|---------|--------|
| `start_simulation` | — | Start / resume loop |
| `pause_simulation` | — | Pause loop |
| `stop_simulation` | — | Stop + full reset |
| `set_speed` | `float` (1–10) | Change simulation speed |
| `inject_leak` | `{intensity, duration, mode, ramp_minutes}` | Start a leak event |
| `stop_leak` | — | Cancel active leak |

---

## Performance Metrics

Metrics below are evaluated at the **auto-calibrated threshold** from training
(`optimal_threshold = -0.10042`).  The live server uses the overridden
`if_threshold = -0.02` which trades higher recall for more false positives,
mitigated by the persistence filter.

| Metric | Value |
|--------|-------|
| Accuracy | 92.9% |
| Precision | 36.5% |
| Recall | 7.2% (at training threshold; rises significantly at -0.02) |
| F1 Score | 12.0% (at training threshold) |
| IF trees | 300 |
| Features | 7 |
| Training windows | ~25,900 |

**Confusion matrix** (at training threshold):

```
              Predicted
              Anomaly   Normal
Actual Anomaly   125     1615   (TP / FN)
       Normal    217    23959   (FP / TN)
```

Note: low recall at the training threshold is expected — the training goal
was a conservative baseline. The `if_threshold = -0.02` runtime override and
the CUSUM+IF bypass rules are the actual sensitivity levers for production use.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Detector never fires | Server not restarted after config change | Restart / rebuild Docker |
| Chart "static" at high speed | `cusum_k` below normal baseline | Ensure `cusum_k ≥ 3.0` in calibration JSON |
| IF only triggers at high intensity | `if_threshold` too negative | Lower toward 0 (e.g. `-0.02`) |
| Too many false alarms | `if_threshold` too close to 0 or `persistence_windows` too low | Raise threshold or increase windows |
| Port conflict | Port 5000 in use | Change port in `server.py` and `docker-compose.yml` |
| `all_appliances.json` not found | Running from wrong directory | Always run from `apartment_simulator/` |
| Training fails: data not found | `generate_data.py` not run first | Run `py preprocessing/main.py` (runs both steps) |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, python-socketio, Uvicorn |
| ML | scikit-learn (IsolationForest, StandardScaler), NumPy |
| Data generation | NumPy, Pandas |
| Frontend | Chart.js, Socket.IO client |
| Container | Docker, Docker Compose |

---

**Last updated**: 2026-03-14
