# Water Leak Detection System

Real-time, AI-powered water leak detection for household and apartment building water systems using hybrid anomaly detection (CUSUM + Isolation Forest).

## Overview

This project provides two complementary simulators for detecting water leaks:

1. **Household Simulator** - Single-unit real-time leak detection
   - Monitors individual household water flow
   - Real-time anomaly detection with web dashboard
   - Interactive leak injection testing

2. **Apartment Building Simulator** - Multi-unit building-level detection
   - Aggregates water flow from 50 independent apartments
   - Building-level anomaly detection and alerting
   - Real-time monitoring with web dashboard

Both systems use realistic, India-calibrated water usage patterns and hybrid detection algorithms.

## Problem Statement

Water leaks are challenging to detect because:
- Normal water usage varies by time of day, day of week, and season
- Small leaks can go unnoticed for days, especially during peak usage
- Occupancy fluctuates seasonally
- Real-time detection requires low latency and minimal false positives

## Solution

Both simulators use **Hybrid CUSUM + Isolation Forest** anomaly detection:
- CUSUM: Detects persistent low-flow deviations (gradual leaks)
- Isolation Forest: Detects statistical outliers
- Trained on normal data only (unsupervised)
- Real-time streaming with web-based dashboards

## Project Structure

### Household Simulator
```
household_simulator/
├── backend/
│   ├── server.py          # FastAPI server (port 5000)
│   ├── live_simulator.py  # Single-unit flow generator
│   ├── model.py           # Hybrid CUSUM + IF detector
│   └── isolation_forest.py
├── frontend/              # Web dashboard (index.html, app.js, style.css)
├── visualization/         # Analysis tools
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

### Apartment Building Simulator
```
apartment_simulator/
├── backend/
│   ├── server.py          # FastAPI + Socket.IO server (port 5000)
│   ├── live_simulator.py  # 50-apartment aggregation
│   └── model.py           # Hybrid CUSUM + IF detector
├── frontend/              # Web dashboard
├── preprocessing/         # Data pipeline
├── artifacts/             # Model calibration files
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.training
└── requirements.txt
```

## Detection Models

Both simulators use the same hybrid approach:

### Hybrid CUSUM + Isolation Forest

**CUSUM (Level 1: Change Detection)**
- Detects persistent deviations from baseline
- Sensitive to gradual/sustained leaks
- Parameters: k (detection threshold), h (alarm threshold)

**Isolation Forest (Level 2: Anomaly Detection)**
- Tree-based ensemble anomaly detector
- Detects statistical outliers in feature space
- Fast training and prediction
- Works well with high-dimensional data

**Feature Engineering (5-7 features per minute)**
- **Household**: flow_rate, duration, hour_of_day, day_of_week, inter-event patterns
- **Apartment**: minimum_flow, inter_appliance_mean, flow_fraction, mean_flow, inter_std, flow_trend, baseline_elevation

**Fusion & Thresholding**
- Weighted combination of CUSUM and IF scores
- Persistence filter: requires consecutive anomalies to reduce false positives
- Decision threshold tuned per environment

## Sensors

### Primary Sensor
- **Water Flow Rate** (L/min) - Main anomaly indicator
  - Household: 0-15 L/min range
  - Building: 0-750 L/min aggregate
  - Minute-by-minute sampling with Gaussian noise


## Installation & Quick Start

### Household Simulator

```bash
cd household_simulator
docker-compose up --build
# Visit: http://localhost:3000
```

### Apartment Building Simulator

```bash
cd apartment_simulator
docker-compose up --build
# Visit: http://localhost:3000
```

### Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run household simulator
python household_simulator/backend/server.py

# Run apartment simulator
python apartment_simulator/backend/server.py
```

## Requirements

- Python 3.8+
- NumPy, Pandas, scikit-learn
- TensorFlow (optional, for training)
- FastAPI, Python-socketio (for server)
- Matplotlib (for visualization)
- Docker & Docker Compose (recommended)

## Features

### Real-Time Monitoring
- **Live data streaming** to web dashboard
- **Minute-by-minute** anomaly scores
- **Historical trend** visualization
- **Leak injection** testing (manually trigger simulated leaks)

### Web Dashboard (Both Simulators)
- Flow rate trend chart
- Anomaly score visualization
- Real-time alerts
- Model parameters display
- Configurable thresholds

### Data Calibration
- **Source**: Real 617-day water usage data from India
- **Standards**: MoHUA and BIS specifications  
- **Fixtures**: Shower, toilet, bidet, washbasin, kitchen faucet, washing machine, dishwasher
- **Household daily volume**: 100-160 L (calibrated for Indian usage)
- **Building daily volume**: 5,000-8,000 L (50 apartments)

## Technical Configuration

### Household Simulator Parameters
- **CUSUM k**: 0.01 (detection threshold)
- **CUSUM h**: 2.0 (alarm threshold)
- **IF threshold**: -0.05 (anomaly sensitivity)
- **Persistence windows**: 2 (false-positive filter)
- **Decision threshold**: 0.65 (fusion score)

### Apartment Building Simulator Parameters
- **CUSUM k**: 3.0 (67th percentile of normal inter-appliance flow)
- **CUSUM h**: 8.0-15.0 (alarm threshold)
- **IF threshold**: -0.02 (aggressive, catches 2-5 L/min leaks)
- **Persistence windows**: 4 (stricter false-positive guard)
- **Decision threshold**: 0.40 (lower threshold for building-level detection)

## Workspace Structure

```
water-detection-simulation-synthetic/
├── household_simulator/          # Single-unit detection system
│   ├── backend/
│   │   ├── server.py            # FastAPI server
│   │   ├── live_simulator.py    # Real-time flow generator
│   │   ├── model.py             # Hybrid CUSUM + IF detector
│   │   └── isolation_forest.py
│   ├── frontend/                # Web dashboard
│   ├── visualization/           # Analysis & debugging
│   ├── artifacts/               # Calibration & trained models
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── requirements.txt
│
├── apartment_simulator/          # 50-unit building detection system
│   ├── backend/
│   │   ├── server.py            # FastAPI + Socket.IO server
│   │   ├── live_simulator.py    # 50-apartment flow aggregator
│   │   └── model.py             # Building-level detector
│   ├── frontend/                # Web dashboard
│   ├── preprocessing/           # Training data pipeline
│   ├── artifacts/               # Calibration & trained models
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── Dockerfile.training
│   └── requirements.txt
│
├── priors_india/                # Real-world usage patterns (JSON)
│   ├── shower.json
│   ├── toilet.json
│   ├── bidet.json
│   ├── washbasin.json
│   ├── kitchenfaucet.json
│   ├── washingmachine.json
│   └── dishwasher.json
│
├── README.md                    # This file
└── documentation/               # Additional guides
```

## How It Works

### Household Simulator Flow
1. **Data Generation**: Live flow generator creates realistic minute-by-minute household water usage
2. **Feature Extraction**: 5 features computed per window (flow rate, duration, time context, etc.)
3. **Detection**: Hybrid CUSUM + Isolation Forest scores aggregated
4. **Streaming**: Results pushed to web dashboard via WebSocket
5. **Visualization**: Real-time charts show flow, anomaly scores, and alerts

### Apartment Building Simulator Flow
1. **Multi-Unit Generation**: 50 independent household generators with unique random seeds
2. **Aggregation**: Flows combined to building-level (0-750 L/min)
3. **Feature Extraction**: 7 building-level features computed
4. **Detection**: Hybrid detector with building-specific thresholds
5. **Streaming**: Building manager dashboard receives aggregated data
6. **Alerting**: Anomalies trigger apartment-level backtracking to identify source

## Model Details

### Isolation Forest Component
- **Trees**: 200 (apartment) / configurable (household)
- **Max Samples**: Auto-computed
- **Contamination**: Auto-calibrated from training data
- **Scoring**: Normalized anomaly scores [-1, 1]

### CUSUM Component
- **Algorithm**: Cumulative Sum Control Chart
- **Detection**: Tracks cumulative deviation from baseline
- **Reset**: Partial resets on appliance events (prevents drift)
- **Output**: Continuous score fed to fusion layer

### Fusion & Decision
- **Weighted combination**: w_cusum × CUSUM + w_if × IF
- **Thresholding**: Alerts when fusion score > decision_threshold
- **Persistence filter**: Requires N consecutive flagged minutes

## Data Source & Calibration

Both simulators generate synthetic water usage based on real-world patterns from India:

### Priors Dataset (WEUSEDTO Data)
- **Source**: [WEUSEDTO-Data Repository](https://github.com/AnnaDiMauro/WEUSEDTO-Data)
- **Duration**: 617 days of real-world multi-apartment building monitoring
- **Coverage**: Shower, toilet, bidet, washbasin, kitchen faucet, washing machine, dishwasher
- **Standards**: Calibrated to MoHUA and BIS specifications

### Appliance Event Distributions
Each fixture has realistic:
- **Event frequency**: events/day (Poisson distribution)
- **Timing patterns**: Hourly probability distributions (morning peaks, evening peaks, etc.)
- **Duration**: Appliance-specific (fixed or lognormal)
- **Flow rates**: Lognormal distributions calibrated for Indian fixtures

## Running Tests

### Household Simulator Test
```bash
cd household_simulator
npm start  # or access via port 3000 after docker-compose up
# Use web dashboard to:
# - View real-time flow
# - Inject manual leaks
# - Monitor anomaly scores
```

### Apartment Building Test
```bash
cd apartment_simulator
npm start  # or access via port 3000 after docker-compose up
# Use web dashboard to:
# - Monitor 50-apartment aggregated flow
# - Test sensitivity to different leak sizes
# - View per-apartment backtracking
```

## License

Educational/demonstration purposes.

## Authors

Developed as a proof-of-concept for intelligent building water management systems.
