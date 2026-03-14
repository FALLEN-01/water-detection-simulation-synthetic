# Apartment Building Model Training Pipeline

## Overview

The household Isolation Forest model was trained on **single-unit data** (100-160 L/day). Simply scaling thresholds doesn't account for the different feature distributions at building aggregate scale.

This pipeline generates **building-scale training data** and trains a dedicated Isolation Forest model optimized for 50-apartment buildings.

## Problem Statement

- **Household Model**: Trained on 1 unit, 0-15 L/min, 100-160 L/day
- **Apartment Building**: 50 units aggregated, 0-200 L/min, 5000-8000 L/day
- **Issue**: Feature distributions, anomaly patterns, and decision boundaries are completely different
- **Solution**: Generate synthetic building data and retrain

## Pipeline Steps

### 1. Generate Building-Scale Synthetic Data

**Script**: `generate_data.py`

Creates 6 months of synthetic water flow for 50-apartment building:
- **Training set**: 3 months of normal-only operation (no leaks)
- **Test set**: 3 months with 8 injected leak events

Leak patterns are realistic:
- **Stress-induced** (40%): Peak hours (7-8 AM, 7-9 PM)
- **Seasonal** (30%): Winter months
- **Night-time** (30%): Midnight-6 AM (longer duration)

**Output**:
```
preprocessing/data/
├── water_train_building.csv    # Normal-only training data
└── water_test_building.csv     # Test data with leaks
```

**Run**:
```bash
cd apartment_simulator
py preprocessing/generate_data.py
```

### 2. Train Isolation Forest Model

**Script**: `train_model.py`

Trains Isolation Forest on building aggregate data:
- Extracts 5-feature windows (mnf, inter_mean, inter_frac, mean_flow, inter_std)
- Auto-calibrates contamination parameter from leak frequency
- Evaluates on test set with metrics (accuracy, precision, recall, F1)
- Determines optimal decision thresholds

**Output**:
```
preprocessing/artifacts/
├── isolation_forest_building.pkl    # Trained model
├── scaler_building.pkl              # StandardScaler
├── calibration_building.json        # Optimal thresholds
└── metrics_building.json            # Performance metrics
```

**Run**:
```bash
cd apartment_simulator
py preprocessing/train_model.py
```

### 3. Run Complete Pipeline

**Script**: `main.py`

Orchestrates all steps (data generation → model training).

**Run**:
```bash
cd apartment_simulator
py preprocessing/main.py
```

## Model Calibration (Building Scale)

**calibration_building.json** contains:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `window_minutes` | 20 | Feature extraction window size |
| `cusum_k` | 0.5 | CUSUM sensitivity (10× household) |
| `cusum_h` | 20.0 | CUSUM trigger threshold |
| `appliance_flow_thresh` | 8.0 L/min | Appliance discriminator (10× household) |
| `noise_floor` | 0.2 L/min | Noise filtering threshold |
| `if_threshold` | (auto-calibrated) | Isolation Forest decision boundary |
| `if_score_scale` | (auto-calibrated) | IF score normalization |

## Deployment

After training:

1. **Copy building-scale models**:
   ```bash
   cp preprocessing/artifacts/isolation_forest_building.pkl artifacts/
   cp preprocessing/artifacts/scaler_building.pkl artifacts/
   cp preprocessing/artifacts/calibration_building.json artifacts/
   ```

2. **Update server.py** to use building models:
   ```python
   if_model = load_pickle(find_model_path("isolation_forest_building.pkl"))
   if_scaler = load_pickle(find_model_path("scaler_building.pkl"))
   cal = load_json(ARTIFACTS_DIR / "calibration_building.json")
   ```

3. **Restart simulator**:
   ```bash
   docker-compose restart
   # or
   py backend/server.py
   ```

## Expected Performance

Building-scale model should achieve:
- **Accuracy**: 85-95% (depends on leak intensity)
- **Precision**: 80-90% (few false positives)
- **Recall**: 75-85% (catches most leaks)
- **F1 Score**: 80-88%

Performance metrics saved to `preprocessing/artifacts/metrics_building.json`

## Feature Extraction (Building Scale)

Features extracted from 20-minute windows:

1. **mnf** (minimum normal flow): 10th percentile of non-zero flows
   - Indicates baseline consumption level
   - Higher for buildings vs households

2. **inter_mean** (inter-appliance mean): Average flow during non-peak periods
   - Baseline consumption pattern
   - Scales with building size

3. **inter_frac** (inter-appliance fraction): % of time in baseline range
   - Occupancy indicator
   - Similar across scales (~0.6-0.7)

4. **mean_flow** (window average): Total consumption
   - Primary discriminator
   - 10-50× higher for buildings

5. **inter_std** (inter-appliance std): Baseline variability
   - Indicates occupancy stability
   - Higher for buildings with more apartments

## Differences: Household Model vs Building Model

| Aspect | Household | Building |
|--------|-----------|----------|
| **Flow scale** | 0-15 L/min | 0-200 L/min |
| **Feature ranges** | ~[0.1, 1, 0.5, 2, 0.3] | ~[1, 8, 0.5, 80, 4] |
| **Anomaly threshold** | IF score -0.11 | (recalibrated) |
| **Training samples** | ~200K minutes | ~200K windows (3M minutes) |
| **Leak intensity** | 0.1-1.0 L/min | 2-10 L/min |
| **Detection window** | 20 min (household) | 20 min (building) |

## Troubleshooting

### Issue: Model accuracy is low
- Check if leak intensities are realistic (2-10 L/min for 50 apartments)
- Verify features are scaled correctly (appliance_flow_thresh=8.0)
- Increase training data (12 months instead of 6)

### Issue: Too many false positives
- Reduce `if_threshold` (make detection stricter)
- Increase `persistence_windows` (require 3+ consecutive anomalies)
- Increase `cusum_h` threshold

### Issue: Detection misses leaks
- Increase `if_score_scale` (less strict normalization)
- Lower `cusum_k` (more sensitive to baseline drift)
- Check if leak injection is working correctly

## Files Structure

```
apartment_simulator/
├── preprocessing/
│   ├── main.py              ← Run complete pipeline
│   ├── generate_data.py     ← Step 1: Generate synthetic data
│   ├── train_model.py       ← Step 2: Train Isolation Forest
│   ├── data/                ← Generated CSV datasets
│   ├── artifacts/           ← Output models & calibration
│   └── README.md            ← This file
│
├── backend/
│   ├── server.py            ← Uses if_calibration.json
│   ├── model.py             ← HybridWaterAnomalyDetector
│   └── live_simulator.py    ← 50-apartment generator
│
└── artifacts/
    ├── isolation_forest_building.pkl  ← (copy here after training)
    ├── scaler_building.pkl            ← (copy here after training)
    └── calibration_building.json      ← (copy here after training)
```

---

**Note**: Always keep backup of household models before retraining.
