## Apartment Simulator Visualization Suite

Real-time water flow visualization and anomaly detection analysis for the 50-unit building simulator.

### Overview

This directory contains four visualization scripts for analyzing building-level water consumption and anomaly detection performance:

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| **visualize.py** | Appliance activation patterns (6-panel plot) | `all_appliances.json` | `appliance_activation_patterns.png` |
| **sanity_check.py** | Flow statistics validation & distribution | CSV flow data | `sanity_check_analysis.png` |
| **flow_analysis.py** | Time-series with detection thresholds | CSV flow data | 3x PNG files (timeseries, inter-apt flow, distributions) |
| **anomaly_detection_viz.py** | Detection results: CUSUM, IF, Fusion scores | Detection output CSV | 3x PNG files (scores, overlay, performance metrics) |

### Quick Start

#### 1. Appliance Activation Patterns

View hourly activation probabilities for each appliance type across the building:

```bash
python visualize.py
```

**Output:** `figures/appliance_activation_patterns.png`

**Shows:**
- 6 subplots (Shower, Washing Machine, Bidet, Wash Basin, Kitchen Faucet, Toilet)
- Hourly activation probability curves
- Peak usage hours by appliance type

---

#### 2. Sanity Check (Flow Statistics)

Validate simulated flow data against expected building-level patterns:

```bash
python sanity_check.py ../preprocessing/artifacts/building_flow_365d.csv
```

**Parameters:**
- Optional: CSV file path (defaults to preprocessing/artifacts/building_flow_365d.csv)

**Output:** `figures/sanity_check_analysis.png` + console statistics

**Analyzes:**
- Daily usage distribution (L/day)
- Night baseline (0-5 AM) for leak detection validation
- Peak/night usage ratio
- Inter-appliance flow (< 400 L/min)
- Coefficient of variation (should be 0.05–0.15 for building)

**Example Output:**
```
=== DAILY WATER USAGE (L/day) ===
Mean      : 5857.32
Median    : 5678.40
Std Dev   :  456.20

=== NIGHT BASELINE (0–5 AM) ===
Median   : 12.451 L/min
95th pct : 28.345 L/min

=== INTER-APPLIANCE FLOW ===
Mean   : 120.45 L/min
Max    : 385.20 L/min
```

---

#### 3. Flow Analysis (Time-Series & Thresholds)

Detailed time-series visualization with anomaly detection thresholds overlaid:

```bash
python flow_analysis.py ../preprocessing/artifacts/building_flow_365d.csv
```

**Output:** 3 PNG files
- `flow_timeseries_thresholds.png` — Raw flow with thresholds
- `inter_appliance_flow_analysis.png` — Baseline drift detection
- `flow_distribution_analysis.png` — Statistical distributions

**Thresholds Shown:**
- **Appliance Threshold (400 L/min)**: Boundary between appliance/inter-apt flow
- **CUSUM Reference (150 L/min)**: 67th percentile of normal inter-apt flow
- **Baseline Mean (120 L/min)**: Expected inter-apt mean (training median × 50)

**Key Visualizations:**
1. Raw aggregate flow with threshold boundaries
2. Rolling inter-appliance baseline (20-min windows) vs expected
3. Anomaly deviation level (baseline drift)
4. Flow distributions (raw, inter-apt, log-scale)
5. Percentile curves

---

#### 4. Anomaly Detection Visualization

Analyze detection model performance (CUSUM, Isolation Forest, Fusion):

```bash
python anomaly_detection_viz.py detection_results.csv
```

**Expected CSV Columns:**
- `flow_lpm`: Aggregate flow
- `cusum_score`, `cusum_triggered`: CUSUM results
- `if_score`, `if_triggered`: Isolation Forest results
- `fusion_score`: Weighted combination (0–1)
- `candidate_anomaly`: Pre-persistence filter flag
- `anomaly_alarm`: Final detection (post-persistence)
- Optional: `leak_injected`, `leak_flow_lpm` (for validation)

**Output:** 3 PNG files (if detection data available)
- `detection_scores.png` — CUSUM, IF, Fusion scores over time
- `flow_with_detections.png` — Flow overlay with alarm markers
- `detection_performance.png` — Confusion matrix, metrics, ROC curve

**Console Output:**
```
DETECTION SUMMARY
CUSUM Triggers     : 1,234 windows
IF Triggers        :   856 windows
Candidate Anomalies:   342 windows
Final Alarms (4x)  :    15 events

Performance:
  Recall    : 0.920
  Precision : 0.867
  F1 Score  : 0.893
```

---

### Data Preparation

#### Generate Flow Data (50-apartment building)

```bash
cd ../preprocessing
python generate_data.py      # Generates building_flow_365d.csv
python train_model.py        # Trains detection models
```

#### Run Detection on Pre-Generated Data

```bash
cd ../backend
python server.py --batch-mode <csv_file>
```

This will output a detection results CSV with all anomaly scores.

---

### Dataset Specifications

**Building Configuration:**
- 50 independent apartments
- Appliance types: Shower, Washing Machine, Bidet, Wash Basin, Kitchen Faucet, Toilet
- Daily consumption per apartment: 100–160 L/day
- Aggregate building daily: 5,000–8,000 L/day

**Key Thresholds (50-apartment building):**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Appliance Threshold | 400 L/min | Flow >= this = appliance event |
| CUSUM Reference (k) | 150 L/min | 67th percentile of normal inter-apt |
| Baseline Mean | 120 L/min | Expected inter-apt baseline |
| CUSUM Trigger (h) | 400 | Accumulated slack threshold |
| IF Threshold | -0.02 | Anomaly gate (aggressive) |
| Decision Threshold | 0.40 | Fusion score gate |
| Persistence Windows | 4 | Min consecutive anomaly windows for alarm |
| Window Size | 20 min | Analysis window (1,200 flow samples) |

---

### Output File Locations

All visualizations are saved to: `apartment_simulator/visualization/figures/`

```
figures/
  ├── appliance_activation_patterns.png
  ├── sanity_check_analysis.png
  ├── flow_timeseries_thresholds.png
  ├── inter_appliance_flow_analysis.png
  ├── flow_distribution_analysis.png
  ├── detection_scores.png
  ├── flow_with_detections.png
  └── detection_performance.png
```

---

### Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

All scripts handle missing dependencies gracefully with informative error messages.

---

### Example Workflow

**Complete analysis pipeline:**

```bash
# 1. Generate simulation data
cd apartment_simulator/preprocessing
python generate_data.py

# 2. Train model
python train_model.py

# 3. Visualize appliance patterns
cd ../visualization
python visualize.py

# 4. Validate sanity check
python sanity_check.py ../preprocessing/artifacts/building_flow_365d.csv

# 5. Analyze flow patterns
python flow_analysis.py ../preprocessing/artifacts/building_flow_365d.csv

# 6. Run detection (if available)
# python anomaly_detection_viz.py /path/to/detection_results.csv
```

---

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: all_appliances.json` | Run from preprocessing first: `python generate_data.py` |
| No timestamp data | Scripts auto-detect and use sequential indexing |
| CSV doesn't have expected columns | Check format; scripts print required columns on error |
| Low detection performance | Verify calibration config in `artifacts/calibration_building.json` |
| Memory issues with large datasets | Process files in chunks; see pandas `chunksize` parameter |

---

### Contact & Support

For questions about visualization scripts or apartment simulator, see: `apartment_simulator/documentation/README.md`
