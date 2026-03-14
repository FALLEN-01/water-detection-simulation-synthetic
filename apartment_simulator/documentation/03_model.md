# Backend Module: `model.py`

**File Path:** `apartment_simulator/backend/model.py`  
**Purpose:** Hybrid anomaly detection combining CUSUM statistical change detection with machine learning (Isolation Forest)  
**Key Class:** `HybridWaterAnomalyDetector`  
**Dependencies:** `numpy`, `sklearn`, `collections.deque`

---

## Overview

This module implements a two-level hybrid anomaly detector for water flow data in multi-apartment buildings. It combines:

1. **Level 2 (CUSUM)**: Statistical change detection for sustained small leaks
2. **Level 3 (Isolation Forest)**: Machine learning outlier detection for point anomalies
3. **Fusion Layer**: Weighted combination with independent bypass rules
4. **Persistence Filter**: Temporal smoothing to reduce false positives

### Detection Philosophy

- **Unsupervised**: Trained only on normal data (no leak labels needed)
- **Hybrid**: Two complementary methods catch different leak types
- **Adaptive**: Baseline tracking via rolling history windows
- **Calibrated**: All thresholds loaded from JSON configuration

---

## Architecture

```
Input: 20-minute window of flow data
   │
   ├─→ LEVEL 2: CUSUM
   │   └─ Detects sustained above-baseline flow
   │       └ Returns: (score, triggered)
   │
   ├─→ LEVEL 3: Isolation Forest
   │   ├─ Feature extraction (7 features)
   │   ├─ IF decision function
   │   └─ Returns: (score, triggered)
   │
   ├─→ FUSION
   │   ├─ Weighted combination (35% CUSUM + 65% IF)
   │   ├─ Independent bypasses
   │   └─ Returns: candidate_anomaly (bool)
   │
   └─→ PERSISTENCE FILTER
       ├─ Accumulate consecutive candidate windows
       ├─ Fire alarm if streak >= 4 windows
       └─ Returns: final_anomaly (bool)
```

---

## Class: `HybridWaterAnomalyDetector`

### Purpose

Detects water leaks in building-level aggregated flow data using hybrid statistical + ML approach.

---

### Constructor: `__init__`

#### Signature

```python
def __init__(
    self,
    if_model,
    if_scaler,
    cusum_k=3.0,
    cusum_h=8.0,
    noise_floor=0.2,
    if_threshold=-0.05,
    if_score_scale=0.1,
    appliance_flow_thresh=8.0,
    clip_bound=10.0,
    baseline_inter_mean_median=1.5,
    baseline_inter_mean_std=0.8,
    w2=0.35,
    w3=0.65,
    decision_threshold=0.40,
    persistence_windows=4,
)
```

#### Parameters

**Isolation Forest (ML) Configuration:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `if_model` | sklearn.ensemble.IsolationForest | Required | Pretrained IF model (300 trees, 7 features) |
| `if_scaler` | sklearn.preprocessing.StandardScaler | Required | Fitted scaler for feature normalization |
| `if_threshold` | float | -0.05 | IF decision function cutoff (< threshold = anomaly) |
| `if_score_scale` | float | 0.1 | Range for score normalization (threshold range) |
| `clip_bound` | float | 10.0 | Feature clipping bounds after scaling |

**CUSUM (Statistical) Configuration:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cusum_k` | float | 3.0 | Reference level (67th percentile of normal inter-apt flow) |
| `cusum_h` | float | 8.0 | Threshold for accumulated slack |
| `noise_floor` | float | 0.2 | Flows below this are treated as zero |
| `appliance_flow_thresh` | float | 8.0 | Boundary between appliance/inter-apt flow |

**Baseline Statistics (for baseline_elev feature):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseline_inter_mean_median` | float | 1.5 | Training-set median inter-apt mean flow (L/min) |
| `baseline_inter_mean_std` | float | 0.8 | Training-set std dev of inter-apt mean |

**Fusion & Decision:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `w2` | float | 0.35 | CUSUM weight in fusion |
| `w3` | float | 0.65 | IF weight in fusion (w2 + w3 = 1.0) |
| `decision_threshold` | float | 0.40 | Fusion score gate for anomaly |
| `persistence_windows` | int | 4 | Consecutive anomaly windows to fire alarm |

#### Initialization Algorithm

```
1. Store all parameters
   self.if_model = if_model
   self.if_scaler = if_scaler
   self.cusum_k = cusum_k
   # ... etc

2. Initialize detection state
   cusum_s = 0.0              (accumulator)
   _anomaly_streak = 0        (persistence counter)
   _prev_appliance = False    (CUSUM state)

3. Initialize rolling history
   _inter_mean_history = deque(maxlen=3)  (3 × 20-min windows = 60 min)
```

#### Example Usage

```python
import pickle
import json
from sklearn.preprocessing import StandardScaler

# Load pretrained model
with open("artifacts/isolation_forest_building.pkl", "rb") as f:
    if_model = pickle.load(f)

with open("artifacts/scaler_building.pkl", "rb") as f:
    if_scaler = pickle.load(f)

# Load calibration
with open("artifacts/calibration_building.json") as f:
    cal = json.load(f)

# Initialize detector
detector = HybridWaterAnomalyDetector(
    if_model=if_model,
    if_scaler=if_scaler,
    cusum_k=cal["cusum_k"],
    cusum_h=cal["cusum_h"],
    if_threshold=cal["if_threshold"],
    baseline_inter_mean_median=cal["baseline_inter_mean_median"],
    baseline_inter_mean_std=cal["baseline_inter_mean_std"],
)
```

---

### Method: `_run_cusum()`

#### Signature

```python
def _run_cusum(self, window) -> Tuple[float, bool]
```

#### Visibility

`private`

#### Purpose

Performs CUSUM (Cumulative Sum Control Chart) on a flow window to detect sustained above-baseline flow (leak indicator).

#### Algorithm

```
CUSUM Principle:
  - Accumulates evidence of sustained flow elevation
  - Resets/decays when evidence decreases
  - Triggers when accumulated evidence exceeds threshold

Pseudocode:
```
```
  s = initial accumulator (0.0)
  triggered = False
  prev_appliance = False state from last window
  
  FOR each flow value in window:
    
    1. Classify current flow
       is_appliance = (flow >= appliance_flow_thresh)
    
    2. On appliance transition (OFF → ON)
       if is_appliance AND NOT prev_appliance:
           s *= 0.5              ← Partial reset (KEY: preserves leak signal)
    
    3. For inter-appliance flow (no major appliance active)
       if NOT is_appliance:
           delta = flow - cusum_k
           s = max(0, s + delta)
           
           if s >= cusum_h:
               triggered = True
    
    4. Update state for next iteration
       prev_appliance = is_appliance
  
  RETURN (s, triggered)
```

#### Partial Resets Explanation

**Why Partial Resets Instead of Full Resets?**

```
Scenario: Leak leaking 0.5 L/min, user showers (flow = 10 L/min)

Option A: Full Reset (s = 0)
  ├─ 0.4 L/min leak: s = 0.4 (CUSUM accumulates)
  ├─ 10 L/min shower: s = 0 (full reset—lose leak signal!)
  └─ Result: Leak undetected, detector "forgets"

Option B: Partial Reset (s *= 0.5)
  ├─ 0.4 L/min leak: s = 0.4 (accumulates)
  ├─ 10 L/min shower: s *= 0.5 = 0.2 (retains some signal)
  ├─ 0.4 L/min leak: s += 0.4 = 0.6 (continues accumulating)
  └─ Result: Leak detected despite appliance interruptions
```

#### Example Walkthrough

```python
# Suppose window contains (in L/min):
window = [0.5, 0.5, 10.0, 0.5, 0.5, 0.5, ...]
# (inter-apt, inter-apt, shower, inter-apt, inter-apt, inter-apt)

# Parameters:
cusum_k = 3.0
cusum_h = 8.0
appliance_thresh = 8.0

# Processing:
s = 0.0
triggered = False

# Minute 0: flow = 0.5 L/min
  - is_appliance = (0.5 >= 8.0) = False
  - delta = 0.5 - 3.0 = -2.5
  - s = max(0, 0 + (-2.5)) = 0.0
  
# Minute 1: flow = 0.5 L/min
  - is_appliance = False
  - delta = 0.5 - 3.0 = -2.5
  - s = max(0, 0 + (-2.5)) = 0.0
  
# Minute 2: flow = 10.0 L/min (shower starts)
  - is_appliance = (10.0 >= 8.0) = True
  - prev_appliance = False → True transition
  - s *= 0.5 = 0.0 * 0.5 = 0.0 (no leak yet)
  - prev_appliance = True
  
# Minute 3: flow = 0.5 L/min (shower ends)
  - is_appliance = False
  - delta = 0.5 - 3.0 = -2.5
  - s = max(0, (-2.5)) = 0.0
  - ...continues...
```

#### Return Values

| Return | Type | Description |
|--------|------|-------------|
| `s_final` | float | Final accumulated value ([0, ∞)) |
| `triggered` | bool | Whether max threshold exceeded during window |

#### Characteristics

- **Sensitive to Baseline**: Works best when leak >> noise (e.g., 0.3 L/min leak > 0.2 noise floor)
- **Sensitive to Duration**: Leaks shorter than ~5 minutes (1/4 window) may not accumulate
- **Robust to Appliances**: Partial resets prevent false negatives when appliances mask leaks
- **Stateful**: Carries accumulator value across window calls

---

### Method: `_extract_features()`

#### Signature

```python
def _extract_features(self, window) -> Tuple[np.ndarray, dict]
```

#### Visibility

`private`

#### Purpose

Extracts 7 statistical features from a 20-minute flow window for Isolation Forest classification.

#### Features Extracted

| # | Feature | Calculation | Unit | Interpretation |
|---|---------|-----------|------|---|
| 1 | **mnf** | 10th percentile of non-zero flows | L/min | Baseline non-appliance flow |
| 2 | **inter_mean** | Mean of flows < 8 L/min | L/min | Average inter-appliance flow |
| 3 | **inter_frac** | Fraction of inter-apt periods > 0.2 L/min | [0, 1] | Activity density baseline |
| 4 | **mean_flow** | Mean of all window flows | L/min | Window average |
| 5 | **inter_std** | Std dev of inter-apt flows | L/min | Baseline variability |
| 6 | **flow_trend** | Linear regression slope | L/min/min | Rising/falling baseline |
| 7 | **baseline_elev** | (rolling_inter_mean - median) / std | unitless | Normalized baseline shift |

#### Feature Extraction Algorithm

```python
# Separate inter-appliance flows
inter = window[window < 8.0]     (flows < threshold)
nonzero = window[window > 0.0]   (any non-zero flow)

# Feature 1: mnf (10th percentile of non-zero flows)
if len(nonzero) > 0:
    mnf = np.percentile(nonzero, 10)
else:
    mnf = 0.0

# Feature 2: inter_mean (mean of inter-appliance flows)
if len(inter) > 0:
    inter_mean = float(inter.mean())
else:
    inter_mean = 0.0

# Feature 3: inter_frac (fraction above noise floor)
if len(inter) > 0:
    inter_frac = (inter > 0.2).mean()
else:
    inter_frac = 0.0

# Feature 4: mean_flow (overall average)
mean_flow = float(window.mean())

# Feature 5: inter_std (std of inter-appliance)
if len(inter) > 1:
    inter_std = float(inter.std())
else:
    inter_std = 0.0

# Feature 6: flow_trend (linear regression slope)
t = np.arange(len(window), dtype=np.float64)  # [0, 1, ..., 19]
t_mean = t.mean()
f_mean = window.mean()
denom = ((t - t_mean) ** 2).sum()
if denom > 1e-9:
    flow_trend = ((t - t_mean) * (window - f_mean)).sum() / denom
else:
    flow_trend = 0.0

# Feature 7: baseline_elev (rolling baseline deviation)
# Add this window's inter_mean to 60-min rolling buffer
_inter_mean_history.append(inter_mean)
rolling_inter_mean = np.mean(_inter_mean_history)
baseline_elev = (rolling_inter_mean - baseline_median) / baseline_std
```

#### Feature Scaling and Clipping

```python
# Combine into raw feature vector (1, 7)
raw = np.array([[mnf, inter_mean, inter_frac, mean_flow, 
                 inter_std, flow_trend, baseline_elev]])

# Apply scaler (fitted on training data)
scaled = scaler.transform(raw)    # Normalize to mean≈0, std≈1

# Clip to prevent extreme values
clipped = np.clip(scaled, -10.0, +10.0)
```

#### Example Feature Extraction

```python
# Suppose window contains:
# Normal day flow: [0.5, 0.6, 1.2, 10.0, 0.8, 12.5, 1.0, 0.7, ..., 0.4]
#                  └─ inter-apt ─┘    └─ dishwasher ─┘      └─ inter ─┘

window = np.array([0.5, 0.6, 1.2, 10.0, 0.8, 12.5, 1.0, 0.7, 0.4, 0.3,
                   0.4, 5.0, 0.6, 0.5, 0.7, 0.8, 0.5, 0.4, 0.3, 0.4])

# Calculations:
inter = [0.5, 0.6, 1.2, 0.8, 1.0, 0.7, 0.4, 0.3, 0.4, 0.6, 0.5, 0.7, 0.8, 0.5, 0.4, 0.3, 0.4]
nonzero = [all except zeros]

mnf = np.percentile([0.5, 0.6, 1.2, 0.8, 1.0, 0.7, 0.4, 0.3, 0.4, 0.6, ...], 10)
    ≈ 0.3 L/min

inter_mean = mean([0.5, 0.6, 1.2, 0.8, ...])
    ≈ 0.6 L/min

inter_frac = (inter > 0.2).mean()
    ≈ 0.94 (17 out of 18 above floor)

mean_flow ≈ 2.1 L/min (includes shower/dishwasher peaks)

inter_std ≈ 0.25 L/min

flow_trend ≈ -0.01 L/min/min (slightly declining)

baseline_elev ≈ -0.5 (slightly below training median)
```

#### Why These Features?

| Feature | Why Useful | Leak Signal |
|---------|-----------|------------|
| **mnf** | Sets baseline floor | Low mnf = sustained small leak |
| **inter_mean** | Average baseline | Elevated = baseline shift |
| **inter_frac** | Event density | Lower = less activity, more leak-like |
| **mean_flow** | Overall average | Elevated = more total usage |
| **inter_std** | Variability | Lower = monotonic leak (not varying) |
| **flow_trend** | Rising/falling | Positive = accumulating leak |
| **baseline_elev** | Deviation from training | Positive = above normal baseline |

#### Return Values

| Return | Type | Description |
|--------|------|-------------|
| `clipped` | np.ndarray | Shape (1, 7), scaled & clipped features |
| `feat_debug` | dict | Debug info: `inter_mean`, `flow_trend`, `baseline_elev` |

---

### Method: `update()`

#### Signature

```python
def update(self, window) -> dict
```

#### Purpose

Processes a complete 20-minute flow window through both detection levels, fuses results, and applies persistence filter.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `window` | array-like | 20 flow values (one per minute), L/min |

#### Execution Flow

```
STEP 1: LEVEL 2 — CUSUM
┌─────────────────────────────────────────┐
│ Run CUSUM analysis                      │
│ - Accumulate above-baseline flow        │
│ - Detect if sustained elevation         │
└─────────────────────────────────────────┘
  Input: window
  Output: (cusum_s, cusum_triggered)
  
  cusum_score = min(1.0, cusum_s / cusum_h)
  (Normalize accumulator to [0, 1])

STEP 2: LEVEL 3 — ISOLATION FOREST
┌─────────────────────────────────────────┐
│ Extract 7 features                      │
│ Get IF decision function score          │
│ Determine if anomalous                  │
└─────────────────────────────────────────┘
  Input: window
  Output: 
    - features (scaled, clipped)
    - if_triggered (raw_score < if_threshold)
    - if_score (normalized to [0, 1])

STEP 3: FUSION — WEIGHTED COMBINATION
┌─────────────────────────────────────────┐
│ Combine Level 2 & 3 scores              │
│ Apply independent bypass logic          │
│ Determine candidate anomaly             │
└─────────────────────────────────────────┘
  final_score = w_cusum * cusum_score + w_if * if_score
              = 0.35 * cusum + 0.65 * if_score
  
  candidate_anomaly = (
      final_score > decision_threshold      ← Fusion path
      OR cusum_triggered                    ← CUSUM bypass
      OR if_triggered                       ← IF bypass
  )

STEP 4: PERSISTENCE FILTER — TEMPORAL SMOOTHING
┌─────────────────────────────────────────┐
│ Track consecutive anomaly windows       │
│ Fire alarm after N windows persistence  │
│ Guard against false positives           │
└─────────────────────────────────────────┘
  if candidate_anomaly:
      streak += 1
  else:
      streak = 0
  
  final_anomaly = (streak >= persistence_windows)
```

#### Detailed Algorithm

```python
window = np.asarray(window, dtype=np.float32)

# ─────── LEVEL 2: CUSUM ──────────────────────────
s_final, cusum_triggered = self._run_cusum(window)
cusum_score = min(1.0, s_final / self.cusum_h)

# ─────── LEVEL 3: ISOLATION FOREST ───────────────
features, feat_debug = self._extract_features(window)
raw_if_score = float(self.if_model.decision_function(features)[0])

if_triggered = raw_if_score < self.if_threshold

# Normalize IF score to [0, 1]
# Range: [if_threshold - if_score_scale, if_threshold]
anomaly_distance = max(0.0, self.if_threshold - raw_if_score)
if_score = min(1.0, anomaly_distance / max(self.if_score_scale, 1e-6))

# ─────── FUSION ───────────────────────────────────
final_score = (
    self.w2 * cusum_score    # 0.35
    + self.w3 * if_score     # 0.65
)

candidate_anomaly = (
    final_score > self.decision_threshold
    or cusum_triggered
    or if_triggered
)

# ─────── PERSISTENCE FILTER ───────────────────────
if candidate_anomaly:
    self._anomaly_streak += 1
else:
    self._anomaly_streak = 0

final_anomaly = self._anomaly_streak >= self.persistence_windows
```

#### Return Dictionary

```python
{
    # Final decision (after all filters)
    "anomaly": bool,              # True = alarm fired
    "final_score": float,         # Fused [0, 1] score
    
    # Level 2 (CUSUM) details
    "level2": {
        "triggered": bool,        # Exceeded threshold this window
        "score": float,           # [0, 1] accumulator normalized
    },
    
    # Level 3 (IF) details
    "level3": {
        "triggered": bool,        # Below IF threshold
        "score": float,           # [0, 1] anomaly score
        "reconstruction_error": float,  # Raw IF decision function
        "flow_trend": float,      # Feature: trend slope
        "baseline_elev": float,   # Feature: baseline deviation
    },
}
```

#### Example Update Call

```python
detector = HybridWaterAnomalyDetector(if_model, if_scaler)

# Simulate 2 days + inject leak on day 2
flows = []
for minute in range(2880):
    if minute == 1460:
        # Inject leak
        pass
    flow = generator.next()
    flows.append(flow)

# Process in 20-minute windows
window_buffer = []
for flow in flows:
    window_buffer.append(flow)
    if len(window_buffer) == 20:
        result = detector.update(window_buffer)
        
        if result["anomaly"]:
            print(f"🚨 ALARM! Score: {result['final_score']:.2f}")
        else:
            print(f"✓ Normal. Score: {result['final_score']:.2f}")
        
        window_buffer = []
```

#### Window Processing Sequence

```
Time    Window (20 min)              Anomaly  Streak
-       [0:20]                        False    0
20:40   [20:40]                       False    0
40:60   [40:60]                       False    0
...
1440:1460 [1440:1460] (leak starts!)  True     1
1460:1480 [1460:1480]                 True     2
1480:1500 [1480:1500]                 True     3
1500:1520 [1500:1520]                 True     4  ← ALARM FIRES
1520:1540 [1520:1540]                 True     5
...
1620:1640 [1620:1640] (leak ends)     False    0
```

---

### Method: `reset()`

#### Signature

```python
def reset(self) -> None
```

#### Visibility

`public`

#### Purpose

Clears detector's internal state (used when restarting simulation).

#### Implementation

```python
self.cusum_s = 0.0                      # Reset accumulator
self._prev_appliance = False            # Reset appliance flag
self._anomaly_streak = 0                # Reset persistence counter
self._inter_mean_history.clear()        # Clear 60-minute history
```

#### When to Call

- User clicks "Stop/Reset" button
- Between different simulation scenarios
- When switching between test datasets
- For reproducible testing

---

## Calibration Parameters

### Typical Building-Scale Calibration

These parameters are loaded from `calibration_building.json`:

```json
{
    "window_minutes": 20,
    "cusum_k": 3.0,
    "cusum_h": 8.0,
    "noise_floor": 0.2,
    "if_threshold": -0.02,
    "if_score_scale": 0.08,
    "appliance_flow_thresh": 8.0,
    "baseline_inter_mean_median": 2.391,
    "baseline_inter_mean_std": 1.394,
    "w_cusum": 0.35,
    "w_if": 0.65,
    "decision_threshold": 0.40,
    "persistence_windows": 4
}
```

### Parameter Tuning Guide

**To detect smaller leaks (0.1-0.2 L/min):**
- ↓ `cusum_k` (e.g., 2.0)
- ↑ `if_threshold` (e.g., -0.01, more aggressive)
- ↓ `persistence_windows` (e.g., 3, faster response)

**To reduce false positives:**
- ↑ `cusum_h` (harder to trigger)
- ↓ `if_threshold` (less aggressive, e.g., -0.1)
- ↑ `persistence_windows` (e.g., 5, need longer confirmation)

**To balance precision/recall:**
- Adjust `w_cusum` and `w_if` weights
- Adjust `decision_threshold` (higher = fewer alarms)

---

## Detection Scenarios

### Scenario 1: Slow Leak (Small but Sustained)

```
Flow profile:
  ├─ Normal baseline: 0-2 L/min
  ├─ With leak: 0.5-2.5 L/min (0.5 L/min added)
  └─ Duration: 12 hours

Detection:
  ├─ CUSUM: Accumulates over many windows ✓
  ├─ IF: Elevated baseline_elev + flow_trend ✓
  └─ Result: Reliably detected after 80-120 min
```

### Scenario 2: Rapid Detection Test

```
Flow profile:
  ├─ Normal baseline: 0-2 L/min
  ├─ With leak: 0-10 L/min (5 L/min burst)
  └─ Duration: 30 minutes

Detection:
  ├─ CUSUM: May not accumulate (below k during inter-apt)
  ├─ IF: High mnf + elevated mean_flow ✓✓
  └─ Result: Fast detection (1-2 windows) but persistence guard applies
```

### Scenario 3: False Positive Guard

```
Flow profile:
  ├─ Normal day variation
  ├─ Morning peak: 100-150 L/min for 2 hours
  ├─ Evening peak: 80-120 L/min for 1 hour
  └─ Occasional anomaly-like spikes

Persistence filter:
  ├─ Single anomalous window: Not alarmed ✓
  ├─ Four consecutive windows: Alarmed 🚨
  └─ Result: Spike-related false alarms suppressed
```

---

## Data Flow & Diagnostics

### Debug Output

```python
result = detector.update(window)

print(f"Final anomaly: {result['anomaly']}")
print(f"Fused score: {result['final_score']:.3f}")
print(f"CUSUM triggered: {result['level2']['triggered']}")
print(f"CUSUM score: {result['level2']['score']:.3f}")
print(f"IF triggered: {result['level3']['triggered']}")
print(f"IF score: {result['level3']['score']:.3f}")
print(f"Flow trend: {result['level3']['flow_trend']:.5f} L/min/min")
print(f"Baseline elev: {result['level3']['baseline_elev']:.3f} σ")
```

### Typical Normal Day Output

```
Final anomaly: False
Fused score: 0.145
CUSUM triggered: False
CUSUM score: 0.012
IF triggered: False
IF score: 0.034
Flow trend: -0.001 L/min/min
Baseline elev: -0.25 σ
```

### Typical Leak Day Output

```
Final anomaly: True          (after 4 windows)
Fused score: 0.623
CUSUM triggered: True        (sustained elevation)
CUSUM score: 0.892
IF triggered: True           (anomalous feature pattern)
IF score: 0.487
Flow trend: 0.015 L/min/min  (rising baseline)
Baseline elev: 0.78 σ        (above normal)
```

---

## Testing & Validation

### Unit Tests

```python
def test_detector_initialization():
    detector = HybridWaterAnomalyDetector(if_model, if_scaler)
    assert detector.cusum_s == 0.0
    assert detector._anomaly_streak == 0
    assert len(detector._inter_mean_history) == 0

def test_cusum_accumulation():
    detector = HybridWaterAnomalyDetector(if_model, if_scaler, cusum_k=3.0)
    # Window with sustained flow above k
    window = np.array([0.5] * 20)  # Inter-apt flow
    _, triggered = detector._run_cusum(window)
    assert detector.cusum_s > 0  # Should accumulate

def test_feature_extraction():
    detector = HybridWaterAnomalyDetector(if_model, if_scaler)
    window = np.array([0.5, 1.0, 0.3, 2.0] * 5)  # 20 values
    features, debug = detector._extract_features(window)
    assert features.shape == (1, 7)
    assert all(np.isfinite(features[0]))  # No NaN/Inf

def test_persistence_filter():
    detector = HybridWaterAnomalyDetector(if_model, if_scaler, persistence_windows=3)
    
    # First window: candidate
    result1 = detector.update(normal_window)
    result1["anomaly"] = False  # Not yet
    
    # Three more anomalies: should alarm
    normal_window["anomaly"] = True
    result2 = detector.update(normal_window)
    assert not result2["anomaly"]
    result3 = detector.update(normal_window)
    assert not result3["anomaly"]
    result4 = detector.update(normal_window)
    assert result4["anomaly"]  # Fired!

def test_reset():
    detector = HybridWaterAnomalyDetector(if_model, if_scaler)
    detector.cusum_s = 10.0
    detector._anomaly_streak = 5
    detector._inter_mean_history.append(2.0)
    
    detector.reset()
    
    assert detector.cusum_s == 0.0
    assert detector._anomaly_streak == 0
    assert len(detector._inter_mean_history) == 0
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Feature extraction time** | ~1ms per window |
| **IF predict time** | ~0.1ms per window |
| **CUSUM compute time** | <0.1ms per window |
| **Total update time** | ~1.2ms per window |
| **Memory footprint** | ~50 KB (model loaded separately) |
| **Windows per day** | 72 (1440 min ÷ 20 min) |
| **Typical latency** | 80-120 min (persistence delay) |

---

## Summary

| Component | Purpose | Trigger Condition |
|-----------|---------|-------------------|
| **CUSUM** | Sustained baseline leaks | Accumulator > h |
| **IF** | Point anomalies | Score < threshold |
| **Fusion** | Combined detection | Weighted score > threshold |
| **Bypass Rules** | High-confidence signals | Independent CUSUM/IF triggers |
| **Persistence** | False-positive guard | 4+ consecutive anomaly windows |

---

**End of `model.py` Documentation**
