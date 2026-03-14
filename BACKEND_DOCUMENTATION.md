# Apartment Simulator Backend - Detailed Function Documentation

**Last Updated**: March 2026  
**Project**: Water Leak Detection System - Apartment Building Simulator  
**Module**: `apartment_simulator/backend`

---

## Table of Contents

1. [Overview](#overview)
2. [Module: `__init__.py`](#module-initpy)
3. [Module: `live_simulator.py`](#module-live_simulatorpy)
   - [Class: LiveWaterFlowGenerator](#class-livewaterflowgenerator)
   - [Class: LiveApartmentBuildingDataGenerator](#class-liveapartmentbuildingdatagenerator)
4. [Module: `model.py`](#module-modelpy)
   - [Class: HybridWaterAnomalyDetector](#class-hybridwateranomalydetector)
5. [Module: `server.py`](#module-serverpy)
   - [Setup & Configuration](#setup--configuration)
   - [FastAPI Routes](#fastapi-routes)
   - [Socket.IO Event Handlers](#socketio-event-handlers)
   - [Main Simulation Loop](#main-simulation-loop)

---

## Overview

The apartment simulator backend provides real-time water flow simulation for a 50-unit apartment building with integrated anomaly detection. It implements:

- **Live water flow generation**: Minute-by-minute simulation of realistic water consumption across 50 independent apartments
- **Hybrid anomaly detection**: Combines CUSUM statistical analysis with machine learning (Isolation Forest) for leak detection
- **Real-time API**: FastAPI web server with WebSocket streaming to frontend dashboard
- **Leak injection**: Support for synthetic leak scenarios in instant or ramped modes

**Architecture Flow:**
```
LiveApartmentBuildingDataGenerator (50 apartments)
  └─> Generates minute-by-minute aggregate flow
       └─> Server collects 20-minute windows
           └─> HybridWaterAnomalyDetector processes window
               └─> Results streamed to frontend via Socket.IO
```

---

## Module: `__init__.py`

### Description
Package initialization file that marks the `backend` directory as a Python package.

### Content
```python
# Backend package initialization
```

**Purpose:** Enables the backend directory to be imported as a Python module.

---

## Module: `live_simulator.py`

### Overview
Implements realistic water flow simulation for individual apartments and multi-apartment buildings. Each apartment is modeled as an independent generator with its own random seed, appliance event scheduling, and leak injection capabilities.

---

### Class: `LiveWaterFlowGenerator`

**Purpose:** Simulates minute-by-minute water flow for a single apartment using realistic appliance event generation based on India-specific usage patterns.

#### Constructor: `__init__(...)`

```python
def __init__(
    self,
    priors_path,
    daily_min_l=DEFAULT_DAILY_MIN_L,
    daily_max_l=DEFAULT_DAILY_MAX_L,
    max_flow_lpm=15.0,
    noise_sigma=0.03,
    max_regen_attempts=10,
    seed=None,
)
```

**Parameters:**
- `priors_path` (str): Path to `all_appliances.json` containing appliance configurations (event frequency, flow rates, timing patterns, duration constraints)
- `daily_min_l` (float): Minimum daily water consumption per apartment in liters (default: 100 L)
- `daily_max_l` (float): Maximum daily water consumption per apartment in liters (default: 160 L)
- `max_flow_lpm` (float): Maximum instantaneous flow rate in liters per minute (default: 15.0 L/min)
- `noise_sigma` (float): Standard deviation of Gaussian noise as proportion of flow (default: 0.03 = 3%)
- `max_regen_attempts` (int): Maximum attempts to regenerate a day's events if volume constraints violated (default: 10)
- `seed` (int): Random seed for reproducibility; used to set `np.random.seed()` if provided

**Initialization Process:**
1. Sets random seed if provided
2. Generates random washing machine offset (0-6 days) for weekly cycle variance
3. Loads appliance priors from JSON file
4. Stores configuration parameters
5. Initializes state variables (`current_day=0`, `current_minute=0`, `day_flow=None`, `injected_leak=None`)
6. Calls `_generate_new_day()` to create first day's schedule

**Key Pattern:** Uses state machine pattern with minute-level granularity — each call to `next()` advances one minute.

---

#### Method: `next()`

```python
def next(self) -> float
```

**Purpose:** Returns the next minute's flow in liters per minute (L/min).

**Behavior:**
1. If current minute exceeds 1440 (minutes per day):
   - Increment day counter
   - Reset minute counter to 0
   - Generate new day's schedule via `_generate_new_day()`

2. Retrieve flow value from pre-generated day array at current minute index

3. **Apply injected leak** (if active):
   - Check if global minute is within leak window [leak_start, leak_end)
   - If within window: add `leak_flow_lpm` to flow value
   - Leak is additive to normal appliance flow

4. **Apply sensor noise** (only when flow > 0):
   - Generate Gaussian noise: `noise ~ N(0, noise_sigma * flow)`
   - Add noise to flow (clamped to non-negative)
   - Ensures realistic measurement variability without negative values

5. Increment current minute counter

6. Return flow as float

**Example Usage:**
```python
gen = LiveWaterFlowGenerator("priors.json", seed=42)
for i in range(1440):
    flow_lpm = gen.next()  # Get next minute's flow
    print(f"Minute {i}: {flow_lpm:.2f} L/min")
```

**Return Type:** `float` - Flow in L/min for current minute

---

#### Method: `inject_leak(duration_minutes=180, flow_lpm=0.4)`

```python
def inject_leak(self, duration_minutes=180, flow_lpm=0.4) -> None
```

**Purpose:** Injects a synthetic water leak into the apartment starting immediately.

**Parameters:**
- `duration_minutes` (int): How long the leak persists in minutes (default: 180 = 3 hours)
- `flow_lpm` (float): Leak flow rate in L/min (default: 0.4 L/min, typical small drip)

**Behavior:**
1. Calculate start minute: `current_global_minute`
2. Calculate end minute: `start + duration_minutes`
3. Store leak configuration as dictionary: `{"start": start, "end": end, "flow_lpm": flow_lpm}`

**Notes:**
- Leak is additive to normal appliance flow (not replacement)
- Once injected, leak persists until `clear_leak()` is called or end time is reached
- Multiple calls overwrite previous leak (no queueing)
- Useful for testing anomaly detector response to known leak scenarios

**Example:**
```python
gen.inject_leak(duration_minutes=240, flow_lpm=0.6)  # 4-hour leak at 0.6 L/min
```

---

#### Method: `clear_leak()`

```python
def clear_leak(self) -> None
```

**Purpose:** Removes any active or pending leak injection.

**Behavior:**
- Sets `self.injected_leak = None`
- Flow returns to baseline (no leak addition)

**Idempotent:** Safe to call even if no leak is active.

---

#### Method: `global_minute() → int`

```python
def global_minute(self) -> int
```

**Purpose:** Returns the current global simulation minute (day-independent counter).

**Calculation:** `global_minute = current_day * MINUTES_PER_DAY + current_minute`

**Use Case:** Provides consistent timeline reference for leak injection timing and logging.

**Range:** Grows unbounded as simulation continues (0 at start, increases indefinitely).

---

#### Method: `_generate_new_day()`

```python
def _generate_new_day(self) -> None
```

**Purpose:** Generates a complete day's worth of appliance events and renders flow for the day.

**Algorithm:**
1. **Regeneration Loop** (up to `max_regen_attempts` iterations):
   - Clear event list
   - For each appliance in priors:
     - Call `_generate_events_for_day()` to create events for that appliance
     - Append to master event list
   
   - Call `_render_day()` to convert events to flow array (1440 values)
   
   - Calculate total daily volume from flow array (`sum(day_flow)`)
   
   - **Validation Check:**
     - If volume is within `[daily_min_l, daily_max_l]`: Break from loop (success)
     - If not and attempts remain: Continue loop (regenerate)
     - If max attempts reached: Issue warning but accept anyway
   
   - Store final `day_flow` array

2. Store generated flow in `self.day_flow`

**Constraints:** 
- Ensures generated days are realistic (within specified volume bounds)
- Warnings issued for pathological cases (>10 failed attempts)
- Day regeneration preserves washing machine weekly cycle (via `wm_offset`)

**State Modified:** `self.day_flow` (and internal PRNG state via appliance event generation)

---

#### Method: `_generate_events_for_day(appliance, day) → List[dict]`

```python
def _generate_events_for_day(self, appliance, day) -> List[dict]
```

**Purpose:** Generates all appliance usage events for a single appliance on a specified day.

**Parameters:**
- `appliance` (dict): Appliance configuration from JSON priors (contains event rates, timing, flow, duration info)
- `day` (int): Day number (0-indexed)

**Appliance-Specific Logic:**

1. **Event Count Generation:**
   - **Shower**: `max(1, Poisson(lambda))` — At least 1 daily
   - **Toilet**: `max(2, Poisson(lambda))` — At least 2 daily  
   - **Bidet**: `Poisson(lambda)` — 0 or more
   - **Washing Machine**: `1 if (day + wm_offset) % 7 == 0 else 0` — Weekly cycle, offset for variety
   - **Others** (bidet, kitchenfaucet, dishwasher): `Poisson(lambda)`

2. **For Each Event:**
   
   - **Start Time:**
     - Sample start hour from hour probability distribution: `hour ~ Categorical(hour_probs)`
     - Add random minute offset (0-59)
     - Convert to absolute minute: `start_min = day * 1440 + hour * 60 + rnd_minute`
   
   - **Duration:**
     - If appliance has `fixed` duration type: Use exact value
     - If appliance supports `normal` distribution (washbasin, kitchenfaucet, etc.):
       - Generate: `N(scale, 0.25*scale)` clipped to [min, max]
     - Otherwise (showers, washing machine):
       - Generate: `Lognormal(log(scale), shape)` clipped to [min, max]
   
   - **Flow Rate:**
     - Generate: `Lognormal(log(scale), shape)` from appliance flow config
     - Returns mean flow in ml/s
   
   - **Shape:**
     - Appliance shape type (trapezoid, pulsed, or step)
     - Configuration parameters (ramp times, pulse patterns)
   
   - Append event dict to list: `{"start_min", "duration_s", "mean_flow_ml_s", "shape", "shape_cfg"}`

3. **Return:** List of event dicts for this appliance on this day

**Notes:**
- Uses Poisson distributions for realistic event frequency variance
- Lognormal distributions for flow and duration match real-world observations
- Clipping ensures physical constraints (min/max times and flows) are respected
- Hour probability distributions encode peak usage times

**Example Output:**
```python
[
  {
    "start_min": 480,           # 8:00 AM on day 0
    "duration_s": 480.5,        # ~8 minutes
    "mean_flow_ml_s": 18.3,     # ~18.3 ml/s
    "shape": "trapezoid",
    "shape_cfg": {"ramp_up_s": 5, "ramp_down_s": 5}
  },
  # ... more events
]
```

---

#### Method: `_render_day(events, day) → np.ndarray`

```python
def _render_day(self, events, day) -> np.ndarray
```

**Purpose:** Converts a list of appliance events into a 1440-element flow array (one value per minute).

**Parameters:**
- `events` (list): List of event dicts from `_generate_events_for_day()`
- `day` (int): Day number (used to extract day-relative times from global event times)

**Algorithm:**
1. Initialize `flow` array: `np.zeros(1440)` (one entry per minute of day)

2. For each event:
   - Calculate day-relative start: `start = event["start_min"] - day * 1440`
   - Calculate duration in minutes: `dur = ceil(duration_s / 60)`
   - Calculate flow in L/min: `lpm = mean_flow_ml_s * 60 / 1000`
   - Calculate end minute: `end = min(start + dur, 1440)` (cap at day boundary)
   - Calculate actual duration: `actual_dur = end - start`
   
   - If `actual_dur <= 0`: Skip event (event is outside this day)
   
   - Generate flow shape curve via `_make_shape_curve()` for normalized profile
   
   - Add shaped flow to array: `flow[start:end] += lpm * curve`
   
   - Clip to max flow: `flow[start:end] = min(flow[start:end], MAX_FLOW_LPM)`

3. Return `flow` array (shape: (1440,))

**Output:** Array of 1440 float values (one per minute), cumulatively summed flows when events overlap.

---

#### Method: `_make_shape_curve(shape, shape_cfg, dur) → np.ndarray`

```python
def _make_shape_curve(self, shape, shape_cfg, dur) -> np.ndarray
```

**Purpose:** Generates a normalized flow profile curve for realistic appliance flow shapes.

**Parameters:**
- `shape` (str): Type of flow shape — `"trapezoid"`, `"pulsed"`, or default `"step"`
- `shape_cfg` (dict): Configuration parameters (ramp times for trapezoid, pulse pattern for pulsed)
- `dur` (int): Duration in minutes

**Shape Types:**

**1. Trapezoid Shape** (showers, washing machines):
```
Flow
  ^
  |   ___________
  |  /           \
  |_/             \___
  └────────────────────> Time

Ramp up: Gradual increase (e.g., 5 seconds)
Plateau: Constant flow
Ramp down: Gradual decrease (e.g., 5 seconds)
```
- Extract ramp times: `ramp_up_s`, `ramp_down_s` from config
- Convert to minute bins
- Build curve: `[linspace(0.5, 1.0, ramp_bins), ones(plateau_bins), linspace(1.0, 0.5, fall_bins)]`
- Normalize to mean=1.0 for consistent amplitude

**2. Pulsed Shape** (dishwasher spray cycles):
```
Flow
  ^
  |  _   _   _
  | | | | | | |
  |_|_|_|_|_|_|___
  └────────────────> Time

Pattern: 2 minutes on, 2 minutes off
```
- Create pulse pattern: `(arange(dur) % 4 < 2)`
- Return normalized to mean=1.0

**3. Step Shape** (default/fallback):
- Return all ones: `np.ones(dur)`

**Normalization:** Divide by `curve.mean()` to ensure energy conservation (total flow units stay consistent regardless of shape).

**Return:** Normalized curve array (length=`dur`, mean≈1.0)

---

### Class: `LiveApartmentBuildingDataGenerator`

**Purpose:** Orchestrates 50 independent apartment generators and aggregates their flows into building-level water consumption data.

#### Constructor: `__init__(...)`

```python
def __init__(
    self,
    priors_path,
    num_apartments=50,
    daily_min_l=None,
    daily_max_l=None,
    max_flow_lpm=None,
    noise_sigma=0.03,
    seed=None,
)
```

**Parameters:**
- `priors_path` (str): Path to appliance JSON file
- `num_apartments` (int): Number of apartments in building (default: 50)
- `daily_min_l` (float): Min daily building consumption (default: `100 * 50 = 5000` L)
- `daily_max_l` (float): Max daily building consumption (default: `160 * 50 = 8000` L)
- `max_flow_lpm` (float): Max building-level flow (default: `15 * 50 = 750` L/min)
- `noise_sigma` (float): Per-apartment noise std dev (default: 0.03)
- `seed` (int): Master random seed; each apartment gets `seed + apartment_index`

**Initialization:**
1. Store parameters
2. Set per-apartment constraints (100-160 L/day, 15 L/min max)
3. Set building constraints (5000-8000 L/day, 750 L/min max)
4. **Create 50 independent generators** with unique seeds:
   ```python
   for i in range(num_apartments):
       apt_seed = (seed + i) if seed else None
       gen = LiveWaterFlowGenerator(..., seed=apt_seed)
       self.generators.append(gen)
   ```
5. Initialize leak tracking and minute counter

**Design:** Each apartment is fully independent with its own random sequence, enabling realistic variance in timing and volume across the building.

---

#### Method: `next() → float`

```python
def next(self) -> float
```

**Purpose:** Returns the next minute's aggregated building-level flow.

**Algorithm:**
1. **Sum apartment flows:**
   ```python
   flows = [gen.next() for gen in self.generators]
   aggregated_flow = sum(flows)
   ```

2. **Clip to building maximum:** `aggregated_flow = min(aggregated_flow, MAX_FLOW_LPM)`

3. **Apply building-level leak** (if active):
   - Check if `injected_leak` is set and current minute is in leak window
   - If yes: `aggregated_flow += leak_flow_lpm`
   - Clip again to max: `aggregated_flow = min(aggregated_flow, MAX_FLOW_LPM)`

4. Increment `current_minute`

5. Return aggregated flow as float

**Semantics:** Building leak is independent of apartment-level events — it adds to whatever normal usage occurs.

---

#### Method: `inject_leak(duration_minutes=180, flow_lpm=0.4) → None`

```python
def inject_leak(self, duration_minutes=180, flow_lpm=0.4) -> None
```

**Purpose:** Injects a building-level leak (e.g., main pipe, common area).

**Parameters:**
- `duration_minutes` (int): Leak duration (default: 180 minutes = 3 hours)
- `flow_lpm` (float): Leak flow rate (default: 0.4 L/min)

**Behavior:**
- Store leak config: `{"start": current_minute, "end": current_minute + duration, "flow_lpm": flow_lpm}`
- Leak is active from current minute until end minute

**Difference from apartment-level:** Building leak applies to aggregate data, useful for testing detector on infrastructure failures.

---

#### Method: `clear_leak() → None`

```python
def clear_leak(self) -> None
```

**Purpose:** Stops any active building-level leak.

**Behavior:** Sets `self.injected_leak = None`

---

#### Method: `reset() → None`

```python
def reset(self) -> None
```

**Purpose:** Resets simulator to initial state (used when restarting simulation).

**Behavior:**
1. For each apartment generator:
   - Reset to day 0, minute 0
   - Clear injected leak
   - Regenerate day 0

2. Reset building state:
   - `current_minute = 0`
   - `injected_leak = None`

**Use Case:** Called when user clicks "Stop/Reset" in dashboard to start fresh simulation.

---

#### Method: `global_minute() → int`

```python
def global_minute(self) -> int
```

**Purpose:** Returns current global minute (building-level time).

**Return:** `self.current_minute`

---

## Module: `model.py`

### Overview
Implements `HybridWaterAnomalyDetector`, a two-level anomaly detection system combining CUSUM statistical change detection with machine learning (Isolation Forest).

**Detection Levels:**
- **Level 2 (CUSUM):** Catches sustained small leaks by accumulating evidence of above-baseline flow
- **Level 3 (Isolation Forest):** Catches point anomalies through statistical outlier detection
- **Fusion:** Weighted combination with independent bypass rules for high-confidence anomalies

---

### Class: `HybridWaterAnomalyDetector`

#### Constructor: `__init__(...)`

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

**Parameters:**

**Isolation Forest Configuration:**
- `if_model`: Pre-trained Isolation Forest model (sklearn) with 7 features, 300 trees
- `if_scaler`: StandardScaler fitted on training data (normalizes features to mean=0, std=1)

**CUSUM Configuration:**
- `cusum_k` (float): Reference level for CUSUM — typically set above normal baseline flow (~3.0 L/min for inter-appliance flow in 50-apt building)
- `cusum_h` (float): Trigger threshold for accumulated slack (default: 8.0)
- `noise_floor` (float): Flow below this (default: 0.2 L/min) treated as zero
- `appliance_flow_thresh` (float): Flow >= this (default: 8.0 L/min) classified as appliance event, excluded from CUSUM

**Isolation Forest Tuning:**
- `if_threshold` (float): Decision function cutoff (default: -0.05); raw score < this = anomaly
- `if_score_scale` (float): Range over which normalized IF score transitions 0→1 (default: 0.1)
- `clip_bound` (float): Clip scaled features to [-bound, +bound] before IF (default: 10.0)

**Baseline Statistics** (loaded from training data):
- `baseline_inter_mean_median` (float): Training-set median of inter-appliance mean flow (default: 1.5 L/min)
- `baseline_inter_mean_std` (float): Training-set std dev of inter-appliance mean (default: 0.8 L/min)

**Fusion Configuration:**
- `w2` (float): CUSUM weight in fusion (default: 0.35)
- `w3` (float): IF weight in fusion (default: 0.65); weights sum to 1.0
- `decision_threshold` (float): Fused score gate (default: 0.40); fusion path triggers if score > threshold
- `persistence_windows` (int): Consecutive candidate-anomaly windows before alarm fires (default: 4)

**Initialization:**
1. Store all parameters
2. Initialize state variables:
   - `cusum_s = 0.0` (CUSUM accumulator)
   - `_anomaly_streak = 0` (consecutive anomaly window counter)
   - `_prev_appliance = False` (previous minute appliance flag)
   - `_inter_mean_history = deque(maxlen=3)` (rolling 60-minute inter-appliance history)

---

#### Method: `_run_cusum(window) → Tuple[float, bool]`

```python
def _run_cusum(self, window) -> Tuple[float, bool]
```

**Purpose:** Performs CUSUM (Cumulative Sum Control Chart) on a window to detect sustained above-baseline flow.

**Parameters:**
- `window` (array): Flow values for the window (typically 20 minutes)

**Algorithm:**

The CUSUM algorithm accumulates evidence of sustained deviation from baseline:

```
For each minute's flow in window:
  
  1. Classify as appliance or inter-appliance:
     - If flow >= appliance_flow_thresh (8.0 L/min):
       Appliance event (shower, washing machine, etc.)
     - Else:
       Inter-appliance baseline (slow leaks, small usage)
  
  2. On appliance START (transition from normal to appliance):
     - Apply partial reset: s *= 0.5  ← Preserves leak signal
     - (Old behavior was full reset: s = 0.0)
  
  3. For inter-appliance flow:
     - Calculate slack: delta = flow - cusum_k
     - Update accumulator: s = max(0, s + delta)
     - Check if triggered: if s >= cusum_h: triggered = True

Return (final_s, triggered)
```

**Key Insight — Partial Resets:**
- When an appliance event occurs during a leak, the CUSUM partially resets but retains evidence
- This preserves leak signal even when users are running water during the leak
- Full reset would lose the sustained leak signal

**Return:**
- `s_final` (float): Final CUSUM accumulator value
- `triggered` (bool): Whether any minute exceeded threshold `h`

---

#### Method: `_extract_features(window) → Tuple[np.ndarray, dict]`

```python
def _extract_features(self, window) -> Tuple[np.ndarray, dict]
```

**Purpose:** Extracts 7 statistical features from a flow window for Isolation Forest.

**Parameters:**
- `window` (array): Flow values for 20-minute window

**Features:**

| # | Name | Calculation | Interpretation |
|---|------|-----------|-----------------|
| 1 | **mnf** | 10th percentile of non-zero flows | Baseline non-appliance flow (low flows indicate leak) |
| 2 | **inter_mean** | Mean of flows < 8.0 L/min | Average inter-appliance flow |
| 3 | **inter_frac** | Fraction of inter-appliance periods > 0.2 L/min | Activity density in baseline periods |
| 4 | **mean_flow** | Mean of all window flows | Overall average flow |
| 5 | **inter_std** | Std dev of inter-appliance flows | Variability in baseline periods |
| 6 | **flow_trend** | Linear regression slope [L/min per minute] | Rising/falling baseline (leak indicator) |
| 7 | **baseline_elev** | (rolling_inter_mean - training_median) / training_std | Deviation from trained baseline |

**Algorithm:**

```python
# Separate inter-appliance flows
inter = window[window < 8.0]
nonzero = window[window > 0.0]

# Feature 1: mnf (10th percentile)
mnf = percentile(nonzero, 10) if len(nonzero) > 0 else 0.0

# Feature 2: inter_mean
inter_mean = mean(inter) if len(inter) > 0 else 0.0

# Feature 3: inter_frac
inter_frac = fraction(inter > 0.2) if len(inter) > 0 else 0.0

# Feature 4: mean_flow
mean_flow = mean(window)

# Feature 5: inter_std
inter_std = std(inter) if len(inter) > 1 else 0.0

# Feature 6: flow_trend (linear regression slope)
t = arange(len(window))  # Time indices [0, 1, ..., 19]
# Slope = Σ((t - t_mean) * (flow - flow_mean)) / Σ((t - t_mean)²)
# Positive slope: rising flow (leak)
# Negative slope: falling flow (normal)
flow_trend = regression_slope(t, window)

# Feature 7: baseline_elev (standardized deviation)
rolling_inter_mean = mean(inter_mean_history[-3 windows])
baseline_elev = (rolling_inter_mean - baseline_median) / baseline_std

# Scale and clip
raw_features = [mnf, inter_mean, inter_frac, mean_flow, inter_std, flow_trend, baseline_elev]
scaled = scaler.transform([raw_features])
clipped = clip(scaled, -10, +10)
```

**Return:**
- `clipped` (array): Shape (1, 7), scaled and clipped features ready for IF model
- `feat_debug` (dict): Debug info including `inter_mean`, `flow_trend`, `baseline_elev`

---

#### Method: `update(window) → dict`

```python
def update(self, window) -> dict
```

**Purpose:** Processes a 20-minute flow window through both detection levels, fuses results, applies persistence filter.

**Parameters:**
- `window` (array): 20 flow values (one per minute)

**Algorithm:**

```
┌──────────────────────────────────────────────┐
│ LEVEL 2: CUSUM Change Detection             │
└──────────────────────────────────────────────┘
  Run CUSUM on window → (s_final, cusum_triggered)
  Normalize score: cusum_score = min(1.0, s_final / h)

┌──────────────────────────────────────────────┐
│ LEVEL 3: Isolation Forest Anomaly Detection │
└──────────────────────────────────────────────┘
  Extract 7 features → scaled vector
  Get IF decision function score
  Check if score < if_threshold → if_triggered
  Normalize to [0, 1]: if_score = min(1.0, (threshold - score) / score_scale)

┌──────────────────────────────────────────────┐
│ FUSION: Weighted Combination                 │
└──────────────────────────────────────────────┘
  final_score = w_cusum * cusum_score + w_if * if_score
              = 0.35 * cusum_score + 0.65 * if_score

  candidate_anomaly = (
      final_score > decision_threshold         ← Fusion path
      OR cusum_triggered                       ← CUSUM bypass
      OR if_triggered                          ← IF bypass
  )

┌──────────────────────────────────────────────┐
│ PERSISTENCE FILTER: False Positive Guard    │
└──────────────────────────────────────────────┘
  if candidate_anomaly:
      streak += 1
  else:
      streak = 0
  
  final_anomaly = (streak >= persistence_windows)
```

**Bypass Rationale:**
- CUSUM alone (weight 0.35) cannot cross 0.40 threshold without IF support
- IF at edge of detectability may not generate enough fusion score
- Independent bypasses ensure high-confidence signals are not lost
- Persistence filter (4 consecutive windows ≈ 80 minutes) is primary false-positive guard

**Return Dictionary:**
```python
{
    "anomaly": bool,                    # Final decision after persistence
    "final_score": float,               # Fused [0, 1] score
    "level2": {
        "triggered": bool,              # CUSUM triggered this window
        "score": float,                 # CUSUM amplitude score
    },
    "level3": {
        "triggered": bool,              # IF triggered this window
        "score": float,                 # IF normalized [0, 1] score
        "reconstruction_error": float,  # Raw IF decision function
        "flow_trend": float,            # Feature: trend slope
        "baseline_elev": float,         # Feature: baseline deviation
    },
}
```

---

#### Method: `reset() → None`

```python
def reset(self) -> None
```

**Purpose:** Clears detector's internal state (used when restarting simulation).

**Behavior:**
```python
self.cusum_s = 0.0                       # Reset accumulator
self._prev_appliance = False             # Reset appliance flag
self._anomaly_streak = 0                 # Reset persistence counter
self._inter_mean_history.clear()         # Clear 60-minute history
```

---

## Module: `server.py`

### Overview
FastAPI + Socket.IO web server providing real-time simulation and anomaly detection via WebSocket streaming. Serves frontend dashboard and handles simulation controls (start/pause/stop, leak injection).

---

### Setup & Configuration

#### Global Constants

```python
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = Path("/models")  # Pre-trained models location
```

#### Model Loading Functions

##### `load_pickle(path) → object`

```python
def load_pickle(path) -> object
```

**Purpose:** Loads a Python pickle file.

**Return:** Unpickled object (typically IF model or scaler)

---

##### `load_json(path) → dict`

```python
def load_json(path) -> dict
```

**Purpose:** Loads a JSON configuration file.

**Return:** Parsed dictionary

---

##### `find_model_path(name, alternates=None) → Path`

```python
def find_model_path(name, alternates=None) -> Path
```

**Purpose:** Locates model file in either `artifacts/` or `/models/` directory with fallback support.

**Algorithm:**
```
1. Try artifacts/{name}
2. Try artifacts/{alternate_names}
3. Try /models/{name}
4. Try /models/{alternate_names}
5. If all fail: Raise FileNotFoundError
```

**Use Case:** Handles multiple model naming conventions and storage locations.

---

#### Model & Detector Initialization

```python
# Load pretrained Isolation Forest model (prioritize building-scale)
if_model = load_pickle(ARTIFACTS_DIR / "isolation_forest_building.pkl")
# Fallback to household model if building model not found

# Load feature scaler
if_scaler = load_pickle(ARTIFACTS_DIR / "scaler_building.pkl")

# Load calibration (building or household)
cal = load_json(ARTIFACTS_DIR / "calibration_building.json")
WINDOW_MINUTES = cal["window_minutes"]  # Typically 20

# Initialize generator (50 apartments, seed=42 for reproducibility)
generator = LiveApartmentBuildingDataGenerator(
    ARTIFACTS_DIR / "all_appliances.json",
    num_apartments=50,
    seed=42,
)

# Initialize detector with calibrated parameters
detector = HybridWaterAnomalyDetector(
    if_model=if_model,
    if_scaler=if_scaler,
    cusum_k=cal.get("cusum_k", 3.0),
    cusum_h=cal.get("cusum_h", 8.0),
    # ... other parameters from calibration JSON
)
```

---

### Global State

```python
simulation_running = False          # Simulation pause/resume flag
simulation_speed = 1.0              # Simulation speed multiplier (1-10x)
sim_minutes = 0                     # Current simulation time (minutes)

window_buffer = deque(maxlen=20)    # Rolling 20-minute flow window

last_result = {...}                 # Latest detector result

leak_active = False                 # Leak injection active flag
leak_intensity = 0.0                # Leak flow rate (L/min)
leak_end_minute = None              # Leak end time
leak_start_minute = None            # Leak start time
leak_mode = "instant"               # "instant" or "ramp"
leak_ramp_minutes = 5               # Ramp duration for ramping leaks
```

---

### FastAPI Routes

#### `GET /`

```python
@app.get("/")
async def serve_index() -> FileResponse
```

**Purpose:** Serves the main frontend HTML page.

**Response:** `frontend/index.html` file

---

### Socket.IO Event Handlers

Socket.IO provides real-time bidirectional communication between server and frontend.

#### Event: `start_simulation`

```python
@sio.event
async def start_simulation(sid)
```

**Purpose:** Starts or resumes the simulation.

**Behavior:**
```python
simulation_running = True
emit("simulation_state", {"state": "running"})
```

**Frontend Receives:** `{"state": "running"}`

---

#### Event: `pause_simulation`

```python
@sio.event
async def pause_simulation(sid)
```

**Purpose:** Pauses the simulation (preserves all state).

**Behavior:**
```python
simulation_running = False
emit("simulation_state", {"state": "paused"})
```

---

#### Event: `stop_simulation`

```python
@sio.event
async def stop_simulation(sid)
```

**Purpose:** Stops and resets simulation to initial state.

**Behavior:**
```python
simulation_running = False
sim_minutes = 0
leak_active = False
leak_end_minute = None
leak_start_minute = None
window_buffer.clear()
generator.reset()
detector.reset()
emit("simulation_state", {"state": "stopped"})
```

---

#### Event: `set_speed`

```python
@sio.event
async def set_speed(sid, data)
```

**Purpose:** Sets simulation speed multiplier.

**Parameters:**
- `data` (float): Desired speed (1-10x), clamped to valid range

**Behavior:**
```python
simulation_speed = max(1.0, min(float(data), 10.0))
emit("speed_update", {"speed": simulation_speed})
```

**Effect:** Increases frequency of `data_update` emissions (skips `asyncio.sleep(1.0 / simulation_speed)`)

---

#### Event: `inject_leak`

```python
@sio.event
async def inject_leak(sid, data)
```

**Purpose:** Injects a synthetic leak into the simulation.

**Parameters (from `data` dict):**
- `"intensity"` (float): Leak flow (L/min), default 0.5, clamped [0.1, 20.0]
- `"duration"` (int): Leak duration (minutes), default 60, minimum 1
- `"mode"` (str): `"instant"` or `"ramp"`, default `"instant"`
- `"ramp_minutes"` (int): Ramp duration (only for ramp mode), default 5, minimum 1

**Server-Side Behavior:**
1. Validate and clamp parameters
2. Store leak metadata
3. During simulation loop: apply ramping if mode="ramp"
   - Ramp progress: `elapsed / ramp_minutes`
   - Effective intensity: `intensity * progress`

**Frontend Receives:**
```python
{
    "active": True,
    "mode": "instant" | "ramp",
    "intensity": 0.5  # L/min
}
```

---

#### Event: `stop_leak`

```python
@sio.event
async def stop_leak(sid)
```

**Purpose:** Stops any active leak injection.

**Behavior:**
```python
leak_active = False
leak_end_minute = None
leak_start_minute = None
emit("leak_status", {"active": False})
```

---

### Main Simulation Loop

#### Function: `simulation_loop()`

```python
async def simulation_loop()
```

**Purpose:** Main background loop that runs continuously, generating flow data and detecting anomalies.

**Algorithm:**

```
LOOP forever:
  
  IF simulation_running:
    
    ┌─ GENERATE FLOW ─────────────────────────┐
    │ flow = generator.next()                 │
    │ Returns single minute's aggregate flow  │
    └─────────────────────────────────────────┘
    
    ┌─ APPLY SERVER-SIDE LEAK ────────────────┐
    │ IF leak_active:                         │
    │   IF current_minute < leak_end_minute:  │
    │     Apply ramp progression              │
    │     flow += effective_intensity         │
    │     clip to MAX_FLOW_LPM                │
    │   ELSE:                                 │
    │     leak_active = False                 │
    └─────────────────────────────────────────┘
    
    ┌─ BUFFER WINDOW ─────────────────────────┐
    │ window_buffer.append(flow)              │
    │ IF len(window_buffer) == 20:            │
    │   result = detector.update(window)      │
    │   last_result = result                  │
    └─────────────────────────────────────────┘
    
    ┌─ BUILD RESPONSE ────────────────────────┐
    │ result["flow"] = current_minute_flow    │
    │ result["sim_time"] = formatted HH:MM    │
    │ result["sim_minutes"] = total_minutes   │
    │ result["leak_active"] = bool            │
    │ result["leak_intensity"] = intensity    │
    │ result["leak_remaining"] = seconds_left │
    └─────────────────────────────────────────┘
    
    ┌─ EMIT TO FRONTEND ──────────────────────┐
    │ sio.emit("data_update", result)         │
    └─────────────────────────────────────────┘
    
    ┌─ ADVANCE TIME ──────────────────────────┐
    │ sim_minutes += 1                        │
    │ await sleep(1.0 / simulation_speed)     │
    └─────────────────────────────────────────┘
  
  ELSE:
    await sleep(0.1)  # Sleep while paused
```

**Data Emitted (`data_update` event):**

```javascript
{
    // Current measurements
    "flow": 12.5,                              // L/min
    
    // Timing
    "sim_time": "08:00",                       // HH:MM format
    "sim_minutes": 480,                        // Total minutes
    
    // Anomaly detection results
    "anomaly": false,                          // Final decision
    "final_score": 0.35,                       // Fused [0, 1] score
    
    "level2": {
        "triggered": false,
        "score": 0.1
    },
    
    "level3": {
        "triggered": false,
        "score": 0.2,
        "reconstruction_error": -0.05,
        "flow_trend": 0.001,
        "baseline_elev": 0.5
    },
    
    // Leak information
    "leak_active": false,
    "leak_mode": "instant",
    "leak_intensity": 0.0,
    "leak_remaining": 0
}
```

**Frequency:** ~1 message/second at 1x speed, ~10 messages/second at 10x speed.

**Frontend Action:** Dashboard updates real-time flow chart and anomaly indicators based on these messages.

---

#### Startup Event

```python
@app.on_event("startup")
async def startup_event()
```

**Purpose:** Executed when server starts up.

**Behavior:**
```python
asyncio.create_task(simulation_loop())  # Start background simulation
```

---

## Summary Table

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| **LiveWaterFlowGenerator** | Single apartment flow | `next()`, `inject_leak()`, `clear_leak()` |
| **LiveApartmentBuildingDataGenerator** | 50-apartment building | `next()`, `inject_leak()`, `clear_leak()`, `reset()` |
| **HybridWaterAnomalyDetector** | Anomaly detection | `update()`, `reset()`, `_run_cusum()`, `_extract_features()` |
| **FastAPI Server** | Web interface & API | `serve_index()` |
| **Socket.IO Handlers** | Real-time control | `start_simulation()`, `inject_leak()`, `set_speed()` |
| **simulation_loop()** | Main engine | Streams data & anomaly results |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ frontend/index.html (web dashboard)                              │
│  ├─ Emits events: start_simulation, inject_leak, set_speed     │
│  └─ Receives: data_update via Socket.IO                         │
└────────────────────────┬────────────────────────────────────────┘
                         │ Socket.IO (WebSocket)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ server.py (FastAPI + Socket.IO)                                  │
│  ├─ Event Handlers                                              │
│  ├─ simulation_loop() — Main engine                             │
│  └─ Model/Generator instances                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐  ┌───────────┐  ┌─────────────────┐
    │Generator│  │ Detector  │  │ Database        │
    │ (Flow)  │  │(Anomaly)  │  │ (Calibration)   │
    └────┬────┘  └─────┬─────┘  └─────────────────┘
         │             │
         └─────┬───────┘
               ▼
         Result Dict
         (emitted to frontend)
```

---

## Notes on Design Decisions

1. **Independent Apartment Generators:** Each uses its own seed for realistic variance while maintaining reproducibility.

2. **Partial CUSUM Resets:** Leaks overlapping with appliance usage are preserved; full resets would lose evidence.

3. **Server-Side Leak Injection:** Added at aggregation level separate from generator, for simulating infrastructure failures.

4. **Persistence Filter:** 4-window persistence (~80 minutes) is primary false-positive guard when IF threshold is set aggressively.

5. **Feature Scaling:** Features clipped post-scaling to [-10, +10] to prevent extreme values from dominating IF model.

6. **WebSocket Streaming:** ~1 msg/sec at 1x, scales to 10x for fast-forward testing without overwhelming network.

---

**End of Backend Documentation**
