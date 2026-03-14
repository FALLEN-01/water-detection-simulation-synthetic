# Water Leak Detection System

An intelligent anomaly detection system for identifying water leaks in apartment buildings using machine learning and synthetic sensor data.

## Overview

This project simulates and detects water leaks in a 50-unit apartment building by analyzing water flow patterns, temperature, and turbidity sensor data over a 6-month period. The system uses two complementary machine learning approaches to identify abnormal water consumption patterns that indicate potential leaks.

## Problem Statement

Water leaks in multi-unit buildings are challenging to detect because:
- Normal usage varies significantly by time of day, day of week, and season
- Occupancy fluctuates (vacations, seasonal residents, visitors)
- Small leaks can go unnoticed for days, especially during peak usage hours
- Manual monitoring is impractical with 1-minute sampling frequency

## Solution

The system uses **pattern-based anomaly detection** with realistic leak scenarios:

### Leak Patterns (Realistic, Not Random)

1. **Stress-Induced Leaks (40%)** - Occur during peak usage hours (7-8 AM, 7-9 PM) when high water flow stresses pipes, causing gradual failures. Duration: 1-3 days

2. **Seasonal Leaks (30%)** - Occur in winter months (Jan, Feb, Dec) when temperature changes cause pipe expansion/contraction. Duration: 4 hours to 2 days

3. **Night-Time Leaks (30%)** - Occur between midnight-6 AM, often going unnoticed longer due to low baseline usage. Duration: 2-5 days

## Architecture

### Data Generation ([emulate_data.py](emulate_data.py))
- **6 months** of synthetic data (259,200 samples at 1 sample/minute)
- **India-based realistic usage patterns** from actual water consumption data:
  - **Shower**: 0.88 events/day, peak hours 6-7 AM (mean flow ~18 ml/s)
  - **Toilet**: 1.93 events/day, peak hours 6-8 AM (fixed 360s duration)
  - **Bidet**: 1.98 events/day, distributed throughout day (mean flow ~34 ml/s)
  - **Washbasin**: 13.99 events/day, morning peak 6-7 AM (mean flow ~43 ml/s)
  - **Kitchen Faucet**: 11.55 events/day, evening peak 7-8 PM (mean flow ~43 ml/s)
  - **Washing Machine**: 0.39 events/day, morning hours 7-9 AM (mean flow ~38 ml/s)
  - **Dishwasher**: 0.08 events/day, evening hours (fixed 1800s duration)
- Appliance usage follows Poisson distributions with hourly probability distributions
- Flow rates sampled from lognormal distributions calibrated for Indian fixtures (MoHUA/BIS standards)
- Seasonal variations:
  - Summer (Apr-Jun): 70-85% occupancy (travel season)
  - Winter holidays (Dec-Jan): 65-80% occupancy
  - Regular months: 80-95% occupancy
- **Training data**: Normal operations only
- **Test data**: Normal operations + 8 pattern-based leak events

### Sensors

**Required:**
- **Flow Rate** (L/min): Primary indicator normalized to 0-15 L/min range


**Auxiliary/Optional:**
- **Turbidity** (NTU): Water quality indicator, increases during leaks

### Temporal Features:**
- Hour of day, day of week, flow duration, weekend flag

### Data Calibration
All water usage patterns are calibrated for **Indian domestic fixtures** based on:
- **MoHUA** (Ministry of Housing and Urban Affairs) guidelines
- **BIS** (Bureau of Indian Standards) specifications
- Real-world dataset spanning 617 days of multi-apartment monitoring
- Flow rates scaled to reflect Indian fixture characteristics (e.g., low-flow showerheads, manual faucets)

### Detection Models

#### 1. Autoencoder ([autoencoder_water.py](autoencoder_water.py))

**Architecture: LSTM Encoder-Decoder with Windowed Input**

```
Input: (10 timesteps, 5 features) - 10-minute sliding window
│
├─ Encoder
│  ├─ LSTM(32 units, return_sequences=True) → (10, 32)
│  ├─ Dropout(0.2)
│  ├─ LSTM(16 units, return_sequences=False) → (16,)
│  └─ Dropout(0.2)
│
├─ Latent Space: 16-dimensional compressed representation
│
└─ Decoder
   ├─ RepeatVector(10) → (10, 16)
   ├─ LSTM(16 units, return_sequences=True) → (10, 16)
   ├─ Dropout(0.2)
   ├─ LSTM(32 units, return_sequences=True) → (10, 32)
   └─ TimeDistributed(Dense(5)) → (10, 5)

Output: Reconstructed 10-minute window

Total Parameters: ~20,000
```

**Training Configuration:**
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning_rate=0.001)
- **Batch Size**: 128 windows
- **Epochs**: 50 (with early stopping, patience=5)
- **Validation Split**: 10% of training data
- **Threshold**: 99th percentile of training reconstruction errors

**How It Works:**
1. **Learning Phase**: Trained only on normal flow patterns (no leaks)
2. **Compression**: LSTM encoder compresses 10-minute sequence into 16 values
3. **Reconstruction**: LSTM decoder attempts to recreate original sequence
4. **Anomaly Detection**: High reconstruction error indicates unusual pattern (leak)

**Why Windows?**
- Captures temporal context (not just single-minute spikes)
- Distinguishes sustained leaks from brief normal usage (showers)
- 10-minute window: balance between context and responsiveness

**Features Used:**
- Flow rate (normalized), Turbidity (optional), Flow duration, Hour of day, Weekend flag

#### 2. Isolation Forest ([isolation_water.py](isolation_water.py))

**Architecture: Ensemble Tree-Based Anomaly Detection**

```
Algorithm: Isolation Forest (scikit-learn)
│
├─ Forest Configuration
│  ├─ n_estimators: 200 trees
│  ├─ contamination: Auto-calibrated (expected leak frequency × 1.5)
│  ├─ max_samples: 'auto' (256 or dataset size)
│  └─ random_state: 42 (reproducibility)
│
└─ Detection Strategy
   ├─ Random Feature Selection
   ├─ Random Split Points
   └─ Path Length Measurement
      └─ Shorter paths = Anomalies (easier to isolate)
```

**Training Configuration:**
- **Input**: Single-timestep features (not windowed)
- **n_jobs**: -1 (use all CPU cores)
- **Prediction**: -1 = anomaly, +1 = normal
- **Scoring**: Lower anomaly scores indicate higher suspicion

**How It Works:**
1. **Isolation Principle**: Anomalies are rare and different, thus easier to isolate
2. **Random Partitioning**: Each tree randomly splits feature space
3. **Path Length**: Measures how many splits needed to isolate a sample
4. **Ensemble Voting**: 200 trees vote on anomaly likelihood

**Advantages:**
- ✓ Fast training and prediction
- ✓ Works well with high-dimensional data
- ✓ No assumption about data distribution
- ✓ Memory efficient

**Disadvantages:**
- ✗ No temporal context (single-point analysis)
- ✗ May miss gradual anomalies
- ✗ Requires contamination parameter tuning

### Model Comparison

| Aspect | LSTM Autoencoder | Isolation Forest |
|--------|------------------|------------------|
| **Input Type** | 10-minute windows | Single timesteps |
| **Temporal Context** | ✓ Yes (LSTM) | ✗ No |
| **Training Time** | ~5-10 minutes | ~30 seconds |
| **Memory Usage** | Higher | Lower |
| **Detection Type** | Pattern deviation | Statistical outlier |
| **Best For** | Sustained leaks | Point anomalies |
| **Parameters** | ~20,000 | None (tree-based) |
| **Interpretability** | Low (deep learning) | Medium (feature importance) |

### Orchestration ([main.py](main.py))
Executes the complete pipeline:
1. Generate synthetic datasets
2. Train autoencoder model
3. Train isolation forest model
4. Output performance metrics and visualizations

## Installation

### Option 1: Local Python Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the system
python main.py
```

### Option 2: Docker

```bash
# Build and run
docker-compose up --build

# Or using Docker directly
docker build -t water-detection .
docker run -v ${PWD}:/app water-detection
```

## Requirements

- Python 3.8+
- NumPy 1.24.3
- Pandas 2.0.3
- scikit-learn 1.3.0
- TensorFlow 2.13.0 (CPU version)
- Matplotlib 3.7.2

## Output

### Datasets
- `water_train.csv` - 259,200 samples of normal operation (6 months)
- `water_test.csv` - 259,200 samples with injected leak events

### Visualizations
- `water_flow_analysis.png` - Multi-panel analysis of flow patterns, temperature correlations, turbidity distribution
- `water_flow_timeline.png` - 6-month overview showing leak occurrences
- `water_flow_24hour_patterns.png` - Daily patterns for each month
- `autoencoder_results.png` - Reconstruction errors, confusion matrix, ROC curve
- `isolation_forest_results.png` - Anomaly scores, feature importance, ROC curve

### Performance Metrics
Both models report:
- **Accuracy**: Overall correct predictions
- **Precision**: Of detected leaks, how many are real
- **Recall**: Of real leaks, how many are detected
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP/FP/TN/FN breakdown

## Model Architecture Details

### LSTM Autoencoder Layer Specifications

**Complete Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────┐
│                    LSTM AUTOENCODER                          │
├─────────────────────────────────────────────────────────────┤
│ INPUT                                                         │
│  └─ Shape: (batch, 10, 5)                                   │
│     10 timesteps × 5 features                                │
│                                                               │
│ ENCODER                                                       │
│  ├─ LSTM(32 units, return_sequences=True)                   │
│  │   └─ Output: (batch, 10, 32)   [4,864 params]           │
│  ├─ Dropout(0.2)                                            │
│  ├─ LSTM(16 units, return_sequences=False)                  │
│  │   └─ Output: (batch, 16)       [3,136 params]           │
│  └─ Dropout(0.2)                                            │
│                                                               │
│ LATENT SPACE                                                  │
│  └─ 16-dimensional compressed representation                 │
│     [Captures essential flow patterns]                       │
│                                                               │
│ DECODER                                                       │
│  ├─ RepeatVector(10)                                         │
│  │   └─ Output: (batch, 10, 16)   [0 params]               │
│  ├─ LSTM(16 units, return_sequences=True)                   │
│  │   └─ Output: (batch, 10, 16)   [2,112 params]           │
│  ├─ Dropout(0.2)                                            │
│  ├─ LSTM(32 units, return_sequences=True)                   │
│  │   └─ Output: (batch, 10, 32)   [6,272 params]           │
│  └─ TimeDistributed(Dense(5))                               │
│      └─ Output: (batch, 10, 5)    [165 params]             │
│                                                               │
│ RECONSTRUCTED OUTPUT                                          │
│  └─ Shape: (batch, 10, 5)                                   │
│     Same as input                                            │
└─────────────────────────────────────────────────────────────┘

Total Parameters: 16,549 (trainable)
Model Size: ~65 KB
```

**Parameter Calculation:**
- LSTM_1: 4 × (32 × (5 + 32 + 1)) = 4,864
- LSTM_2: 4 × (16 × (32 + 16 + 1)) = 3,136
- LSTM_3: 4 × (16 × (16 + 16 + 1)) = 2,112
- LSTM_4: 4 × (32 × (16 + 32 + 1)) = 6,272
- Dense: (32 + 1) × 5 = 165

**Training Strategy:**
```
Loss Function: Mean Squared Error (MSE)
├─ Measures reconstruction quality
├─ Lower MSE = Better reconstruction
└─ Anomalies have high MSE (can't reconstruct unusual patterns)

Optimizer: Adam (lr=0.001)
├─ Adaptive learning rate
├─ Momentum-based optimization
└─ Efficient for LSTM networks

Early Stopping: patience=5
├─ Monitors training loss
├─ Stops if no improvement for 5 epochs
└─ Restores best weights

Regularization:
├─ Dropout: 0.2 (prevents overfitting)
├─ Validation split: 10% (monitors generalization)
└─ Window shuffling (randomizes temporal order)
```

### Feature Importance & Selection

**Input Features (per timestep):**

1. **flow_normalized** (0-1): Primary anomaly indicator
2. **turbidity** (NTU): Auxiliary sensor (optional, increases sediment during leaks)
3. **flow_duration** (seconds): Cumulative daily flow time
4. **hour** (0-23): Time-of-day context (peak hours)
5. **is_weekend** (0-1): Weekend usage patterns

**Why These Features?**
- Combination of **direct measurements** (flow, turbidity) and **contextual** (time, duration)
- Normalized to prevent feature dominance
- Captures both magnitude and temporal patterns

### Detection Philosophy

Both models use **unsupervised anomaly detection**:
- ✓ Trained only on **normal data** (no leak examples needed)
- ✓ Simulates real-world deployment (leak patterns unknown/evolving)
- ✓ Threshold auto-calibrated from normal behavior
- ✓ Generalizes to novel leak types

## Project Structure

```
├── main.py                    # Pipeline orchestrator
├── emulate_data.py           # India-based synthetic data generator
├── autoencoder_water.py      # Deep learning detector
├── isolation_water.py        # Ensemble detector
├── priors_india/             # Real-world India water usage patterns (JSON)
│   ├── shower.json
│   ├── toilet.json
│   ├── bidet.json
│   ├── washbasin.json
│   ├── kitchenfaucet.json
│   ├── washingmachine.json
│   └── dishwasher30.json
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Multi-container setup
├── pyrightconfig.json       # Python type checking
└── README.md               # This file
```

## Technical Details

### Data Generation Strategy
- **India-specific calibration**: Flow rates and usage patterns from real 617-day dataset
- **Occupancy modeling**: 55-98% occupancy with seasonal variations (Indian climate patterns)
- **Appliance-level simulation**: Each fixture type modeled independently with realistic:
  - Event frequency (Poisson distribution)
  - Hourly timing (categorical distribution from actual data)
  - Duration (lognormal or fixed based on appliance type)
  - Flow rate (lognormal distribution, scaled for Indian fixtures)
- **Realistic noise**: Gaussian noise added to all sensor readings
- **Leak injection**: Pattern-based placement with varying duration and severity

### Model Training

**Autoencoder Training Process:**

```python
# 1. Data Preparation (10-min sliding windows)
Training windows: 259,141 (from 259,200 samples)
Window shape: (259141, 10, 5)  # samples × timesteps × features

# 2. Model Architecture
Encoder: (10,5) → LSTM(32) → LSTM(16) → [Compressed: 16]
Decoder: [16] → Repeat(10) → LSTM(16) → LSTM(32) → (10,5)

# 3. Training Configuration
- Learns to reconstruct NORMAL patterns only
- Early stopping monitors training loss
- Validation split ensures no overfitting
- Dropout layers (0.2) prevent memorization

# 4. Threshold Calibration
- Calculate reconstruction error on training set
- Set threshold = 99th percentile
- Samples exceeding threshold = anomalies

# 5. Evaluation
- Test on 6 months with 8 injected leak events
- Metrics: Accuracy, Precision, Recall, F1 Score
```

**Isolation Forest Training Process:**

```python
# 1. Data Preparation (single samples)
Training samples: 259,200
Features: 5-6 (depending on turbidity availability)

# 2. Forest Construction
- Build 200 isolation trees
- Each tree: random feature splits
- Path length tracked for each sample

# 3. Contamination Estimation
- Expected leaks / total samples
- Multiply by 1.5 safety factor
- Cap at 0.1 (10% max contamination)

# 4. Scoring
- Aggregate path lengths across trees
- Normalize to anomaly score [-1, 1]
- Score < 0 typically indicates anomaly

# 5. Prediction
- Apply threshold at decision boundary
- -1 = anomaly, +1 = normal
```


## Dataset Attribution

This project uses realistic water consumption patterns from the **priors_india** dataset:

- **Source:** [WEUSEDTO-Data Repository](https://github.com/AnnaDiMauro/WEUSEDTO-Data)
- **Description:** Real-world water usage patterns from 617-day monitoring of multi-unit buildings in India
- **Coverage:** Fixture types include shower, toilet, bidet, washbasin, kitchen faucet, washing machine, and dishwasher
- **Standards:** Calibrated to MoHUA (Ministry of Housing and Urban Affairs) guidelines and BIS (Bureau of Indian Standards) specifications
- **Usage:** All synthetic data generation uses these India-specific priors to ensure realistic domestic water consumption patterns

### Priors Data Structure (`priors_india/`)

Each fixture's prior contains:
- Event frequency (events per day using Poisson distribution)
- Timing patterns (hourly probability distributions)
- Duration characteristics (normal or lognormal distributions with constraints)
- Flow rate distributions (calibrated for Indian fixture types)
- Shape profiles (trapezoid, pulsed, or step function)

**Note:** We gratefully acknowledge the use of these real-world priors in generating our synthetic datasets.

---

## License

This is a simulation/demonstration project for educational purposes.

## Authors

Developed as a proof-of-concept for intelligent building management systems.
