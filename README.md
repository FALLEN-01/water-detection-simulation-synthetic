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
Deep learning approach using TensorFlow/Keras:
- **Architecture**: 16→8→4→8→16 neurons with dropout layers
- **Training**: Learns to reconstruct normal flow patterns
- **Detection**: Flags samples where reconstruction error exceeds 99th percentile
- **Threshold**: Adaptive based on training data distribution

#### 2. Isolation Forest ([isolation_water.py](isolation_water.py))
Ensemble method using scikit-learn:
- **Algorithm**: Isolates anomalies by partitioning feature space
- **Configuration**: 200 estimators with contamination tuning
- **Detection**: Identifies samples that are easily isolated (anomalous)
- **Score**: Lower anomaly scores indicate higher likelihood of leaks

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
- **Autoencoder**: 50 epochs with early stopping (patience=5), validation split=10%
- **Isolation Forest**: Auto-calibrated contamination rate based on expected leak frequency
- **Feature normalization**: Flow rate normalized to 0-1 range

### Detection Philosophy
Both models use **unsupervised learning** - trained only on normal data without seeing leak examples. This simulates real-world deployment where leak patterns may be unknown or evolve over time.

## Future Enhancements

- [ ] Real-time streaming data processing
- [ ] Integration with IoT sensor hardware
- [ ] Alert notification system (SMS/email)
- [ ] Mobile app for maintenance staff
- [ ] Historical trend analysis dashboard
- [ ] Multi-building deployment support
- [ ] Automated valve shutoff integration

## License

This is a simulation/demonstration project for educational purposes.

## Authors

Developed as a proof-of-concept for intelligent building management systems.
