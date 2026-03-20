# Water Simulator Testing Framework

Automated testing framework for leak detection accuracy on both household and apartment simulators.

## What It Does

- **Automated Testing**: Runs 20-50 simulations per simulator automatically
- **Randomized Scenarios**: Each run has random leak time, intensity (≥2.2 L/min), and duration
- **Metrics Calculation**: Computes accuracy, precision, recall, F1-score, detection rate, false alarm rate
- **Visualizations**: 
  - Confusion matrix heatmap
  - Accuracy dashboard (bar chart)
  - Individual run traces (flow + CUSUM + leak/detection markers)

## Running Tests

### Using Docker (Recommended)

```bash
cd testing

# Run both simulators (default: 20 runs each)
docker-compose up

# Run household only (30 runs)
docker-compose run test python run_tests.py --simulator household --runs 30

# Run apartment only (25 runs)
docker-compose run test python run_tests.py --simulator apartment --runs 25
```

### Direct Python (local)

```bash
cd testing
pip install -r requirements.txt

# Run both
python run_tests.py --simulator both --runs 20

# Run specific simulator
python run_tests.py --simulator household --runs 20
```

On Windows/PowerShell you can also use `py`:

```bash
py run_tests.py --simulator both --runs 20
```

## Test Parameters

**Household Simulator** (single unit):
- **Leak Time**: Random minute (100-450 min, allows sustained 100+ min leak)
- **Leak Intensity**: Random 0.5-2.0 L/min (matches server.py cap of 0.1-2.0 on UI)
- **Leak Duration**: Random 100-200 minutes (sustained signal for clear detection)
- **Leak Mode**: Randomized (50/50 instant vs ramp)
  - Instant: Full intensity immediately
  - Ramp: Gradual increase over 5-15 minutes
- Model tuned: CUSUM k=0.01, h=1.0

**Apartment Building Simulator** (50 units aggregated):
- **Leak Time**: Random minute (100-450 min, allows sustained 100+ min leak)
- **Leak Intensity**: Random 5.0-40.0 L/min (building scale, larger leaks)
- **Leak Duration**: Random 100-200 minutes (sustained signal for clear detection)
- **Leak Mode**: Randomized (50/50 instant vs ramp)
  - Instant: Full intensity immediately
  - Ramp: Gradual increase over 5-15 minutes
- Model tuned: CUSUM k=3.0, h=8.0 (less sensitive, baseline ~2.0 L/min per apartment)

**Simulation Window**: 600 minutes (10 hours)
**Detection Window**: 20-minute sliding window

## Output Structure

```
results/
├── household/
│   ├── results.csv              # All run data
│   ├── confusion_matrix.png     # TP/FP/FN/TN heatmap
│   ├── accuracy_dashboard.png   # Metrics bar chart
│   └── metrics_summary.txt      # Human-readable results
├── apartment/
│   ├── results.csv
│   ├── confusion_matrix.png
│   ├── accuracy_dashboard.png
│   └── metrics_summary.txt
```

## Key Metrics

- **Detection Rate**: % of leaks successfully detected
- **False Alarm Rate**: % of false positive detections
- **Accuracy**: Overall correct predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **Avg Delay**: Average detection latency in minutes

## Results CSV Columns

- `leak_time`: When leak was injected (minute)
- `detection_time`: When alarm triggered (minute)
- `delay`: Detection latency (detection_time - leak_time)
- `false_alarm`: Boolean, true if alarm before leak
- `missed_detection`: Boolean, true if leak with no alarm

## Important Notes

1. **Different Models, Different Tests**: Household and apartment simulators use completely different models:
   - Household: Sensitive to small leaks (0.5-3.0 L/min range), CUSUM k=0.01
   - Apartment: Tuned for building scale, requires larger leaks (5.0-40.0 L/min), CUSUM k=3.0
   - Tests are NOT identical because the models are tuned independently

2. **State Reset**: Each run fully resets generator, detector state, and leak injection to avoid data pollution

3. **Randomization**: Each run uses different random seed for realistic variation

4. **Sample Size**: 20-30 runs recommended for statistical significance
