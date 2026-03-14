# Technical Details: Water Usage Simulation Layers

## Overview
This document details all assumptions, constraints, and implementation layers used in the India-based water usage simulation for the 50-unit apartment building leak detection system.

---

## Data Source: India Priors

### Source Dataset
- **Region**: India
- **Duration**: 617 days (1550048311 to 1603297596 timestamp)
- **Calibration Method**: Deterministic scaling based on MoHUA (Ministry of Housing and Urban Affairs) and BIS (Bureau of Indian Standards) specifications
- **Single Person Usage**: All priors represent **single person/household** usage patterns

### Appliance Definitions

#### 1. Toilet
- **Events/Day**: 1.93 (Poisson λ)
- **Duration**: Fixed 360 seconds (6 minutes)
- **Flow Rate**: N/A (flush-based)
- **Peak Hours**: 6-8 AM (12.2%), 7-11 PM (distributed evening usage)

#### 2. Shower
- **Events/Day**: 0.88 (Poisson λ)
- **Duration**: Lognormal (shape=2.548, scale=7.343 seconds)
- **Mean Flow**: Lognormal (shape=1.617, scale=18.19 ml/s) → ~1.09 L/min
- **Peak Flow**: Lognormal (shape=1.716, scale=29.40 ml/s) → ~1.76 L/min
- **Peak Hours**: 6-7 AM (10.1%), 5-6 PM (5.9%)
- **Flow Scale Factor**: 2.0× for Indian fixtures (lower pressure, manual controls)

#### 3. Bidet
- **Events/Day**: 1.98 (Poisson λ)
- **Duration**: Lognormal (shape=1.638, scale=4.319 seconds)
- **Mean Flow**: Lognormal (shape=1.390, scale=33.94 ml/s) → ~2.04 L/min
- **Peak Flow**: Lognormal (shape=1.490, scale=51.53 ml/s) → ~3.09 L/min
- **Peak Hours**: 6-7 AM (10.2%), distributed throughout day
- **Flow Scale Factor**: 3.0× for Indian fixtures

#### 4. Washbasin
- **Events/Day**: 13.99 (Poisson λ) - highest frequency
- **Duration**: Lognormal (shape=1.093, scale=varies seconds)
- **Mean Flow**: Lognormal (shape=1.434, scale=43.35 ml/s) → ~2.60 L/min
- **Peak Flow**: Lognormal (shape=1.534, scale=68.61 ml/s) → ~4.12 L/min
- **Peak Hours**: 6-7 AM (14.2%), 3-5 PM (6.6% each)
- **Flow Scale Factor**: 3.0× for Indian fixtures

#### 5. Kitchen Faucet
- **Events/Day**: 11.55 (Poisson λ) - second highest frequency
- **Duration**: Lognormal (shape=1.043, scale=3.343 seconds)
- **Mean Flow**: Lognormal (shape=1.434, scale=43.35 ml/s) → ~2.60 L/min
- **Peak Flow**: Lognormal (shape=1.534, scale=68.61 ml/s) → ~4.12 L/min
- **Peak Hours**: 7-8 PM (8.8%), 11 AM-1 PM (6.7% each), 5-7 PM (6.0% each)
- **Flow Scale Factor**: 3.0× for Indian fixtures

#### 6. Washing Machine
- **Events/Day**: 0.39 (Poisson λ)
- **Duration**: Lognormal (shape=2.055, scale=45.02 seconds)
- **Mean Flow**: Lognormal (shape=1.697, scale=38.09 ml/s) → ~2.29 L/min
- **Peak Flow**: Lognormal (shape=1.623, scale=127.24 ml/s) → ~7.63 L/min
- **Peak Hours**: 7-8 AM (28.2%), 8-9 AM (27.0%) - morning concentrated
- **Flow Scale Factor**: 1.2× for Indian machines (semi-automatic common)

#### 7. Dishwasher
- **Events/Day**: 0.08 (Poisson λ) - lowest frequency (rare in India)
- **Duration**: Fixed 1800 seconds (30 minutes)
- **Mean Flow**: N/A (cycle-based)
- **Peak Hours**: 8 PM (13.5%), 6 PM-11 PM (distributed)
- **Flow Scale Factor**: 1.5× for Indian machines

---

## Building Configuration

### Physical Structure
```
Number of Apartments: 50 units
Building Type: Multi-family residential
Average Household Size: 2.5-3.5 persons (randomized per calculation)
Expected Total Occupants: 127-166 people (depending on occupancy rate)
```

### Occupancy Model

#### Seasonal Variations
```
Regular Months (Feb-Mar, Jul-Nov): 80-95% occupancy
Summer (Apr-Jun):                   70-85% occupancy (travel season)
Winter Holidays (Dec-Jan):          65-80% occupancy (vacation/festivals)
Daily Variation:                    ±5-10% random adjustment
```

#### Constraints
```
Minimum Occupancy: 55% (never less)
Maximum Occupancy: 98% (always some vacancy/maintenance)
```

---

## Appliance Availability Randomization

### Layer 1: Building-Wide Availability
Not all apartments have all appliances. Availability is randomized within realistic ranges:

```python
Appliance          | Min    | Max    | Typical | Reasoning
-------------------|--------|--------|---------|---------------------------
Toilet             | 98%    | 100%   | ~100%   | Essential fixture
Washbasin          | 98%    | 100%   | ~100%   | Essential fixture
Kitchen Faucet     | 98%    | 100%   | ~100%   | Essential fixture
Shower             | 90%    | 98%    | ~95%    | Most have, some use bucket
Washing Machine    | 65%    | 75%    | ~70%    | Common but not universal
Bidet              | 25%    | 35%    | ~30%    | Less common in India
Dishwasher         | 5%    | 8%    | ~4%    | Rare in Indian households
```

**Implementation**: Each simulation run randomizes availability once for the entire building, representing the aggregate fixture distribution.

---

## Peak Hour Categorization & Randomization

### Layer 2: Appliance Category Peaks

Appliances grouped by usage patterns with randomized multipliers:

#### Category 1: Morning Personal Care
**Appliances**: Shower, Toilet, Washbasin, Bidet  
**Peak Period**: 6-9 AM  
**Multiplier**: 1.8-2.4× (randomized per calculation)  
**Reasoning**: Morning hygiene routine dominates before work/school

**Off-Peak Multipliers**:
- Lunch (12-2 PM): 0.6-0.9×
- Night (10 PM-6 AM): 0.2-0.5×
- Regular hours: 0.7-1.1×

#### Category 2: Kitchen Activities
**Appliances**: Kitchen Faucet, Dishwasher  
**Peak Periods**: 
- Lunch (12-2 PM): 1.5-2.0×
- Evening (6-10 PM): 1.8-2.5×  
**Reasoning**: Meal preparation and cleanup times

**Off-Peak Multipliers**:
- Morning (6-9 AM): 1.2-1.6× (breakfast prep)
- Night (10 PM-6 AM): 0.2-0.5×
- Regular hours: 0.7-1.1×

#### Category 3: Laundry
**Appliances**: Washing Machine  
**Peak Periods**: 
- Weekday Morning (7-9 AM): 0.9-1.2×
- Weekend Morning: 1.4-1.8×  
**Reasoning**: Done before work or on weekends

**Off-Peak Multipliers**:
- Afternoon: 1.0-1.4×
- Evening: 0.8-1.2×
- Night: 0.2-0.5×

#### Category 4: Continuous Use
**Appliances**: Toilet, Washbasin  
**Behavior**: Used throughout day with moderate variation (0.7-1.1×)  
**Reasoning**: Biological needs and hygiene not restricted to specific hours

### Weekend Adjustment
**Multiplier**: 1.15-1.35× (randomized)  
**Applied To**: All categories  
**Reasoning**: People home more on weekends, less time-constrained usage patterns

---

## Multi-Layer Randomization Structure

### Layer 3: Flow Calculation Randomization

```
Final Flow = Baseline + (Expected Flow × Multipliers × Variations)

Where:
  Baseline        = 0.1-0.25 L/min × (people/100)
  Expected Flow   = f(hour, people, India priors)
  Multipliers     = {
                      Peak Hour Category Multiplier (1.8-2.5×),
                      Weekend Multiplier (1.15-1.35×),
                      Daily Pattern (0.7-1.3×)
                    }
  Variations      = {
                      Appliance Individual Variation (±15%),
                      Gamma Distribution (shape=2.0, scale=0.5),
                      Measurement Noise (σ=0.1 L/min)
                    }
```

#### Daily Pattern Variation
**Distribution**: Normal (μ=1.0, σ=0.15)  
**Clipped Range**: 0.7-1.3×  
**Duration**: Entire day (1440 minutes)  
**Reasoning**: Some days naturally busier (guests, events, holidays, sick days)

#### Gamma Distribution Variation
**Shape**: 2.0  
**Scale**: 0.5  
**Mean**: 1.0  
**Purpose**: Positive-skewed variation (realistic spikes without negative flows)  
**Reasoning**: Usage patterns naturally have occasional high spikes, rarely symmetric

---

## Sensor Models

### Primary Sensors

#### 1. Flow Rate Sensor
```
Range:              0.1 - 15.0 L/min
Resolution:         0.01 L/min
Measurement Noise:  Gaussian (μ=0, σ=0.1)
Normalization:      Flow / MAX_FLOW (15.0)
Physical Constraint: Cannot exceed pipe capacity (15.0 L/min)
```

### Auxiliary Sensor

#### 2. Turbidity Sensor (Optional)
```
Range:              0.1 - 3.0 NTU
Base Formula:       0.5 + (flow/MAX_FLOW) × 0.8
Flow Correlation:   Positive (higher flow → more sediment disturbance)
Measurement Noise:  Gaussian (μ=0, σ=0.15)
Reasoning:          Higher flow rates stir sediment in pipes
```

---

## Validation Constraints

### Layer 4: Validation & Optimization

#### Per-Person Daily Usage
```
Expected Range:     135-150 liters/day (BIS standard for India)
Warning Threshold:  < 100 L/day or > 200 L/day
Calculation:        Total daily flow / total people in building
Reference:          Bureau of Indian Standards residential water consumption guidelines
```

#### Peak-to-Night Flow Ratio
```
Expected Range:     4-8× (morning peak / night average)
Warning Threshold:  < 3× or > 12×
Reasoning:          Too low = unrealistic usage pattern
                    Too high = excessive concentration (not human behavior)
```

#### Peak Flow Rate
```
Maximum:            15.0 L/min (pipe capacity)
99th Percentile:    Should be < 13.5 L/min
Reasoning:          Building meter has physical capacity limits
```

#### Hourly Pattern Checks
```
Morning Peak (6-8 AM):   Expected 3-6 L/min average
Evening Peak (7-9 PM):   Expected 3-7 L/min average
Night Time (0-5 AM):     Expected 0.5-1.5 L/min average
Reasoning:              Validates realistic usage distribution
```

---

## Leak Injection Patterns

### Pattern-Based Leak Events (NOT Random)

#### Type 1: Stress-Induced Leaks (40% of events)
```
Trigger:            High flow periods (top 25th percentile) during peak hours
Peak Hours:         7-8 AM, 7-9 PM
Duration:           1440-4320 minutes (1-3 days)
Severity:           0.20-0.40 × MAX_FLOW
Reasoning:          Pipe stress during peak usage causes gradual failures
Turbidity Effect:   +0.5 to +2.0 NTU (increased sediment)
```

#### Type 2: Seasonal Leaks (30% of events)
```
Trigger:            Cold months (Jan, Feb, Dec)
Cause:              Temperature-induced pipe expansion/contraction
Duration:           240-2880 minutes (4 hours to 2 days)
Severity:           0.25-0.45 × MAX_FLOW
Reasoning:          Material fatigue from thermal cycles
Turbidity Effect:   +0.5 to +2.0 NTU
```

#### Type 3: Night-Time Leaks (30% of events)
```
Trigger:            Night hours (0-6 AM)
Duration:           2880-7200 minutes (2-5 days)
Severity:           0.15-0.30 × MAX_FLOW
Reasoning:          Go unnoticed longer due to low baseline usage
Detection:          Easier to detect but harder to notice initially
Turbidity Effect:   +0.5 to +2.0 NTU
```

### Total Leak Events
```
6-Month Period:     8 leak events (reduced from 12-month 15 events)
Distribution:       3 stress-induced, 2 seasonal, 3 night-time (target distribution)
Selection:          Pattern-based sampling from candidates meeting criteria
Constraint:         No random placement - all follow realistic trigger patterns
```

---

## Assumptions & Limitations

### Assumptions
1. **Single Meter**: All water usage measured at single building meter (aggregate flow)
2. **Instant Aggregation**: Multiple concurrent uses sum linearly at meter
3. **No Temporal Lag**: Water usage immediately reflected in meter reading
4. **Stationary Behavior**: Usage patterns don't evolve over 6-month period
5. **Independent Apartments**: No correlation in usage between apartments
6. **Perfect Occupancy Knowledge**: System knows occupancy rate (in reality, estimated)
7. **No External Events**: Festivals, maintenance, infrastructure issues not modeled

### Limitations
1. **Appliance-Level Detail**: Not tracking individual appliances, only aggregate patterns
2. **Sub-Minute Dynamics**: 1-minute sampling misses very short usage events
3. **Leak Overlap**: Multiple simultaneous leaks not modeled
4. **Repair Time**: Leaks continue until event duration ends (no repair simulation)
5. **Water Quality Events**: Only turbidity modeled, not other quality parameters
6. **Pressure Variations**: Assumes constant supply pressure
7. **Seasonal Events**: Major festivals (Diwali, Holi) not explicitly modeled

### Known Constraints
1. **Maximum Flow**: Hard cap at 15.0 L/min (pipe capacity)
2. **Minimum Flow**: Floor at 0.1 L/min (always some background flow)
3. **People Count**: Must be positive integer
4. **Occupancy Bounds**: 55-98% enforced at all times
5. **No Negative Flows**: All distributions and variations ensure positive values
6. **Fixed Building Size**: 50 apartments constant (not growing/shrinking)

---

## Implementation Notes

### Computational Efficiency
- **No Per-Appliance Loops**: Aggregate calculation avoids nested loops
- **Vectorized Operations**: Numpy used for array operations where possible
- **Pre-Generated Daily Patterns**: 180 daily multipliers generated once
- **Single-Pass Generation**: Each minute calculated once, no backtracking

### Reproducibility
- **Random Seeds**: Not set - each run produces different realistic data
- **Appliance Availability**: Randomized once per dataset, fixed for all samples
- **Daily Patterns**: Randomized once per dataset (180 values)
- **Minute-Level Variation**: New random values each minute

### Data Storage
```
Training Dataset:   259,200 samples (6 months × 1 min/sample)
Test Dataset:       259,200 samples (copy with 8 leak events)
Memory per sample:  ~80 bytes (9 numeric fields + timestamp)
Total size:         ~40 MB uncompressed CSV
```

---

## References

### Standards
- **Bureau of Indian Standards (BIS)**: Residential water consumption guidelines
- **Ministry of Housing and Urban Affairs (MoHUA)**: Plumbing fixture specifications
- **IS 1172**: Indian Standard Code of Basic Requirements for Water Supply, Drainage and Sanitation

### Dataset
- **Original Dataset**: 617-day multi-household monitoring (India region)
- **Calibration**: Deterministic scaling to match Indian fixture flow rates
- **Version**: 1.1-india for all appliance priors

---

## Change Log

### Version 1.0 (Initial)
- Generic apartment building usage patterns
- 12-month simulation period
- Random leak placement

### Version 2.0 (India-Based)
- Integrated India priors data (617-day dataset)
- Reduced to 6-month simulation
- Pattern-based leak injection
- Appliance diversity modeling

### Version 3.0 (Optimized)
- Aggregate flow focus (not individual appliances)
- Multi-layer randomization
- Peak hour categorization
- Validation constraints added
- Daily usage pattern variation

---

*Document Generated: January 28, 2026*  
*Simulation Version: 3.0*  
*Dataset: India Priors v1.1*
