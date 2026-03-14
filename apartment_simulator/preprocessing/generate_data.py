"""
Generate synthetic training data for apartment building anomaly detection.

Creates 6 months of synthetic water flow data for 50 apartments (aggregated),
split into training (normal only) and test (with injected leaks) datasets.

Leak coverage in test set:
- Intensity range: 0.1 - 15 L/min (wide spectrum)
- Types: sustained_drip, slow_leak, stress, seasonal, night, ramp
- 20 leak events for good label coverage across diverse scenarios
- Ramp-style leaks included for gradual onset detection

Output: CSV files compatible with model training
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from live_simulator import LiveApartmentBuildingDataGenerator

# Configuration
DAYS = 180  # 6 months of data
APARTMENTS = 50
APPLIANCES_PATH = Path(__file__).parent.parent / "artifacts" / "all_appliances.json"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

MINUTES_PER_DAY = 1440


def generate_building_training_data(days=180, seed=42):
    """
    Generate clean (normal) training data for building aggregate.
    No leaks, just normal appliance usage.
    """
    print(f"Generating {days} days of normal building flow data...")

    generator = LiveApartmentBuildingDataGenerator(
        APPLIANCES_PATH,
        num_apartments=APARTMENTS,
        seed=seed
    )

    samples = []
    for day in range(days):
        for minute in range(MINUTES_PER_DAY):
            flow = generator.next()
            samples.append({
                'timestamp': day * MINUTES_PER_DAY + minute,
                'day': day,
                'minute_of_day': minute,
                'hour': minute // 60,
                'flow_lpm': flow
            })

        if (day + 1) % 30 == 0:
            print(f"  Generated day {day + 1}/{days}...")

    df = pd.DataFrame(samples)
    return df


def inject_leaks_into_data(df, num_leaks=20, seed=99):
    """
    Inject diverse realistic leak events into test data.

    Leak types:
    - sustained_drip:   0.1 - 1.5 L/min, 4-24 hours
    - slow_leak:        0.5 - 3.0 L/min, 3-8 hours
    - stress:           3.0 - 10.0 L/min, 1-3 hours (peak hours)
    - seasonal:         1.0 - 8.0 L/min, 6 hours (winter)
    - night:            0.5 - 5.0 L/min, midnight-6 AM
    - ramp:             gradual onset 0.2 → 5.0 L/min over 60-120 min

    This gives the Isolation Forest exposure to the full spectrum of leak sizes,
    especially low-intensity sustained leaks that the original model never saw.
    """
    rng = np.random.default_rng(seed)
    print(f"Injecting {num_leaks} diverse leak events into test data...")

    df = df.copy()
    df['is_leak'] = False
    df['leak_intensity'] = 0.0

    # Proportional distribution of leak types
    leak_types = [
        'sustained_drip',   # ~25%
        'sustained_drip',
        'sustained_drip',
        'sustained_drip',
        'slow_leak',        # ~25%
        'slow_leak',
        'slow_leak',
        'slow_leak',
        'stress',           # ~20%
        'stress',
        'stress',
        'stress',
        'seasonal',         # ~10%
        'seasonal',
        'night',            # ~10%
        'night',
        'ramp',             # ~10%
        'ramp',
        'slow_leak',
        'sustained_drip',
    ]

    total_days = len(df) // MINUTES_PER_DAY
    injected = 0

    for i in range(num_leaks):
        leak_type = leak_types[i % len(leak_types)]

        if leak_type == 'sustained_drip':
            # Key new type: very low intensity, very long duration
            intensity = float(rng.uniform(0.1, 1.5))
            duration = int(rng.integers(240, 1440))  # 4-24 hours
            day = int(rng.integers(5, total_days - 2))
            start_min = day * MINUTES_PER_DAY + int(rng.integers(0, MINUTES_PER_DAY - duration))

        elif leak_type == 'slow_leak':
            intensity = float(rng.uniform(0.5, 3.0))
            duration = int(rng.integers(180, 480))  # 3-8 hours
            day = int(rng.integers(5, total_days - 2))
            start_min = day * MINUTES_PER_DAY + int(rng.integers(0, MINUTES_PER_DAY))

        elif leak_type == 'stress':
            intensity = float(rng.uniform(3.0, 10.0))
            duration = int(rng.integers(60, 180))  # 1-3 hours
            hour = int(rng.choice([7, 8, 19, 20]))
            day = int(rng.integers(5, total_days - 2))
            start_min = day * MINUTES_PER_DAY + hour * 60 + int(rng.integers(0, 60))

        elif leak_type == 'seasonal':
            intensity = float(rng.uniform(1.0, 8.0))
            duration = int(rng.integers(240, 720))  # 4-12 hours
            day = int(rng.integers(0, min(60, total_days - 2)))
            start_min = day * MINUTES_PER_DAY + int(rng.integers(0, MINUTES_PER_DAY))

        elif leak_type == 'night':
            intensity = float(rng.uniform(0.5, 5.0))
            duration = int(rng.integers(60, 360))  # 1-6 hours
            hour = int(rng.integers(0, 5))
            day = int(rng.integers(5, total_days - 2))
            start_min = day * MINUTES_PER_DAY + hour * 60 + int(rng.integers(0, 60))

        elif leak_type == 'ramp':
            # Gradual onset: starts near zero, ramps up to peak
            peak_intensity = float(rng.uniform(2.0, 8.0))
            duration = int(rng.integers(120, 360))  # 2-6 hours
            day = int(rng.integers(5, total_days - 2))
            start_min = day * MINUTES_PER_DAY + int(rng.integers(0, MINUTES_PER_DAY))
            # Will handle ramp separately below
            end_min = min(start_min + duration, len(df) - 1)
            for m in range(start_min, end_min):
                if m < len(df):
                    progress = (m - start_min) / max(1, duration)
                    eff_intensity = peak_intensity * progress
                    df.loc[m, 'is_leak'] = True
                    df.loc[m, 'leak_intensity'] = eff_intensity
                    df.loc[m, 'flow_lpm'] += eff_intensity
            injected += 1
            print(f"  Leak {i+1:2d}: {leak_type:16s} | Day {start_min//MINUTES_PER_DAY:3d} | "
                  f"Peak {peak_intensity:.1f} L/min | Duration {duration} min")
            continue

        else:
            intensity = float(rng.uniform(0.5, 5.0))
            duration = 360
            day = int(rng.integers(5, total_days - 2))
            start_min = day * MINUTES_PER_DAY

        end_min = min(start_min + duration, len(df) - 1)

        df.loc[start_min:end_min, 'is_leak'] = True
        df.loc[start_min:end_min, 'leak_intensity'] = intensity
        df.loc[start_min:end_min, 'flow_lpm'] += intensity

        injected += 1
        print(f"  Leak {i+1:2d}: {leak_type:16s} | Day {start_min//MINUTES_PER_DAY:3d} | "
              f"Intensity {intensity:.1f} L/min | Duration {duration} min")

    # Clip flow to realistic range
    df['flow_lpm'] = df['flow_lpm'].clip(lower=0.0)

    print(f"\n  Total leak minutes: {df['is_leak'].sum():,} / {len(df):,} "
          f"({100 * df['is_leak'].mean():.2f}%)")
    return df


def save_dataset(df, filename, description=""):
    """Save dataset to CSV."""
    path = OUTPUT_DIR / f"{filename}.csv"
    df.to_csv(path, index=False)
    print(f"[OK] Saved {filename}: {len(df)} samples -> {path}")
    if description:
        print(f"     {description}")


if __name__ == "__main__":
    print("=" * 70)
    print("APARTMENT BUILDING SYNTHETIC DATA GENERATION")
    print("=" * 70)

    # Generate training data (normal only)
    train_df = generate_building_training_data(days=DAYS // 2, seed=42)
    save_dataset(
        train_df,
        "water_train_building",
        f"90 days × 50 apartments = {len(train_df):,} normal samples"
    )

    # Generate test data (normal + diverse leaks)
    test_df = generate_building_training_data(days=DAYS // 2, seed=123)
    test_df = inject_leaks_into_data(test_df, num_leaks=20, seed=99)
    save_dataset(
        test_df,
        "water_test_building",
        f"90 days × 50 apartments + 20 diverse leaks = {len(test_df):,} samples "
        f"({test_df['is_leak'].sum():,} leak minutes)"
    )

    print("\n" + "=" * 70)
    print("Data generation complete!")
    print(f"Training set: {len(train_df):,} samples (normal only)")
    print(f"Test set: {len(test_df):,} samples "
          f"({test_df['is_leak'].sum():,} with leaks, "
          f"{100*test_df['is_leak'].mean():.2f}%)")
    print("=" * 70)
