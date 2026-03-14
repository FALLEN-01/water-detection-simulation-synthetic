"""
Generate synthetic training data for apartment building anomaly detection.

Creates 6 months of synthetic water flow data for 50 apartments (aggregated),
split into training (normal only) and test (with injected leaks) datasets.

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


def inject_leaks_into_data(df, num_low_sustained=6, num_medium_sustained=6, num_high_burst=6):
    """
    Inject diverse leak patterns into test data for better model training.

    Three leak categories:
    1. Low-sustained: 0.5-2.0 L/min for 3-7 days (hard to detect!)
    2. Medium-sustained: 2.0-5.0 L/min for 1-3 days
    3. High-burst: 10-20 L/min for 6-24 hours (easy to detect)
    """
    df = df.copy()
    df['is_leak'] = False
    df['leak_intensity'] = 0.0
    df['leak_type'] = ''

    leak_id = 0

    # LOW-SUSTAINED LEAKS (hardest to detect - focus here!)
    print(f"\nInjecting {num_low_sustained} low-sustained leaks (0.5-2.0 L/min for days)...")
    for i in range(num_low_sustained):
        intensity = np.random.uniform(0.5, 2.0)  # Subtle
        duration_days = np.random.randint(3, 8)  # 3-7 days
        duration_minutes = duration_days * MINUTES_PER_DAY

        day = np.random.randint(0, (len(df) // MINUTES_PER_DAY) - duration_days)
        start_min = day * MINUTES_PER_DAY
        end_min = min(start_min + duration_minutes, len(df))

        df.loc[start_min:end_min, 'is_leak'] = True
        df.loc[start_min:end_min, 'leak_intensity'] = intensity
        df.loc[start_min:end_min, 'flow_lpm'] += intensity
        df.loc[start_min:end_min, 'leak_type'] = 'low-sustained'

        leak_id += 1
        print(f"  [{leak_id}] Low-sustained: Day {day:3d} | "
              f"Intensity {intensity:.2f} L/min | Duration {duration_days} days ({duration_minutes} min)")

    # MEDIUM-SUSTAINED LEAKS
    print(f"\nInjecting {num_medium_sustained} medium-sustained leaks (2-5 L/min for days)...")
    for i in range(num_medium_sustained):
        intensity = np.random.uniform(2.0, 5.0)
        duration_days = np.random.randint(1, 4)  # 1-3 days
        duration_minutes = duration_days * MINUTES_PER_DAY

        day = np.random.randint(0, (len(df) // MINUTES_PER_DAY) - duration_days)
        start_min = day * MINUTES_PER_DAY + np.random.randint(0, MINUTES_PER_DAY)
        end_min = min(start_min + duration_minutes, len(df))

        df.loc[start_min:end_min, 'is_leak'] = True
        df.loc[start_min:end_min, 'leak_intensity'] = intensity
        df.loc[start_min:end_min, 'flow_lpm'] += intensity
        df.loc[start_min:end_min, 'leak_type'] = 'medium-sustained'

        leak_id += 1
        print(f"  [{leak_id}] Medium-sustained: Day {day:3d} | "
              f"Intensity {intensity:.2f} L/min | Duration {duration_days} days ({duration_minutes} min)")

    # HIGH-BURST LEAKS (easy to detect)
    print(f"\nInjecting {num_high_burst} high-burst leaks (10-20 L/min for hours)...")
    for i in range(num_high_burst):
        intensity = np.random.uniform(10.0, 20.0)
        duration_hours = np.random.randint(6, 25)  # 6-24 hours
        duration_minutes = duration_hours * 60

        # Place during peak hours for realism
        hour = np.random.choice([7, 14, 19])
        day = np.random.randint(0, len(df) // MINUTES_PER_DAY)
        start_min = day * MINUTES_PER_DAY + hour * 60 + np.random.randint(0, 60)
        end_min = min(start_min + duration_minutes, len(df))

        df.loc[start_min:end_min, 'is_leak'] = True
        df.loc[start_min:end_min, 'leak_intensity'] = intensity
        df.loc[start_min:end_min, 'flow_lpm'] += intensity
        df.loc[start_min:end_min, 'leak_type'] = 'high-burst'

        leak_id += 1
        print(f"  [{leak_id}] High-burst: Day {day:3d} Hour {hour:2d} | "
              f"Intensity {intensity:.1f} L/min | Duration {duration_hours} hours ({duration_minutes} min)")

    total_leak_minutes = (df['is_leak']).sum()
    print(f"\n✓ Total injected leak minutes: {total_leak_minutes:,} ({100*total_leak_minutes/len(df):.2f}%)")

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
        f"6 months × 50 apartments = {len(train_df):,} normal samples"
    )

    # Generate test data (normal + leaks)
    test_df = generate_building_training_data(days=DAYS // 2, seed=123)
    test_df = inject_leaks_into_data(test_df, num_low_sustained=6, num_medium_sustained=6, num_high_burst=6)
    save_dataset(
        test_df,
        "water_test_building",
        f"6 months × 50 apartments + 18 leaks = {len(test_df):,} samples "
        f"({test_df['is_leak'].sum():,} leak minutes)"
    )

    print("\n" + "=" * 70)
    print(f"Data generation complete!")
    print(f"Training set: {len(train_df):,} samples (normal only)")
    print(f"Test set: {len(test_df):,} samples ({test_df['is_leak'].sum():,} with leaks)")
    print("=" * 70)
