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


def inject_leaks_into_data(df, num_leaks=8, leak_duration_minutes=360):
    """
    Inject realistic leak events into test data.

    Leak patterns:
    - Stress-induced: During peak hours (7-8 AM, 7-9 PM)
    - Seasonal: Winter months
    - Night-time: Midnight-6 AM
    """
    print(f"Injecting {num_leaks} leak events into test data...")

    df = df.copy()
    df['is_leak'] = False
    df['leak_intensity'] = 0.0

    leak_types = ['stress', 'seasonal', 'night']

    for i in range(num_leaks):
        leak_type = leak_types[i % len(leak_types)]

        # Randomize leak parameters
        intensity = np.random.uniform(2.0, 10.0)  # 2-10 L/min for building

        if leak_type == 'stress':
            # Peak hours: 7-8 AM or 7-9 PM
            hour = np.random.choice([7, 19])
            day = np.random.randint(0, len(df) // MINUTES_PER_DAY)
            start_min = day * MINUTES_PER_DAY + hour * 60 + np.random.randint(0, 60)

        elif leak_type == 'seasonal':
            # Winter months only (assume first 60 days)
            day = np.random.randint(0, 60)
            start_min = day * MINUTES_PER_DAY + np.random.randint(0, MINUTES_PER_DAY)

        else:  # night_time
            # Midnight to 6 AM
            hour = np.random.randint(0, 6)
            day = np.random.randint(0, len(df) // MINUTES_PER_DAY)
            start_min = day * MINUTES_PER_DAY + hour * 60 + np.random.randint(0, 60)

        end_min = min(start_min + leak_duration_minutes, len(df))

        df.loc[start_min:end_min, 'is_leak'] = True
        df.loc[start_min:end_min, 'leak_intensity'] = intensity
        df.loc[start_min:end_min, 'flow_lpm'] += intensity

        print(f"  Leak {i+1}: {leak_type:10s} | Day {start_min//MINUTES_PER_DAY:3d} | "
              f"Intensity {intensity:.1f} L/min | Duration {leak_duration_minutes} min")

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
    test_df = inject_leaks_into_data(test_df, num_leaks=8, leak_duration_minutes=360)
    save_dataset(
        test_df,
        "water_test_building",
        f"6 months × 50 apartments + 8 leaks = {len(test_df):,} samples "
        f"({test_df['is_leak'].sum():,} leak minutes)"
    )

    print("\n" + "=" * 70)
    print(f"Data generation complete!")
    print(f"Training set: {len(train_df):,} samples (normal only)")
    print(f"Test set: {len(test_df):,} samples ({test_df['is_leak'].sum():,} with leaks)")
    print("=" * 70)
