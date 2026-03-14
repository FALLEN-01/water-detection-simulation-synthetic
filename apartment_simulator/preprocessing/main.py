"""
Run the complete apartment building model training pipeline.

Steps:
1. Generate building-scale synthetic training data (6 months of normal + test with leaks)
2. Train Isolation Forest model on building aggregate data
3. Calibrate detection thresholds for apartment buildings
4. Output models ready for deployment
"""

import subprocess
import sys
from pathlib import Path

PREPROCESSING_DIR = Path(__file__).parent

def run_step(name, script):
    """Run a preprocessing step."""
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script],
        cwd=PREPROCESSING_DIR,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\nERROR in {name}!")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 70)
    print("APARTMENT BUILDING MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Generate data
    run_step(
        "Generate Building-Scale Training Data",
        "generate_data.py"
    )

    # Step 2: Train model
    run_step(
        "Train Isolation Forest (Building Scale)",
        "train_model.py"
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Copy new models to apartment_simulator/artifacts/:")
    print("   - isolation_forest_building.pkl")
    print("   - scaler_building.pkl")
    print("   - calibration_building.json")
    print("\n2. Update backend/server.py to use building-scale models")
    print("\n3. Restart the simulator for improved building-level detection")
