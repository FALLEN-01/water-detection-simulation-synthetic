"""
Run the complete apartment building model training pipeline.

Steps:
1. Generate building-scale synthetic training data (90 days normal + test with 20 diverse leaks)
2. Train Isolation Forest model (7 features) on building aggregate data
3. Auto-copy artifacts to apartment_simulator/artifacts/ for immediate server use
"""

import shutil
import subprocess
import sys
from pathlib import Path

PREPROCESSING_DIR = Path(__file__).parent
ARTIFACTS_SRC = PREPROCESSING_DIR / "artifacts"
ARTIFACTS_DST = PREPROCESSING_DIR.parent / "artifacts"


def run_step(name, script):
    """Run a preprocessing step and exit on failure."""
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script],
        cwd=PREPROCESSING_DIR,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\nERROR in {name}!  (exit code {result.returncode})")
        sys.exit(1)


def copy_artifacts():
    """Copy freshly trained artifacts to apartment_simulator/artifacts/."""
    print("\n" + "=" * 70)
    print("COPYING ARTIFACTS → apartment_simulator/artifacts/")
    print("=" * 70)

    files = [
        "isolation_forest_building.pkl",
        "scaler_building.pkl",
        "calibration_building.json",
        "metrics_building.json",
    ]

    ARTIFACTS_DST.mkdir(exist_ok=True)

    for fname in files:
        src = ARTIFACTS_SRC / fname
        dst = ARTIFACTS_DST / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  [OK] {fname}")
        else:
            print(f"  [WARN] {fname} not found in preprocessing/artifacts/")

    print(f"\n  All artifacts ready at: {ARTIFACTS_DST}")


if __name__ == "__main__":
    print("=" * 70)
    print("APARTMENT BUILDING MODEL TRAINING PIPELINE")
    print("7-feature Isolation Forest — diverse leak coverage")
    print("=" * 70)

    # Step 1: Generate data
    run_step(
        "Generate Building-Scale Training Data (diverse leaks)",
        "generate_data.py"
    )

    # Step 2: Train model
    run_step(
        "Train Isolation Forest (7 features)",
        "train_model.py"
    )

    # Step 3: Auto-copy to runtime artifacts
    copy_artifacts()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nNext step: restart the simulator to use the new model.")
    print("  Local:  python backend/server.py")
    print("  Docker: docker-compose up --build simulator")
