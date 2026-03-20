"""
Legacy training pipeline (deprecated).

This directory contains an older, monolithic training pipeline that generates
datasets from `priors_india/` and trains:
- LSTM autoencoder
- Isolation Forest

The newer codepaths under `household_simulator/` and `apartment_simulator/`
provide the current real-time demo systems.

This script intentionally executes by importing the step modules in order.
Run it from within `old_model_training/`:

```bash
py main.py
```
"""

from __future__ import annotations

import os


def main() -> None:
    # Reduce TensorFlow logging noise (when used by legacy autoencoder module).
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    print("\nWater Leak Detection System — Full Pipeline")
    print("=" * 55)

    print("\nStep 1: Generating datasets from priors_india...")
    import emulate_data  # noqa: F401

    print("\nStep 2: Training LSTM Autoencoder...")
    import autoencoder_water  # noqa: F401

    print("\nStep 3: Training Isolation Forest...")
    import isolation_water  # noqa: F401

    print("\n" + "=" * 55)
    print("Pipeline complete!")
    print("  water_train.csv / water_test.csv — datasets")
    print("  models/autoencoder_lstm.keras    — LSTM model")
    print("  models/autoencoder_lstm_threshold.txt")
    print("  models/isolation_forest_model.pkl")
    print("  autoencoder_results.png / isolation_forest_results.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
