import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("\nWater Leak Detection System — Full Pipeline")
print("=" * 55)

print("\nStep 1: Generating datasets from priors_india...")
import emulate_data

print("\nStep 2: Training LSTM Autoencoder...")
import autoencoder_water

print("\nStep 3: Training Isolation Forest...")
import isolation_water

print("\n" + "=" * 55)
print("Pipeline complete!")
print("  water_train.csv / water_test.csv — datasets")
print("  models/autoencoder_lstm.keras    — LSTM model")
print("  models/autoencoder_lstm_threshold.txt")
print("  models/isolation_forest_model.pkl")
print("  autoencoder_results.png / isolation_forest_results.png")
print("=" * 55)
