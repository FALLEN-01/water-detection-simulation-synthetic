"""
ESP32-S3 Optimized Water Leak Detection Model
==============================================
This script creates a smaller, optimized LSTM autoencoder specifically for ESP32-S3:
- Reduced model size (fewer parameters)
- Quantized to int8 for memory efficiency
- Converts to TFLite format
- Generates C array for ESP32 deployment

Target: ESP32-S3 with 512KB SRAM, 8MB PSRAM
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

print("\n" + "="*60)
print("ESP32-S3 OPTIMIZED MODEL")
print("="*60)

# Load datasets
print("\nLoading datasets...")
train_df = pd.read_csv('water_train.csv')
test_df = pd.read_csv('water_test.csv')

# Configuration for ESP32-S3
WINDOW_SIZE = 10  # 10-minute sliding window
ESP32_MODEL_DIR = "esp32_s3_model"

# Check if turbidity sensor is available
has_turbidity = 'turbidity' in train_df.columns

if has_turbidity:
    feature_cols = ['flow_normalized', 'turbidity', 'flow_duration', 'hour', 'is_weekend']
    print("Using turbidity sensor (auxiliary)")
else:
    feature_cols = ['flow_normalized', 'flow_duration', 'hour', 'is_weekend']
    print("Using flow only (no turbidity)")

def create_windows(data, window_size):
    """Create sliding windows from time series data"""
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)

print("\nPreparing data...")
train_data = train_df[feature_cols].values
test_data = test_df[feature_cols].values
test_labels = test_df['label'].values

train_windows = create_windows(train_data, WINDOW_SIZE)
test_windows = create_windows(test_data, WINDOW_SIZE)
test_window_labels = test_labels[WINDOW_SIZE - 1:]

print(f"Training samples: {len(train_windows):,} | Test samples: {len(test_windows):,} | Leaks: {(test_window_labels==1).sum():,}")

# ============================================================================
# ESP32-S3 OPTIMIZED MODEL
# ============================================================================
print("\n" + "="*60)
print("Building Model")
print("="*60)

timesteps = WINDOW_SIZE
n_features = len(feature_cols)

# REDUCED architecture for ESP32-S3
# Original: 32->16->16->32 (16,549 params, 271KB)
# ESP32-S3: 16->8->8->16 (~4,000 params, ~70KB)

encoder_input = Input(shape=(timesteps, n_features), name='encoder_input')
x = LSTM(16, activation='relu', return_sequences=True, name='encoder_lstm1')(encoder_input)
x = Dropout(0.2, name='encoder_dropout1')(x)
x = LSTM(8, activation='relu', return_sequences=False, name='encoder_lstm2')(x)
encoded = Dropout(0.2, name='encoder_dropout2')(x)

# Decoder
x = RepeatVector(timesteps, name='repeat_vector')(encoded)
x = LSTM(8, activation='relu', return_sequences=True, name='decoder_lstm1')(x)
x = Dropout(0.2, name='decoder_dropout1')(x)
x = LSTM(16, activation='relu', return_sequences=True, name='decoder_lstm2')(x)
decoded = TimeDistributed(Dense(n_features), name='output')(x)

autoencoder = Model(encoder_input, decoded, name='esp32_autoencoder')
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

print(f"\nModel Parameters: {autoencoder.count_params():,}")
print("Target: <5,000 parameters for ESP32-S3")

# Train the model
print("\n" + "="*60)
print("Training ESP32-S3 Model")
print("="*60)

early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

history = autoencoder.fit(
    train_windows, train_windows,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Calculate threshold
train_reconstructed = autoencoder.predict(train_windows, verbose=0)
train_errors = np.mean(np.mean((train_windows - train_reconstructed) ** 2, axis=2), axis=1)
THRESHOLD = np.percentile(train_errors, 99)

print(f"\nThreshold: {THRESHOLD:.6f}")

# Test the model
print("\nTesting...")
test_reconstructed = autoencoder.predict(test_windows, verbose=0)
test_errors = np.mean(np.mean((test_windows - test_reconstructed) ** 2, axis=2), axis=1)
anomalies_detected = test_errors > THRESHOLD

# Calculate metrics
true_positives = np.sum((anomalies_detected) & (test_window_labels == 1))
false_positives = np.sum((anomalies_detected) & (test_window_labels == 0))
true_negatives = np.sum((~anomalies_detected) & (test_window_labels == 0))
false_negatives = np.sum((~anomalies_detected) & (test_window_labels == 1))

accuracy = (true_positives + true_negatives) / len(test_windows)
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n" + "="*60)
print("ESP32-S3 Model Performance")
print("="*60)
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"TP: {true_positives:,} | FP: {false_positives:,} | TN: {true_negatives:,} | FN: {false_negatives:,}")

# ============================================================================
# SAVE MODELS FOR ESP32-S3
# ============================================================================
print(f"\n" + "="*60)
print("Exporting Models")
print("="*60)

# Create directory if it doesn't exist
os.makedirs(ESP32_MODEL_DIR, exist_ok=True)

# 1. Save Keras model (for reference)
keras_path = os.path.join(ESP32_MODEL_DIR, "esp32_autoencoder.h5")
autoencoder.save(keras_path)
keras_size = os.path.getsize(keras_path) / 1024
print(f"\n✓ Saved Keras model: {keras_path} ({keras_size:.1f} KB)")

# 2. Convert to TFLite (float32)
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
# Fix LSTM conversion issues
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Enable select TF ops for LSTM
]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

tflite_path = os.path.join(ESP32_MODEL_DIR, "esp32_autoencoder_float32.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
tflite_size = os.path.getsize(tflite_path) / 1024
print(f"✓ Saved TFLite (float32): {tflite_path} ({tflite_size:.1f} KB)")

# 3. Convert to Quantized TFLite (int8) - CRITICAL for ESP32-S3
print("Quantizing to int8...")

def representative_dataset():
    """Representative dataset for quantization"""
    for i in range(100):
        # Use random samples from training data
        idx = np.random.randint(0, len(train_windows))
        sample = train_windows[idx:idx+1].astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# Fix LSTM conversion issues for quantization
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_quant_model = converter.convert()
    
    tflite_quant_path = os.path.join(ESP32_MODEL_DIR, "esp32_autoencoder_int8.tflite")
    with open(tflite_quant_path, "wb") as f:
        f.write(tflite_quant_model)
    tflite_quant_size = os.path.getsize(tflite_quant_path) / 1024
    print(f"✓ Saved TFLite (int8): {tflite_quant_path} ({tflite_quant_size:.1f} KB)")
    print(f"  Size reduction: {((tflite_size - tflite_quant_size) / tflite_size * 100):.1f}%")
except Exception as e:
    print(f"✗ Quantization failed: {e}")
    print("  Using float32 model instead")
    tflite_quant_path = tflite_path

# 4. Save model metadata
metadata = {
    "model_name": "ESP32-S3 Water Leak Detector",
    "version": "1.0",
    "window_size": WINDOW_SIZE,
    "features": feature_cols,
    "n_features": n_features,
    "threshold": float(THRESHOLD),
    "parameters": int(autoencoder.count_params()),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

import json
metadata_path = os.path.join(ESP32_MODEL_DIR, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\n✓ Saved metadata: {metadata_path}")

print(f"\n" + "="*60)
print("ESP32-S3 Model Export Complete!")
print("="*60)
print(f"\nFiles created in '{ESP32_MODEL_DIR}/':")
print(f"  1. esp32_autoencoder.h5 ({keras_size:.1f} KB) - Keras model")
print(f"  2. esp32_autoencoder_float32.tflite ({tflite_size:.1f} KB) - TFLite float32")
print(f"  3. esp32_autoencoder_int8.tflite ({tflite_quant_size:.1f} KB) - TFLite int8 (RECOMMENDED)")
print(f"  4. model_metadata.json - Model configuration")
print(f"\nNext step: Convert TFLite to C array using tflite_to_c_array.py")
