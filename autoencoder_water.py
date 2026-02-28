"""
Autoencoder (LSTM) — Water Leak Detection
==========================================
Trained on data from MULTIPLE building profiles so it learns
generalized "normal" usage patterns, not just one building.

Architecture: LSTM encoder-decoder with slightly larger capacity
(64→32 units) for better temporal pattern learning without
over-fitting to one building's quirks.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                     RepeatVector, TimeDistributed,
                                     BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

print("\nAutoencoder (LSTM) — Water Leak Detection")
print("=" * 50)
print("Loading datasets...")

train_df = pd.read_csv('water_train.csv')
test_df  = pd.read_csv('water_test.csv')

# Window configuration — 10 minutes captures one full usage event cycle
WINDOW_SIZE = 10

# Feature columns — use all available sensors
# turbidity is an auxiliary sensor (not always present)
has_turbidity = 'turbidity' in train_df.columns

if has_turbidity:
    feature_cols = ['flow_normalized', 'turbidity',
                    'flow_duration', 'hour', 'is_weekend']
    print("Using turbidity (auxiliary sensor) for enhanced detection")
else:
    feature_cols = ['flow_normalized', 'flow_duration', 'hour', 'is_weekend']
    print("Turbidity not available — using flow features only")

n_features = len(feature_cols)
print(f"Features ({n_features}): {feature_cols}")
print(f"Window size: {WINDOW_SIZE} minutes")

# =====================================================
# Sliding window creation
# =====================================================
def create_windows(data, window_size):
    """Create overlapping sliding windows from time series"""
    n = len(data) - window_size + 1
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=(window_size, data.shape[1])
    ).reshape(n, window_size, data.shape[1])
    return windows.astype(np.float32)

print("\nCreating sliding windows...")
train_data = train_df[feature_cols].values.astype(np.float32)
test_data  = test_df[feature_cols].values.astype(np.float32)
test_labels = test_df['label'].values

train_windows = create_windows(train_data, WINDOW_SIZE)
test_windows  = create_windows(test_data,  WINDOW_SIZE)
test_window_labels = test_labels[WINDOW_SIZE - 1:]

print(f"Training windows: {len(train_windows):,}  shape: {train_windows.shape}")
print(f"Testing windows:  {len(test_windows):,}  "
      f"({(test_window_labels==1).sum():,} leaks / "
      f"{(test_window_labels==0).sum():,} normal)")

# =====================================================
# Model Architecture
# =====================================================
print("\nBuilding LSTM Autoencoder...")

inp = Input(shape=(WINDOW_SIZE, n_features), name='encoder_input')

# --- Encoder ---
x = LSTM(64, activation='tanh', return_sequences=True, name='enc_lstm1')(inp)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = LSTM(32, activation='tanh', return_sequences=False, name='enc_lstm2')(x)
x = BatchNormalization()(x)
encoded = Dropout(0.15)(x)   # compressed representation

# --- Decoder ---
x = RepeatVector(WINDOW_SIZE)(encoded)
x = LSTM(32, activation='tanh', return_sequences=True, name='dec_lstm1')(x)
x = BatchNormalization()(x)
x = Dropout(0.15)(x)
x = LSTM(64, activation='tanh', return_sequences=True, name='dec_lstm2')(x)
decoded = TimeDistributed(Dense(n_features), name='output')(x)

autoencoder = Model(inp, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print(f"Parameters: {autoencoder.count_params():,}")
autoencoder.summary(print_fn=lambda x: None)  # silent summary

# =====================================================
# Training
# =====================================================
print("\nTraining...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-5, verbose=1),
]

history = autoencoder.fit(
    train_windows, train_windows,
    epochs=60,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# =====================================================
# Threshold — 99th percentile of training reconstruction error
# =====================================================
print("\nCalculating anomaly threshold...")
train_recon = autoencoder.predict(train_windows, batch_size=512, verbose=0)
train_errors = np.mean(np.mean((train_windows - train_recon) ** 2, axis=2), axis=1)
THRESHOLD = float(np.percentile(train_errors, 99))
print(f"Threshold (99th pct of training errors): {THRESHOLD:.6f}")

# =====================================================
# Evaluation on test set
# =====================================================
print("Evaluating on test data...")
test_recon = autoencoder.predict(test_windows, batch_size=512, verbose=0)
test_errors = np.mean(np.mean((test_windows - test_recon) ** 2, axis=2), axis=1)
anomalies = test_errors > THRESHOLD

TP = int(np.sum(anomalies & (test_window_labels == 1)))
FP = int(np.sum(anomalies & (test_window_labels == 0)))
TN = int(np.sum(~anomalies & (test_window_labels == 0)))
FN = int(np.sum(~anomalies & (test_window_labels == 1)))

accuracy  = (TP + TN) / len(test_windows)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\n{'='*40}")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  TP: {TP:,} | FP: {FP:,} | TN: {TN:,} | FN: {FN:,}")
print(f"{'='*40}")

# =====================================================
# Visualizations
# =====================================================
print("\nGenerating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training loss
ax = axes[0, 0]
ax.plot(history.history['loss'], label='Train Loss')
if 'val_loss' in history.history:
    ax.plot(history.history['val_loss'], label='Val Loss')
ax.set_title('Autoencoder Training Loss', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
ax.legend(); ax.grid(True, alpha=0.3)

# Reconstruction error distribution
ax = axes[0, 1]
normal_err = test_errors[test_window_labels == 0]
leak_err   = test_errors[test_window_labels == 1]
ax.hist(normal_err, bins=100, alpha=0.6, label='Normal', color='steelblue', density=True)
ax.hist(leak_err,   bins=100, alpha=0.6, label='Leak',   color='crimson',   density=True)
ax.axvline(THRESHOLD, color='lime', linestyle='--', lw=2,
           label=f'Threshold ({THRESHOLD:.4f})')
ax.set_title('Reconstruction Error Distribution', fontweight='bold')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)

# Error over time
ax = axes[1, 0]
n = min(15000, len(test_errors))
colors = ['crimson' if l else 'steelblue' for l in test_window_labels[:n]]
ax.scatter(range(n), test_errors[:n], c=colors, s=0.5, alpha=0.5)
ax.axhline(THRESHOLD, color='lime', lw=2, linestyle='--', label='Threshold')
ax.set_title('Reconstruction Error Over Time', fontweight='bold')
ax.set_xlabel('Window'); ax.set_ylabel('MSE'); ax.legend(); ax.grid(True, alpha=0.3)

# Confusion matrix
ax = axes[1, 1]
cm = np.array([[TN, FP], [FN, TP]])
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred Normal', 'Pred Leak'])
ax.set_yticklabels(['Actual Normal', 'Actual Leak'])
ax.set_title('Confusion Matrix', fontweight='bold')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('autoencoder_results.png', dpi=150, bbox_inches='tight')
print("Saved: autoencoder_results.png")
plt.close()

# =====================================================
# Save model + threshold
# =====================================================
print("\nSaving model...")
os.makedirs("models", exist_ok=True)

autoencoder.save("models/autoencoder_lstm.keras")
print("Saved: models/autoencoder_lstm.keras")

with open("models/autoencoder_lstm_threshold.txt", 'w') as f:
    f.write(str(THRESHOLD))
print(f"Saved: models/autoencoder_lstm_threshold.txt ({THRESHOLD:.6f})")

print("\nAutoencoder training complete!")
