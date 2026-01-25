import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

print("\nAutoencoder Model")
print("Loading datasets...")

train_df = pd.read_csv('water_train.csv')
test_df = pd.read_csv('water_test.csv')

feature_cols = ['flow_normalized', 'pressure', 'turbidity', 'temperature', 
                'flow_duration', 'hour', 'is_weekend']

train_data = train_df[feature_cols].values
test_data = test_df[feature_cols].values
test_labels = test_df['label'].values

print(f"Training: {len(train_data):,} samples")
print(f"Testing: {len(test_data):,} samples ({(test_labels==1).sum():,} leaks)")


print("\nBuilding model...")
input_dim = len(feature_cols)

inp = Input(shape=(input_dim,))
x = Dense(16, activation="relu")(inp)
x = Dropout(0.2)(x)
x = Dense(8, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(4, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(16, activation="relu")(x)
out = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

print("Training...")
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

history = autoencoder.fit(
    train_data, train_data,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

train_reconstructed = autoencoder.predict(train_data, verbose=0)
train_errors = np.mean((train_data - train_reconstructed) ** 2, axis=1)
THRESHOLD = np.percentile(train_errors, 99)

print(f"\nThreshold: {THRESHOLD:.6f}")
print("Testing...")
test_reconstructed = autoencoder.predict(test_data, verbose=0)
test_errors = np.mean((test_data - test_reconstructed) ** 2, axis=1)
anomalies_detected = test_errors > THRESHOLD

# Calculate metrics
true_positives = np.sum((anomalies_detected) & (test_labels == 1))
false_positives = np.sum((anomalies_detected) & (test_labels == 0))
true_negatives = np.sum((~anomalies_detected) & (test_labels == 0))
false_negatives = np.sum((~anomalies_detected) & (test_labels == 1))

accuracy = (true_positives + true_negatives) / len(test_data)
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nResults:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1 Score:  {f1:.3f}")
print(f"  TP: {true_positives:,} | FP: {false_positives:,} | TN: {true_negatives:,} | FN: {false_negatives:,}")

print("\nGenerating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training Loss
ax1 = axes[0, 0]
ax1.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Autoencoder Training Loss', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Mean Squared Error')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Reconstruction Error Distribution
ax2 = axes[0, 1]
normal_errors = test_errors[test_labels == 0]
leak_errors = test_errors[test_labels == 1]
ax2.hist(normal_errors, bins=100, alpha=0.6, label='Normal', color='blue', density=True)
ax2.hist(leak_errors, bins=100, alpha=0.6, label='Leaks', color='red', density=True)
ax2.axvline(THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Threshold ({THRESHOLD:.4f})')
ax2.set_title('Reconstruction Error Distribution', fontweight='bold')
ax2.set_xlabel('Reconstruction Error')
ax2.set_ylabel('Density')
ax2.legend()
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# Plot 3: Error over time (first 10000 samples)
ax3 = axes[1, 0]
sample_size = min(10000, len(test_errors))
x_range = range(sample_size)
colors = ['red' if label == 1 else 'blue' for label in test_labels[:sample_size]]
ax3.scatter(x_range, test_errors[:sample_size], c=colors, s=0.5, alpha=0.5)
ax3.axhline(THRESHOLD, color='green', linestyle='--', linewidth=2, label='Threshold')
ax3.set_title('Reconstruction Error Over Time (First 10k samples)', fontweight='bold')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Reconstruction Error')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix
ax4 = axes[1, 1]
conf_matrix = np.array([[true_negatives, false_positives],
                        [false_negatives, true_positives]])
im = ax4.imshow(conf_matrix, cmap='Blues', aspect='auto')
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Predicted Normal', 'Predicted Leak'])
ax4.set_yticklabels(['Actual Normal', 'Actual Leak'])
ax4.set_title('Confusion Matrix', fontweight='bold')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax4.text(j, i, f'{conf_matrix[i, j]:,}',
                       ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax4)
plt.tight_layout()
plt.savefig('autoencoder_results.png', dpi=150, bbox_inches='tight')
print("Saved: autoencoder_results.png")
plt.close()

print("Autoencoder complete!")
