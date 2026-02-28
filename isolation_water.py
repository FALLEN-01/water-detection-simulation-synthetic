import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

print("\nIsolation Forest Model")
print("Loading datasets...")

train_df = pd.read_csv('water_train.csv')
test_df = pd.read_csv('water_test.csv')

# Check if turbidity sensor is available (auxiliary sensor)
has_turbidity = 'turbidity' in train_df.columns

if has_turbidity:
    feature_cols = ['flow_normalized', 'turbidity', 
                    'flow_duration', 'hour', 'is_weekend']
    print("Using turbidity sensor (auxiliary) for enhanced detection")
else:
    feature_cols = ['flow_normalized', 
                    'flow_duration', 'hour', 'is_weekend']
    print("Turbidity sensor not available - using flow only")

train_data = train_df[feature_cols].values
test_data = test_df[feature_cols].values
test_labels = test_df['label'].values

print(f"Training: {len(train_data):,} samples")
print(f"Testing: {len(test_data):,} samples ({(test_labels==1).sum():,} leaks)")

expected_contamination = (test_labels == 1).sum() / len(test_labels)
print(f"\nTraining...")
model = IsolationForest(
    n_estimators=200,
    contamination=min(0.1, expected_contamination * 1.5),  # Slightly higher than expected
    max_samples='auto',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(train_data)
print("Testing...")
# Predict: -1 for anomalies, 1 for normal
predictions = model.predict(test_data)
anomalies_detected = predictions == -1

# Get anomaly scores (lower score = more anomalous)
anomaly_scores = model.score_samples(test_data)

# Calculate metrics
true_positives = np.sum((anomalies_detected) & (test_labels == 1))
false_positives = np.sum((anomalies_detected) & (test_labels == 0))
true_negatives = np.sum((~anomalies_detected) & (test_labels == 0))
false_negatives = np.sum((~anomalies_detected) & (test_labels == 1))

print(f"\nTotal samples tested: {len(test_data):,}")
print(f"Anomalies detected: {np.sum(anomalies_detected):,}")
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
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Anomaly Score Distribution
ax1 = axes[0, 0]
normal_scores = anomaly_scores[test_labels == 0]
leak_scores = anomaly_scores[test_labels == 1]
ax1.hist(normal_scores, bins=100, alpha=0.6, label='Normal', color='blue', density=True)
ax1.hist(leak_scores, bins=100, alpha=0.6, label='Leaks', color='red', density=True)
ax1.set_title('Anomaly Score Distribution', fontweight='bold')
ax1.set_xlabel('Anomaly Score (lower = more anomalous)')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: ROC Curve
ax2 = axes[0, 1]
# Note: For Isolation Forest, lower scores indicate anomalies, so we negate the scores
fpr, tpr, thresholds = roc_curve(test_labels, -anomaly_scores)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# Plot 3: Anomaly Scores over time (first 10000 samples)
ax3 = axes[1, 0]
sample_size = min(10000, len(anomaly_scores))
x_range = range(sample_size)
colors = ['red' if label == 1 else 'blue' for label in test_labels[:sample_size]]
ax3.scatter(x_range, anomaly_scores[:sample_size], c=colors, s=0.5, alpha=0.5)
ax3.set_title('Anomaly Scores Over Time (First 10k samples)', fontweight='bold')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Anomaly Score')
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix
ax4 = axes[1, 1]
conf_matrix = np.array([[true_negatives, false_positives],
                        [false_negatives, true_positives]])
im = ax4.imshow(conf_matrix, cmap='Greens', aspect='auto')
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
plt.savefig('isolation_forest_results.png', dpi=150, bbox_inches='tight')
print("  Saved: isolation_forest_results.png")
plt.close()

# Sample detections
print("\n--- Sample Detections (First 20) ---")
for i in range(min(20, len(test_data))):
    score = anomaly_scores[i]
    pred = predictions[i]
    actual = 'LEAK' if test_labels[i] == 1 else 'Normal'
    predicted = 'LEAK' if pred == -1 else 'Normal'
    status = '✓' if (actual == predicted) else '✗'
    
    print(f"{status} Sample {i+1}: Score={score:.6f} | Actual: {actual:6s} | Predicted: {predicted:6s}")

print("\n" + "="*60)
print("ISOLATION FOREST ANALYSIS COMPLETE")
print("="*60)

print("Saved: isolation_forest_results.png")
plt.close()

# === Save model for live inference ===
print("\nSaving model for live inference...")
import pickle
import os
os.makedirs("models", exist_ok=True)

# Save Isolation Forest model
with open("models/isolation_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved: models/isolation_forest_model.pkl")

print("Isolation Forest complete!")