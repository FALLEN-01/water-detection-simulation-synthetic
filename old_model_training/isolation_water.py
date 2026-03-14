"""
Isolation Forest — Water Leak Detection
========================================
Trained on multi-building data with 500 trees for robust
anomaly boundary estimation across diverse usage patterns.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle

print("\nIsolation Forest — Water Leak Detection")
print("=" * 50)
print("Loading datasets...")

train_df = pd.read_csv('water_train.csv')
test_df  = pd.read_csv('water_test.csv')

has_turbidity = 'turbidity' in train_df.columns
if has_turbidity:
    feature_cols = ['flow_normalized', 'turbidity',
                    'flow_duration', 'hour', 'is_weekend']
    print("Using turbidity (auxiliary sensor)")
else:
    feature_cols = ['flow_normalized', 'flow_duration', 'hour', 'is_weekend']
    print("Turbidity not available")

print(f"Features ({len(feature_cols)}): {feature_cols}")

train_data  = train_df[feature_cols].values
test_data   = test_df[feature_cols].values
test_labels = test_df['label'].values

print(f"Training:  {len(train_data):,} samples (multi-building, no leaks)")
print(f"Testing:   {len(test_data):,} samples ({(test_labels==1).sum():,} leaks)")

# =====================================================
# Model
# =====================================================
expected_contamination = (test_labels == 1).sum() / len(test_labels)
contamination = float(np.clip(expected_contamination * 1.5, 0.01, 0.15))
print(f"\nExpected contamination: {expected_contamination:.4f} → using {contamination:.4f}")
print("Training Isolation Forest (500 trees)...")

model = IsolationForest(
    n_estimators=500,
    contamination=contamination,
    max_samples=2048,       # larger sub-samples for better boundary estimation
    max_features=1.0,       # all features per tree
    bootstrap=False,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
model.fit(train_data)

# =====================================================
# Evaluation
# =====================================================
print("Evaluating...")
predictions    = model.predict(test_data)
anomaly_scores = model.score_samples(test_data)
anomalies      = predictions == -1

TP = int(np.sum(anomalies & (test_labels == 1)))
FP = int(np.sum(anomalies & (test_labels == 0)))
TN = int(np.sum(~anomalies & (test_labels == 0)))
FN = int(np.sum(~anomalies & (test_labels == 1)))

accuracy  = (TP + TN) / len(test_data)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

fpr, tpr, _ = roc_curve(test_labels, -anomaly_scores)
roc_auc     = auc(fpr, tpr)

print(f"\n{'='*40}")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC AUC:   {roc_auc:.4f}")
print(f"  TP: {TP:,} | FP: {FP:,} | TN: {TN:,} | FN: {FN:,}")
print(f"{'='*40}")

# Sample detections
print("\n--- Sample Detections (first 15) ---")
for i in range(min(15, len(test_data))):
    actual    = 'LEAK  ' if test_labels[i] == 1 else 'Normal'
    predicted = 'LEAK  ' if predictions[i] == -1 else 'Normal'
    match = '✓' if actual.strip() == predicted.strip() else '✗'
    print(f"  {match} Score={anomaly_scores[i]:.4f} | Actual: {actual} | Pred: {predicted}")

# =====================================================
# Visualizations
# =====================================================
print("\nGenerating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Anomaly score distribution
ax = axes[0, 0]
normal_sc = anomaly_scores[test_labels == 0]
leak_sc   = anomaly_scores[test_labels == 1]
ax.hist(normal_sc, bins=100, alpha=0.6, label='Normal', color='steelblue', density=True)
ax.hist(leak_sc,   bins=100, alpha=0.6, label='Leak',   color='crimson',   density=True)
ax.set_title('Anomaly Score Distribution', fontweight='bold')
ax.set_xlabel('Score (lower = more anomalous)'); ax.set_ylabel('Density')
ax.legend(); ax.grid(True, alpha=0.3)

# ROC Curve
ax = axes[0, 1]
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1], color='navy', lw=2, linestyle='--', label='Random')
ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
ax.set_title('ROC Curve', fontweight='bold')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.legend(); ax.grid(True, alpha=0.3)

# Scores over time
ax = axes[1, 0]
n = min(15000, len(anomaly_scores))
colors = ['crimson' if l else 'steelblue' for l in test_labels[:n]]
ax.scatter(range(n), anomaly_scores[:n], c=colors, s=0.5, alpha=0.5)
ax.set_title('Anomaly Scores Over Time', fontweight='bold')
ax.set_xlabel('Sample'); ax.set_ylabel('Score'); ax.grid(True, alpha=0.3)

# Confusion matrix
ax = axes[1, 1]
cm = np.array([[TN, FP], [FN, TP]])
im = ax.imshow(cm, cmap='Greens', aspect='auto')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Pred Normal', 'Pred Leak'])
ax.set_yticklabels(['Actual Normal', 'Actual Leak'])
ax.set_title('Confusion Matrix', fontweight='bold')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('isolation_forest_results.png', dpi=150, bbox_inches='tight')
print("Saved: isolation_forest_results.png")
plt.close()

# =====================================================
# Save
# =====================================================
print("\nSaving model...")
os.makedirs("models", exist_ok=True)
with open("models/isolation_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved: models/isolation_forest_model.pkl")

print("\nIsolation Forest training complete!")