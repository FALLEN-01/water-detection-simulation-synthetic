"""
Train Isolation Forest model on apartment building water flow data.

Trains separate models for building-scale detection vs original household scale,
allowing comparison and validation of performance.

Output:
- isolation_forest_building.pkl: Model trained on building data
- scaler_building.pkl: StandardScaler fitted on building data
- calibration_building.json: Optimal thresholds
- performance_metrics.json: Accuracy, precision, recall, F1
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

OUTPUT_DIR = Path(__file__).parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent / "data"


def extract_features(df, window_size=20, stride=5):
    """
    Extract statistical features from flow data for Isolation Forest.

    Features:
    - mnf: 10th percentile flow (minimum normal flow)
    - inter_mean: Mean of inter-appliance flows
    - inter_frac: Fraction of inter-appliance periods
    - mean_flow: Average window flow
    - inter_std: Std dev of inter-appliance flows

    appliance_flow_thresh: 8.0 L/min (building scale)
    """
    appliance_flow_thresh = 8.0
    noise_floor = 0.2

    features_list = []
    labels_list = []

    # Create windows
    for i in range(0, len(df) - window_size, stride):
        window = df.iloc[i:i+window_size]
        flows = window['flow_lpm'].values
        is_leak = window['is_leak'].values.any()

        # Extract features
        inter = flows[flows < appliance_flow_thresh]
        nonzero = flows[flows > 0.0]

        mnf = float(np.percentile(nonzero, 10)) if len(nonzero) > 0 else 0.0
        inter_mean = float(inter.mean()) if len(inter) > 0 else 0.0
        inter_frac = float((inter > noise_floor).mean()) if len(inter) > 0 else 0.0
        mean_flow = float(flows.mean())
        inter_std = float(inter.std()) if len(inter) > 0 else 0.0

        features_list.append([mnf, inter_mean, inter_frac, mean_flow, inter_std])
        labels_list.append(1 if is_leak else 0)  # 1 = leak, 0 = normal

    return np.array(features_list), np.array(labels_list)


def train_isolation_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Isolation Forest model on building-scale data.

    Auto-calibrates contamination based on leak frequency.
    """
    print("\n" + "=" * 70)
    print("TRAINING ISOLATION FOREST (Building Scale)")
    print("=" * 70)

    # Fit scaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Auto-calibrate contamination
    leak_frac = y_train.sum() / len(y_train) if len(y_train) > 0 else 0.05
    contamination = min(leak_frac * 1.5, 0.15)  # 1.5x with cap at 15%

    print(f"\nTraining parameters:")
    print(f"  Training samples: {len(X_train_scaled):,}")
    print(f"  Leak fraction: {leak_frac:.4f}")
    print(f"  Contamination: {contamination:.4f}")

    # Train model
    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples='auto',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_scaled)

    print(f"  Model trained: {model.n_estimators} trees")

    # Predict and evaluate
    print("\nEvaluating on test set...")
    y_pred_train = model.predict(X_train_scaled)  # -1 = anomaly, +1 = normal
    y_pred_test = model.predict(X_test_scaled)

    # Convert predictions: -1 (anomaly) → 1, +1 (normal) → 0
    y_pred_train_binary = (y_pred_train == -1).astype(int)
    y_pred_test_binary = (y_pred_test == -1).astype(int)

    # Metrics
    acc_test = accuracy_score(y_test, y_pred_test_binary)
    prec_test = precision_score(y_test, y_pred_test_binary, zero_division=0)
    rec_test = recall_score(y_test, y_pred_test_binary, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test_binary, zero_division=0)

    cm = confusion_matrix(y_test, y_pred_test_binary)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {acc_test:.4f}")
    print(f"  Precision: {prec_test:.4f}")
    print(f"  Recall:    {rec_test:.4f}")
    print(f"  F1 Score:  {f1_test:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TN: {tn:4d}")

    # Calculate thresholds
    scores_train = model.decision_function(X_train_scaled)
    scores_test = model.decision_function(X_test_scaled)

    threshold_99 = np.percentile(scores_train[y_train == 0], 1)  # 99th percentile of normal
    threshold_95 = np.percentile(scores_train[y_train == 0], 5)  # 95th percentile of normal
    score_scale = np.percentile(scores_test[y_test == 1], 50) - threshold_99 if (y_test == 1).sum() > 0 else 0.1

    print(f"\nThreshold Calibration:")
    print(f"  99th percentile (normal): {threshold_99:.6f}")
    print(f"  95th percentile (normal): {threshold_95:.6f}")
    print(f"  Score scale: {score_scale:.6f}")

    metrics = {
        "accuracy": float(acc_test),
        "precision": float(prec_test),
        "recall": float(rec_test),
        "f1_score": float(f1_test),
        "confusion_matrix": {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        },
        "contamination": float(contamination),
        "n_estimators": int(model.n_estimators)
    }

    calibration = {
        "window_minutes": 20,
        "stride_minutes": 5,
        "appliance_flow_thresh": 8.0,
        "noise_floor": 0.2,
        "cusum_k": 0.5,
        "cusum_h": 20.0,
        "if_threshold": float(threshold_99),
        "if_score_scale": float(max(score_scale, 0.1)),
        "if_threshold_95": float(threshold_95)
    }

    return model, scaler, metrics, calibration


if __name__ == "__main__":
    print("=" * 70)
    print("APARTMENT BUILDING ISOLATION FOREST MODEL TRAINING")
    print("=" * 70)

    # Load data
    print("\nLoading datasets...")
    train_path = DATA_DIR / "water_train_building.csv"
    test_path = DATA_DIR / "water_test_building.csv"

    if not train_path.exists() or not test_path.exists():
        print("ERROR: Training data not found!")
        print(f"  Expected: {train_path}")
        print(f"  Expected: {test_path}")
        print("\nRun: py preprocessing/generate_data.py")
        exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"  Train: {len(train_df):,} samples")
    print(f"  Test: {len(test_df):,} samples ({test_df['is_leak'].sum():,} leaks)")

    # Extract features
    print("\nExtracting features...")
    X_train, y_train = extract_features(train_df, window_size=20, stride=5)
    X_test, y_test = extract_features(test_df, window_size=20, stride=5)

    print(f"  Train features: {X_train.shape}")
    print(f"  Test features: {X_test.shape}")

    # Train model
    model, scaler, metrics, calibration = train_isolation_forest(X_train, y_train, X_test, y_test)

    # Save artifacts
    print("\nSaving artifacts...")
    with open(OUTPUT_DIR / "isolation_forest_building.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  [OK] Model saved: isolation_forest_building.pkl")

    with open(OUTPUT_DIR / "scaler_building.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  [OK] Scaler saved: scaler_building.pkl")

    with open(OUTPUT_DIR / "calibration_building.json", "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"  [OK] Calibration saved: calibration_building.json")

    with open(OUTPUT_DIR / "metrics_building.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [OK] Metrics saved: metrics_building.json")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nNew models ready at: {OUTPUT_DIR}/")
    print(f"Copy to: apartment_simulator/artifacts/")
    print(f"\nUpdate server.py to use building-scale model:")
    print(f"  if_model = load_pickle(...isolation_forest_building.pkl)")
    print(f"  if_scaler = load_pickle(...scaler_building.pkl)")
    print(f"  cal = load_json(...calibration_building.json)")
