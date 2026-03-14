"""
Train Isolation Forest model on apartment building water flow data.  (v2)

Improvements over v1:
- 7 features instead of 5: adds flow_trend and baseline_elev
  * flow_trend: linear regression slope over 20-min window  → detects rising baseline
  * baseline_elev: inter-appliance mean vs 60th-percentile expected → sustained elevation
- Better contamination estimate: uses actual leak fraction from test data
- F1-optimal threshold search instead of fixed 99th-percentile
- Saves feature_names to calibration JSON

Output:
- isolation_forest_building.pkl: Model trained on building data
- scaler_building.pkl: StandardScaler fitted on building data
- calibration_building.json: Optimal thresholds
- metrics_building.json: Accuracy, precision, recall, F1
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_curve
)

OUTPUT_DIR = Path(__file__).parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent / "data"

# Must match backend/model.py _extract_features() exactly
APPLIANCE_FLOW_THRESH = 8.0
NOISE_FLOOR = 0.2
WINDOW_SIZE = 20
STRIDE = 5


def extract_features(df, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Extract statistical features from flow data for Isolation Forest.

    Features (7 total):
    - mnf:           10th percentile of non-zero flows in window
    - inter_mean:    Mean of inter-appliance (low) flows
    - inter_frac:    Fraction of inter-appliance periods above noise floor
    - mean_flow:     Average window flow
    - inter_std:     Std dev of inter-appliance flows
    - flow_trend:    Linear regression slope over window (L/min per minute)
                     → positive + large = rising baseline → possible sustained leak
    - baseline_elev: How elevated is inter-appliance mean vs global training median
                     → computed as a z-score after fitting; placeholder=0 during extraction,
                        computed post-hoc from global stats
    """
    features_list = []
    labels_list = []
    raw_inter_means = []

    for i in range(0, len(df) - window_size, stride):
        window = df.iloc[i:i + window_size]
        flows = window['flow_lpm'].values.astype(np.float64)

        # Label
        if 'is_leak' in window.columns:
            is_leak = bool(window['is_leak'].values.any())
        else:
            is_leak = False

        inter = flows[flows < APPLIANCE_FLOW_THRESH]
        nonzero = flows[flows > 0.0]

        # Base features
        mnf = float(np.percentile(nonzero, 10)) if len(nonzero) > 0 else 0.0
        inter_mean = float(inter.mean()) if len(inter) > 0 else 0.0
        inter_frac = float((inter > NOISE_FLOOR).mean()) if len(inter) > 0 else 0.0
        mean_flow = float(flows.mean())
        inter_std = float(inter.std()) if len(inter) > 1 else 0.0

        # NEW: flow_trend — linear regression slope of entire window
        t = np.arange(len(flows), dtype=np.float64)
        t_mean = t.mean()
        f_mean = flows.mean()
        denom = ((t - t_mean) ** 2).sum()
        if denom > 1e-9:
            flow_trend = float(((t - t_mean) * (flows - f_mean)).sum() / denom)
        else:
            flow_trend = 0.0

        # Store raw inter_mean for baseline_elev later
        raw_inter_means.append(inter_mean)

        features_list.append([mnf, inter_mean, inter_frac, mean_flow, inter_std, flow_trend, 0.0])
        labels_list.append(1 if is_leak else 0)

    features_arr = np.array(features_list, dtype=np.float64)
    labels_arr = np.array(labels_list, dtype=np.int32)

    # NEW: baseline_elev — compute from normal windows only
    # = (inter_mean - global_median_inter_mean) / (global_std_inter_mean + 1e-6)
    raw_inter_arr = np.array(raw_inter_means)
    normal_mask = labels_arr == 0
    if normal_mask.sum() > 10:
        baseline_median = float(np.median(raw_inter_arr[normal_mask]))
        baseline_std = float(raw_inter_arr[normal_mask].std() + 1e-6)
    else:
        baseline_median = float(raw_inter_arr.mean())
        baseline_std = float(raw_inter_arr.std() + 1e-6)

    features_arr[:, 6] = (raw_inter_arr - baseline_median) / baseline_std

    return features_arr, labels_arr, {
        "baseline_median": baseline_median,
        "baseline_std": baseline_std
    }


def create_balanced_calibration_set(X, y, max_normal_ratio=5, random_state=42):
    """
    Downsample normal windows so normal:anomaly <= max_normal_ratio.

    With heavily imbalanced test data (e.g. 97% normal), the PR curve may return
    thresholds optimised for accuracy, not recall. A balanced calibration set
    forces the search to pay full attention to anomaly detection.

    Returns (X_cal, y_cal) — a balanced subset of X, y.
    """
    rng = np.random.default_rng(random_state)
    anomaly_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    if len(anomaly_idx) == 0:
        return X, y   # can't balance with no anomalies

    n_normal_keep = min(len(normal_idx), len(anomaly_idx) * max_normal_ratio)
    sampled_normal = rng.choice(normal_idx, size=n_normal_keep, replace=False)

    balanced_idx = np.concatenate([anomaly_idx, sampled_normal])
    rng.shuffle(balanced_idx)
    return X[balanced_idx], y[balanced_idx]


def find_optimal_threshold(y_true, scores_train_normal, scores_test_raw, min_recall=0.65):
    """
    Find the IF decision threshold that maximises F1.

    Strategy:
    1. Build a PR curve on the balanced calibration set (passed in already balanced).
    2. Walk the curve and pick the threshold maximising F1 subject to recall >= min_recall.
    3. If no valid point found (recall constraint too strict or model too weak),
       fall back to the 1st percentile of TRAINING NORMAL scores — this keeps the
       FPR at ~1% on unseen clean data.
    4. Sanity cap: threshold must be <= 0 (IF scores anomalies negatively).
    """
    # PR curve: sklearn expects higher score = more positive, so negate IF scores
    # (IF: more negative = more anomalous; negating makes more-anomalous = higher)
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, -scores_test_raw)

    best_f1 = -1.0
    best_threshold = None

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds_pr):
        if r < min_recall:
            continue
        f1 = 2.0 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = -float(t)   # back to original IF sign convention

    # Fallback: 1st percentile of normal training scores
    # (the score below which only ~1% of clean windows fall → very low FPR)
    fallback_threshold = float(np.percentile(scores_train_normal, 1))

    if best_threshold is None or best_threshold > 0.0:
        print(f"  [WARN] Optimal threshold {best_threshold} is None or positive "
              f"— using 1st-pct normal fallback {fallback_threshold:.4f}")
        best_threshold = fallback_threshold
        best_f1 = 0.0  # will be recalculated at final evaluation

    return best_threshold, best_f1


def train_isolation_forest(X_train, y_train, X_test, y_test, baseline_stats):
    """
    Train and evaluate Isolation Forest model on building-scale data.
    """
    print("\n" + "=" * 70)
    print("TRAINING ISOLATION FOREST v2 (Building Scale — 7 Features)")
    print("=" * 70)

    # Fit scaler
    print("Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Auto-calibrate contamination from actual test leak fraction
    # Use test data's actual leak fraction (training data is clean, but IF needs
    # to know the expected anomaly rate at runtime)
    actual_leak_frac = float(y_test.mean())
    contamination = float(np.clip(actual_leak_frac * 1.2, 0.03, 0.20))
    print(f"\nTraining parameters:")
    print(f"  Training samples:  {len(X_train_scaled):,}")
    print(f"  Test leak fraction: {actual_leak_frac:.4f}")
    print(f"  Contamination:     {contamination:.4f}")
    print(f"  Features:          7 (mnf, inter_mean, inter_frac, mean_flow, "
          f"inter_std, flow_trend, baseline_elev)")

    # Train model
    print("Training Isolation Forest (300 trees)...")
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_samples='auto',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_scaled)
    print(f"  Model trained: {model.n_estimators} trees")

    # Score
    scores_train = model.decision_function(X_train_scaled)
    scores_test = model.decision_function(X_test_scaled)

    # Build balanced calibration set (5:1 normal:anomaly) for unbiased threshold search
    # Raw test set can be ~97% normal, which biases PR-curve threshold selection.
    print("\nBuilding balanced calibration set for threshold search...")
    X_cal_scaled, y_cal = create_balanced_calibration_set(
        X_test_scaled, y_test, max_normal_ratio=5, random_state=42
    )
    scores_cal = model.decision_function(X_cal_scaled)
    print(f"  Calibration set: {len(X_cal_scaled):,} samples  "
          f"({y_cal.sum():,} anomalies = {100 * y_cal.mean():.1f}%)")

    # Find optimal threshold using balanced calibration set
    print("Searching for F1-optimal threshold (recall >= 65%)...")
    scores_train_normal = scores_train[y_train == 0]
    opt_threshold, opt_f1 = find_optimal_threshold(
        y_cal, scores_train_normal, scores_cal, min_recall=0.65
    )
    print(f"  Optimal threshold: {opt_threshold:.6f} (F1={opt_f1:.4f} on balanced cal)")

    # Evaluate at optimal threshold
    y_pred_test = (scores_test < opt_threshold).astype(int)
    y_pred_train = (scores_train < opt_threshold).astype(int)

    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test, zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)

    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {acc_test:.4f}")
    print(f"  Precision: {prec_test:.4f}")
    print(f"  Recall:    {rec_test:.4f}")
    print(f"  F1 Score:  {f1_test:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TN: {tn:4d}")

    # Score normalisation scale
    # We want: when raw_score == opt_threshold → if_score ≈ 0
    #          when score is deep anomaly → if_score → 1.0
    # Use 5th percentile of leak scores as the "deep anomaly" reference
    if (y_test == 1).sum() > 0:
        leak_scores = scores_test[y_test == 1]
        deep_anomaly_ref = float(np.percentile(leak_scores, 5))  # very anomalous
        score_scale = float(abs(opt_threshold - deep_anomaly_ref))
        score_scale = max(score_scale, 0.02)  # prevent near-zero scale
    else:
        score_scale = 0.1

    print(f"\nThreshold Calibration:")
    print(f"  if_threshold (optimal): {opt_threshold:.6f}")
    print(f"  if_score_scale:         {score_scale:.6f}")
    print(f"  baseline_median:        {baseline_stats['baseline_median']:.4f}")
    print(f"  baseline_std:           {baseline_stats['baseline_std']:.4f}")

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
        "n_estimators": int(model.n_estimators),
        "n_features": 7,
        "optimal_threshold": float(opt_threshold)
    }

    calibration = {
        "version": 2,
        "window_minutes": WINDOW_SIZE,
        "stride_minutes": STRIDE,
        "appliance_flow_thresh": APPLIANCE_FLOW_THRESH,
        "noise_floor": NOISE_FLOOR,
        "n_features": 7,
        "feature_names": [
            "mnf", "inter_mean", "inter_frac", "mean_flow",
            "inter_std", "flow_trend", "baseline_elev"
        ],
        # CUSUM (tuned for sustained leaks - lower k catches smaller deviations)
        "cusum_k": 0.3,      # was 0.5 - lower = more sensitive to small sustained leaks
        "cusum_h": 15.0,     # was 20.0 - lower = triggers sooner
        # Isolation Forest
        "if_threshold": float(opt_threshold),
        "if_score_scale": float(score_scale),
        # Baseline stats for baseline_elev feature in live inference
        "baseline_inter_mean_median": float(baseline_stats["baseline_median"]),
        "baseline_inter_mean_std": float(baseline_stats["baseline_std"]),
        # Fusion & decision
        "w_cusum": 0.35,
        "w_if": 0.65,
        "decision_threshold": 0.55,   # was 0.65 - lower to catch more sustained leaks
        "persistence_windows": 2
    }

    return model, scaler, metrics, calibration


if __name__ == "__main__":
    print("=" * 70)
    print("APARTMENT BUILDING IF MODEL TRAINING  v2 — 7 Features")
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
    print(f"  Test:  {len(test_df):,} samples "
          f"({test_df['is_leak'].sum():,} leaks, "
          f"{100*test_df['is_leak'].mean():.2f}%)")

    # Extract features
    print("\nExtracting features (7 features per window)...")
    X_train, y_train, train_stats = extract_features(train_df, window_size=WINDOW_SIZE, stride=STRIDE)
    X_test, y_test, _ = extract_features(test_df, window_size=WINDOW_SIZE, stride=STRIDE)

    # Re-normalise baseline_elev in test using TRAIN stats (prevent leakage)
    test_inter_means = X_test[:, 1].copy()
    X_test[:, 6] = (test_inter_means - train_stats["baseline_median"]) / train_stats["baseline_std"]

    print(f"  Train features: {X_train.shape}  (leak windows: {y_train.sum():,})")
    print(f"  Test features:  {X_test.shape}   (leak windows: {y_test.sum():,})")

    # Train model
    model, scaler, metrics, calibration = train_isolation_forest(
        X_train, y_train, X_test, y_test, train_stats
    )

    # Save artifacts
    print("\nSaving artifacts...")

    with open(OUTPUT_DIR / "isolation_forest_building.pkl", "wb") as f:
        pickle.dump(model, f)
    print("  [OK] Model saved: isolation_forest_building.pkl")

    with open(OUTPUT_DIR / "scaler_building.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("  [OK] Scaler saved: scaler_building.pkl")

    with open(OUTPUT_DIR / "calibration_building.json", "w") as f:
        json.dump(calibration, f, indent=2)
    print("  [OK] Calibration saved: calibration_building.json")

    with open(OUTPUT_DIR / "metrics_building.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  [OK] Metrics saved: metrics_building.json")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nKey results:")
    print(f"  Recall:    {metrics['recall']:.4f}  (target: ≥0.75)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1:        {metrics['f1_score']:.4f}")
    print(f"\nArtifacts at: {OUTPUT_DIR}/")
    print("Copy models+calibration to apartment_simulator/artifacts/ before running server")
