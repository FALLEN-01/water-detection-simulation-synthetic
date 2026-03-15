"""
Anomaly Detection Results Visualization

DESCRIPTION:
    Visualizes anomaly detection output including:
    1. CUSUM scores and thresholds
    2. Isolation Forest anomaly scores
    3. Fusion-based detection decisions
    4. Detected events with timestamps
    5. ROC-style performance metrics (if ground truth available)

    This script reads detection output CSV with columns:
    - timestamp: Event timestamp
    - flow_lpm: Raw flow value
    - window_idx: 20-minute window index
    - cusum_score: CUSUM accumulator value
    - cusum_triggered: Boolean flag
    - if_score: Isolation Forest decision function output
    - if_triggered: Boolean flag
    - fusion_score: Weighted combination (0-1)
    - candidate_anomaly: Pre-persistence flag
    - anomaly_alarm: Final detection (post-persistence filter)
    - Optional: leak_injected, leak_flow_lpm (ground truth for validation)

USAGE:
    python anomaly_detection_viz.py <detection_output_csv>
    Example: python anomaly_detection_viz.py detection_results_2026_03.csv

DEPENDENCIES:
    - pandas, numpy: Data processing
    - matplotlib: Plotting
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ======================================================
# CONFIG
# ======================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Model thresholds
CUSUM_H = 8.0 * 50  # 400 (scaled for 50 apartments)
IF_THRESHOLD = -0.02
DECISION_THRESHOLD = 0.40

# ======================================================
# PARSE INPUT
# ======================================================
if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])
else:
    print("[ERROR] No CSV file provided")
    print("Usage: python anomaly_detection_viz.py <detection_output_csv>")
    print("\nExample: python anomaly_detection_viz.py detection_results.csv")
    sys.exit(1)

if not csv_path.exists():
    print(f"[ERROR] CSV file not found: {csv_path}")
    sys.exit(1)

print(f"[OK] Loading detection results: {csv_path}")

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(csv_path)

# Validate required columns
required = ["flow_lpm"]
has_detection = all(col in df.columns for col in ["cusum_score", "if_score", "fusion_score"])
has_timestamps = "timestamp" in df.columns or "datetime" in df.columns
has_ground_truth = "leak_injected" in df.columns

if has_timestamps:
    ts_col = "timestamp" if "timestamp" in df.columns else "datetime"
    try:
        df[ts_col] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
        if df[ts_col].isna().sum() > 0:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    except:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col)

print(f"[OK] Loaded {len(df):,} records")
print(f"    Detection data available: {has_detection}")
print(f"    Timestamps available: {has_timestamps}")
print(f"    Ground truth available: {has_ground_truth}")

# ======================================================
# SUMMARY STATISTICS
# ======================================================
print("\n" + "=" * 70)
print("DETECTION SUMMARY")
print("=" * 70)

if has_detection:
    n_cusum = df["cusum_triggered"].sum() if "cusum_triggered" in df.columns else 0
    n_if = df["if_triggered"].sum() if "if_triggered" in df.columns else 0
    n_candidates = df["candidate_anomaly"].sum() if "candidate_anomaly" in df.columns else 0
    n_alarms = df["anomaly_alarm"].sum() if "anomaly_alarm" in df.columns else 0
    
    print(f"\nCUSUM Triggers     : {n_cusum:,} windows")
    print(f"IF Triggers        : {n_if:,} windows")
    print(f"Candidate Anomalies: {n_candidates:,} windows")
    print(f"Final Alarms (4x)  : {n_alarms:,} events")
    
    if has_ground_truth:
        true_leaks = df["leak_injected"].sum() if "leak_injected" in df.columns else 0
        tp = (df["anomaly_alarm"] & df["leak_injected"]).sum() if "leak_injected" in df.columns else 0
        fp = (df["anomaly_alarm"] & ~df["leak_injected"]).sum() if "leak_injected" in df.columns else 0
        fn = (~df["anomaly_alarm"] & df["leak_injected"]).sum() if "leak_injected" in df.columns else 0
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nTrue Leak Windows  : {true_leaks:,}")
        print(f"  True Positives : {tp:,} (detected)")
        print(f"  False Negatives: {fn:,} (missed)")
        print(f"  False Positives: {fp:,} (false alarms)")
        print(f"\nPerformance:")
        print(f"  Recall    : {recall:.3f}")
        print(f"  Precision : {precision:.3f}")
        print(f"  F1 Score  : {f1:.3f}")

# ======================================================
# PLOT 1: CUSUM & IF SCORES
# ======================================================
if has_detection:
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    x = np.arange(len(df))
    
    # CUSUM
    ax = axes[0]
    ax.plot(x, df["cusum_score"], linewidth=1, color="steelblue", alpha=0.7, label="CUSUM Score")
    ax.axhline(CUSUM_H, color="red", linestyle="--", linewidth=2, label=f"Threshold ({CUSUM_H:.0f})")
    
    if "cusum_triggered" in df.columns:
        triggered = df["cusum_triggered"].astype(bool)
        ax.scatter(x[triggered], df["cusum_score"][triggered], color="red", s=20, label="CUSUM Trigger", zorder=5)
    
    ax.set_ylabel("Score")
    ax.set_title("CUSUM: Sustained Flow Elevation Detection", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # Isolation Forest
    ax = axes[1]
    ax.plot(x, df["if_score"], linewidth=1, color="darkgreen", alpha=0.7, label="IF Decision Score")
    ax.axhline(IF_THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"Threshold ({IF_THRESHOLD})")
    
    if "if_triggered" in df.columns:
        triggered = df["if_triggered"].astype(bool)
        ax.scatter(x[triggered], df["if_score"][triggered], color="red", s=20, label="IF Trigger", zorder=5)
    
    ax.set_ylabel("Score")
    ax.set_title("Isolation Forest: Anomalous Pattern Detection", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Fusion & Decision
    ax = axes[2]
    ax.plot(x, df["fusion_score"], linewidth=1.5, color="purple", alpha=0.8, label="Fusion Score")
    ax.axhline(DECISION_THRESHOLD, color="orange", linestyle="--", linewidth=2, label=f"Decision Gate ({DECISION_THRESHOLD})")
    
    if "candidate_anomaly" in df.columns:
        candidates = df["candidate_anomaly"].astype(bool)
        ax.scatter(x[candidates], df["fusion_score"][candidates], color="orange", s=15, label="Candidate", alpha=0.6, zorder=4)
    
    if "anomaly_alarm" in df.columns:
        alarms = df["anomaly_alarm"].astype(bool)
        ax.scatter(x[alarms], df["fusion_score"][alarms], color="red", s=30, marker="*", label="ALARM", zorder=5)
    
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Score")
    ax.set_title("Fusion & Persistence Filter: Final Decision", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = FIGURES_DIR / "detection_scores.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {output_file}")
    plt.close()


# ======================================================
# PLOT 2: FLOW WITH ALARMS OVERLAY
# ======================================================
if "flow_lpm" in df.columns:
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x = np.arange(len(df))
    ax.plot(x, df["flow_lpm"], linewidth=0.8, color="navy", alpha=0.8, label="Flow (L/min)")
    
    if "anomaly_alarm" in df.columns:
        alarms = df["anomaly_alarm"].astype(bool)
        ax.scatter(x[alarms], df["flow_lpm"][alarms], color="red", s=100, marker="*", 
                  label="Detection Alarm", zorder=5, edgecolor="darkred", linewidth=1.5)
    
    if has_ground_truth and "leak_injected" in df.columns:
        leaks = df["leak_injected"].astype(bool)
        ax.scatter(x[leaks], df["flow_lpm"][leaks], color="orange", s=50, marker="^", 
                  label="Injected Leak", zorder=4, alpha=0.7)
    
    ax.set_xlabel("Window Index" if not has_timestamps else "Time")
    ax.set_ylabel("Flow (L/min)")
    ax.set_title("Detection Results Overlay on Flow Time-Series", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = FIGURES_DIR / "flow_with_detections.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {output_file}")
    plt.close()

# ======================================================
# PLOT 3: PERFORMANCE METRICS (if ground truth available)
# ======================================================
if has_ground_truth and has_detection:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion Matrix
    if "leak_injected" in df.columns and "anomaly_alarm" in df.columns:
        tp = (df["anomaly_alarm"] & df["leak_injected"]).sum()
        fp = (df["anomaly_alarm"] & ~df["leak_injected"]).sum()
        fn = (~df["anomaly_alarm"] & df["leak_injected"]).sum()
        tn = (~df["anomaly_alarm"] & ~df["leak_injected"]).sum()
        
        cm = np.array([[tn, fp], [fn, tp]])
        im = ax1.imshow(cm, cmap="Blues", aspect="auto")
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(["Negative", "Positive"])
        ax1.set_yticklabels(["Negative", "Positive"])
        ax1.set_ylabel("True")
        ax1.set_xlabel("Predicted")
        ax1.set_title("Confusion Matrix")
        
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12, fontweight="bold")
        
        plt.colorbar(im, ax=ax1)
    
    # Metrics
    metrics = ["Recall", "Precision", "Specificity", "F1"]
    if "leak_injected" in df.columns and "anomaly_alarm" in df.columns:
        tp = (df["anomaly_alarm"] & df["leak_injected"]).sum()
        fp = (df["anomaly_alarm"] & ~df["leak_injected"]).sum()
        fn = (~df["anomaly_alarm"] & df["leak_injected"]).sum()
        tn = (~df["anomaly_alarm"] & ~df["leak_injected"]).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        values = [recall, precision, specificity, f1]
        colors_bar = ["#2ecc71" if v >= 0.8 else "#f39c12" if v >= 0.6 else "#e74c3c" for v in values]
        
        ax2.bar(metrics, values, color=colors_bar, edgecolor="black", linewidth=1.5)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel("Score")
        ax2.set_title("Performance Metrics")
        ax2.grid(True, alpha=0.3, axis="y")
        
        for i, (m, v) in enumerate(zip(metrics, values)):
            ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    
    # ROC Curve (simplified)
    if "if_score" in df.columns and "leak_injected" in df.columns:
        thresholds = np.linspace(df["if_score"].min(), df["if_score"].max(), 100)
        tpr = []
        fpr = []
        
        for t in thresholds:
            predicted = (df["if_score"] < t).astype(int)
            actual = df["leak_injected"].astype(int)
            
            tp = (predicted & actual).sum()
            fp = (predicted & ~actual).sum()
            fn = (~predicted & actual).sum()
            tn = (~predicted & ~actual).sum()
            
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        
        ax3.plot(fpr, tpr, linewidth=2, color="steelblue", label="IF Score ROC")
        ax3.plot([0, 1], [0, 1], "r--", linewidth=1, alpha=0.5, label="Random")
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curve (IF Based)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Event timeline
    if has_timestamps:
        ts_col = "timestamp" if "timestamp" in df.columns else "datetime"
        alarms_df = df[df["anomaly_alarm"] == True] if "anomaly_alarm" in df.columns else df.head(0)
        
        ax4.barh(range(len(alarms_df)), [1]*len(alarms_df), color="red", alpha=0.7)
        ax4.set_yticks(range(min(10, len(alarms_df))))
        ax4.set_yticklabels([str(ts)[:10] if hasattr(ts, '__str__') else str(ts) for ts in alarms_df[ts_col].head(10)])
        ax4.set_xlabel("Alarm Event")
        ax4.set_title(f"Top 10 Detection Events")
        ax4.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    output_file = FIGURES_DIR / "detection_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {output_file}")
    plt.close()

print("\n" + "=" * 70)
print("ANOMALY DETECTION VISUALIZATION COMPLETE")
print("=" * 70)
print(f"Plots saved to: {FIGURES_DIR}")
