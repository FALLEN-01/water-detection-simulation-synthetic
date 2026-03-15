"""
Building-Level Flow Analysis & Anomaly Detection Visualization

DESCRIPTION:
    Generates time-series visualizations with anomaly detection thresholds overlaid.
    Shows:
    1. Raw flow time-series with appliance_flow_thresh (400 L/min) boundary
    2. Window-based CUSUM scores and IF anomaly scores
    3. Detection results (fusion scores, alarms, leak events)

    Expected CSV format:
    - timestamp: Unix seconds or datetime
    - flow_lpm: Aggregate flow in liters per minute
    - Optional: cusum_score, if_score, anomaly_score, alarm (from detection output)

USAGE:
    python flow_analysis.py <csv_file>
    Example: python flow_analysis.py ../preprocessing/artifacts/building_flow_365d.csv

DEPENDENCIES:
    - pandas, numpy: Data processing
    - matplotlib: Plotting
    - Optional: sklearn (for visualization of detection if scores available)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# ======================================================
# CONFIG
# ======================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DEFAULT_CSV = SCRIPT_DIR.parent / "preprocessing" / "artifacts" / "building_flow_365d.csv"

# Apartment simulator parameters
NUM_APARTMENTS = 50
APPLIANCE_THRESH = 8.0 * NUM_APARTMENTS  # 400 L/min
CUSUM_K = 3.0 * NUM_APARTMENTS           # ~150 L/min (67th percentile of inter-apt)
INTER_MEAN_MEDIAN = 2.391 * NUM_APARTMENTS  # ~120 L/min

# ======================================================
# PARSE INPUT
# ======================================================
if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])
else:
    csv_path = DEFAULT_CSV

if not csv_path.exists():
    print(f"[ERROR] CSV file not found: {csv_path}")
    print(f"Usage: python flow_analysis.py <path_to_csv>")
    sys.exit(1)

print(f"[OK] Loading CSV: {csv_path}")

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(csv_path)

# Handle timestamp conversion
if "timestamp" in df.columns:
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        if df["timestamp"].isna().sum() > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    df = df.sort_values("timestamp")
    has_timestamp = True
else:
    print("[WARN] No 'timestamp' column found. Using sequential index.")
    has_timestamp = False

flow = df["flow_lpm"].astype(float).values

print(f"[OK] Loaded {len(df):,} samples")
if has_timestamp:
    print(f"    Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# ======================================================
# FEATURE EXTRACTION (for anomaly visualization)
# ======================================================
print(f"\nExtracting features...")

# Window size (20 minutes for building)
window_size = 20

# Extract inter-appliance flow
inter_vals = flow[flow < APPLIANCE_THRESH]
inter_mean_rolling = np.full(len(flow), np.nan)

for i in range(len(flow) - window_size + 1):
    window = flow[i:i+window_size]
    inter = window[window < APPLIANCE_THRESH]
    if len(inter) > 0:
        inter_mean_rolling[i+window_size-1] = inter.mean()

# Compute anomaly indicators
anomaly_level = np.full(len(flow), np.nan)
for i in range(len(flow)):
    if not np.isnan(inter_mean_rolling[i]):
        # Simple anomaly: deviation from normal baseline
        deviation = inter_mean_rolling[i] - INTER_MEAN_MEDIAN
        anomaly_level[i] = max(0, deviation)

# ======================================================
# GENERATE PLOTS
# ======================================================
print("Generating plots...")

time_axis = np.arange(len(flow))
if has_timestamp:
    time_axis_dt = df["timestamp"].values

# ======================================================
# PLOT 1: Raw Flow Time-Series with Thresholds
# ======================================================
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(time_axis, flow, linewidth=0.8, color="navy", alpha=0.8, label="Aggregate Flow")
ax.axhline(APPLIANCE_THRESH, color="red", linestyle="--", linewidth=2, 
           label=f"Appliance Threshold ({APPLIANCE_THRESH:.0f} L/min)", alpha=0.7)
ax.axhline(CUSUM_K, color="orange", linestyle=":", linewidth=2, 
           label=f"CUSUM Reference ({CUSUM_K:.0f} L/min)", alpha=0.7)
ax.axhline(INTER_MEAN_MEDIAN, color="green", linestyle="-.", linewidth=1.5, 
           label=f"Baseline Mean ({INTER_MEAN_MEDIAN:.0f} L/min)", alpha=0.7)

ax.set_xlabel("Time (minutes)" if not has_timestamp else "Time")
ax.set_ylabel("Flow (L/min)")
ax.set_title("Building-Level Water Flow with Detection Thresholds (50 Apartments)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)

if has_timestamp and len(df) <= 1440:
    # If data is short enough, show timestamps
    ticks = np.linspace(0, len(flow)-1, 10, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([df["timestamp"].iloc[t].strftime("%H:%M") for t in ticks], rotation=45)

plt.tight_layout()
output_file = FIGURES_DIR / "flow_timeseries_thresholds.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"[OK] Saved: {output_file}")
plt.close()

# ======================================================
# PLOT 2: Inter-Appliance Flow Analysis
# ======================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

# Top: Rolling inter-appliance mean
ax1.plot(time_axis, inter_mean_rolling, linewidth=1.2, color="steelblue", alpha=0.8, label="Rolling Inter-Apt Mean")
ax1.axhline(INTER_MEAN_MEDIAN, color="green", linestyle="--", linewidth=2, label=f"Expected Baseline ({INTER_MEAN_MEDIAN:.0f} L/min)", alpha=0.7)
ax1.fill_between(time_axis, INTER_MEAN_MEDIAN * 0.8, INTER_MEAN_MEDIAN * 1.2, alpha=0.2, color="green", label="Normal Range ±20%")
ax1.set_ylabel("Flow (L/min)")
ax1.set_title("Inter-Appliance Baseline (Leak Detection Zone)", fontsize=13, fontweight="bold")
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(True, alpha=0.3)

# Bottom: Anomaly level
colors = np.where(anomaly_level > 0, "red", "green")
ax2.scatter(time_axis, anomaly_level, s=10, c=colors, alpha=0.5)
ax2.plot(time_axis, anomaly_level, linewidth=0.5, color="darkred", alpha=0.5, label="Deviation from Baseline")
ax2.axhline(0, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Normal")
ax2.set_xlabel("Time (minutes)" if not has_timestamp else "Time")
ax2.set_ylabel("Anomaly Level (L/min)")
ax2.set_title("Deviation from Normal Baseline", fontsize=13, fontweight="bold")
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(True, alpha=0.3)

if has_timestamp and len(df) <= 1440:
    ticks = np.linspace(0, len(flow)-1, 10, dtype=int)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([df["timestamp"].iloc[t].strftime("%H:%M") for t in ticks], rotation=45)

plt.tight_layout()
output_file = FIGURES_DIR / "inter_appliance_flow_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"[OK] Saved: {output_file}")
plt.close()

# ======================================================
# PLOT 3: Distribution Comparison
# ======================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# All flow distribution
ax1.hist(flow, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
ax1.axvline(np.percentile(flow, 50), color="red", linestyle="--", linewidth=2, label=f"Median: {np.percentile(flow, 50):.1f}")
ax1.set_xlabel("Flow (L/min)")
ax1.set_ylabel("Frequency")
ax1.set_title("All Flow Distribution")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Inter-appliance flow distribution
inter_only = flow[flow < APPLIANCE_THRESH]
ax2.hist(inter_only, bins=50, color="darkgreen", alpha=0.7, edgecolor="black")
ax2.axvline(inter_only.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {inter_only.mean():.1f}")
ax2.axvline(INTER_MEAN_MEDIAN, color="orange", linestyle="--", linewidth=2, label=f"Expected: {INTER_MEAN_MEDIAN:.1f}")
ax2.set_xlabel("Flow (L/min)")
ax2.set_ylabel("Frequency")
ax2.set_title(f"Inter-Appliance Flow Distribution (< {APPLIANCE_THRESH:.0f} L/min)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Percentile plot
percentiles = np.arange(0, 101, 5)
values = np.percentile(flow, percentiles)
ax3.plot(percentiles, values, marker="o", linewidth=2, markersize=6, color="steelblue")
ax3.axvline(50, color="red", linestyle="--", alpha=0.5)
ax3.axhline(INTER_MEAN_MEDIAN, color="green", linestyle="--", alpha=0.5, label=f"Expected Baseline")
ax3.set_xlabel("Percentile")
ax3.set_ylabel("Flow (L/min)")
ax3.set_title("Flow Percentile Curve")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Log-scale distribution
non_zero = flow[flow > 1e-3]
ax4.hist(np.log10(non_zero), bins=40, color="purple", alpha=0.7, edgecolor="black")
ax4.set_xlabel("log₁₀(Flow L/min)")
ax4.set_ylabel("Frequency")
ax4.set_title("Flow Distribution (Log Scale)")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_file = FIGURES_DIR / "flow_distribution_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"[OK] Saved: {output_file}")
plt.close()

# ======================================================
# SUMMARY STATISTICS
# ======================================================
print("\n" + "=" * 70)
print("FLOW STATISTICS (50-Unit Building)")
print("=" * 70)
print(f"\nTotal flow samples: {len(flow):,}")
print(f"\nFull Flow Statistics:")
print(f"  Mean  : {flow.mean():.2f} L/min")
print(f"  Median: {np.percentile(flow, 50):.2f} L/min")
print(f"  Std   : {flow.std():.2f} L/min")
print(f"  Min   : {flow.min():.2f} L/min")
print(f"  Max   : {flow.max():.2f} L/min")

print(f"\nInter-Appliance Flow (< {APPLIANCE_THRESH:.0f} L/min):")
print(f"  Samples: {len(inter_only):,} ({100*len(inter_only)/len(flow):.1f}%)")
print(f"  Mean   : {inter_only.mean():.2f} L/min (expected: {INTER_MEAN_MEDIAN:.2f})")
print(f"  Std    : {inter_only.std():.2f} L/min")

appliance_only = flow[flow >= APPLIANCE_THRESH]
print(f"\nAppliance Flow (>= {APPLIANCE_THRESH:.0f} L/min):")
print(f"  Samples: {len(appliance_only):,} ({100*len(appliance_only)/len(flow):.1f}%)")
if len(appliance_only) > 0:
    print(f"  Mean   : {appliance_only.mean():.2f} L/min")
    print(f"  Max    : {appliance_only.max():.2f} L/min")
else:
    print(f"  Mean   : N/A (no appliance events)")
    print(f"  Max    : N/A (no appliance events)")

print("\n" + "=" * 70)
print("PLOTS GENERATED")
print("=" * 70)
print(f"  1. flow_timeseries_thresholds.png")
print(f"  2. inter_appliance_flow_analysis.png")
print(f"  3. flow_distribution_analysis.png")
print(f"\nSaved to: {FIGURES_DIR}")
