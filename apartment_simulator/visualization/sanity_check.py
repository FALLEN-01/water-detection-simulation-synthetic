"""
Building-Level Sanity Check & Statistics Validation

DESCRIPTION:
    Analyzes building-level aggregate water flow data from CSV to validate
    simulation statistics: daily usage, night baseline, peak/night ratios,
    and inter-appliance flow patterns. Outputs statistics summary and saves
    detailed validation plots.

    Expected CSV format:
    - timestamp: Unix seconds (60s resolution) or datetime
    - flow_lpm: Aggregate flow in liters per minute

USAGE:
    python sanity_check.py <csv_file>
    Example: python sanity_check.py ../preprocessing/artifacts/building_flow_365d.csv

DEPENDENCIES:
    - pandas, numpy: Data processing
    - matplotlib: Plotting
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DEFAULT_CSV = SCRIPT_DIR.parent / "preprocessing" / "artifacts" / "building_flow_365d.csv"

# ======================================================
# PARSE INPUT
# ======================================================
if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])
else:
    csv_path = DEFAULT_CSV

if not csv_path.exists():
    print(f"[ERROR] CSV file not found: {csv_path}")
    print(f"Usage: python sanity_check.py <path_to_csv>")
    sys.exit(1)

print(f"[OK] Loading CSV: {csv_path}")

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(csv_path)

# Handle timestamp conversion
if "timestamp" in df.columns:
    try:
        # Try Unix seconds first
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        if df["timestamp"].isna().sum() > 0:
            # Fall back to string parsing
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    df = df.sort_values("timestamp")
else:
    print("[WARN] No 'timestamp' column found. Using row index as time.")

flow = df["flow_lpm"].astype(float)

print(f"[OK] Loaded {len(df):,} samples")
if "timestamp" in df.columns:
    print(f"    Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# ======================================================
# DAILY USAGE ANALYSIS
# ======================================================
print("\n" + "=" * 70)
print("DAILY WATER USAGE (L/day)")
print("=" * 70)

if "timestamp" in df.columns:
    daily = df.set_index("timestamp")["flow_lpm"].resample("D").sum()
else:
    # Fallback if no timestamps
    daily = df["flow_lpm"].values.reshape(-1, 1440).sum(axis=1)

print(daily.tail(10))

mean = daily.mean()
std = daily.std()
cv = std / mean if mean > 0 else 0

print("\n" + "-" * 70)
print("SUMMARY STATISTICS")
print("-" * 70)
print(f"Mean      : {mean:8.2f} L/day")
print(f"Std Dev   : {std:8.2f} L/day")
print(f"Min       : {daily.min():8.2f} L/day")
print(f"10th pct  : {np.percentile(daily, 10):8.2f} L/day")
print(f"Median    : {np.percentile(daily, 50):8.2f} L/day")
print(f"90th pct  : {np.percentile(daily, 90):8.2f} L/day")
print(f"Max       : {daily.max():8.2f} L/day")

# ======================================================
# VARIABILITY CHECK
# ======================================================
print("\n" + "-" * 70)
print("VARIABILITY CHECK (Coefficient of Variation)")
print("-" * 70)
print(f"Coeff. of variation: {cv:.3f}")
print("Expected (building-level):")
print("  Typical : 0.05 – 0.15  (more stable than household)")
print("  Accept  : 0.03 – 0.25")

if cv < 0.03 or cv > 0.25:
    print("  ⚠ WARNING: CV outside acceptable range!")
else:
    print("  ✓ PASS")

# ======================================================
# NIGHT BASELINE (LEAK DETECTION CRUCIAL)
# ======================================================
print("\n" + "-" * 70)
print("NIGHT BASELINE (0–5 AM) - Leak Detection Window")
print("-" * 70)

if "timestamp" in df.columns:
    night = df[df["timestamp"].dt.hour < 5]["flow_lpm"]
else:
    print("[WARN] Cannot compute night baseline without timestamps")
    night = pd.Series([])

if len(night) > 0:
    night_median = night.median()
    night_p95 = np.percentile(night, 95)
    night_p99 = np.percentile(night, 99)
    
    print(f"Median   : {night_median:.3f} L/min")
    print(f"95th pct : {night_p95:.3f} L/min")
    print(f"99th pct : {night_p99:.3f} L/min")
    print("Expected (50-unit building):")
    print("  Median < 10 L/min")
    print("  95th pct < 30 L/min")
    
    # Detection threshold context
    print("\nLeak Detection: Night baseline drift indicates water leaks")
    print(f"  Small leak (0.5 L/m): ~{30 + 0.5:.1f} L/min")
    print(f"  Medium leak (1.0 L/m): ~{30 + 1.0:.1f} L/min")
else:
    print("No night data available")

# ======================================================
# PEAK / NIGHT RATIO
# ======================================================
print("\n" + "-" * 70)
print("PEAK / NIGHT RATIO")
print("-" * 70)

if "timestamp" in df.columns:
    morning = df[
        (df["timestamp"].dt.hour >= 6) &
        (df["timestamp"].dt.hour <= 9)
    ]["flow_lpm"]
    
    if len(night) > 0 and len(morning) > 0:
        night_mean = night.mean()
        morning_mean = morning.mean()
        ratio = morning_mean / (night_mean + 1e-6)
        
        print(f"Morning avg (6-9 AM): {morning_mean:8.2f} L/min")
        print(f"Night avg (0-5 AM) : {night_mean:8.2f} L/min")
        print(f"Peak / Night Ratio : {ratio:8.2f}x")
        print("Expected: 2x – 5x")
else:
    print("[WARN] Cannot compute peak/night ratio without timestamps")

# ======================================================
# INTER-APPLIANCE FLOW
# ======================================================
print("\n" + "-" * 70)
print("INTER-APPLIANCE FLOW (Continuous baseline)")
print("-" * 70)

# Appliance threshold for building (50 apartments * 8.0 L/min = 400 L/min)
appliance_thresh = 8.0 * 50  # 400 L/min

inter_vals = flow[flow < appliance_thresh]
print(f"Flow below {appliance_thresh} L/min (inter-appliance): {len(inter_vals):,} samples")
print(f"Mean   : {inter_vals.mean():.2f} L/min")
print(f"Median : {inter_vals.median():.2f} L/min")
print(f"Std    : {inter_vals.std():.2f} L/min")
print(f"Max    : {inter_vals.max():.2f} L/min")

# ======================================================
# GENERATE PLOTS
# ======================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Daily Usage Distribution
ax = axes[0, 0]
ax.hist(daily, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.1f}")
ax.axvline(np.percentile(daily, 50), color="green", linestyle="--", linewidth=2, label=f"Median: {np.percentile(daily, 50):.1f}")
ax.set_xlabel("Daily Usage (L/day)")
ax.set_ylabel("Frequency")
ax.set_title("Daily Water Usage Distribution")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Daily Usage Over Time
ax = axes[0, 1]
ax.plot(daily.values, linewidth=1.5, color="steelblue")
ax.axhline(mean, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Mean")
ax.set_xlabel("Day")
ax.set_ylabel("Daily Usage (L/day)")
ax.set_title("Daily Water Usage Trend")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Hourly Profile
if "timestamp" in df.columns:
    hourly = df.set_index("timestamp")["flow_lpm"].resample("h").mean()
    ax = axes[1, 0]
    hours = range(len(hourly))
    ax.plot(hours, hourly.values, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean Flow (L/min)")
    ax.set_title("Hourly Flow Profile")
    ax.grid(True, alpha=0.3)
else:
    ax = axes[1, 0]
    ax.text(0.5, 0.5, "No hourly data available\n(requires timestamps)", 
            ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")

# Plot 4: Flow Distribution (log scale)
ax = axes[1, 1]
non_zero_flow = flow[flow > 0]
ax.hist(np.log10(non_zero_flow), bins=40, color="darkgreen", alpha=0.7, edgecolor="black")
ax.set_xlabel("log₁₀(Flow L/min)")
ax.set_ylabel("Frequency")
ax.set_title("Flow Distribution (Log Scale)")
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_file = FIGURES_DIR / "sanity_check_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"[OK] Saved: {output_file}")
plt.close()

print("\n" + "=" * 70)
print("SANITY CHECK COMPLETE")
print("=" * 70)
