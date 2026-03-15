"""
Building-Level Appliance Activation Pattern Visualization

DESCRIPTION:
    Visualizes appliance activation probability patterns for a 50-unit building.
    Generates a 3x2 subplot grid showing hourly activation patterns for each appliance type.
    Output saved as PNG to figures/ directory.

DEPENDENCIES:
    - json: Load appliance priors
    - matplotlib: Plotting

USAGE:
    python visualize.py
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PRIORS_PATH = SCRIPT_DIR.parent / "artifacts" / "all_appliances.json"
FIGURES_DIR = SCRIPT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

# ======================================================
# LOAD PRIORS
# ======================================================
try:
    with open(PRIORS_PATH) as f:
        priors = json.load(f)["appliances"]
    print(f"[OK] Loaded appliance priors from {PRIORS_PATH}")
except FileNotFoundError:
    print(f"[ERROR] Priors file not found: {PRIORS_PATH}")
    print("Make sure all_appliances.json exists in apartment_simulator/artifacts/")
    exit(1)

# ======================================================
# DISPLAY NAMES MAPPING
# ======================================================
NAME_MAP = {
    "shower": "Shower",
    "washingmachine": "Washing Machine",
    "bidet": "Bidet",
    "washbasin": "Wash Basin",
    "kitchenfaucet": "Kitchen Faucet",
    "toilet": "Toilet"
}

# ======================================================
# PLOT CONFIGURATION
# ======================================================
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

# ======================================================
# GENERATE PLOTS
# ======================================================
for i, ap in enumerate(priors):
    probs = ap["timing"]["start_hour"]["p"]
    name = NAME_MAP.get(ap["appliance"], ap["appliance"].title())

    ax = axes[i]

    # Plot activation probability by hour
    ax.plot(range(24), probs, marker="o", linewidth=2.5, markersize=6, color="steelblue")
    ax.fill_between(range(24), probs, alpha=0.3, color="steelblue")
    
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, 23.5)
    ax.grid(True, alpha=0.3)

# ======================================================
# CONFIGURE AXES
# ======================================================
for ax in axes:
    ax.set_xlabel("Hour of Day", fontsize=10)
    ax.set_ylabel("Activation Probability", fontsize=10)
    ax.set_xticks(range(0, 24, 3))

fig.suptitle("Building-Level Appliance Activation Patterns (50-Unit Building)", 
             fontsize=14, fontweight="bold", y=0.995)

plt.tight_layout()

# ======================================================
# SAVE FIGURE
# ======================================================
output_file = FIGURES_DIR / "appliance_activation_patterns.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"[OK] Saved: {output_file}")
plt.close()
