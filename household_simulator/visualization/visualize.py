"""
Visualize appliance timing priors (hour-of-day activation probabilities).

This module reads `all_appliances.json` (merged prior file) and plots the
categorical probability distribution for `timing.start_hour.p` for each
appliance.

Intended use:
- Run from a simulator directory that contains `all_appliances.json`, or pass
  an explicit path via `--priors`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_PRIORS_PATH = Path("all_appliances.json")

# Proper display names (fallback is `.title()`).
NAME_MAP = {
    "shower": "Shower",
    "washingmachine": "Washing Machine",
    "bidet": "Bidet",
    "washbasin": "Wash Basin",
    "kitchenfaucet": "Kitchen Faucet",
    "toilet": "Toilet",
}


def plot_start_hour_priors(priors_path: Path) -> None:
    """Plot start-hour probability vectors for each appliance in `priors_path`."""
    with open(priors_path) as f:
        priors = json.load(f)["appliances"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ap in enumerate(priors[: len(axes)]):
        probs = ap["timing"]["start_hour"]["p"]
        name = NAME_MAP.get(ap["appliance"], ap["appliance"].title())

        ax = axes[i]
        ax.plot(range(24), probs, marker="o")
        ax.set_title(name)
        ax.set_xlim(0, 23)
        ax.set_xlabel("Hour")
        ax.set_ylabel("P")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot appliance start-hour priors.")
    parser.add_argument("--priors", default=str(DEFAULT_PRIORS_PATH), help="Path to all_appliances.json")
    args = parser.parse_args()
    plot_start_hour_priors(Path(args.priors))


if __name__ == "__main__":
    main()