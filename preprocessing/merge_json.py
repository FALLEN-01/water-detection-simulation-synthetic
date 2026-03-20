"""
Merge individual appliance prior JSON files into `all_appliances.json`.

This utility is used after calibrating or editing per-appliance prior files in
`priors_india/` to produce a single JSON blob consumed by the simulators.

Input:
- `priors_india/*.json` (each file either a dict for one appliance, or a list of
  appliance dicts)

Output:
- `all_appliances.json` with keys: schema_version, count, appliances
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_INPUT_DIR = Path("priors_india")  # directory containing individual JSON files
DEFAULT_OUTPUT_FILE = Path("all_appliances.json")

# -----------------------------
# LOAD + MERGE
# -----------------------------
def merge_appliance_json(input_dir: Path) -> list:
    """Load and merge appliance JSON payloads from `input_dir`."""
    all_appliances = []

    for file in input_dir.glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        if isinstance(data, dict):
            all_appliances.append(data)
        elif isinstance(data, list):
            all_appliances.extend(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {file}")

    return all_appliances

# -----------------------------
# FINAL WRAP
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Merge appliance prior JSON files.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory of per-appliance JSON files")
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE), help="Output JSON filename")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    all_appliances = merge_appliance_json(input_dir)

    output = {
        "schema_version": "appliance_priors_v1",
        "count": len(all_appliances),
        "appliances": all_appliances,
    }

# -----------------------------
# WRITE OUTPUT
# -----------------------------
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Merged {len(all_appliances)} appliances into {output_file}")


if __name__ == "__main__":
    main()
