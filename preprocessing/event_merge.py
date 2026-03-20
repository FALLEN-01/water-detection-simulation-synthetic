"""
Merge per-appliance event CSVs into consolidated events.

This script reads `events/*_events.csv` files and merges consecutive events that are
separated by a small time gap (appliance-specific). The primary use is to reduce
"chattering" in event extraction where a single real-world usage episode may be
split into multiple adjacent events.

Inputs:
- `events/*_events.csv` with columns:
  `event_id,start_ts,end_ts,duration_s,mean_flow_ml_s,peak_flow_ml_s,total_volume_ml`

Outputs:
- `events_merged/*_events.csv` with the same columns, but fewer rows.

Notes:
- The merge window is in seconds and is configured per appliance via `MERGE_WINDOWS`.
- This file is designed to be run as a script (not imported).
"""

import csv
from pathlib import Path
from typing import Any, Dict, List

# ----------------------------
# CONFIG
# ----------------------------
EVENTS_DIR = Path("events")
OUTPUT_DIR = Path("events_merged")
OUTPUT_DIR.mkdir(exist_ok=True)

MERGE_WINDOWS = {
    "shower": 120,
    "washingmachine": 180,
    "kitchenfaucet": 0,
    "washbasin": 0,
    "bidet": 0,
    "toilet": 0,
    "dishwasher30": 0,
}

# ----------------------------
# MERGE FUNCTION
# ----------------------------
def merge_events(events: List[Dict[str, Any]], gap_threshold: int) -> List[Dict[str, Any]]:
    """Merge adjacent events when the gap between them is <= `gap_threshold` seconds."""
    if not events or gap_threshold == 0:
        return events

    merged = []
    current = events[0].copy()

    for nxt in events[1:]:
        gap = nxt["start_ts"] - current["end_ts"]

        if gap <= gap_threshold:
            # merge
            current["end_ts"] = nxt["end_ts"]
            current["total_volume_ml"] += nxt["total_volume_ml"]
            current["peak_flow_ml_s"] = max(
                current["peak_flow_ml_s"], nxt["peak_flow_ml_s"]
            )
        else:
            # finalize current
            duration = current["end_ts"] - current["start_ts"] + 1
            current["duration_s"] = duration
            current["mean_flow_ml_s"] = (
                current["total_volume_ml"] / duration if duration > 0 else 0
            )
            merged.append(current)
            current = nxt.copy()

    # finalize last
    duration = current["end_ts"] - current["start_ts"] + 1
    current["duration_s"] = duration
    current["mean_flow_ml_s"] = (
        current["total_volume_ml"] / duration if duration > 0 else 0
    )
    merged.append(current)

    return merged

# ----------------------------
# MAIN LOOP
# ----------------------------
def main() -> None:
    """Entry point for merging all appliance event CSVs found in `EVENTS_DIR`."""
    for csv_file in EVENTS_DIR.glob("*_events.csv"):
        appliance = csv_file.stem.replace("_events", "")
        gap = MERGE_WINDOWS.get(appliance)

        if gap is None:
            print(f"[SKIP] No merge rule for {appliance}")
            continue

        # Load events
        events: List[Dict[str, Any]] = []
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(
                    {
                        "event_id": int(row["event_id"]),
                        "start_ts": int(row["start_ts"]),
                        "end_ts": int(row["end_ts"]),
                        "duration_s": int(row["duration_s"]),
                        "mean_flow_ml_s": float(row["mean_flow_ml_s"]),
                        "peak_flow_ml_s": float(row["peak_flow_ml_s"]),
                        "total_volume_ml": float(row["total_volume_ml"]),
                    }
                )

        before = len(events)
        merged = merge_events(events, gap)
        after = len(merged)

        # Write merged events
        out_file = OUTPUT_DIR / csv_file.name
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "event_id",
                    "start_ts",
                    "end_ts",
                    "duration_s",
                    "mean_flow_ml_s",
                    "peak_flow_ml_s",
                    "total_volume_ml",
                ]
            )

            for i, ev in enumerate(merged):
                writer.writerow(
                    [
                        i,
                        ev["start_ts"],
                        ev["end_ts"],
                        ev["duration_s"],
                        round(float(ev["mean_flow_ml_s"]), 3),
                        round(float(ev["peak_flow_ml_s"]), 3),
                        round(float(ev["total_volume_ml"]), 3),
                    ]
                )

        print(f"[{appliance}] {before} → {after} (gap={gap}s)")


if __name__ == "__main__":
    main()
