"""
Extract event rows from interval-form flow data.

Input format:
- `processed/<fixture>_intervals.csv` with columns `start_ts,end_ts,flow_ml_s`

Output format:
- `events/<fixture>_events.csv` with columns:
  `event_id,start_ts,end_ts,duration_s,mean_flow_ml_s,peak_flow_ml_s,total_volume_ml`

This script is a utility used during prior-building. It treats each interval row
as a single event (duration = end_ts - start_ts) and computes total volume as
flow_ml_s × duration_s.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def extract_interval_events(input_file: Path, output_file: Path) -> None:
    """Convert an interval CSV into an event CSV."""
    output_file.parent.mkdir(exist_ok=True)

    with open(input_file, newline="") as f, open(output_file, "w", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.writer(out)

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

        for i, row in enumerate(reader):
            start = int(row["start_ts"])
            end = int(row["end_ts"])
            flow = float(row["flow_ml_s"])
            duration = max(0, end - start)

            writer.writerow([i, start, end, duration, flow, flow, flow * duration])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract events from interval CSV.")
    parser.add_argument("--fixture", required=True, help="Fixture name, e.g. toilet")
    parser.add_argument("--processed-dir", default="processed", help="Input directory")
    parser.add_argument("--events-dir", default="events", help="Output directory")
    args = parser.parse_args()

    input_file = Path(args.processed_dir) / f"{args.fixture}_intervals.csv"
    output_file = Path(args.events_dir) / f"{args.fixture}_events.csv"

    extract_interval_events(input_file, output_file)
    print(f"Extracted events → {output_file}")


if __name__ == "__main__":
    main()
