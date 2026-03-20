"""
Extract event rows from point-form flow data (1 Hz samples).

Input format:
- `processed/<fixture>_points.csv` with columns `timestamp,flow_ml_s`

Output format:
- `events/<fixture>_events.csv` with columns:
  `event_id,start_ts,end_ts,duration_s,mean_flow_ml_s,peak_flow_ml_s,total_volume_ml`

Event definition:
- A new event starts when flow becomes > 0.
- The event continues while samples stay contiguous at 1-second intervals.
- A gap (> 1 second) or flow == 0 closes the current event.

Volume estimate:
- Assumes 1 Hz samples, so total_volume_ml ≈ sum(flow_ml_s) over event samples.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


EventRow = Tuple[int, int, int, int, float, float, float]


def points_to_events(input_file: Path) -> List[EventRow]:
    """Parse a point CSV and return extracted events."""
    events: List[EventRow] = []
    event_id = 0

    current_start: Optional[int] = None
    current_end: Optional[int] = None
    flows: List[float] = []

    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)
        prev_ts: Optional[int] = None

        for row in reader:
            ts = int(row["timestamp"])
            flow = float(row["flow_ml_s"])

            if flow > 0:
                if current_start is None:
                    current_start = ts
                    flows = [flow]
                else:
                    if prev_ts is not None and ts - prev_ts > 1:
                        # Gap: close previous event and start a new one.
                        assert current_end is not None
                        duration = current_end - current_start + 1
                        events.append(
                            (
                                event_id,
                                current_start,
                                current_end,
                                duration,
                                sum(flows) / len(flows),
                                max(flows),
                                sum(flows),
                            )
                        )
                        event_id += 1
                        current_start = ts
                        flows = [flow]
                    else:
                        flows.append(flow)

                current_end = ts
            else:
                if current_start is not None:
                    assert current_end is not None
                    duration = current_end - current_start + 1
                    events.append(
                        (
                            event_id,
                            current_start,
                            current_end,
                            duration,
                            sum(flows) / len(flows),
                            max(flows),
                            sum(flows),
                        )
                    )
                    event_id += 1
                    current_start = None
                    current_end = None
                    flows = []

            prev_ts = ts

    # Close last event if open.
    if current_start is not None:
        assert current_end is not None
        duration = current_end - current_start + 1
        events.append(
            (
                event_id,
                current_start,
                current_end,
                duration,
                sum(flows) / len(flows),
                max(flows),
                sum(flows),
            )
        )

    return events


def write_events(output_file: Path, events: Sequence[EventRow]) -> None:
    """Write event rows to a standard event CSV."""
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", newline="") as f:
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
        writer.writerows(events)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract events from point CSV (1 Hz).")
    parser.add_argument("--fixture", required=True, help="Fixture name, e.g. washingmachine")
    parser.add_argument("--processed-dir", default="processed", help="Input directory")
    parser.add_argument("--events-dir", default="events", help="Output directory")
    args = parser.parse_args()

    input_file = Path(args.processed_dir) / f"{args.fixture}_points.csv"
    output_file = Path(args.events_dir) / f"{args.fixture}_events.csv"

    events = points_to_events(input_file)
    write_events(output_file, events)

    print(f"Extracted {len(events)} events → {output_file}")


if __name__ == "__main__":
    main()
