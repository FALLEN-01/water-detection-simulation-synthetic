"""
Compute first/last timestamps for raw feed CSV files.

This script scans `feed*.csv` files in the current directory and writes a small
report (`dataset_time_ranges.txt`) containing:
- filename
- first timestamp + UTC datetime
- last timestamp + UTC datetime

Use this to sanity-check dataset coverage before extracting events/priors.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

OUTPUT_FILE = Path("dataset_time_ranges.txt")


def extract_timestamp(line: str) -> Optional[int]:
    """Parse the first numeric field of a line as a UNIX timestamp (seconds)."""
    parts = line.replace(",", " ").split()
    if not parts:
        return None
    try:
        return int(float(parts[0]))
    except ValueError:
        return None


def fmt_utc(ts: int) -> str:
    """Format a UNIX timestamp in UTC."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d-%m-%Y %H:%M:%S")


def main() -> None:
    """Write the timestamp coverage report to `OUTPUT_FILE`."""
    with open(OUTPUT_FILE, "w") as out:
        out.write(
            f"{'filename':<30} | "
            f"{'first_ts':<12} | "
            f"{'first_datetime_utc':<19} | "
            f"{'last_ts':<12} | "
            f"{'last_datetime_utc':<19}\n"
        )
        out.write("-" * 110 + "\n")

        for fname in sorted(os.listdir(".")):
            if not fname.startswith("feed") or not fname.endswith(".csv"):
                continue

            first_ts = None
            last_ts = None

            with open(fname) as f:
                for line in f:
                    ts = extract_timestamp(line)
                    if ts is None:
                        continue
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

            if first_ts is None or last_ts is None:
                continue

            out.write(
                f"{fname:<30} | "
                f"{first_ts:<12} | "
                f"{fmt_utc(first_ts):<19} | "
                f"{last_ts:<12} | "
                f"{fmt_utc(last_ts):<19}\n"
            )

    print(f"Saved timestamp ranges to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
