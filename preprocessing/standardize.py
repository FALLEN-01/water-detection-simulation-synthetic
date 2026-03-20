"""
Standardize raw feed CSVs into point and interval datasets.

The raw dataset may contain rows of two shapes:
- `timestamp,flow` (a point measurement)
- `start,flow,end` (an interval measurement)

This script converts each raw file in `RAW_DIR` into two clean CSVs:
- `{base}_points.csv`   with columns `timestamp,flow_ml_s`
- `{base}_intervals.csv` with columns `start_ts,end_ts,flow_ml_s`

Intended usage:
- Place raw files under `data/` (default `RAW_DIR`)
- Run `python preprocessing/standardize.py`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

RAW_DIR = Path("data")
OUT_DIR = Path("processed")


def parse_fields(line: str) -> List[str]:
    """Split a line into fields, tolerant of commas or whitespace separators."""
    return line.replace(",", " ").split()


def looks_like_header(line: str) -> bool:
    """Return True if the line appears to contain non-numeric header text."""
    return any(c.isalpha() for c in line)


def iter_feed_files(directory: Path) -> Iterable[Path]:
    """Yield raw `feed*.csv` files in a stable (sorted) order."""
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("feed") and fname.endswith(".csv"):
            yield directory / fname


def normalize_basename(filename: str) -> str:
    """Map raw filenames to a consistent lowercase base name."""
    return (
        filename.replace("feed_", "")
        .replace("feed.", "")
        .replace(".MYD", "")
        .replace(".csv", "")
        .lower()
    )


def process_file(raw_path: Path, out_dir: Path) -> None:
    """Convert a single raw feed file into `{base}_points.csv` and `{base}_intervals.csv`."""
    base = normalize_basename(raw_path.name)
    out_dir.mkdir(exist_ok=True)

    points_path = out_dir / f"{base}_points.csv"
    intervals_path = out_dir / f"{base}_intervals.csv"

    with open(points_path, "w", newline="") as point_out, open(
        intervals_path, "w", newline=""
    ) as interval_out:
        point_out.write("timestamp,flow_ml_s\n")
        interval_out.write("start_ts,end_ts,flow_ml_s\n")

        with open(raw_path) as f:
            for line in f:
                if looks_like_header(line):
                    continue

                parts = parse_fields(line)

                if len(parts) == 2:
                    ts, flow = parts
                    point_out.write(f"{ts},{flow}\n")
                elif len(parts) == 3:
                    start, flow, end = parts
                    interval_out.write(f"{start},{end},{flow}\n")

    print(f"Processed {raw_path.name}")


def main() -> None:
    """Process all raw feed files under `RAW_DIR`."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR.resolve()}")

    for raw_path in iter_feed_files(RAW_DIR):
        process_file(raw_path, OUT_DIR)


if __name__ == "__main__":
    main()
