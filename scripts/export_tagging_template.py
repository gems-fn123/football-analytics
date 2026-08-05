"""Hand-tagging bridge.

Route: tag the match in LongoMatch or Kinovea, export, run this, get a SPADL-shaped
event table. This bypasses stages 6 and 7 and is the faster path to Opta-like events.

Both tools are separate GPL applications. Reading their exported files creates no
licensing obligation on this repo.

Usage:
    python scripts/export_tagging_template.py --make-template out.csv
    python scripts/export_tagging_template.py --from-longomatch tags.xml --out events.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TEMPLATE_COLUMNS = [
    "period",
    "clock_mmss",
    "team",
    "shirt",
    "type_name",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "result_name",
    "bodypart_name",
    "note",
]

EXAMPLE_ROWS = [
    [1, "03:12", "home", 10, "pass", 41.0, 22.5, 58.0, 30.0, "success", "foot", ""],
    [1, "03:19", "home", 7, "shot", 88.0, 30.0, 105.0, 34.0, "fail", "foot", "blocked"],
]


def make_template(path: str | Path) -> Path:
    df = pd.DataFrame(EXAMPLE_ROWS, columns=TEMPLATE_COLUMNS)
    out = Path(path)
    df.to_csv(out, index=False)
    return out


def from_longomatch(xml_path: str | Path) -> pd.DataFrame:
    """TODO: parse the LongoMatch project XML, map category names to SPADL types."""
    raise NotImplementedError


def from_kinovea(csv_path: str | Path) -> pd.DataFrame:
    """TODO: Kinovea exports timestamped keyframe comments, not structured events."""
    raise NotImplementedError


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template")
    ap.add_argument("--from-longomatch")
    ap.add_argument("--from-kinovea")
    ap.add_argument("--out", default="data/interim/events_manual.parquet")
    args = ap.parse_args()

    if args.make_template:
        print(f"wrote {make_template(args.make_template)}")
        return 0
    if args.from_longomatch:
        from_longomatch(args.from_longomatch).to_parquet(args.out, index=False)
        return 0
    if args.from_kinovea:
        from_kinovea(args.from_kinovea).to_parquet(args.out, index=False)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
