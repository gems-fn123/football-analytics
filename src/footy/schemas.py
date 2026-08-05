"""Column contracts between stages.

Every stage reads a frame of one shape and writes a frame of another. Keeping the
contract in one place is what stops stage 6 silently breaking stage 9.
"""

from __future__ import annotations

import pandas as pd

DETECTIONS = {
    "frame": "int64",
    "det_id": "int64",
    "cls": "string",       # player | goalkeeper | referee | ball
    "conf": "float32",
    "x1": "float32",
    "y1": "float32",
    "x2": "float32",
    "y2": "float32",
}

TRACKS_PX = {**DETECTIONS, "track_id": "int64"}

TRACKS_M = {
    "frame": "int64",
    "t_s": "float32",
    "track_id": "int64",
    "team": "string",      # home | away | referee | ball
    "shirt": "Int16",      # nullable, OCR is unreliable
    "player": "string",    # resolved from the squad list, nullable
    "x_m": "float32",
    "y_m": "float32",
    "speed_ms": "float32",
    "conf": "float32",
}

# socceraction SPADL-compatible subset
EVENTS = {
    "match_id": "string",
    "period": "int8",
    "t_s": "float32",
    "team": "string",
    "player": "string",
    "type_name": "string",
    "start_x": "float32",
    "start_y": "float32",
    "end_x": "float32",
    "end_y": "float32",
    "result_name": "string",
    "bodypart_name": "string",
}


def validate(df: pd.DataFrame, schema: dict[str, str], name: str) -> pd.DataFrame:
    missing = set(schema) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    return df.astype({k: v for k, v in schema.items() if k in df.columns})
