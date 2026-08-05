"""Distance, speed, and load metrics from the tracking table."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPRINT_THRESHOLD_MS = 7.0
HIGH_SPEED_THRESHOLD_MS = 5.5


def add_kinematics(tracks: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    """Add speed and acceleration per track. Smooth first, differentiate second."""
    out = tracks.sort_values(["track_id", "frame"]).copy()
    grp = out.groupby("track_id", group_keys=False)
    for col in ("x_m", "y_m"):
        out[col] = grp[col].transform(
            lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean()
        )
    dt = grp["t_s"].diff()
    dx = grp["x_m"].diff()
    dy = grp["y_m"].diff()
    out["speed_ms"] = np.hypot(dx, dy) / dt.replace(0, np.nan)
    out["accel_ms2"] = out.groupby("track_id")["speed_ms"].diff() / dt.replace(0, np.nan)
    return out


def summarise_player(tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-player totals. Only meaningful if identity resolved for that track."""
    df = tracks.dropna(subset=["speed_ms"])
    dt = df.groupby("track_id")["t_s"].diff().fillna(0)
    df = df.assign(step_m=df["speed_ms"] * dt)
    return df.groupby(["team", "player"], dropna=False).agg(
        minutes=("t_s", lambda s: (s.max() - s.min()) / 60),
        distance_m=("step_m", "sum"),
        top_speed_ms=("speed_ms", "max"),
        sprints=("speed_ms", lambda s: int((s > SPRINT_THRESHOLD_MS).sum())),
    ).reset_index()
