"""Distance, speed, and load metrics from the tracking table."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPRINT_THRESHOLD_MS = 7.0
HIGH_SPEED_THRESHOLD_MS = 5.5


def add_kinematics(tracks: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    """Add speed and acceleration per track. Smooth first, differentiate second.

    Rows with track_id < 0 (untracked detections, the ball) share one bucket that is
    not a trajectory. They are excluded from smoothing entirely - a rolling mean over
    unrelated objects writes invented positions - and their kinematics stay NaN.
    """
    out = tracks.copy()
    out["speed_ms"] = np.nan
    out["accel_ms2"] = np.nan

    tracked = out[out["track_id"] >= 0].sort_values(["track_id", "frame"])
    if tracked.empty:
        return out

    grp = tracked.groupby("track_id", group_keys=False)
    for col in ("x_m", "y_m"):
        tracked[col] = grp[col].transform(
            lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean()
        )
    grp = tracked.groupby("track_id", group_keys=False)
    dt = grp["t_s"].diff()
    dx = grp["x_m"].diff()
    dy = grp["y_m"].diff()
    tracked["speed_ms"] = np.hypot(dx, dy) / dt.replace(0, np.nan)
    tracked["accel_ms2"] = tracked.groupby("track_id")["speed_ms"].diff() / dt.replace(0, np.nan)

    cols = ["x_m", "y_m", "speed_ms", "accel_ms2"]
    out.loc[tracked.index, cols] = tracked[cols]
    return out


def summarise_player(tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-player totals. Only meaningful if identity resolved for that track.

    Grouped by track as well as player: while OCR abstains, player is null for
    everyone, and grouping by (team, player) alone would melt a whole team's
    anonymous tracks into one row with their summed distance.
    """
    df = tracks.dropna(subset=["speed_ms"])
    if df.empty:  # uncalibrated run: no speeds, no physical metrics
        return pd.DataFrame(
            columns=[
                "team",
                "player",
                "track_id",
                "minutes",
                "distance_m",
                "top_speed_ms",
                "sprints",
            ]
        )
    dt = df.groupby("track_id")["t_s"].diff().fillna(0)
    df = df.assign(step_m=df["speed_ms"] * dt)
    return (
        df.groupby(["team", "player", "track_id"], dropna=False)
        .agg(
            minutes=("t_s", lambda s: (s.max() - s.min()) / 60),
            distance_m=("step_m", "sum"),
            top_speed_ms=("speed_ms", "max"),
            sprints=("speed_ms", lambda s: int((s > SPRINT_THRESHOLD_MS).sum())),
        )
        .reset_index()
    )
