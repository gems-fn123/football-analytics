"""Possession from tracking. Nearest player to the ball, with hysteresis."""

from __future__ import annotations

import numpy as np
import pandas as pd

POSSESSION_RADIUS_M = 2.0
MIN_HOLD_FRAMES = 3


def assign_possession(
    tracks: pd.DataFrame,
    ball: pd.DataFrame,
    radius_m: float = POSSESSION_RADIUS_M,
) -> pd.DataFrame:
    """Return frame, possessor_track_id, team. Null when the ball is loose."""
    merged = tracks.merge(ball[["frame", "x_m", "y_m"]], on="frame", suffixes=("", "_ball"))
    merged["dist_m"] = np.hypot(
        merged["x_m"] - merged["x_m_ball"], merged["y_m"] - merged["y_m_ball"]
    )
    nearest = merged.loc[merged.groupby("frame")["dist_m"].idxmin()]
    nearest.loc[nearest["dist_m"] > radius_m, ["track_id", "team"]] = pd.NA
    # TODO: hysteresis so a single noisy frame does not flip possession
    return nearest[["frame", "track_id", "team", "dist_m"]].rename(
        columns={"track_id": "possessor_track_id"}
    )


def possession_share(possession: pd.DataFrame) -> pd.Series:
    return possession["team"].value_counts(normalize=True)
