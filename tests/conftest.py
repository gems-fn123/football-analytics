from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tracks_m() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_frames, n_players = 50, 4
    rows = []
    for tid in range(n_players):
        x, y = rng.uniform(10, 95), rng.uniform(10, 58)
        for f in range(n_frames):
            x += rng.normal(0, 0.12)
            y += rng.normal(0, 0.12)
            rows.append({
                "frame": f,
                "t_s": f / 25.0,
                "track_id": tid,
                "team": "home" if tid < 2 else "away",
                "shirt": pd.NA,
                "player": f"P{tid}",
                "x_m": np.clip(x, 0, 105),
                "y_m": np.clip(y, 0, 68),
                "speed_ms": np.nan,
                "conf": 0.9,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def shots() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 400
    sx = rng.uniform(70, 104, n)
    sy = rng.uniform(14, 54, n)
    dist = np.hypot(105 - sx, 34 - sy)
    p = np.clip(0.75 - dist * 0.035, 0.01, 0.9)
    return pd.DataFrame({
        "start_x": sx,
        "start_y": sy,
        "is_goal": rng.binomial(1, p),
    })
