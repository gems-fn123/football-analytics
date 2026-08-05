import numpy as np
import pandas as pd
import pytest

from footy.analytics.physical import add_kinematics, summarise_player


def two_object_frame() -> pd.DataFrame:
    """A ball and a player interleaved in the shared track_id=-1 bucket plus one
    real track. The -1 rows alternate between two far-apart objects."""
    rows = []
    for f in range(6):
        rows.append({"frame": f, "t_s": f / 25, "track_id": -1, "x_m": 10.0, "y_m": 30.0})
        rows.append({"frame": f, "t_s": f / 25, "track_id": -1, "x_m": 90.0, "y_m": 5.0})
        rows.append({"frame": f, "t_s": f / 25, "track_id": 3, "x_m": 50.0 + 0.1 * f, "y_m": 30.0})
    return pd.DataFrame(rows)


def test_untracked_bucket_positions_are_never_smoothed():
    """The critical review finding: a rolling mean across the -1 bucket used to
    blend a ball at x=10/90 into invented positions around 40-60."""
    df = two_object_frame()
    out = add_kinematics(df, smooth_window=5)
    untracked = out[out["track_id"] == -1].sort_index()
    original = df[df["track_id"] == -1].sort_index()
    pd.testing.assert_series_equal(untracked["x_m"], original["x_m"])
    pd.testing.assert_series_equal(untracked["y_m"], original["y_m"])
    assert untracked["speed_ms"].isna().all()


def test_tracked_rows_get_speed():
    out = add_kinematics(two_object_frame(), smooth_window=1)
    tracked = out[(out["track_id"] == 3) & (out["frame"] > 0)]
    # 0.1 m per 0.04 s = 2.5 m/s
    assert tracked["speed_ms"].dropna().iloc[0] == pytest.approx(2.5, abs=0.01)


def test_row_order_is_preserved():
    df = two_object_frame()
    out = add_kinematics(df)
    assert list(out.index) == list(df.index)


def test_summarise_player_keeps_anonymous_tracks_separate():
    """Two identity-abstained tracks of the same team must not melt into one row
    with their distances summed."""
    rows = []
    for tid in (1, 2):
        for f in range(10):
            rows.append(
                {
                    "frame": f,
                    "t_s": f / 25,
                    "track_id": tid,
                    "team": "home",
                    "player": pd.NA,
                    "x_m": 10.0 + tid + 0.2 * f,  # 0.2 m per frame -> 5 m/s
                    "y_m": 30.0,
                }
            )
    out = add_kinematics(pd.DataFrame(rows), smooth_window=1)
    summary = summarise_player(out)
    assert len(summary) == 2
    # Each track covers 1.6 m on its own: 10 frames -> 9 speed intervals, minus the
    # first retained row whose dt is zeroed -> 8 counted steps x 0.2 m. The old
    # (team, player) grouping reported one row with double this.
    for dist in summary["distance_m"]:
        assert dist == pytest.approx(1.6, abs=0.05)


def test_summarise_player_empty_when_uncalibrated():
    df = pd.DataFrame(
        {
            "frame": [0, 1],
            "t_s": [0.0, 0.04],
            "track_id": [1, 1],
            "team": ["home", "home"],
            "player": [pd.NA, pd.NA],
            "x_m": [np.nan, np.nan],
            "y_m": [np.nan, np.nan],
        }
    )
    out = add_kinematics(df)
    assert summarise_player(out).empty
