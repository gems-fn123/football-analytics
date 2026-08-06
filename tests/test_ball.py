import pandas as pd
import pytest
from conftest import FakeVideo

from footy.config import Config
from footy.schemas import DETECTIONS
from footy.stages.ball import BallTracker
from footy.stages.base import StageResult


def ball_detections(frames_xy: dict[int, tuple[float, float, float]]) -> pd.DataFrame:
    """frame -> (cx, cy, conf), as 8px ball boxes in the DETECTIONS shape."""
    rows = []
    for f, (cx, cy, conf) in sorted(frames_xy.items()):
        rows.append(
            {
                "frame": f,
                "det_id": 0,
                "cls": "ball",
                "conf": conf,
                "x1": cx - 4,
                "y1": cy - 4,
                "x2": cx + 4,
                "y2": cy + 4,
            }
        )
    return pd.DataFrame(rows, columns=list(DETECTIONS))


def run_ball(
    frames_xy: dict,
    n_frames: int = 40,
    grid: list[int] | None = None,
    homography: list | None = None,
) -> StageResult:
    stage = BallTracker({"max_speed_ms": 45.0, "max_gap_frames": 12, "reacquire_s": 1.0})
    stage.setup()
    ctx = {
        "video": FakeVideo({}, n_frames=n_frames, width=768),
        "detect": StageResult(
            "detect",
            ball_detections(frames_xy),
            artifacts={"frames": grid if grid is not None else list(range(n_frames))},
            stats={"n_frames": n_frames},
        ),
        "config": Config(raw={"pitch": {"length_m": 105.0}}, match={}),
    }
    if homography is not None:
        import numpy as np

        ctx["calibrate"] = StageResult(
            "calibrate", pd.DataFrame(), artifacts={"homography": np.array(homography)}
        )
    return stage.run(ctx)


def steady_path(frames: list[int]) -> dict[int, tuple[float, float, float]]:
    return {f: (100.0 + 10.0 * f, 200.0, 0.8) for f in frames}


def test_steady_ball_is_fully_kept():
    result = run_ball(steady_path(list(range(20))))
    df = result.table
    assert len(df) == 20
    assert not df["interpolated"].any()
    assert result.stats["n_rejected_speed"] == 0


def test_teleporting_candidate_is_rejected():
    path = steady_path(list(range(20)))
    # Frame 5's only candidate is a confident false positive across the pitch;
    # it must die at the speed gate, and the hole it leaves gets interpolated.
    path[5] = (700.0, 40.0, 0.99)
    result = run_ball(path)
    df = result.table
    assert result.stats["n_rejected_speed"] == 1
    frame5 = df[df["frame"] == 5].iloc[0]
    assert bool(frame5["interpolated"])  # refilled from the neighbours, not measured
    # The refill is the linear midpoint of frames 4 and 6, not the outlier.
    assert frame5["x_px"] == pytest.approx(150.0)
    assert frame5["y_px"] == pytest.approx(200.0)


def test_short_gap_is_interpolated_long_gap_is_not():
    frames = list(range(9)) + [13, 30]
    result = run_ball(steady_path(frames))
    df = result.table.set_index("frame")
    # Gap 9-12 (4 frames) is within max_gap_frames=12: filled, marked, conf NaN,
    # and positions sit on the straight line between the endpoints.
    for f in range(9, 13):
        assert bool(df.loc[f, "interpolated"])
        assert pd.isna(df.loc[f, "conf"])
        assert df.loc[f, "x_px"] == pytest.approx(100.0 + 10.0 * f)
    # Gap 14-29 (16 frames) is beyond the limit: absent entirely.
    assert not set(range(14, 30)) & set(df.index)


def test_gap_semantics_follow_the_processed_frame_grid():
    """With frame_stride 2 only even frames exist; a 20-raw-frame hole is 10
    processed frames and must still interpolate (10 <= max_gap_frames=12)."""
    grid = list(range(0, 80, 2))
    frames = [0, 2, 4, 6, 26, 28]
    result = run_ball(steady_path(frames), n_frames=80, grid=grid)
    df = result.table.set_index("frame")
    filled = [f for f in range(8, 26, 2)]
    assert all(f in df.index and bool(df.loc[f, "interpolated"]) for f in filled)
    # Nothing is invented on frames the run never decoded.
    assert not (set(df.index) - set(grid))
    assert result.stats["coverage_fraction"] <= 1.0


def test_reacquire_after_long_silence_accepts_the_next_candidate():
    """After reacquire_s with nothing accepted, a jump must not be vetoed forever."""
    path = steady_path(list(range(5)))
    # 60 frames of silence (2.4 s > reacquire_s=1.0), then the ball reappears far
    # away: accepted as a fresh acquisition, not speed-gated against frame 4.
    path[65] = (700.0, 40.0, 0.9)
    result = run_ball(path, n_frames=80)
    df = result.table.set_index("frame")
    assert 65 in df.index and not bool(df.loc[65, "interpolated"])
    assert result.stats["n_rejected_speed"] == 0
    # The 60-frame hole is beyond max_gap_frames: no interpolation across it.
    assert not (set(df.index) & set(range(5, 65)))


def test_calibrated_ball_gets_metres_via_homography():
    H = [[105 / 768, 0.0, 0.0], [0.0, -68 / 432, 68.0], [0.0, 0.0, 1.0]]
    df = run_ball(steady_path(list(range(5))), homography=H).table
    assert df["x_m"].notna().all()
    row = df[df["frame"] == 0].iloc[0]
    assert row["x_m"] == pytest.approx(100.0 * 105 / 768, abs=1e-3)
    assert row["y_m"] == pytest.approx(68 - 200.0 * 68 / 432, abs=1e-3)


def test_uncalibrated_metres_are_null():
    df = run_ball(steady_path(list(range(10)))).table
    assert df["x_m"].isna().all() and df["y_m"].isna().all()


def test_per_frame_metres_respect_the_solve_gap_cap():
    """A ball seen long after the last successful solve stays null - the same
    'null, not stale' rule the calibrate stage applies to players."""
    import numpy as np

    stage = BallTracker({"max_speed_ms": 45.0, "max_gap_frames": 12, "reacquire_s": 1.0})
    stage.setup()
    n_frames = 40
    frames_xy = {0: (100.0, 200.0, 0.9), 30: (400.0, 200.0, 0.9)}
    ctx = {
        "video": FakeVideo({}, n_frames=n_frames, width=768),
        "detect": StageResult(
            "detect",
            ball_detections(frames_xy),
            artifacts={"frames": list(range(n_frames))},
            stats={"n_frames": n_frames},
        ),
        "config": Config(raw={"pitch": {"length_m": 105.0}}, match={}),
        "calibrate": StageResult(
            "calibrate",
            pd.DataFrame(),
            artifacts={
                "homography": None,
                "homographies_px_to_m": {0: np.eye(3)},
                "solve_gap_frames": 6,
            },
        ),
    }
    result = stage.run(ctx)
    df = result.table
    assert df[df["frame"] == 0]["x_m"].notna().all()
    assert df[df["frame"] == 30]["x_m"].isna().all()
    assert result.stats["calibrated"] is True


def test_no_candidates_is_survivable():
    result = run_ball({})
    assert result.table.empty
    assert result.stats["coverage_fraction"] == 0.0
