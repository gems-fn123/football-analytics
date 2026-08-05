import pandas as pd
import pytest

from footy.schemas import DETECTIONS
from footy.stages.base import StageResult
from footy.stages.track import Tracker


def make_detections(n_frames: int = 12, n_players: int = 3, with_ball: bool = True):
    """Players drifting steadily right. Easy association, so any id churn is a bug."""
    rows = []
    for frame in range(n_frames):
        det_id = 0
        for p in range(n_players):
            x = 100.0 + 120.0 * p + 4.0 * frame
            rows.append(
                {
                    "frame": frame,
                    "det_id": det_id,
                    "cls": "player",
                    "conf": 0.9,
                    "x1": x,
                    "y1": 300.0,
                    "x2": x + 40.0,
                    "y2": 400.0,
                }
            )
            det_id += 1
        if with_ball:
            rows.append(
                {
                    "frame": frame,
                    "det_id": det_id,
                    "cls": "ball",
                    "conf": 0.8,
                    "x1": 500.0 + 9.0 * frame,
                    "y1": 350.0,
                    "x2": 512.0 + 9.0 * frame,
                    "y2": 362.0,
                }
            )
    # Columns even when empty: that is what stages.detect emits, so it is the contract
    # stages.track actually has to survive.
    return pd.DataFrame(rows, columns=list(DETECTIONS))


def run_tracker(
    detections: pd.DataFrame, cfg: dict | None = None, grid: list[int] | None = None
) -> pd.DataFrame:
    stage = Tracker(cfg or {"name": "bytetrack", "frame_rate": 25})
    stage.setup()
    artifacts = {"frames": grid} if grid is not None else {}
    ctx = {"detect": StageResult("detect", detections, artifacts=artifacts)}
    return stage.run(ctx).table


def test_steady_players_keep_stable_ids():
    df = run_tracker(make_detections())
    players = df[df["cls"] == "player"]
    assert (players["track_id"] >= 0).all()
    # Three players, three ids, and nobody swaps.
    assert players["track_id"].nunique() == 3
    per_frame = players.groupby("frame")["track_id"].apply(frozenset)
    assert per_frame.nunique() == 1


def test_ball_is_left_untracked():
    """stages.ball owns the ball. Feeding it to ByteTrack pollutes the motion model."""
    df = run_tracker(make_detections(with_ball=True))
    assert (df.loc[df["cls"] == "ball", "track_id"] == -1).all()


def test_empty_detections_survive():
    df = run_tracker(make_detections(n_frames=0, n_players=0, with_ball=False))
    assert df.empty
    assert "track_id" in df.columns


def test_id_survives_a_short_occlusion():
    """A player hidden for a few frames must come back with the same track id."""
    full = make_detections(n_frames=20, n_players=2, with_ball=False)
    # Player 1 vanishes for frames 8-11 (well inside lost_track_buffer).
    occluded = full[~((full["det_id"] == 1) & full["frame"].between(8, 11))]
    df = run_tracker(occluded, grid=list(range(20)))
    p1 = df[(df["det_id"] == 1)]
    assert p1["track_id"].nunique() == 1
    assert (p1["track_id"] >= 0).all()


def test_empty_frames_age_the_tracker():
    """Detector silence must age lost tracks in wall-clock frames: after a dropout
    longer than lost_track_buffer, a player at a new position is a NEW track, not a
    stale id resurrected from before the dropout."""
    early = make_detections(n_frames=3, n_players=1, with_ball=False)
    late = make_detections(n_frames=3, n_players=1, with_ball=False)
    late["frame"] += 100  # far beyond lost_track_buffer=60 at 25 fps
    late[["x1", "x2"]] += 400.0  # and somewhere else entirely
    df = run_tracker(pd.concat([early, late], ignore_index=True), grid=list(range(103)))
    early_ids = set(df[df["frame"] < 3]["track_id"]) - {-1}
    late_ids = set(df[df["frame"] >= 100]["track_id"]) - {-1}
    assert early_ids and late_ids
    assert not early_ids & late_ids


def test_botsort_backend_is_refused_not_silently_downgraded():
    with pytest.raises(NotImplementedError, match="botsort"):
        Tracker({"name": "botsort"}).setup()


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown tracker backend"):
        Tracker({"name": "deepsort"}).setup()
