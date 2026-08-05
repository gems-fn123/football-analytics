"""Contract test for the full stage chain minus the detector.

Synthetic detections flow through track -> team -> identity -> calibrate ->
kinematics, and the result must satisfy the TRACKS_M contract exactly. This is the
test that would have caught the original tracks.parquet contract violations
(missing t_s and speed_ms, object dtypes for team/shirt/player).
"""

import json

import numpy as np
import pandas as pd
from conftest import (
    KIT_BLUE_BGR,
    KIT_BLUE_HEX,
    KIT_RED_BGR,
    KIT_RED_HEX,
    FakeVideo,
    synthetic_tracks_px,
)

from footy.analytics.physical import add_kinematics
from footy.config import Config
from footy.pipeline import _final_tracks_m
from footy.schemas import TRACKS_M
from footy.stages.base import StageResult
from footy.stages.calibrate import Calibrator
from footy.stages.identity import IdentityResolver
from footy.stages.team import TeamAssigner
from footy.stages.track import Tracker

BOXES = {
    0: (KIT_RED_BGR, 50, 100, 80, 160),
    1: (KIT_BLUE_BGR, 350, 100, 380, 160),
}
SCALE_H = [[105 / 640, 0.0, 0.0], [0.0, -68 / 360, 68.0], [0.0, 0.0, 1.0]]


def test_chain_produces_contract_conformant_tracks_m(tmp_path):
    detections = synthetic_tracks_px(BOXES, n_frames=10).drop(columns=["track_id"])
    cfg = Config(
        raw={},
        match={"home": {"kit_colour": KIT_RED_HEX}, "away": {"kit_colour": KIT_BLUE_HEX}},
    )
    ctx: dict = {"video": FakeVideo(BOXES, n_frames=10), "config": cfg}
    ctx["detect"] = StageResult(
        "detect", detections, artifacts={"frames": list(range(10))}, stats={"n_frames": 10}
    )

    tracker = Tracker({"name": "bytetrack", "frame_rate": 25})
    tracker.setup()
    ctx["track"] = tracker.run(ctx)

    team = TeamAssigner({"method": "kit_colour", "sample_stride": 1, "min_crops": 3})
    team.setup()
    ctx["team"] = team.run(ctx)

    identity = IdentityResolver({})
    identity.setup()
    ctx["identity"] = identity.run(ctx)

    pts = tmp_path / "points.json"
    pts.write_text(json.dumps({"solved_homography": SCALE_H}))
    calib = Calibrator({"homography": {"mode": "static", "manual_points": str(pts)}})
    calib.setup()
    ctx["calibrate"] = calib.run(ctx)

    # The same assembly the orchestrator itself uses, not a reimplementation.
    final = _final_tracks_m(ctx)

    # Contract: exact columns, exact dtypes.
    assert list(final.columns) == list(TRACKS_M)
    assert str(final["team"].dtype) == "string"
    assert str(final["shirt"].dtype) == "Int16"
    assert str(final["player"].dtype) == "string"

    players = final[final["team"].isin(["home", "away"])]
    assert set(players["team"]) == {"home", "away"}
    # Static boxes: speeds exist after the first frame and are ~0, not noise.
    speeds = players["speed_ms"].dropna()
    assert len(speeds) > 0
    assert float(speeds.max()) < 0.5

    # The ball is not motion-tracked, so its pseudo-track must carry no kinematics.
    ball = final[final["team"] == "ball"]
    assert ball["speed_ms"].isna().all()

    # Positions land on the pitch under the known homography.
    assert players["x_m"].between(0, 105).all()
    assert players["y_m"].between(0, 68).all()


def test_kinematics_are_nan_for_untracked_bucket():
    df = pd.DataFrame(
        {
            "frame": [0, 1, 0, 1],
            "t_s": [0.0, 0.04, 0.0, 0.04],
            "track_id": [-1, -1, 3, 3],
            "x_m": [10.0, 90.0, 50.0, 50.4],
            "y_m": [30.0, 5.0, 30.0, 30.0],
        }
    )
    out = add_kinematics(df, smooth_window=1)
    assert out.loc[out["track_id"] == -1, "speed_ms"].isna().all()
    tracked = out[(out["track_id"] == 3) & (out["frame"] == 1)]
    assert np.isfinite(tracked["speed_ms"].iloc[0])
