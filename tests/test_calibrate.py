import json

import numpy as np
import pandas as pd
import pytest
from conftest import FakeVideo, synthetic_tracks_px

from footy.schemas import TRACKS_M
from footy.stages.base import StageResult
from footy.stages.calibrate import Calibrator, solve_homography

# Maps 640x360 pixels onto a 105x68 pitch with the y-axis flipped: origin
# bottom-left in metres, top-left in pixels.
SCALE_H = [[105 / 640, 0.0, 0.0], [0.0, -68 / 360, 68.0], [0.0, 0.0, 1.0]]


def test_apply_identity_homography():
    H = np.eye(3)
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(Calibrator.apply(H, pts), pts)


def test_apply_translation():
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    pts = np.array([[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_allclose(Calibrator.apply(H, pts), [[5.0, -3.0], [7.0, -1.0]])


def make_identity_ctx() -> dict:
    boxes = {0: ((0, 0, 200), 100, 100, 130, 160), 1: ((200, 0, 0), 400, 200, 430, 260)}
    tracks = synthetic_tracks_px(boxes, n_frames=5, with_ball=False)
    tracks = tracks.assign(
        team=pd.Series("home", index=tracks.index, dtype="string"),
        shirt=pd.Series(pd.NA, index=tracks.index, dtype="Int16"),
        player=pd.Series(pd.NA, index=tracks.index, dtype="string"),
    )
    return {"video": FakeVideo(boxes, n_frames=5), "identity": StageResult("identity", tracks)}


def run_calibrator(tmp_path, mode: str, H: list | None) -> StageResult:
    cfg: dict = {"homography": {"mode": mode}}
    if H is not None:
        pts = tmp_path / "points.json"
        pts.write_text(json.dumps({"solved_homography": H}))
        cfg["homography"]["manual_points"] = str(pts)
    stage = Calibrator(cfg)
    stage.setup()
    return stage.run(make_identity_ctx())


def test_static_mode_projects_ground_contact_point(tmp_path):
    result = run_calibrator(tmp_path, "static", SCALE_H)
    df = result.table
    row = df[(df["track_id"] == 0) & (df["frame"] == 0)].iloc[0]
    # Bottom-centre of box (100,100,130,160): pixel (115, 160).
    assert row["x_m"] == pytest.approx(115 * 105 / 640, abs=1e-3)
    assert row["y_m"] == pytest.approx(68 - 160 * 68 / 360, abs=1e-3)
    assert result.stats["calibrated"] is True


def test_t_s_comes_from_frame_and_fps(tmp_path):
    df = run_calibrator(tmp_path, "static", SCALE_H).table
    assert df[df["frame"] == 4]["t_s"].iloc[0] == pytest.approx(4 / 25.0)


def test_output_satisfies_tracks_m_contract(tmp_path):
    df = run_calibrator(tmp_path, "static", SCALE_H).table
    assert list(df.columns) == list(TRACKS_M)
    assert str(df["shirt"].dtype) == "Int16"
    assert str(df["team"].dtype) == "string"


def test_uncalibrated_static_abstains_with_nan(tmp_path):
    result = run_calibrator(tmp_path, "static", None)
    assert result.table["x_m"].isna().all()
    assert result.stats["calibrated"] is False


def test_per_frame_mode_abstains_with_nan(tmp_path):
    result = run_calibrator(tmp_path, "per_frame", None)
    assert result.table["x_m"].isna().all()
    assert result.artifacts["homography"] is None


def test_solve_homography_recovers_known_mapping():
    H_true = np.array(SCALE_H)
    image_points = [[0.0, 0.0], [640.0, 0.0], [640.0, 360.0], [0.0, 360.0], [320.0, 180.0]]
    pitch_points = [Calibrator.apply(H_true, np.array([p]))[0].tolist() for p in image_points]
    H, err = solve_homography(image_points, pitch_points)
    assert err < 1e-6
    probe = np.array([[100.0, 100.0], [500.0, 300.0]])
    np.testing.assert_allclose(
        Calibrator.apply(H, probe), Calibrator.apply(H_true, probe), atol=1e-6
    )


def test_solve_homography_needs_four_points():
    with pytest.raises(ValueError, match="at least 4"):
        solve_homography([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])


def test_solve_homography_error_ignores_ransac_rejected_outliers():
    """One mis-clicked landmark must not fail an otherwise exact calibration."""
    H_true = np.array(SCALE_H)
    image_points = [
        [0.0, 0.0],
        [640.0, 0.0],
        [640.0, 360.0],
        [0.0, 360.0],
        [320.0, 180.0],
        [160.0, 90.0],
    ]
    pitch_points = [Calibrator.apply(H_true, np.array([p]))[0].tolist() for p in image_points]
    image_points.append([500.0, 300.0])
    pitch_points.append([5.0, 60.0])  # wildly wrong pair, ~8 m residual
    H, err = solve_homography(image_points, pitch_points)
    assert err < 0.5  # inlier error, not the rejected outlier's residual
    probe = np.array([[100.0, 100.0]])
    np.testing.assert_allclose(
        Calibrator.apply(H, probe), Calibrator.apply(H_true, probe), atol=0.1
    )
