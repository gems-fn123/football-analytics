"""Orchestration test: run_pipeline end to end with a stubbed detector.

The real detector needs a model download; everything else - stage sequencing, the
skip toggles, kinematics, output files, the report - is exercised for real against
a tiny generated video.
"""

import json

import cv2
import numpy as np
import pandas as pd
import pytest
from conftest import KIT_BLUE_BGR, KIT_BLUE_HEX, KIT_RED_BGR, KIT_RED_HEX

import footy.pipeline as pipeline_mod
from footy.config import Config
from footy.schemas import DETECTIONS, TRACKS_M, validate
from footy.stages.base import Stage, StageResult

BOXES = {
    0: (KIT_RED_BGR, 50, 100, 80, 160),
    1: (KIT_BLUE_BGR, 350, 100, 380, 160),
}
N_FRAMES = 10
SCALE_H = [[105 / 640, 0.0, 0.0], [0.0, -68 / 360, 68.0], [0.0, 0.0, 1.0]]


class StubDetector(Stage):
    """Emits the boxes the generated video actually contains, plus a ball."""

    name = "detect"

    def run(self, ctx):
        rows = []
        processed = []
        for frame_idx, _image in ctx["video"]:
            processed.append(frame_idx)
            det_id = 0
            for _tid, (_c, x1, y1, x2, y2) in BOXES.items():
                rows.append(
                    {
                        "frame": frame_idx,
                        "det_id": det_id,
                        "cls": "player",
                        "conf": 0.9,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )
                det_id += 1
            rows.append(
                {
                    "frame": frame_idx,
                    "det_id": det_id,
                    "cls": "ball",
                    "conf": 0.7,
                    "x1": 300.0 + frame_idx,
                    "y1": 200.0,
                    "x2": 308.0 + frame_idx,
                    "y2": 208.0,
                }
            )
        df = validate(pd.DataFrame(rows, columns=list(DETECTIONS)), DETECTIONS, "detect")
        return StageResult(
            "detect", df, artifacts={"frames": processed}, stats={"n_frames": len(processed)}
        )


@pytest.fixture
def video_path(tmp_path):
    # MJPG in an AVI container: encodable by every bundled OpenCV build, unlike mp4v.
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (640, 360))
    for _ in range(N_FRAMES):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (30, 160, 30)
        for _tid, (colour, x1, y1, x2, y2) in BOXES.items():
            frame[y1:y2, x1:x2] = colour
        writer.write(frame)
    writer.release()
    return path


def make_config(tmp_path, stages_overrides: dict | None = None) -> Config:
    pts = tmp_path / "points.json"
    pts.write_text(json.dumps({"solved_homography": SCALE_H}))
    raw = {
        "match_id": "orchestration_test",
        "io": {"output_dir": str(tmp_path / "processed")},
        "camera": {"homography": {"mode": "static", "manual_points": str(pts)}},
        "detector": {},
        "tracker": {"name": "bytetrack"},
        "team": {"method": "kit_colour", "sample_stride": 1, "min_crops": 3},
        "stages": {"events": False, **(stages_overrides or {})},
        "runtime": {"device": "cpu"},
        "outputs": {},
    }
    match = {"home": {"kit_colour": KIT_RED_HEX}, "away": {"kit_colour": KIT_BLUE_HEX}}
    return Config(raw=raw, match=match)


def test_run_pipeline_writes_contract_outputs(tmp_path, video_path, monkeypatch):
    monkeypatch.setattr(
        pipeline_mod,
        "STAGE_ORDER",
        [
            ("detect", StubDetector, "detector"),
            *pipeline_mod.STAGE_ORDER[1:],
        ],
    )
    result = pipeline_mod.run_pipeline(video_path, make_config(tmp_path))
    out = result["outputs"]

    for key in ("tracks", "tracks_px", "ball", "report"):
        assert key in out and out[key].exists(), f"missing output: {key}"

    tracks = pd.read_parquet(out["tracks"])
    assert list(tracks.columns) == list(TRACKS_M)
    validate(tracks, TRACKS_M, "written tracks")  # dtypes survive the round-trip
    players = tracks[tracks["team"].isin(["home", "away"])]
    assert set(players["team"]) == {"home", "away"}
    assert players["x_m"].between(0, 105).all()

    ball = pd.read_parquet(out["ball"])
    assert len(ball) == N_FRAMES
    assert ball["x_m"].notna().all()  # calibrated run projects the ball too

    html = out["report"].read_text(encoding="utf-8")
    assert "orchestration_test" in html


def test_run_pipeline_survives_skipping_enrichment_stages(tmp_path, video_path, monkeypatch):
    """team + identity off: tracks still written, with null team/shirt/player."""
    monkeypatch.setattr(
        pipeline_mod,
        "STAGE_ORDER",
        [
            ("detect", StubDetector, "detector"),
            *pipeline_mod.STAGE_ORDER[1:],
        ],
    )
    cfg = make_config(tmp_path, {"team": False, "identity": False})
    result = pipeline_mod.run_pipeline(video_path, cfg)
    tracks = pd.read_parquet(result["outputs"]["tracks"])
    person = tracks[tracks["track_id"] >= 0]
    assert person["team"].isna().all()
    assert person["shirt"].isna().all()


def test_run_pipeline_names_the_missing_stage(tmp_path, video_path, monkeypatch):
    """Disabling a load-bearing stage fails fast with an actionable message."""
    monkeypatch.setattr(
        pipeline_mod,
        "STAGE_ORDER",
        [
            ("detect", StubDetector, "detector"),
            *pipeline_mod.STAGE_ORDER[1:],
        ],
    )
    cfg = make_config(tmp_path, {"track": False})
    with pytest.raises(RuntimeError, match="needs one of \\['track'\\]"):
        pipeline_mod.run_pipeline(video_path, cfg)
