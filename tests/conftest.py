from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest


@dataclass
class FakeMeta:
    fps: float = 25.0
    width: int = 640
    height: int = 360
    n_frames: int = 10

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps


class FakeVideo:
    """Synthetic VideoSource: green pitch, solid-colour kit rectangles.

    boxes: track_id -> (colour_bgr, x1, y1, x2, y2), drawn identically each frame.
    """

    def __init__(
        self,
        boxes: dict[int, tuple[tuple[int, int, int], int, int, int, int]],
        n_frames: int = 10,
        width: int = 640,
        height: int = 360,
    ) -> None:
        self.boxes = boxes
        self.meta = FakeMeta(width=width, height=height, n_frames=n_frames)
        self.stride = 1

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        for f in range(self.meta.n_frames):
            frame = np.zeros((self.meta.height, self.meta.width, 3), dtype=np.uint8)
            frame[:] = (30, 160, 30)  # saturated pitch green
            for _tid, (colour, x1, y1, x2, y2) in self.boxes.items():
                frame[y1:y2, x1:x2] = colour
            yield f, frame


# BGR kit colours for FakeVideo, with their hex twins for match configs.
KIT_RED_BGR, KIT_RED_HEX = (30, 30, 200), "#C81E1E"
KIT_BLUE_BGR, KIT_BLUE_HEX = (200, 60, 30), "#1E3CC8"
KIT_YELLOW_BGR, KIT_YELLOW_HEX = (30, 220, 220), "#DCDC1E"


def synthetic_tracks_px(
    boxes: dict[int, tuple[tuple[int, int, int], int, int, int, int]],
    n_frames: int = 10,
    with_ball: bool = True,
) -> pd.DataFrame:
    """A tracks_px table consistent with what FakeVideo draws."""
    rows = []
    for f in range(n_frames):
        det_id = 0
        for tid, (_colour, x1, y1, x2, y2) in boxes.items():
            rows.append(
                {
                    "frame": f,
                    "det_id": det_id,
                    "cls": "player",
                    "conf": 0.9,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "track_id": tid,
                }
            )
            det_id += 1
        if with_ball:
            rows.append(
                {
                    "frame": f,
                    "det_id": det_id,
                    "cls": "ball",
                    "conf": 0.7,
                    "x1": 300.0 + f,
                    "y1": 200.0,
                    "x2": 308.0 + f,
                    "y2": 208.0,
                    "track_id": -1,
                }
            )
    # Contract dtypes, so tests exercise what real stages actually emit.
    from footy.schemas import TRACKS_PX, validate

    return validate(pd.DataFrame(rows), TRACKS_PX, "synthetic_tracks_px")


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
            rows.append(
                {
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
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def shots() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 400
    sx = rng.uniform(70, 104, n)
    sy = rng.uniform(14, 54, n)
    dist = np.hypot(105 - sx, 34 - sy)
    p = np.clip(0.75 - dist * 0.035, 0.01, 0.9)
    return pd.DataFrame(
        {
            "start_x": sx,
            "start_y": sy,
            "is_goal": rng.binomial(1, p),
        }
    )
