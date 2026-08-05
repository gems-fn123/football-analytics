"""Stage 5: map image pixels to pitch metres.

Plain version: work out where the camera is looking, so a player at pixel (940, 512)
becomes a player at 62.3 m along and 21.7 m across.

Fixed camera: solve once from four or more known landmarks, reuse all match.
Broadcast: re-solve every frame from detected pitch lines and keypoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footy.stages.base import Stage, StageResult


class Calibrator(Stage):
    name = "calibrate"

    def setup(self) -> None:
        self.mode = self.cfg.get("homography", {}).get("mode", "per_frame")
        self.H: np.ndarray | None = None
        if self.mode == "static":
            pts_file = self.cfg["homography"].get("manual_points")
            if pts_file and Path(pts_file).exists():
                blob = json.loads(Path(pts_file).read_text())
                if blob.get("solved_homography"):
                    self.H = np.array(blob["solved_homography"], dtype=np.float64)
        self.log.info("calibration mode=%s preloaded=%s", self.mode, self.H is not None)

    @staticmethod
    def apply(H: np.ndarray, xy_px: np.ndarray) -> np.ndarray:
        """Project Nx2 pixel points to Nx2 pitch metres."""
        ones = np.ones((xy_px.shape[0], 1))
        homo = np.hstack([xy_px, ones]) @ H.T
        return homo[:, :2] / homo[:, 2:3]

    def run(self, ctx: dict[str, Any]) -> StageResult:
        tracks: pd.DataFrame = ctx["identity"].table
        # TODO: per_frame mode - detect pitch keypoints, solve H per frame
        #       static mode   - use self.H for every frame
        #       Use the bottom-centre of each box as the ground contact point.
        df = tracks.assign(x_m=np.nan, y_m=np.nan)
        return StageResult(self.name, df)
