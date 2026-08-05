"""Stage 5: map image pixels to pitch metres.

Plain version: work out where the camera is looking, so a player at pixel (940, 512)
becomes a player at 62.3 m along and 21.7 m across.

Fixed camera: solve once from four or more known landmarks, reuse all match.
Broadcast: re-solve every frame from detected pitch lines and keypoints. That needs
a pitch keypoint model which is not wired up yet, so per_frame mode currently
abstains: metres come out null with a loud warning rather than a wrong number.

This stage also owns the tracks_m assembly: t_s from the frame index, pixel boxes
collapsed to a ground contact point, contract columns in contract order. speed_ms
is left NaN here and filled by analytics.physical in the orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footy.logging_utils import get_logger
from footy.schemas import TRACKS_M, validate
from footy.stages.base import Stage, StageResult, upstream_table


def solve_homography(
    image_points: list[list[float]], pitch_points: list[list[float]]
) -> tuple[np.ndarray, float]:
    """RANSAC-solve pixels -> metres. Returns (H, max inlier reprojection error, m).

    The error is computed over RANSAC inliers only: RANSAC exists to discard a
    mis-clicked landmark, and reporting the discarded point's residual would fail a
    perfectly good calibration. Rejected points are logged instead.
    """
    import cv2

    if len(image_points) < 4:
        raise ValueError(f"need at least 4 point pairs, got {len(image_points)}")
    src = np.array(image_points, dtype=np.float64)
    dst = np.array(pitch_points, dtype=np.float64)
    # ransacReprojThreshold is in destination units, i.e. metres here.
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if H is None:
        raise ValueError("homography solve failed; are the points collinear?")
    residuals = np.linalg.norm(Calibrator.apply(H, src) - dst, axis=1)
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), dtype=bool)
    n_rejected = int((~inliers).sum())
    if n_rejected:
        get_logger("footy.calibrate").warning(
            "RANSAC rejected %d of %d landmark pairs (worst residual %.1f m); "
            "the fit uses the rest",
            n_rejected,
            len(src),
            float(residuals[~inliers].max()),
        )
    err = float(residuals[inliers].max()) if inliers.any() else float(residuals.max())
    return H, err


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
        tracks = upstream_table(ctx, "identity", "team", "track", stage=self.name).copy()
        fps = float(ctx["video"].meta.fps) or 25.0

        # Tolerate skipped enrichment stages: the contract columns they would have
        # added exist as nulls, which is exactly what "that stage did not run" means.
        if "team" not in tracks.columns:
            tracks["team"] = pd.Series(pd.NA, index=tracks.index, dtype="string")
        if "shirt" not in tracks.columns:
            tracks["shirt"] = pd.Series(pd.NA, index=tracks.index, dtype="Int16")
        if "player" not in tracks.columns:
            tracks["player"] = pd.Series(pd.NA, index=tracks.index, dtype="string")

        x_m = np.full(len(tracks), np.nan)
        y_m = np.full(len(tracks), np.nan)

        if self.mode == "static" and self.H is not None:
            # Bottom-centre of the box is the ground contact point; the homography
            # maps the ground plane, so feet are the only honest anchor.
            ground = np.column_stack(
                [
                    (tracks["x1"].to_numpy() + tracks["x2"].to_numpy()) / 2.0,
                    tracks["y2"].to_numpy(),
                ]
            )
            if len(ground):
                projected = self.apply(self.H, ground)
                x_m, y_m = projected[:, 0], projected[:, 1]
        elif self.mode == "static":
            self.log.warning(
                "static calibration has no solved homography; metres stay null. "
                "Solve once with scripts/calibrate_fixed_camera.py or footy calibrate."
            )
        else:
            self.log.warning(
                "per_frame calibration needs a pitch keypoint model that is not wired "
                "up; metres stay null. Tracks remain valid in pixel space "
                "(tracks_px.parquet); use a fixed camera for the metric chain."
            )

        df = tracks.assign(
            t_s=tracks["frame"].to_numpy() / fps,
            x_m=x_m,
            y_m=y_m,
            speed_ms=np.nan,  # filled by analytics.physical.add_kinematics
        )
        df = validate(df[list(TRACKS_M)], TRACKS_M, self.name)

        calibrated = self.H is not None
        stats: dict[str, Any] = {"mode": self.mode, "calibrated": calibrated}
        if calibrated and len(df):
            inside = (
                df["x_m"].between(-3, 108) & df["y_m"].between(-3, 71)
            ).mean()  # small margin: keepers and throw-ins stand off the pitch
            stats["fraction_on_pitch"] = round(float(inside), 3)
            if inside < 0.9:
                self.log.warning(
                    "only %.0f%% of points land on the pitch; homography or pitch "
                    "dimensions look wrong",
                    inside * 100,
                )
        return StageResult(self.name, df, artifacts={"homography": self.H}, stats=stats)
