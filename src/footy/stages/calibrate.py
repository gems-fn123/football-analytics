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
        hcfg = self.cfg.get("homography", {})
        self.mode = hcfg.get("mode", "per_frame")
        self.H: np.ndarray | None = None
        self.refresh = max(1, int(hcfg.get("refresh_every_n_frames", 1)))
        self._model = None
        self._grid: dict[int, tuple[float, float]] | None = None

        if self.mode == "static":
            pts_file = hcfg.get("manual_points")
            if pts_file and Path(pts_file).exists():
                blob = json.loads(Path(pts_file).read_text())
                if blob.get("solved_homography"):
                    self.H = np.array(blob["solved_homography"], dtype=np.float64)
        elif self.mode == "per_frame":
            self._kp_weights = hcfg.get("keypoint_model", "models/weights/SV_kp.pth")
            self._kp_grid_path = hcfg.get("keypoint_grid", "analysis/calib/keypoint_grid.json")
            self._min_score = float(hcfg.get("min_score", 1.0))
            self._min_points = int(hcfg.get("min_points", 5))
        self.log.info("calibration mode=%s preloaded=%s", self.mode, self.H is not None)

    def _load_keypoint_model(self) -> bool:
        """Lazy: the model is 265 MB and only per_frame mode needs it."""
        if self._model is not None:
            return True
        if not Path(self._kp_weights).exists() or not Path(self._kp_grid_path).exists():
            self.log.warning(
                "per_frame calibration needs %s and %s; run scripts/download_weights.sh "
                "and scripts/derive_keypoint_grid.py. Metres stay null.",
                self._kp_weights,
                self._kp_grid_path,
            )
            return False
        from footy.calib.hrnet import load_checkpoint
        from footy.calib.keypoints import snap_grid

        blob = json.loads(Path(self._kp_grid_path).read_text())
        self._grid = snap_grid({int(k): tuple(v) for k, v in blob["grid"].items()})
        self._model = load_checkpoint(
            self._kp_weights, stem_position=blob.get("stem_position", "last")
        ).eval()
        return True

    def _solve_frame(self, image: np.ndarray) -> np.ndarray | None:
        """One frame -> H (pitch -> image px), or None when not confidently solvable.

        Injectable for tests: monkeypatch this to avoid the model.
        """
        import torch

        from footy.calib.keypoints import extract_peaks, homography_from_keypoints, preprocess
        from footy.calib.line_dlt import project

        with torch.no_grad():
            hm = self._model(preprocess(image))[0].numpy()
        peaks = extract_peaks(hm, image.shape[:2], min_score=self._min_score)
        H = homography_from_keypoints(
            peaks, self._grid, min_score=self._min_score, min_points=self._min_points
        )
        if H is None:
            return None
        # Consistency gate: the solved H must put the peaks back where they were.
        used = [p for p in peaks if int(p["channel"]) in self._grid]
        pit = np.array([self._grid[int(p["channel"])] for p in used])
        img = np.array([[p["x_px"], p["y_px"]] for p in used])
        err = np.linalg.norm(project(H, pit) - img, axis=1)
        # RANSAC already dropped outliers; the median over all candidate points
        # being small says the inlier set dominates.
        if np.median(err) > 8.0:
            return None
        return H

    def _solve_video(self, ctx: dict[str, Any], frames_needed: list[int]) -> dict[int, np.ndarray]:
        """Solve H on a cadence over the video; nearest solve covers the gaps."""
        if not self._load_keypoint_model():
            return {}
        solved: dict[int, np.ndarray] = {}
        wanted = set(frames_needed[:: self.refresh])
        last = max(wanted) if wanted else -1
        for frame_idx, image in ctx["video"]:
            if frame_idx in wanted:
                H = self._solve_frame(image)
                if H is not None:
                    solved[frame_idx] = H
            if frame_idx >= last:
                break
        self.log.info(
            "per-frame calibration: solved %d/%d sampled frames", len(solved), len(wanted)
        )
        return solved

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
        homographies: dict[int, np.ndarray] = {}

        # Bottom-centre of the box is the ground contact point; the homography
        # maps the ground plane, so feet are the only honest anchor.
        ground = np.column_stack(
            [
                (tracks["x1"].to_numpy() + tracks["x2"].to_numpy()) / 2.0,
                tracks["y2"].to_numpy(),
            ]
        )

        if self.mode == "static" and self.H is not None:
            if len(ground):
                # Static solves store pixel->metres; per-frame solves are the
                # inverse convention. Keep the stored orientation here.
                projected = self.apply(self.H, ground)
                x_m, y_m = projected[:, 0], projected[:, 1]
        elif self.mode == "static":
            self.log.warning(
                "static calibration has no solved homography; metres stay null. "
                "Solve once with scripts/calibrate_fixed_camera.py or footy calibrate."
            )
        else:
            frames = sorted(int(f) for f in tracks["frame"].unique())
            homographies = self._solve_video(ctx, frames)
            if homographies:
                solved_frames = np.array(sorted(homographies))
                inverses = {f: np.linalg.inv(homographies[f]) for f in solved_frames}
                frame_col = tracks["frame"].to_numpy()
                for f in frames:
                    # Nearest solved frame covers the refresh gaps; bounded by the
                    # cadence, so at most refresh/2 frames of pan drift.
                    nearest = int(solved_frames[np.argmin(np.abs(solved_frames - f))])
                    if abs(nearest - f) > self.refresh * 3:
                        continue  # long unsolved stretch: stay null, not stale
                    mask = frame_col == f
                    if mask.any():
                        pts = self.apply(inverses[nearest], ground[mask])
                        x_m[mask], y_m[mask] = pts[:, 0], pts[:, 1]

        df = tracks.assign(
            t_s=tracks["frame"].to_numpy() / fps,
            x_m=x_m,
            y_m=y_m,
            speed_ms=np.nan,  # filled by analytics.physical.add_kinematics
        )
        df = validate(df[list(TRACKS_M)], TRACKS_M, self.name)

        calibrated = self.H is not None or bool(homographies)
        stats: dict[str, Any] = {
            "mode": self.mode,
            "calibrated": calibrated,
            "frames_solved": len(homographies),
        }
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
        # Convention note: "homography" is pixels -> metres (the static solve's
        # orientation); "homographies_px_to_m" carries the per-frame equivalents.
        artifacts = {
            "homography": self.H,
            "homographies_px_to_m": (
                {f: np.linalg.inv(H) for f, H in homographies.items()} if homographies else {}
            ),
        }
        return StageResult(self.name, df, artifacts=artifacts, stats=stats)
