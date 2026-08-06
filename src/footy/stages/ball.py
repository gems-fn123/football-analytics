"""Stage 6: track the ball.

Plain version: follow a small, fast, frequently hidden object.

This is the least reliable stage in the whole pipeline. The ball is a handful of
pixels, motion-blurred, and often behind a leg. Expect gaps and false positives.
Strategy here is: detect what you can, reject implausible jumps, interpolate the rest.

Works from the main detector's ball candidates. A dedicated TrackNet-style heatmap
model would beat this; the plumbing (gating, interpolation, the BALL schema) stays
the same when one is added.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from footy.schemas import BALL, validate
from footy.stages.base import Stage, StageResult


class BallTracker(Stage):
    name = "ball"
    requires_gpu = True

    def setup(self) -> None:
        self.max_speed_ms = float(self.cfg.get("max_speed_ms", 45.0))
        # Measured in processed frames (after io.frame_stride), so the wall-clock
        # interpolation window scales with the stride the run actually used.
        self.max_gap_frames = int(self.cfg.get("max_gap_frames", 12))
        # After this long without an accepted point, accept the next candidate
        # unconditionally. Without it, one early false positive could veto every
        # real detection that follows.
        self.reacquire_s = float(self.cfg.get("reacquire_s", 1.0))

    def _speed_gate(
        self, cands: pd.DataFrame, fps: float, max_px_per_s: float
    ) -> tuple[pd.DataFrame, int]:
        """Sequentially accept candidates whose implied speed is plausible."""
        kept_idx: list[Any] = []
        last: tuple[int, float, float] | None = None  # frame, x, y
        n_rejected = 0
        for row in cands.itertuples():
            if last is None:
                kept_idx.append(row.Index)
                last = (row.frame, row.x_px, row.y_px)
                continue
            dt = (row.frame - last[0]) / fps
            if dt <= 0:
                n_rejected += 1
                continue
            if dt > self.reacquire_s:
                kept_idx.append(row.Index)
                last = (row.frame, row.x_px, row.y_px)
                continue
            dist_px = float(np.hypot(row.x_px - last[1], row.y_px - last[2]))
            if dist_px / dt <= max_px_per_s:
                kept_idx.append(row.Index)
                last = (row.frame, row.x_px, row.y_px)
            else:
                n_rejected += 1
        return cands.loc[kept_idx], n_rejected

    def run(self, ctx: dict[str, Any]) -> StageResult:
        if "detect" not in ctx:
            raise RuntimeError(
                "stage 'ball' needs the detect stage; enable it under stages: in "
                "configs/pipeline.yaml"
            )
        detections: pd.DataFrame = ctx["detect"].table
        fps = float(ctx["video"].meta.fps) or 25.0

        # The frame grid the run actually decoded. Gaps, interpolation, and coverage
        # are all measured on this grid: with frame_stride 2 the raw frame numbers
        # step by two, and pretending the skipped frames exist would double-count.
        grid: list[int] = ctx["detect"].artifacts.get("frames") or []
        if not grid and len(detections):
            # Fallback when the detect stage did not record its grid: assume stride 1
            # over the detected span. The table alone cannot reveal skipped frames.
            grid = list(range(int(detections["frame"].min()), int(detections["frame"].max()) + 1))
        grid_pos = {int(f): i for i, f in enumerate(grid)}

        # Pixel-space speed gate. Without calibration the metres bound is scaled by
        # a rough pitch-spans-the-image-width assumption, widened because camera
        # pans add apparent motion on top of real ball motion.
        pitch_len = float(ctx["config"].get("pitch", {}).get("length_m", 105.0))
        px_per_m = ctx["video"].meta.width / pitch_len
        max_px_per_s = self.max_speed_ms * px_per_m * 1.5

        balls = detections[detections["cls"] == "ball"]
        # Best candidate per frame, centre point (the ball is round; bottom-centre
        # ground anchoring does not apply).
        cands = (
            balls.sort_values("conf", ascending=False)
            .drop_duplicates("frame")
            .sort_values("frame")
            .assign(
                x_px=lambda d: (d["x1"] + d["x2"]) / 2.0,
                y_px=lambda d: (d["y1"] + d["y2"]) / 2.0,
            )[["frame", "x_px", "y_px", "conf"]]
        )

        kept, n_rejected = self._speed_gate(cands, fps, max_px_per_s)

        rows: list[dict[str, Any]] = []
        n_interpolated = 0
        prev = None
        for row in kept.itertuples():
            if prev is not None:
                gap = grid_pos[int(row.frame)] - grid_pos[int(prev.frame)]
                if 1 < gap <= self.max_gap_frames:
                    # Linear fill over the processed frames in between; conf NaN
                    # marks these as synthetic. Longer gaps stay absent - null
                    # beats invented.
                    for step in range(1, gap):
                        w = step / gap
                        rows.append(
                            {
                                "frame": int(grid[grid_pos[int(prev.frame)] + step]),
                                "x_px": prev.x_px + w * (row.x_px - prev.x_px),
                                "y_px": prev.y_px + w * (row.y_px - prev.y_px),
                                "conf": np.nan,
                                "interpolated": True,
                            }
                        )
                        n_interpolated += 1
            rows.append(
                {
                    "frame": int(row.frame),
                    "x_px": float(row.x_px),
                    "y_px": float(row.y_px),
                    "conf": float(row.conf),
                    "interpolated": False,
                }
            )
            prev = row

        df = pd.DataFrame(rows, columns=["frame", "x_px", "y_px", "conf", "interpolated"])
        df["t_s"] = df["frame"].astype("float64") / fps

        # Metres via the ground-plane homography, when calibrated. An airborne ball
        # projects long; treat these as approximate even on a calibrated camera.
        H = None
        per_frame: dict[int, np.ndarray] = {}
        if "calibrate" in ctx:
            H = ctx["calibrate"].artifacts.get("homography")
            per_frame = ctx["calibrate"].artifacts.get("homographies_px_to_m") or {}
        df["x_m"] = np.nan
        df["y_m"] = np.nan
        if len(df) and (H is not None or per_frame):
            from footy.stages.calibrate import Calibrator

            if per_frame:
                solved = np.array(sorted(per_frame))
                for i, row in enumerate(df.itertuples()):
                    nearest = int(solved[np.argmin(np.abs(solved - int(row.frame)))])
                    xy = Calibrator.apply(per_frame[nearest], np.array([[row.x_px, row.y_px]]))
                    df.iloc[i, df.columns.get_loc("x_m")] = xy[0, 0]
                    df.iloc[i, df.columns.get_loc("y_m")] = xy[0, 1]
            else:
                xy = Calibrator.apply(H, df[["x_px", "y_px"]].to_numpy(dtype=np.float64))
                df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]

        df = validate(df[list(BALL)], BALL, self.name)
        coverage = len(df) / len(grid) if grid else 0.0
        return StageResult(
            self.name,
            df,
            stats={
                "coverage_fraction": round(coverage, 3),
                "n_candidates": len(cands),
                "n_rejected_speed": n_rejected,
                "n_interpolated": n_interpolated,
                "calibrated": H is not None,
            },
        )
