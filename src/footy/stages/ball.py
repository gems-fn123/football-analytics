"""Stage 6: track the ball.

Plain version: follow a small, fast, frequently hidden object.

This is the least reliable stage in the whole pipeline. The ball is a handful of
pixels, motion-blurred, and often behind a leg. Expect gaps and false positives.
Strategy here is: detect what you can, reject implausible jumps, interpolate the rest.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from footy.stages.base import Stage, StageResult


class BallTracker(Stage):
    name = "ball"
    requires_gpu = True

    def setup(self) -> None:
        self.max_speed_ms = self.cfg.get("max_speed_ms", 45.0)
        self.max_gap_frames = self.cfg.get("max_gap_frames", 12)
        # TODO: TrackNet-style heatmap model, or tiled inference on the main detector

    def run(self, ctx: dict[str, Any]) -> StageResult:
        # TODO: 1) collect ball candidates  2) drop jumps above max_speed_ms
        #       3) interpolate gaps up to max_gap_frames  4) leave longer gaps null
        df = pd.DataFrame(columns=["frame", "t_s", "x_m", "y_m", "conf", "interpolated"])
        return StageResult(self.name, df, stats={"coverage_fraction": 0.0})
