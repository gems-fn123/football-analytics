"""Stage 1: detect players, goalkeepers, referees, and the ball per frame.

Plain version: draw a box around every person and the ball in each still image.
No knowledge of who they are, no memory between frames.

Backend is chosen by config so the AGPL detector stays optional. See NOTICE.md.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from footy.schemas import DETECTIONS, validate
from footy.stages.base import Stage, StageResult


class Detector(Stage):
    name = "detect"
    requires_gpu = True

    def setup(self) -> None:
        backend = self.cfg.get("name", "rfdetr")
        self.log.info("loading detector backend=%s licence=%s", backend, self.cfg.get("licence"))
        # TODO: load weights from self.cfg["weights"].
        # rfdetr  -> transformers / rfdetr package (Apache-2.0)
        # ultralytics -> from ultralytics import YOLO  (AGPL-3.0, opt-in only)
        raise NotImplementedError("wire up a detector backend")

    def run(self, ctx: dict[str, Any]) -> StageResult:
        source = ctx["video"]
        rows: list[dict[str, Any]] = []
        for _frame_idx, _image in source:
            # TODO: batch frames, run inference, append one row per detection
            pass
        df = validate(pd.DataFrame(rows, columns=list(DETECTIONS)), DETECTIONS, self.name)
        return StageResult(self.name, df, stats={"n_detections": len(df)})
