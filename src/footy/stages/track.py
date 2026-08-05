"""Stage 2: link detections across frames into tracks.

Plain version: keep the same number on the same player from one frame to the next.
Pure geometry and appearance, no external data.

Known failure: identity switches when players cross or are hidden. In the box on a
corner, expect swaps.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from footy.schemas import TRACKS_PX, validate
from footy.stages.base import Stage, StageResult


class Tracker(Stage):
    name = "track"

    def setup(self) -> None:
        self.log.info("tracker=%s", self.cfg.get("name", "bytetrack"))
        # TODO: supervision.ByteTrack(...) or BoT-SORT with re-ID weights

    def run(self, ctx: dict[str, Any]) -> StageResult:
        detections: pd.DataFrame = ctx["detect"].table
        # TODO: feed detections frame by frame into the tracker, collect track_id
        df = detections.assign(track_id=-1)
        df = validate(df, TRACKS_PX, self.name)
        return StageResult(
            self.name,
            df,
            stats={"n_tracks": int(df["track_id"].nunique()), "id_switches": None},
        )
