"""Stage 4: shirt number to player name.

Plain version: read the number on the back of the shirt, then look it up in the
squad list the user typed.

This is the weakest identity link. Best published tracklet accuracy is around 0.78,
so roughly one in five tracks gets the wrong number. Two mitigations are built in:
majority vote across a whole track, and a confidence floor below which the shirt is
left null rather than guessed.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from footy.stages.base import Stage, StageResult


class IdentityResolver(Stage):
    name = "identity"
    requires_gpu = True

    def setup(self) -> None:
        self.min_votes = self.cfg.get("min_votes", 5)
        self.min_conf = self.cfg.get("min_conf", 0.6)
        # TODO: load a text detector plus an OCR head, or a small VLM

    @staticmethod
    def majority_vote(readings: list[int], min_votes: int) -> int | None:
        if len(readings) < min_votes:
            return None
        number, count = Counter(readings).most_common(1)[0]
        return number if count / len(readings) >= 0.5 else None

    def run(self, ctx: dict[str, Any]) -> StageResult:
        tracks: pd.DataFrame = ctx["team"].table
        squad = ctx["config"].squad_map()
        # TODO: per track_id, OCR every crop, majority_vote, then map via squad
        df = tracks.assign(shirt=pd.NA, player=pd.NA)
        self.log.info("squad entries available=%d", len(squad))
        return StageResult(self.name, df, stats={"resolved_fraction": 0.0})
