"""Stage 7: events.

Two routes, and the second is usually the right one.

auto     - spot actions from video or derive them from tracking. Coarse compared to
           Opta. Good for shots, goals, corners. Weak on the long tail.
manual   - a human tags the match in LongoMatch or Kinovea, exports XML or CSV, and
           this stage just imports it. Slower per match, far more accurate.
tracking - rule-based derivation from the tracking table: possession changes become
           passes, ball speed spikes become shots or clearances.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from footy.schemas import EVENTS, validate
from footy.stages.base import Stage, StageResult


class EventBuilder(Stage):
    name = "events"

    def setup(self) -> None:
        self.route = self.cfg.get("route", "manual")
        self.log.info("event route=%s", self.route)

    def _from_tracking(self, tracks: pd.DataFrame, ball: pd.DataFrame) -> pd.DataFrame:
        # TODO: nearest-player-to-ball gives possession; a change of possessor
        #       between team-mates is a pass, between opponents a turnover.
        raise NotImplementedError

    def _from_manual(self, path: str) -> pd.DataFrame:
        # TODO: parse LongoMatch XML or Kinovea CSV, map to SPADL type names
        raise NotImplementedError

    def _from_model(self, ctx: dict[str, Any]) -> pd.DataFrame:
        # TODO: action spotting model over the video timeline
        raise NotImplementedError

    def run(self, ctx: dict[str, Any]) -> StageResult:
        df = pd.DataFrame(columns=list(EVENTS))
        df = validate(df, EVENTS, self.name)
        return StageResult(self.name, df, stats={"n_events": len(df)})
