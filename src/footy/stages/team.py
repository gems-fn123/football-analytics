"""Stage 3: split tracks into home, away, referee.

Plain version: crop each player, look at the shirt colours, group them into two
clusters. The match config supplies which cluster is home.

No roster needed. This is why the pipeline works on a league with no public data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from footy.stages.base import Stage, StageResult


class TeamAssigner(Stage):
    name = "team"

    def setup(self) -> None:
        self.method = self.cfg.get("method", "siglip_umap_kmeans")
        self.log.info("team assignment method=%s", self.method)
        # TODO: load SigLIP image encoder, or fall back to HSV histogram k-means

    def run(self, ctx: dict[str, Any]) -> StageResult:
        tracks: pd.DataFrame = ctx["track"].table
        # TODO: embed crops -> reduce -> cluster into 2 -> map cluster to home/away
        #       using the kit colours in the match config. Referees usually fall out
        #       as a third, small cluster.
        df = tracks.assign(team=pd.NA)
        return StageResult(self.name, df)
