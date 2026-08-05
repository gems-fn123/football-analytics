"""Stage 4: shirt number to player name.

Plain version: read the number on the back of the shirt, then look it up in the
squad list the user typed.

This is the weakest identity link. Best published tracklet accuracy is around 0.78,
so roughly one in five tracks gets the wrong number. Two mitigations are built in:
majority vote across a whole track, and a confidence floor below which the shirt is
left null rather than guessed.

No OCR backend is wired up yet, so today this stage abstains everywhere: shirt and
player come out null, with the correct nullable dtypes. That is the designed
degradation, not an error - see CONTRIBUTING.md, "nullable beats guessed". Team-level
analytics keep working; per-player metrics wait for an OCR backend.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from footy.stages.base import Stage, StageResult, upstream_table


class IdentityResolver(Stage):
    name = "identity"
    requires_gpu = True

    def setup(self) -> None:
        self.min_votes = self.cfg.get("min_votes", 5)
        self.min_conf = self.cfg.get("min_conf", 0.6)
        # TODO: load a text detector plus an OCR head, or a small VLM. Until then
        # run() abstains, which downstream code must already tolerate.

    @staticmethod
    def majority_vote(readings: list[int], min_votes: int) -> int | None:
        if len(readings) < min_votes:
            return None
        number, count = Counter(readings).most_common(1)[0]
        return number if count / len(readings) >= 0.5 else None

    def run(self, ctx: dict[str, Any]) -> StageResult:
        tracks = upstream_table(ctx, "team", "track", stage=self.name).copy()
        if "team" not in tracks.columns:  # team stage skipped: unassigned, honestly
            tracks["team"] = pd.Series(pd.NA, index=tracks.index, dtype="string")
        squad = ctx["config"].squad_map()

        # OCR not implemented: abstain on every track, with contract dtypes so the
        # nullability survives the parquet round-trip (Int16, not object).
        shirt = pd.Series(pd.NA, index=tracks.index, dtype="Int16")
        player = pd.Series(pd.NA, index=tracks.index, dtype="string")

        # The squad lookup is live code even while OCR is not: the moment a shirt
        # number exists, (team, shirt) resolves to a name.
        if squad and shirt.notna().any():
            resolved_names = [
                squad.get((team, int(num))) if pd.notna(num) and pd.notna(team) else None
                for team, num in zip(tracks["team"], shirt, strict=True)
            ]
            player = pd.Series(resolved_names, index=tracks.index, dtype="string")

        df = tracks.assign(shirt=shirt, player=player)
        person = df["cls"] != "ball"
        resolved = float(df.loc[person, "player"].notna().mean()) if person.any() else 0.0
        self.log.info("squad entries available=%d resolved=%.2f", len(squad), resolved)
        return StageResult(self.name, df, stats={"resolved_fraction": round(resolved, 3)})
