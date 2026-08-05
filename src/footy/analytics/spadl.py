"""Bridge to kloppy and socceraction.

Once events are in SPADL form, xT and VAEP come for free from socceraction.
"""

from __future__ import annotations

import pandas as pd

from footy.schemas import EVENTS, validate

SPADL_TYPE_MAP = {
    "pass": "pass",
    "cross": "cross",
    "shot": "shot",
    "dribble": "dribble",
    "tackle": "tackle",
    "interception": "interception",
    "clearance": "clearance",
    "foul": "foul",
    "corner": "corner_crossed",
    "throw_in": "throw_in",
    "goalkick": "goalkick",
    "keeper_save": "keeper_save",
}


def to_spadl(events: pd.DataFrame) -> pd.DataFrame:
    """Map the internal event table onto SPADL column and type names."""
    df = validate(events.copy(), EVENTS, "events")
    df["type_name"] = df["type_name"].map(SPADL_TYPE_MAP).fillna(df["type_name"])
    return df


def value_actions(spadl: pd.DataFrame) -> pd.DataFrame:
    """Run VAEP. Requires the analytics extra.

    TODO: socceraction expects game_id, action_id, and a specific column order.
    """
    raise NotImplementedError("wire up socceraction.vaep")
