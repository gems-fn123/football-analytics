"""Static pitch plots. Thin wrapper so the rest of the code never imports mplsoccer."""

from __future__ import annotations

import pandas as pd


def heatmap(tracks: pd.DataFrame, player: str | None = None, ax=None):
    """Positional heatmap for one player or a whole team."""
    from mplsoccer import Pitch

    df = tracks if player is None else tracks[tracks["player"] == player]
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)
    if ax is None:
        _, ax = pitch.draw(figsize=(9, 6))
    pitch.kdeplot(df["x_m"], df["y_m"], ax=ax, fill=True, levels=50, alpha=0.6)
    return ax


def shot_map(shots: pd.DataFrame, xg_col: str = "xg", ax=None):
    from mplsoccer import Pitch

    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, half=True)
    if ax is None:
        _, ax = pitch.draw(figsize=(9, 6))
    pitch.scatter(shots["start_x"], shots["start_y"], s=shots[xg_col] * 900 + 30, ax=ax, alpha=0.75)
    return ax


def pass_network(events: pd.DataFrame, team: str, ax=None):
    """TODO: average position per player, edge width by pass count."""
    raise NotImplementedError
