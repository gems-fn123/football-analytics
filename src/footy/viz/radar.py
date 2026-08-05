"""Top-down minimap overlay. The 2D radar view rendered onto or beside the video."""

from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_L, PITCH_W = 105.0, 68.0


def draw_frame(tracks_frame: pd.DataFrame, width_px: int = 800) -> np.ndarray:
    """Render one frame of the radar as a BGR image.

    TODO: draw pitch lines, then one dot per player coloured by team, then the ball.
    """
    raise NotImplementedError


def render_clip(tracks: pd.DataFrame, out_path: str, fps: int = 25) -> str:
    """Write a radar-only mp4. Expensive, use on highlights not full matches."""
    raise NotImplementedError
