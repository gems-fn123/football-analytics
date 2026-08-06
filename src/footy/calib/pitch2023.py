"""Pitch geometry keyed by the SoccerNet calibration-2023 element names.

Coordinates are our house frame: origin bottom-left, x in [0, 105] along the length,
y in [0, 68] across, metres. "top" in the element names means image-up, which on the
standard main-camera view is the far touchline, i.e. large y.

Goal furniture (posts, crossbars) lives off the ground plane and must never feed a
ground homography; it is deliberately absent from STRAIGHT_LINES.
"""

from __future__ import annotations

PITCH_L = 105.0
PITCH_W = 68.0
CIRCLE_R = 9.15
BOX18_X_L, BOX18_X_R = 16.5, PITCH_L - 16.5
BOX18_Y0, BOX18_Y1 = 13.84, PITCH_W - 13.84
BOX5_X_L, BOX5_X_R = 5.5, PITCH_L - 5.5
BOX5_Y0, BOX5_Y1 = 24.84, PITCH_W - 24.84
PEN_X_L, PEN_X_R = 11.0, PITCH_L - 11.0

# name -> ((x1, y1), (x2, y2)) segment on the ground plane, metres.
STRAIGHT_LINES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "Side line left": ((0.0, 0.0), (0.0, PITCH_W)),
    "Side line right": ((PITCH_L, 0.0), (PITCH_L, PITCH_W)),
    "Side line top": ((0.0, PITCH_W), (PITCH_L, PITCH_W)),
    "Side line bottom": ((0.0, 0.0), (PITCH_L, 0.0)),
    "Middle line": ((PITCH_L / 2, 0.0), (PITCH_L / 2, PITCH_W)),
    "Big rect. left main": ((BOX18_X_L, BOX18_Y0), (BOX18_X_L, BOX18_Y1)),
    "Big rect. left top": ((0.0, BOX18_Y1), (BOX18_X_L, BOX18_Y1)),
    "Big rect. left bottom": ((0.0, BOX18_Y0), (BOX18_X_L, BOX18_Y0)),
    "Big rect. right main": ((BOX18_X_R, BOX18_Y0), (BOX18_X_R, BOX18_Y1)),
    "Big rect. right top": ((BOX18_X_R, BOX18_Y1), (PITCH_L, BOX18_Y1)),
    "Big rect. right bottom": ((BOX18_X_R, BOX18_Y0), (PITCH_L, BOX18_Y0)),
    "Small rect. left main": ((BOX5_X_L, BOX5_Y0), (BOX5_X_L, BOX5_Y1)),
    "Small rect. left top": ((0.0, BOX5_Y1), (BOX5_X_L, BOX5_Y1)),
    "Small rect. left bottom": ((0.0, BOX5_Y0), (BOX5_X_L, BOX5_Y0)),
    "Small rect. right main": ((BOX5_X_R, BOX5_Y0), (BOX5_X_R, BOX5_Y1)),
    "Small rect. right top": ((BOX5_X_R, BOX5_Y1), (PITCH_L, BOX5_Y1)),
    "Small rect. right bottom": ((BOX5_X_R, BOX5_Y0), (PITCH_L, BOX5_Y0)),
}

# name -> (centre, radius). The side "circles" are the penalty arcs, annotated as
# partial rims of the same 9.15 m radius.
CIRCLES: dict[str, tuple[tuple[float, float], float]] = {
    "Circle central": ((PITCH_L / 2, PITCH_W / 2), CIRCLE_R),
    "Circle left": ((PEN_X_L, PITCH_W / 2), CIRCLE_R),
    "Circle right": ((PEN_X_R, PITCH_W / 2), CIRCLE_R),
}
