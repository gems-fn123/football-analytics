"""Canonical football pitch model (IFAB dimensions) and map-view renderer.

Coordinate system: origin at the bottom-left corner of the pitch, x along the length
(0 -> 105 m), y across the width (0 -> 68 m). Everything downstream - positions,
distances, zones, xT - is expressed in this frame, so a metre means the same thing
regardless of where the camera was pointing.

Every landmark here is a *named* geometric primitive, which is what makes automatic
correspondence possible: the line detector finds lines in the image, and each one is
matched to one of these by identity rather than by guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# IFAB Laws of the Game, standard international dimensions.
LENGTH = 105.0
WIDTH = 68.0
CENTRE_CIRCLE_R = 9.15
PENALTY_AREA_LEN = 16.5
PENALTY_AREA_HALF_W = 20.16
GOAL_AREA_LEN = 5.5
GOAL_AREA_HALF_W = 9.16
PENALTY_SPOT_DIST = 11.0
GOAL_HALF_W = 3.66
CORNER_ARC_R = 1.0

CX, CY = LENGTH / 2.0, WIDTH / 2.0


@dataclass(frozen=True)
class Landmark:
    """A named point on the pitch whose metre coordinates are known exactly."""

    name: str
    x: float
    y: float

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


def landmarks() -> dict[str, Landmark]:
    """Every unambiguous point correspondence a calibration can key on."""
    L: list[Landmark] = [
        # corners
        Landmark("corner_bl", 0.0, 0.0),
        Landmark("corner_tl", 0.0, WIDTH),
        Landmark("corner_br", LENGTH, 0.0),
        Landmark("corner_tr", LENGTH, WIDTH),
        # halfway line endpoints and centre
        Landmark("halfway_bottom", CX, 0.0),
        Landmark("halfway_top", CX, WIDTH),
        Landmark("centre_spot", CX, CY),
        # centre circle extremes
        Landmark("circle_left", CX - CENTRE_CIRCLE_R, CY),
        Landmark("circle_right", CX + CENTRE_CIRCLE_R, CY),
        Landmark("circle_bottom", CX, CY - CENTRE_CIRCLE_R),
        Landmark("circle_top", CX, CY + CENTRE_CIRCLE_R),
        # penalty spots
        Landmark("pen_spot_left", PENALTY_SPOT_DIST, CY),
        Landmark("pen_spot_right", LENGTH - PENALTY_SPOT_DIST, CY),
    ]
    # penalty and goal area corners, both ends
    for side, gx in (("left", 0.0), ("right", LENGTH)):
        sgn = 1.0 if side == "left" else -1.0
        px = gx + sgn * PENALTY_AREA_LEN
        gx2 = gx + sgn * GOAL_AREA_LEN
        L += [
            Landmark(f"pen_{side}_bottom_goal", gx, CY - PENALTY_AREA_HALF_W),
            Landmark(f"pen_{side}_top_goal", gx, CY + PENALTY_AREA_HALF_W),
            Landmark(f"pen_{side}_bottom_out", px, CY - PENALTY_AREA_HALF_W),
            Landmark(f"pen_{side}_top_out", px, CY + PENALTY_AREA_HALF_W),
            Landmark(f"goalarea_{side}_bottom_goal", gx, CY - GOAL_AREA_HALF_W),
            Landmark(f"goalarea_{side}_top_goal", gx, CY + GOAL_AREA_HALF_W),
            Landmark(f"goalarea_{side}_bottom_out", gx2, CY - GOAL_AREA_HALF_W),
            Landmark(f"goalarea_{side}_top_out", gx2, CY + GOAL_AREA_HALF_W),
            Landmark(f"goalpost_{side}_bottom", gx, CY - GOAL_HALF_W),
            Landmark(f"goalpost_{side}_top", gx, CY + GOAL_HALF_W),
        ]
    return {lm.name: lm for lm in L}


LANDMARKS = landmarks()


def polylines() -> list[tuple[str, np.ndarray]]:
    """Pitch markings as polylines in metres, for drawing and for overlay checks."""
    out: list[tuple[str, np.ndarray]] = []
    out.append(
        (
            "boundary",
            np.array([[0, 0], [LENGTH, 0], [LENGTH, WIDTH], [0, WIDTH], [0, 0]], float),
        )
    )
    out.append(("halfway", np.array([[CX, 0], [CX, WIDTH]], float)))

    th = np.linspace(0, 2 * np.pi, 181)
    out.append(
        (
            "centre_circle",
            np.column_stack([CX + CENTRE_CIRCLE_R * np.cos(th), CY + CENTRE_CIRCLE_R * np.sin(th)]),
        )
    )

    for side, gx in (("left", 0.0), ("right", LENGTH)):
        sgn = 1.0 if side == "left" else -1.0
        px = gx + sgn * PENALTY_AREA_LEN
        ax = gx + sgn * GOAL_AREA_LEN
        out.append(
            (
                f"penalty_{side}",
                np.array(
                    [
                        [gx, CY - PENALTY_AREA_HALF_W],
                        [px, CY - PENALTY_AREA_HALF_W],
                        [px, CY + PENALTY_AREA_HALF_W],
                        [gx, CY + PENALTY_AREA_HALF_W],
                    ],
                    float,
                ),
            )
        )
        out.append(
            (
                f"goalarea_{side}",
                np.array(
                    [
                        [gx, CY - GOAL_AREA_HALF_W],
                        [ax, CY - GOAL_AREA_HALF_W],
                        [ax, CY + GOAL_AREA_HALF_W],
                        [gx, CY + GOAL_AREA_HALF_W],
                    ],
                    float,
                ),
            )
        )
        # penalty arc: the part of a 9.15 m circle about the spot that lies outside the box
        spot_x = gx + sgn * PENALTY_SPOT_DIST
        a = np.linspace(0, 2 * np.pi, 361)
        arc = np.column_stack([spot_x + CENTRE_CIRCLE_R * np.cos(a), CY + CENTRE_CIRCLE_R * np.sin(a)])
        keep = (arc[:, 0] - px) * sgn > 0
        if keep.any():
            out.append((f"pen_arc_{side}", arc[keep]))
    return out


# ---------------------------------------------------------------- zones / thirds

def thirds(x_m: np.ndarray) -> np.ndarray:
    """Defensive / middle / attacking third for a left-to-right attacking side."""
    return np.select(
        [x_m < LENGTH / 3.0, x_m < 2.0 * LENGTH / 3.0],
        ["defensive", "middle"],
        default="attacking",
    )


def zone_5x4(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Opta-style pitch grid: 5 columns along the length, 4 across the width."""
    col = np.clip((x_m / LENGTH * 5).astype(int), 0, 4)
    row = np.clip((y_m / WIDTH * 4).astype(int), 0, 3)
    return col + 5 * row


def in_bounds(x_m: np.ndarray, y_m: np.ndarray, margin: float = 3.0) -> np.ndarray:
    return (
        (x_m >= -margin) & (x_m <= LENGTH + margin) & (y_m >= -margin) & (y_m <= WIDTH + margin)
    )


# ---------------------------------------------------------------- expected threat

def xt_grid() -> np.ndarray:
    """A 12x8 expected-threat surface, rising toward the attacked goal.

    This is a *modelled* surface, not one fitted to a shot database: with 82 seconds of
    footage there is no basis for fitting one. It encodes the uncontroversial shape -
    value grows sharply near the opposition goal and decays toward the touchlines - so
    it ranks progressions sensibly. It is not a calibrated Opta xT table and is labelled
    as such wherever it is used.
    """
    nx, ny = 12, 8
    gx = (np.arange(nx) + 0.5) / nx  # 0..1 toward the attacked goal
    gy = (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(gx, gy)
    central = 1.0 - 2.2 * np.abs(Y - 0.5) ** 1.7
    surface = (X**4.2) * np.clip(central, 0.05, None)
    return surface / surface.max() * 0.30


def xt_value(x_m: np.ndarray, y_m: np.ndarray, attacking_right: bool = True) -> np.ndarray:
    grid = xt_grid()
    ny, nx = grid.shape
    xr = x_m / LENGTH if attacking_right else 1.0 - x_m / LENGTH
    ix = np.clip((xr * nx).astype(int), 0, nx - 1)
    iy = np.clip((y_m / WIDTH * ny).astype(int), 0, ny - 1)
    return grid[iy, ix]
