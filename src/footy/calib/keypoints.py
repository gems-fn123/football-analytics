"""Keypoint heatmap inference: preprocessing, peak extraction, homography solve.

The model (see hrnet.py) predicts 58 heatmap channels; 57 are pitch keypoints and
channel 57 is background, trained per-pixel softmax-style: a keypoint exists where
its logit beats background. Which channel means which pitch location is not taken
from any reference implementation: scripts/derive_keypoint_grid.py derives the
mapping empirically by back-projecting peaks through ground-truth homographies.

Preprocessing has NO ImageNet normalisation - established empirically: plain
RGB/255 with stem_position='last' is the only combination in a 10-way sweep that
produces any keypoint activations at all (see the derivation script's provenance).
"""

from __future__ import annotations

from typing import Any

import numpy as np

BACKGROUND_CHANNEL = 57


def preprocess(frame_bgr: np.ndarray, size_hw: tuple[int, int] = (540, 960)) -> Any:
    """BGR uint8 frame -> RGB/255 float tensor (1, 3, H, W). No mean/std norm."""
    import cv2
    import torch

    resized = cv2.resize(frame_bgr, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))[None]


def extract_peaks(
    heatmaps: np.ndarray,
    frame_shape_hw: tuple[int, int],
    min_score: float = 0.0,
    background: int = BACKGROUND_CHANNEL,
) -> list[dict[str, float]]:
    """Per-channel best peak with 3x3 subpixel refinement.

    heatmaps: (C, h, w) raw logits for one frame. The score is the peak's margin
    over the background channel at the same pixel - positive means the model calls
    this pixel that keypoint rather than background.
    """
    C, h, w = heatmaps.shape
    scale_x = frame_shape_hw[1] / w
    scale_y = frame_shape_hw[0] / h
    out = []
    for c in range(C):
        if c == background:
            continue
        margin_map = heatmaps[c] - heatmaps[background]
        idx = int(np.argmax(margin_map))
        py, px = divmod(idx, w)
        score = float(margin_map[py, px])
        if score < min_score:
            continue
        hm = margin_map
        # Subpixel: weighted centroid of the 3x3 neighbourhood (shifted positive).
        y0, y1 = max(py - 1, 0), min(py + 2, h)
        x0, x1 = max(px - 1, 0), min(px + 2, w)
        win = hm[y0:y1, x0:x1].astype(np.float64)
        win = win - win.min() + 1e-9
        ys, xs = np.mgrid[y0:y1, x0:x1]
        cy = float((ys * win).sum() / win.sum())
        cx = float((xs * win).sum() / win.sum())
        out.append(
            {
                "channel": c,
                "x_px": (cx + 0.5) * scale_x,
                "y_px": (cy + 0.5) * scale_y,
                "score": score,
            }
        )
    return out


def canonical_anchors() -> list[tuple[float, float]]:
    """Every pitch location a keypoint channel could plausibly mean.

    Pairwise intersections of the marked straight lines, the three spots, the
    circle-line tangency points, and the arc-box intersections. Used to snap
    empirically derived channel anchors (centimetre noise) onto exact geometry.
    """
    from footy.calib.pitch2023 import (
        BOX18_X_L,
        BOX18_X_R,
        CIRCLE_R,
        PEN_X_L,
        PEN_X_R,
        PITCH_L,
        PITCH_W,
        STRAIGHT_LINES,
    )

    anchors: set[tuple[float, float]] = set()
    segs = list(STRAIGHT_LINES.values())
    for i, (a1, a2) in enumerate(segs):
        for b1, b2 in segs[i + 1 :]:
            l1 = np.cross([*a1, 1.0], [*a2, 1.0])
            l2 = np.cross([*b1, 1.0], [*b2, 1.0])
            x = np.cross(l1, l2)
            if abs(x[2]) < 1e-9:
                continue
            px, py = x[0] / x[2], x[1] / x[2]
            if -0.1 <= px <= PITCH_L + 0.1 and -0.1 <= py <= PITCH_W + 0.1:
                anchors.add((round(float(px), 3), round(float(py), 3)))

    mid_y = PITCH_W / 2
    anchors.update(
        {
            (PEN_X_L, mid_y),
            (PEN_X_R, mid_y),
            (PITCH_L / 2, mid_y),
            # centre circle tangencies with the halfway line, and its extremes
            (PITCH_L / 2, mid_y + CIRCLE_R),
            (PITCH_L / 2, mid_y - CIRCLE_R),
            (PITCH_L / 2 + CIRCLE_R, mid_y),
            (PITCH_L / 2 - CIRCLE_R, mid_y),
        }
    )
    # Penalty arcs meeting the front of each big box.
    dy = float(np.sqrt(CIRCLE_R**2 - (BOX18_X_L - PEN_X_L) ** 2))
    anchors.update(
        {
            (BOX18_X_L, mid_y + dy),
            (BOX18_X_L, mid_y - dy),
            (BOX18_X_R, mid_y + dy),
            (BOX18_X_R, mid_y - dy),
        }
    )
    return sorted(anchors)


def snap_grid(
    grid: dict[int, tuple[float, float]], tol_m: float = 0.5
) -> dict[int, tuple[float, float]]:
    """Snap derived anchors to the nearest canonical pitch location within tol."""
    canon = np.array(canonical_anchors())
    out: dict[int, tuple[float, float]] = {}
    for ch, (x, y) in grid.items():
        d = np.linalg.norm(canon - [x, y], axis=1)
        i = int(np.argmin(d))
        out[ch] = tuple(canon[i]) if d[i] <= tol_m else (x, y)
    return out


def homography_from_keypoints(
    peaks: list[dict[str, float]],
    grid: dict[int, tuple[float, float]],
    min_score: float,
    min_points: int = 4,
) -> np.ndarray | None:
    """RANSAC homography (pitch -> image px) from channel peaks and the grid table."""
    import cv2

    img, pit = [], []
    for p in peaks:
        anchor = grid.get(int(p["channel"]))
        if anchor is None or p["score"] < min_score:
            continue
        img.append([p["x_px"], p["y_px"]])
        pit.append(list(anchor))
    if len(img) < min_points:
        return None
    H, _ = cv2.findHomography(
        np.array(pit, dtype=np.float64),
        np.array(img, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
    )
    return H
