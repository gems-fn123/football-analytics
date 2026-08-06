"""Homography from line correspondences, with an optional nonlinear polish.

Why lines and not points: the calibration-2023 annotations mark where each pitch
line crosses the image, so the annotated *endpoints* are frame clips, not physical
landmarks - point DLT on them would be wrong. The lines themselves are exact.

Convention: H maps pitch (metres, homogeneous) -> image (pixels). Under a point map
x' = Hx, a line transforms as l' = H^{-T} l, so an image line l and its pitch line m
satisfy m ~ H^T l, which is linear in H and is what the DLT here solves.

Torch-free on purpose: this must stay importable by light-weight tests and scripts.
"""

from __future__ import annotations

import numpy as np

from footy.calib.pitch2023 import CIRCLES, STRAIGHT_LINES


def _line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Homogeneous line through two 2D points, unit-normalised."""
    line = np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
    norm = np.hypot(line[0], line[1])
    return line / norm if norm > 1e-12 else line


def _normalise_points(pts: np.ndarray) -> np.ndarray:
    """Similarity transform sending pts to zero mean, sqrt(2) mean radius."""
    centre = pts.mean(axis=0)
    scale = np.sqrt(2.0) / max(np.mean(np.linalg.norm(pts - centre, axis=1)), 1e-9)
    return np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1.0]])


def homography_from_lines(
    image_segments: list[tuple[np.ndarray, np.ndarray]],
    pitch_segments: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """DLT over >=4 line correspondences. Returns H (pitch -> image), det-normalised.

    Segments are ((x1, y1), (x2, y2)) endpoint pairs; only the infinite line each
    pair spans is used.
    """
    if len(image_segments) < 4 or len(image_segments) != len(pitch_segments):
        raise ValueError(f"need >=4 paired line correspondences, got {len(image_segments)}")

    img_pts = np.array([p for seg in image_segments for p in seg], dtype=np.float64)
    pit_pts = np.array([p for seg in pitch_segments for p in seg], dtype=np.float64)
    T_img, T_pit = _normalise_points(img_pts), _normalise_points(pit_pts)

    rows = []
    for (ip, iq), (pp, pq) in zip(image_segments, pitch_segments, strict=True):
        # Lines in the normalised frames: for x' = Tx, l' = T^{-T} l, equivalently
        # the line through the transformed endpoints.
        li = _line_through((T_img @ [*ip, 1.0])[:2], (T_img @ [*iq, 1.0])[:2])
        mp = _line_through((T_pit @ [*pp, 1.0])[:2], (T_pit @ [*pq, 1.0])[:2])
        # m ~ H^T l  =>  m x (H^T l) = 0. Two independent rows per correspondence,
        # linear in the 9 entries of H (grouped by H's columns since H^T l = sum).
        a, b, c = li
        u, v, w = mp
        # (H^T l)_k = H[0,k] a + H[1,k] b + H[2,k] c ; cross with (u, v, w):
        rows.append([0, 0, 0, -w * a, -w * b, -w * c, v * a, v * b, v * c])  # row for x-component
        rows.append([w * a, w * b, w * c, 0, 0, 0, -u * a, -u * b, -u * c])

    A = np.array(rows)
    _, _, vt = np.linalg.svd(A)
    # Vector orders H columns first (H[:, 0], H[:, 1], H[:, 2]) per the row layout.
    Hn = vt[-1].reshape(3, 3).T
    H = np.linalg.inv(T_img) @ Hn @ T_pit
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply H to Nx2 points."""
    homo = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return homo[:, :2] / homo[:, 2:3]


def _model_curve_px(H: np.ndarray, name: str, n_arc: int = 180) -> np.ndarray | None:
    """The named pitch element projected into the image as a dense polyline."""
    if name in STRAIGHT_LINES:
        (x1, y1), (x2, y2) = STRAIGHT_LINES[name]
        t = np.linspace(0, 1, 32)[:, None]
        seg = np.array([[x1, y1]]) * (1 - t) + np.array([[x2, y2]]) * t
        return project(H, seg)
    if name in CIRCLES:
        (cx, cy), r = CIRCLES[name]
        th = np.linspace(0, 2 * np.pi, n_arc, endpoint=False)
        rim = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])
        return project(H, rim)
    return None  # goal furniture etc: off the ground plane


def residuals_px(H: np.ndarray, elements: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per element: distance in px from each annotated point to the projected model.

    Straight ground lines project to straight image lines under a homography, so
    their residual is the exact perpendicular distance; circles project to conics,
    approximated by a dense polyline.

    elements: name -> Nx2 annotated image points (pixels).
    """
    out: dict[str, np.ndarray] = {}
    for name, pts in elements.items():
        if not len(pts):
            continue
        if name in STRAIGHT_LINES:
            (a, b) = STRAIGHT_LINES[name]
            ia, ib = project(H, np.array([a, b], dtype=np.float64))
            line = _line_through(ia, ib)
            out[name] = np.abs(pts @ line[:2] + line[2])
        elif name in CIRCLES:
            curve = _model_curve_px(H, name)
            out[name] = np.linalg.norm(pts[:, None, :] - curve[None, :, :], axis=2).min(axis=1)
    return out


def polish(H0: np.ndarray, elements: dict[str, np.ndarray]) -> np.ndarray:
    """Nonlinear refinement of H over all ground evidence, circles included.

    Minimises the point-to-projected-model distances with scipy's trust region
    least squares, parameterising the 8 DOF (H[2,2] pinned to 1).
    """
    from scipy.optimize import least_squares

    def unpack(h8: np.ndarray) -> np.ndarray:
        return np.append(h8, 1.0).reshape(3, 3)

    def cost(h8: np.ndarray) -> np.ndarray:
        res = residuals_px(unpack(h8), elements)
        if not res:
            return np.zeros(1)
        return np.concatenate(list(res.values()))

    h0 = (H0 / H0[2, 2]).reshape(-1)[:8]
    fit = least_squares(cost, h0, method="trf", loss="soft_l1", f_scale=3.0, max_nfev=200)
    return unpack(fit.x)


# 180-degree pitch rotation: how the same view labels from the opposite side.
FLIP = np.array([[-1.0, 0.0, 105.0], [0.0, -1.0, 68.0], [0.0, 0.0, 1.0]])


def _median_residual(res: dict[str, np.ndarray]) -> float:
    all_d = np.concatenate(list(res.values())) if res else np.array([np.inf])
    return float(np.median(all_d))


def solve_gt_homography(
    annotation: dict[str, list[dict[str, float]]],
    width: int,
    height: int,
    min_lines: int = 4,
    refine: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Fit H (pitch -> image px) to one calibration-2023 ground-truth record.

    annotation: element name -> [{'x':, 'y':}, ...] in normalised image coords.
    Returns (H, per-element residuals in px), or None when the view is unusable:
    fewer than min_lines straight ground lines, or lines of only one pitch
    direction (a homography is underdetermined without both).

    Both pitch orientations are tried - some camera positions are annotated from
    the reverse angle - and the better-fitting one wins.
    """
    elements = {
        name: np.array([[p["x"] * width, p["y"] * height] for p in pts])
        for name, pts in annotation.items()
        if len(pts) >= 2
    }

    img_segs, pit_segs, directions = [], [], set()
    for name, pts in elements.items():
        if name not in STRAIGHT_LINES:
            continue
        # Endpoints along the annotated polyline span the same infinite line; use
        # the two most separated points for numerical stability.
        d = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        img_segs.append((pts[i], pts[j]))
        (a, b) = STRAIGHT_LINES[name]
        pit_segs.append((np.array(a), np.array(b)))
        directions.add("x" if a[0] == b[0] else "y")

    if len(img_segs) < min_lines or len(directions) < 2:
        return None

    best: tuple[np.ndarray, dict[str, np.ndarray]] | None = None
    for flip in (None, FLIP):
        segs = (
            pit_segs
            if flip is None
            else [
                (project(FLIP, np.stack(seg))[0], project(FLIP, np.stack(seg))[1])
                for seg in pit_segs
            ]
        )
        try:
            H = homography_from_lines(img_segs, segs)
        except np.linalg.LinAlgError:
            continue
        if flip is not None:
            H = H @ FLIP  # fold the relabelling into H so callers never see it
        if refine:
            H = polish(H, elements)
        res = residuals_px(H, elements)
        if best is None or _median_residual(res) < _median_residual(best[1]):
            best = (H, res)
    return best
