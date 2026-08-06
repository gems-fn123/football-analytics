"""Register a frame to the pitch model by optimising camera pose.

Why not point correspondences: reading landmark pixels by eye is good to ~5 px, and at
these scales that is a quarter-metre of error before anything else goes wrong. Why not
a direct 8-DOF homography solve: with only a touchline, a halfway line and a circle
visible, the linear system is poorly conditioned and happily returns a pitch that is
geometrically absurd.

Instead, following the approach TVCalib and PnLCalib take, the *camera* is the thing
being estimated - position, pan, tilt, roll, focal length - and the objective is the
reprojection distance between the projected pitch markings and the line pixels actually
found in the image. Seven physically meaningful parameters are far better conditioned
than eight abstract ones, and a camera that cannot exist cannot be returned.

Distance is read from a distance transform of the line mask, so the objective is smooth
and a rough starting guess still converges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pitch_model as pm  # noqa: E402


# ------------------------------------------------------------------ camera model

def rotation(pan: float, tilt: float, roll: float) -> np.ndarray:
    """Camera rotation. pan about world z, tilt down from horizontal, roll about axis."""
    cp, sp = np.cos(pan), np.sin(pan)
    ct, st = np.cos(tilt), np.sin(tilt)
    cr, sr = np.cos(roll), np.sin(roll)
    # world -> camera: yaw, then pitch down, then roll, with camera looking along +z_cam
    Rz = np.array([[cp, sp, 0], [-sp, cp, 0], [0, 0, 1.0]])
    Rx = np.array([[1.0, 0, 0], [0, ct, st], [0, -st, ct]])
    Rr = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1.0]])
    # map world (x,y,z) into a camera frame whose +z points forward, +y down
    flip = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    return Rr @ flip @ Rx @ Rz


def homography_from(params: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Ground-plane (z=0) homography: pitch metres -> image pixels."""
    Cx, Cy, Cz, pan, tilt, roll, f = params
    R = rotation(pan, tilt, roll)
    t = -R @ np.array([Cx, Cy, Cz], dtype=float)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return H


def project(H: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    p = np.column_stack([pts_m, np.ones(len(pts_m))]) @ H.T
    z = p[:, 2:3]
    bad = np.abs(z[:, 0]) < 1e-9
    z[bad] = 1e-9
    out = p[:, :2] / z
    out[bad] = 1e6
    return out


# ------------------------------------------------------------------ model points

def model_points(step: float = 0.6) -> np.ndarray:
    """Dense sample of every pitch marking, in metres."""
    chunks = []
    for _, poly in pm.polylines():
        for i in range(len(poly) - 1):
            a, b = poly[i], poly[i + 1]
            d = float(np.hypot(*(b - a)))
            n = max(2, int(d / step))
            chunks.append(np.linspace(a, b, n))
    return np.vstack(chunks)


# ------------------------------------------------------------------ optimisation

def distance_transform(mask: np.ndarray) -> np.ndarray:
    inv = (mask == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


# Physically plausible broadcast camera. Bounds exist to make absurd poses
# unreachable rather than merely unlikely.
BOUNDS_LO = np.array([-80.0, -160.0, 4.0, -1.5, 0.02, -0.45, 250.0])
BOUNDS_HI = np.array([185.0, 160.0, 80.0, 1.5, 1.25, 0.45, 4500.0])


def fit(
    mask: np.ndarray,
    init: np.ndarray,
    trunc: float = 26.0,
    verbose: bool = False,
    n_line_samples: int = 500,
) -> tuple[np.ndarray, dict]:
    from scipy.spatial import cKDTree

    h, w = mask.shape
    cx, cy = w / 2.0, h / 2.0
    dt = distance_transform(mask)
    mp = model_points()

    # Sample of the detected line pixels, used for the reverse chamfer term.
    ys, xs = np.nonzero(mask)
    if len(xs) > n_line_samples:
        sel = np.random.default_rng(0).choice(len(xs), n_line_samples, replace=False)
        ys, xs = ys[sel], xs[sel]
    line_pts = np.column_stack([xs, ys]).astype(float)

    def residuals(p: np.ndarray) -> np.ndarray:
        H = homography_from(p, cx, cy)
        uv = project(H, mp)
        u, v = uv[:, 0], uv[:, 1]
        inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)

        # Forward term: each projected marking should land on a detected line pixel.
        fwd = np.full(len(mp), trunc, dtype=float)
        if inside.sum() >= 30:
            fwd[inside] = np.minimum(dt[v[inside].astype(int), u[inside].astype(int)], trunc)
        else:
            d = np.linalg.norm(uv - np.array([cx, cy]), axis=1)
            fwd = np.clip(d / 50.0, 0, 4 * trunc)

        # Reverse term: each detected line pixel should be explained by some marking.
        # Without this the objective is minimised by shrinking the whole pitch onto a
        # single line fragment, which is a perfect forward score and complete nonsense.
        if inside.sum() >= 30:
            tree = cKDTree(uv[inside])
            rev, _ = tree.query(line_pts, k=1)
            rev = np.minimum(rev, trunc)
        else:
            rev = np.full(len(line_pts), trunc, dtype=float)

        return np.concatenate([fwd, 1.4 * rev])

    res = least_squares(
        residuals,
        np.clip(init, BOUNDS_LO + 1e-6, BOUNDS_HI - 1e-6),
        bounds=(BOUNDS_LO, BOUNDS_HI),
        loss="soft_l1",
        f_scale=6.0,
        max_nfev=900,
        x_scale=np.array([20, 20, 8, 0.3, 0.3, 0.15, 400.0]),
        verbose=2 if verbose else 0,
    )
    H = homography_from(res.x, cx, cy)
    uv = project(H, mp)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    d = np.full(len(mp), np.nan)
    if inside.any():
        d[inside] = dt[uv[inside, 1].astype(int), uv[inside, 0].astype(int)]
    # Coverage: what share of detected line pixels the fitted model actually explains.
    # This is the number that exposes a collapsed fit, which scores perfectly on
    # median_px_to_line while explaining almost nothing.
    coverage = 0.0
    if inside.sum() >= 3:
        from scipy.spatial import cKDTree

        rev, _ = cKDTree(uv[inside]).query(line_pts, k=1)
        coverage = float((rev < 5.0).mean())

    stats = {
        "cost": float(res.cost),
        "visible_model_pts": int(inside.sum()),
        "median_px_to_line": float(np.nanmedian(d[inside])) if inside.any() else float("nan"),
        "p90_px_to_line": float(np.nanpercentile(d[inside], 90)) if inside.any() else float("nan"),
        "frac_within_3px": float((d[inside] < 3).mean()) if inside.any() else 0.0,
        "line_coverage": coverage,
        "params": res.x.tolist(),
    }
    return H, stats


def multi_start(mask: np.ndarray, guesses: list[np.ndarray], verbose=False) -> tuple[np.ndarray, dict]:
    best_H, best_stats = None, None
    for i, g in enumerate(guesses):
        H, s = fit(mask, g)
        if verbose:
            print(f"    start {i}: cost {s['cost']:9.1f}  median {s['median_px_to_line']:6.2f}px  "
                  f"within3 {s['frac_within_3px']:.2f}  coverage {s['line_coverage']:.2f}  "
                  f"vis {s['visible_model_pts']}")
        # Rank on coverage first: a fit that explains the observed markings beats one
        # that merely sits on top of a few of them.
        key = (round(s["line_coverage"], 3), s["frac_within_3px"], -s["cost"])
        best_key = (
            (round(best_stats["line_coverage"], 3), best_stats["frac_within_3px"], -best_stats["cost"])
            if best_stats
            else None
        )
        if best_key is None or key > best_key:
            best_H, best_stats = H, s
    return best_H, best_stats


def overlay(img: np.ndarray, H: np.ndarray, scale=2.0, colour=(0, 255, 255)) -> np.ndarray:
    """Draw the projected pitch model over the frame to eyeball the fit."""
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for _, poly in pm.polylines():
        uv = project(H, poly) * scale
        pts = uv.astype(np.int32)
        ok = np.isfinite(uv).all(axis=1) & (np.abs(uv) < 1e5).all(axis=1)
        for i in range(len(pts) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), colour, 2, cv2.LINE_AA)
    return vis
