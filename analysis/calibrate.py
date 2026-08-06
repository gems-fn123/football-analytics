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


def params_from_homography(H: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    """Recover camera parameters from a ground-plane homography (Zhang's method).

    A homography by itself cannot be checked against anything that lives off the ground
    plane — player heights, for instance — because it only maps z = 0. Recovering the
    camera makes those checks possible on registrations that were produced as a bare
    matrix, such as the hand-seeded reference.

    Assumes a square-pixel pinhole with the principal point at (cx, cy). Orthonormality
    of the first two rotation columns then fixes the focal length:
        r1 . r2 = 0  =>  f^2 = -(h1x h2x + h1y h2y) / (h1z h2z)
    Returns None when that yields no positive f, which means the matrix is not the
    homography of a real camera looking at a plane.
    """
    h1, h2 = H[:, 0], H[:, 1]
    a = np.array([h1[0] - cx * h1[2], h1[1] - cy * h1[2], h1[2]])
    b = np.array([h2[0] - cx * h2[2], h2[1] - cy * h2[2], h2[2]])
    denom = a[2] * b[2]
    if abs(denom) < 1e-15:
        return None
    f2 = -(a[0] * b[0] + a[1] * b[1]) / denom
    if not np.isfinite(f2) or f2 <= 0:
        return None
    f = float(np.sqrt(f2))

    Kinv = np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1.0]])
    r1, r2 = Kinv @ h1, Kinv @ h2
    s = float(np.sqrt(np.linalg.norm(r1) * np.linalg.norm(r2)))
    if s < 1e-12:
        return None

    # The overall scale of H is defined only up to sign; pick the one that puts the
    # camera above the pitch rather than below it.
    for sign in (1.0, -1.0):
        a1, a2, t = sign * r1 / s, sign * r2 / s, sign * (Kinv @ H[:, 2]) / s
        R = np.column_stack([a1, a2, np.cross(a1, a2)])
        # Nearest true rotation: a1 and a2 are only approximately orthonormal.
        U, _, Vt = np.linalg.svd(R)
        R = U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt
        C = -R.T @ t
        if C[2] > 0:
            break
    else:
        return None

    # Invert `rotation()`'s composition R = Rr @ flip @ Rx @ Rz. Working it through, the
    # bottom row of R is (-cos t sin p, cos t cos p, sin t) and the last column is
    # (-sin r cos t, -cos r cos t, sin t), which gives all three angles directly.
    tilt = float(np.arcsin(np.clip(R[2, 2], -1.0, 1.0)))
    pan = float(np.arctan2(-R[2, 0], R[2, 1]))
    roll = float(np.arctan2(-R[0, 2], -R[1, 2]))
    return np.array([C[0], C[1], C[2], pan, tilt, roll, f], float)


def fit_params_to_homography(H: np.ndarray, cx: float, cy: float, shape=None) -> np.ndarray:
    """Closest camera-model pose to an arbitrary homography, by reprojection.

    A homography has 8 degrees of freedom; the camera model here has 7, and constrains
    square pixels with the principal point at the image centre. So a homography solved
    from point correspondences generally is *not* reachable by the camera model - clip0's
    verified reference is not, and `params_from_homography` returns None for it because
    the focal length comes out imaginary.

    That matters for interpreting accuracy numbers: pose search cannot reproduce such a
    reference exactly, so there is a floor below which the comparison measures the model
    mismatch rather than the search. This finds the best reachable approximation, which
    is what that floor should be measured against.
    """
    from scipy.optimize import least_squares

    h, w = shape if shape is not None else (int(2 * cy), int(2 * cx))
    g = np.stack(np.meshgrid(np.linspace(0, pm.LENGTH, 24),
                             np.linspace(0, pm.WIDTH, 16)), -1).reshape(-1, 2)
    tgt = project(H, g)
    seen = (tgt[:, 0] > -2 * w) & (tgt[:, 0] < 3 * w) & (tgt[:, 1] > -2 * h) & (tgt[:, 1] < 3 * h)
    g, tgt = g[seen], tgt[seen]

    def resid(p):
        d = project(homography_from(p, cx, cy), g) - tgt
        return np.clip(d, -1e4, 1e4).ravel()

    best, best_cost = None, np.inf
    rng = np.random.default_rng(0)
    for _ in range(60):
        p0 = np.array([rng.uniform(-20, pm.LENGTH + 20), rng.uniform(-140, 140),
                       rng.uniform(8, 45), rng.uniform(-np.pi, np.pi),
                       rng.uniform(-1.2, -0.05), rng.uniform(-0.1, 0.1),
                       rng.uniform(400, 2600)])
        try:
            r = least_squares(resid, p0, bounds=(BOUNDS_LO, BOUNDS_HI),
                              loss="soft_l1", f_scale=8.0, max_nfev=300)
        except ValueError:
            continue
        if r.cost < best_cost:
            best, best_cost = r.x, r.cost
    return best


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


# Physically plausible broadcast camera. Bounds exist to make absurd poses unreachable
# rather than merely unlikely. Tilt is *negative* for a camera above the pitch: solving
# the rotation chain for "look-at target lands on the principal point" gives
# tan(tilt) = -Cz / horiz, so the old `tilt >= 0.02` locked the optimiser out of every
# downward-looking camera there is.
BOUNDS_LO = np.array([-90.0, -190.0, 3.0, -np.pi, -1.45, -0.5, 250.0])
BOUNDS_HI = np.array([195.0, 190.0, 90.0, np.pi, -0.005, 0.5, 5000.0])


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
