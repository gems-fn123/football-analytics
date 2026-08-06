"""Automatic pitch registration for an arbitrary camera angle.

The problem with per-clip hand-seeded correspondences is that they do not survive a new
stadium, a new camera position, or a new broadcast director. This module estimates the
field model for a clip with no clip-specific input.

Why the naive search failed
---------------------------
Optimising camera pose from arbitrary starting points finds the wrong basin: the
objective is happy to squash the whole pitch into a bright sliver, or to fit a
"whole pitch visible" pose to a frame that shows only the centre circle. Two changes
fix it:

1. **Constrained sampling.** A pose is not drawn at random. A camera position and a
   look-at point on the pitch are drawn, and pan/tilt are *derived* so the camera is
   always aimed at the pitch. Every sample is therefore a physically sensible broadcast
   camera rather than an arbitrary 7-vector, so the search spends its budget in the
   region where the answer lives.

2. **Coverage-first scoring.** A fit is ranked by how much of the *observed* line
   evidence it explains, not by how close its own projected markings land to something.
   The second is trivially gamed by shrinking; the first is not.

The result is one number per clip that says whether the registration can be trusted,
which matters more than the registration itself: a wrong homography that reports itself
as wrong is recoverable, one that reports itself as right is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pitch_model as pm  # noqa: E402
from calibrate import distance_transform, homography_from, model_points, project  # noqa: E402

TRUNC = 22.0


def pose_looking_at(C: np.ndarray, T: np.ndarray, roll: float, f: float) -> np.ndarray:
    """Camera parameter vector for a camera at C aimed at pitch point T.

    Deriving pan/tilt from a look-at target is what keeps the sampler inside the space
    of cameras that can actually see the pitch.
    """
    d = np.array([T[0] - C[0], T[1] - C[1], -C[2]], dtype=float)
    horiz = float(np.hypot(d[0], d[1]))
    pan = float(np.arctan2(-d[0], d[1]))
    # Solving the rotation chain for "target lands on the principal point" gives
    # tan(tilt) = -Cz / horiz, so a camera above the pitch has *negative* tilt in this
    # convention. Earlier fits constrained tilt to be positive and could therefore never
    # reach a downward-looking camera at all.
    tilt = float(np.arctan2(-C[2], max(horiz, 1e-6)))
    return np.array([C[0], C[1], C[2], pan, tilt, roll, f], float)


def _score(H, mp, dt, line_pts, shape, tol: float = 4.0):
    """Score a registration by the F1 of its agreement with the line evidence.

    Recall alone (what fraction of detected line pixels the model explains) is gamed by
    shrinking the pitch: a half-scale pitch packs many more markings into the frame, so
    every detected pixel ends up near *some* projected line and recall approaches 1 while
    the fit is nonsense.

    Precision - what fraction of the model's own visible markings are actually supported
    by evidence - punishes exactly that, because a shrunken pitch draws penalty boxes and
    arcs across empty grass. Their harmonic mean is what makes the objective honest.
    """
    h, w = shape
    uv = project(H, mp)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    n_in = int(inside.sum())
    if n_in < 25:
        return 0.0, 1e9, n_in, 0.0, 0.0
    d_model = dt[uv[inside, 1].astype(int), uv[inside, 0].astype(int)]
    med = float(np.median(d_model))
    precision = float((d_model < tol).mean())
    rev, _ = cKDTree(uv[inside]).query(line_pts, k=1)
    recall = float((rev < tol).mean())
    f1 = 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)
    return f1, med, n_in, precision, recall


def sample_candidates(rng, n: int) -> list[np.ndarray]:
    """Plausible broadcast cameras aimed at plausible parts of the pitch."""
    out = []
    for _ in range(n):
        side = rng.choice([-1.0, 1.0])  # camera can be on either touchline
        C = np.array(
            [
                rng.uniform(-20.0, pm.LENGTH + 20.0),
                (pm.WIDTH / 2.0) + side * rng.uniform(18.0, 95.0),
                rng.uniform(6.0, 50.0),
            ]
        )
        T = np.array([rng.uniform(0.0, pm.LENGTH), rng.uniform(0.0, pm.WIDTH)])
        f = rng.uniform(400.0, 2600.0)
        roll = rng.uniform(-0.08, 0.08)
        out.append(pose_looking_at(C, T, roll, f))
    return out


def refine(p0, mp, dt, line_pts, shape):
    h, w = shape

    def resid(p):
        H = homography_from(p, w / 2.0, h / 2.0)
        uv = project(H, mp)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        fwd = np.full(len(mp), TRUNC)
        if inside.sum() >= 25:
            fwd[inside] = np.minimum(
                dt[uv[inside, 1].astype(int), uv[inside, 0].astype(int)], TRUNC
            )
            rev, _ = cKDTree(uv[inside]).query(line_pts, k=1)
            rev = np.minimum(rev, TRUNC)
        else:
            rev = np.full(len(line_pts), TRUNC)
        return np.concatenate([fwd, 1.6 * rev])

    lo = np.array([-90.0, -190.0, 3.0, -np.pi, -1.45, -0.5, 250.0])
    hi = np.array([195.0, 190.0, 90.0, np.pi, -0.005, 0.5, 5000.0])
    res = least_squares(
        resid,
        np.clip(p0, lo + 1e-6, hi - 1e-6),
        bounds=(lo, hi),
        loss="soft_l1",
        f_scale=5.0,
        max_nfev=400,
        x_scale=np.array([20, 20, 8, 0.3, 0.3, 0.15, 400.0]),
    )
    return res.x


def auto_fit(mask: np.ndarray, n_samples: int = 4000, n_refine: int = 30, seed: int = 0):
    """Register the pitch model to a frame with no clip-specific input."""
    h, w = mask.shape
    dt = distance_transform(mask)
    mp_full = model_points(step=0.7)
    mp_coarse = model_points(step=2.2)

    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(seed)
    if len(xs) > 700:
        sel = rng.choice(len(xs), 700, replace=False)
        ys, xs = ys[sel], xs[sel]
    line_pts = np.column_stack([xs, ys]).astype(float)

    # --- stage 1: broad sampling, cheap scoring
    cands = sample_candidates(rng, n_samples)
    scored = []
    for p in cands:
        H = homography_from(p, w / 2.0, h / 2.0)
        f1, med, n_in, _, _ = _score(H, mp_coarse, dt, line_pts, (h, w))
        if n_in >= 25:
            scored.append((f1, -med, p))
    scored.sort(key=lambda r: (r[0], r[1]), reverse=True)

    # --- stage 2: refine the most promising, keep the best by coverage
    best = None
    for _f0, _, p0 in scored[:n_refine]:
        p = refine(p0, mp_full, dt, line_pts, (h, w))
        H = homography_from(p, w / 2.0, h / 2.0)
        f1, med, n_in, prec, rec = _score(H, mp_full, dt, line_pts, (h, w))
        if best is None or f1 > best["f1"] + 1e-6:
            best = {"H": H, "params": p, "f1": f1, "precision": prec, "recall": rec,
                    "median_px": med, "visible": n_in}
    return best


def sanity(H: np.ndarray, shape) -> dict:
    """Geometric plausibility checks that do not depend on the image evidence."""
    h, w = shape
    corners = np.array([[0, 0], [pm.LENGTH, 0], [pm.LENGTH, pm.WIDTH], [0, pm.WIDTH]], float)
    uv = project(H, corners)
    # A sane registration keeps the pitch far larger than the frame (we see part of it)
    # and preserves orientation.
    area = 0.5 * abs(
        np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], -1))
    )
    # metres per pixel at the image centre, via a 1 m probe
    c = np.array([[pm.CX, pm.CY], [pm.CX + 1.0, pm.CY]])
    p = project(H, c)
    px_per_m = float(np.hypot(*(p[1] - p[0])))
    return {
        "pitch_area_px": float(area),
        "frame_area_px": float(w * h),
        "pitch_to_frame_area": float(area / (w * h)),
        "px_per_metre_at_centre": round(px_per_m, 2),
    }
