"""Ellipse fitting and measurement, for the one landmark a pitch cannot fake.

A pitch is mostly parallel lines, and parallel lines are exactly what a homography is
free to slide along: many very different camera poses explain a touchline plus a
perpendicular line equally well. The circles - the centre circle and the two penalty
arcs - are different. A circle projects to an ellipse whose size, eccentricity and
orientation together pin down where the camera is, because only one pose makes a 9.15 m
circle land as *that* ellipse. Binding a detected ellipse to a named model circle is
therefore what turns an under-determined fit into a determined one.

Fitting is algebraic (Halir & Flusser) rather than contour-based. `cv2.fitEllipse`
fits the outline of a blob: a painted line is a ridge a few pixels thick, so its contour
runs up one side of the arc and back down the other and the result describes the edge of
the paint rather than the circle the paint lies on. It also needs a closed contour, and
most camera angles show only part of a circle.
"""

from __future__ import annotations

import cv2
import numpy as np


def _normaliser(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: centroid at the origin, mean radius sqrt(2).

    Conic fitting squares the coordinates, so raw pixel values around 700 put entries of
    order 1e11 in the scatter matrix next to entries of order 1. The eigen-solve then
    returns noise. This is not optional.
    """
    m = pts.mean(0)
    r = float(np.sqrt(((pts - m) ** 2).sum(1)).mean())
    s = np.sqrt(2.0) / max(r, 1e-9)
    return np.array([[s, 0.0, -s * m[0]], [0.0, s, -s * m[1]], [0.0, 0.0, 1.0]])


def fit_conic(pts: np.ndarray) -> np.ndarray | None:
    """Best-fit ellipse through `pts` as a 3x3 symmetric conic matrix, or None.

    The ellipse-specific constraint (4ac - b^2 = 1) is what makes this usable on a short
    arc: an unconstrained conic fit to a 90-degree arc happily returns a hyperbola, which
    is a better algebraic fit and a useless answer.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 6:
        return None
    T = _normaliser(pts)
    q = (np.column_stack([pts, np.ones(len(pts))]) @ T.T)[:, :2]
    x, y = q[:, 0], q[:, 1]

    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        Tm = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ Tm
    # Premultiply by inv(C1), C1 = [[0,0,2],[0,-1,0],[2,0,0]], done by row swap.
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
    try:
        w, v = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    v = np.real(v)
    cond = 4.0 * v[0] * v[2] - v[1] ** 2  # >0 selects the ellipse solution
    if not np.isfinite(cond).any() or cond.max() <= 0:
        return None
    a1 = v[:, int(np.argmax(cond))]
    a = np.concatenate([a1, Tm @ a1])
    A, B, C, D, E, F = a
    Cn = np.array([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]])
    Cm = T.T @ Cn @ T  # undo normalisation, back to pixel coordinates
    n = float(np.linalg.norm(Cm))
    if not np.isfinite(n) or n <= 0:
        return None
    return Cm / n


def conic_geometry(C: np.ndarray) -> tuple[np.ndarray, tuple[float, float], float] | None:
    """(centre, (semi_major, semi_minor), angle_rad), or None if C is not a real ellipse."""
    A2 = C[:2, :2]
    if np.linalg.det(A2) <= 1e-14:
        return None  # hyperbola or parabola: cannot be the image of a circle
    centre = np.linalg.solve(A2, -C[:2, 2])
    f_c = float(C[2, 2] + C[:2, 2] @ centre)  # quadratic form evaluated at the centre
    w, v = np.linalg.eigh(A2)
    if np.any(w * (-f_c) <= 0):
        return None  # imaginary ellipse
    ax = np.sqrt(-f_c / w)
    order = np.argsort(-ax)
    e = v[:, order[0]]
    # An axis direction is defined only up to sign, so fold the angle into [0, pi).
    # Rotating by pi maps (u, v) -> (-u, -v), which traces the identical ellipse.
    ang = float(np.arctan2(e[1], e[0])) % np.pi
    return centre, (float(ax[order[0]]), float(ax[order[1]])), ang


def sampson(C: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """First-order approximation to the geometric distance from each point to the conic.

    Used only for the inlier test, where points are within a few pixels and the
    approximation is tight. It is deliberately *not* used as an optimisation residual:
    the estimate degrades badly far from the conic, which is exactly where an optimiser
    starts.
    """
    p = np.column_stack([pts, np.ones(len(pts))])
    num = np.einsum("ij,jk,ik->i", p, C, p)
    grad = 2.0 * (p @ C[:, :2])
    return np.abs(num) / np.maximum(np.linalg.norm(grad, axis=1), 1e-12)


def sample_ellipse(C: np.ndarray, n: int = 180) -> np.ndarray | None:
    """n points evenly spaced in eccentric anomaly around the ellipse."""
    g = conic_geometry(C)
    if g is None:
        return None
    (cx, cy), (a, b), ang = g[0], g[1], g[2]
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    u, v = a * np.cos(t), b * np.sin(t)
    return np.column_stack([cx + u * ca - v * sa, cy + u * sa + v * ca])


def straightness(pts: np.ndarray) -> float:
    """Ratio of the point cloud's minor to major spread: ~0 if collinear, ~1 if round.

    This, not angular support, is what rejects a straight line. A line segment genuinely
    *is* the rim of an arbitrarily eccentric ellipse, so the fit succeeds and its inliers
    spread right around the (degenerate) rim - measured support comes out near 1.0. The
    giveaway is in the evidence rather than the model: the supporting pixels of a real
    arc fill a two-dimensional band, those of a line do not.
    """
    if len(pts) < 3:
        return 0.0
    q = pts - pts.mean(0)
    s = np.linalg.svd(q, compute_uv=False)
    return float(s[1] / max(s[0], 1e-9))


def angular_support(C: np.ndarray, pts: np.ndarray, n_bins: int = 36) -> float:
    """Fraction of the ellipse's rim that has supporting points, in [0, 1].

    Measures how much of the fitted curve is actually observed, which is what separates
    a full centre circle from a penalty arc, and either from a shadow boundary that
    grazes one shallow stretch. It does *not* separate a circle from a straight line -
    see `straightness` for that - because perpendicular noise divided by a near-zero
    minor axis scatters angles over the whole rim.
    """
    g = conic_geometry(C)
    if g is None or len(pts) == 0:
        return 0.0
    (cx, cy), (a, b), ang = g[0], g[1], g[2]
    ca, sa = np.cos(-ang), np.sin(-ang)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    u = (dx * ca - dy * sa) / max(a, 1e-9)
    v = (dx * sa + dy * ca) / max(b, 1e-9)
    th = np.arctan2(v, u)
    idx = ((th + np.pi) / (2.0 * np.pi) * n_bins).astype(int) % n_bins
    return float(len(np.unique(idx))) / n_bins


# Acceptance criteria for "this really is the image of a pitch circle". Kept in one
# place because they are the whole defence against binding the pose to a shadow.
MIN_SEMI_MAJOR = 12.0       # px; below this the fit carries no usable geometry
MIN_SEMI_MINOR = 4.0
MAX_AXIS_RATIO = 14.0       # a very low camera foreshortens hard, but not to a needle
MIN_STRAIGHTNESS = 0.06     # matches MAX_AXIS_RATIO: a 14:1 rim has a spread ratio ~0.07
MIN_SUPPORT = 0.35          # 126 deg of rim; less does not constrain a conic usefully
MIN_POINTS = 40


def rim_precision(C: np.ndarray, mask: np.ndarray, tol: int = 3, n: int = 360) -> float:
    """Fraction of the fitted rim that has evidence under it, in [0, 1].

    The counterpart to `angular_support`, and the measure that rejects a conic draped
    across a straight line for part of its length and empty grass for the rest. Support
    asks whether evidence is spread around the rim; this asks whether the rim is on
    evidence. Either alone is satisfiable by a wrong answer; together they are not.
    """
    pts = sample_ellipse(C, n)
    if pts is None:
        return 0.0
    h, w = mask.shape
    near = cv2.dilate(mask, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8))
    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if ok.sum() < 20:
        return 0.0
    return float((near[y[ok], x[ok]] > 0).mean())


def validate(C: np.ndarray, pts: np.ndarray, shape) -> dict | None:
    """Full acceptance test. Returns measurements if `C` could be a pitch circle."""
    h, w = shape[:2]
    g = conic_geometry(C)
    if g is None or len(pts) < MIN_POINTS:
        return None
    centre, (a, b), ang = g
    if a < MIN_SEMI_MAJOR or a > 1.4 * max(h, w) or b < MIN_SEMI_MINOR:
        return None
    if a / max(b, 1e-9) > MAX_AXIS_RATIO:
        return None
    if not np.all(np.abs(centre) < 3.0 * max(h, w)):
        return None
    st = straightness(pts)
    if st < MIN_STRAIGHTNESS:
        return None
    sup = angular_support(C, pts)
    if sup < MIN_SUPPORT:
        return None
    return {
        "centre": (float(centre[0]), float(centre[1])),
        "axes": (float(2 * a), float(2 * b)),  # full axes, cv2.ellipse convention
        "angle": float(np.rad2deg(ang)),
        "support": float(sup),
        "straightness": float(st),
        "axis_ratio": float(a / max(b, 1e-9)),
        "n_pts": int(len(pts)),
    }
