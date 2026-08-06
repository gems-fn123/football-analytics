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
from calibrate import (  # noqa: E402
    distance_transform,
    homography_from,
    model_points,
    project,
    rotation,
)

TRUNC = 22.0

# The three circles on a pitch, any of which a detected ellipse might be. Trying all
# three and keeping the best is what makes the binding work on a view of a penalty area
# as well as on a view of the centre spot.
CIRCLES = {
    "centre": (pm.CX, pm.CY),
    "pen_left": (pm.PENALTY_SPOT_DIST, pm.CY),
    "pen_right": (pm.LENGTH - pm.PENALTY_SPOT_DIST, pm.CY),
}


def circle_points(name: str, n: int = 96) -> np.ndarray:
    """Model points around a named pitch circle, in metres."""
    cx, cy = CIRCLES[name]
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + pm.CENTRE_CIRCLE_R * np.cos(t), cy + pm.CENTRE_CIRCLE_R * np.sin(t)])


# Truncation for the circle term, much tighter than for the line chamfer. A detected rim
# is never perfectly pure - the best candidate on clip0 is 81% genuine - and a rim pixel
# more than a few px from the model circle is far more likely to be a misclassified pixel
# than a large residual worth minimising. Left at the line term's 22 px, that 19% of
# contamination dragged the pose ~380 px off.
TRUNC_CIRCLE = 8.0


def circle_residual(H, model_circle, ell_pts, tree, trunc: float = TRUNC_CIRCLE) -> np.ndarray:
    """Distance from each detected rim pixel to the projected model circle.

    Measured against the *pixels* rather than the fitted conic. A conic fitted to a
    partial arc extrapolates badly - the algebraic fit is biased toward low eccentricity
    and on a 110-degree arc returns a semi-major axis ~25% short - so scoring against the
    fitted curve would import that error into the pose. The pixels carry no such bias;
    the conic only ever decides *which* pixels are rim.

    One-directional on purpose. The obvious symmetric chamfer also asks every point of
    the model circle to find a rim pixel, and a real detection covers maybe half the rim,
    so that term is minimised by shrinking the model circle onto the observed arc. It is
    the same collapse that the line term already had to be protected against, and it cost
    ~400 px of accuracy here before the direction was dropped. Saying only "every rim
    pixel lies on the model circle" leaves the unobserved half unpenalised, and the
    line chamfer over the rest of the pitch still prevents any collapse.

    This term is what breaks the ambiguity: a pitch is mostly parallel lines, along which
    a homography slides freely, but only one camera makes a 9.15 m circle at a known
    place land as *this* ellipse.
    """
    uv = project(H, model_circle)
    ok = np.isfinite(uv).all(1) & (np.abs(uv) < 1e5).all(1)
    if ok.sum() < 8:
        return np.full(len(ell_pts), trunc)
    r, _ = cKDTree(uv[ok]).query(ell_pts, k=1)
    return np.minimum(r, trunc)


def aim_at_pixel(p, model_pt, target_px, cx, cy, iters: int = 4):
    """Nudge pan and tilt so `model_pt` projects onto `target_px`.

    Applied to every sampled camera, this collapses two of the seven degrees of freedom
    for free: whatever else a candidate gets wrong, it at least puts the hypothesised
    circle where the detected ellipse actually is. The remaining search - position,
    focal length, roll - is then disciplined by the circle's *size and eccentricity*,
    which is exactly the information a set of parallel lines cannot provide.
    """
    p = p.astype(float).copy()
    mp1 = np.asarray(model_pt, float)[None]
    for _ in range(iters):
        q = project(homography_from(p, cx, cy), mp1)[0]
        err = np.asarray(target_px, float) - q
        if not np.isfinite(err).all() or float(np.hypot(*err)) < 0.5:
            break
        J = np.zeros((2, 2))
        for k, i in enumerate((3, 4)):  # pan, tilt
            hi, lo = p.copy(), p.copy()
            hi[i] += 1e-4
            lo[i] -= 1e-4
            J[:, k] = (
                project(homography_from(hi, cx, cy), mp1)[0]
                - project(homography_from(lo, cx, cy), mp1)[0]
            ) / 2e-4
        if abs(np.linalg.det(J)) < 1e-9:
            break
        step = np.linalg.solve(J, err)
        if not np.isfinite(step).all():
            break
        p[3] += float(np.clip(step[0], -0.6, 0.6))
        p[4] += float(np.clip(step[1], -0.6, 0.6))
    return p


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


def refine(p0, mp, dt, line_pts, shape, circle=None, w_circle: float = 3.0):
    h, w = shape
    if circle is not None:
        model_circle, ell_pts, ell_tree = circle

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
        blocks = [fwd, 1.6 * rev]
        if circle is not None:
            # Weighted above the chamfer terms: this is a *named* correspondence, worth
            # more per residual than an anonymous nearest-line distance.
            blocks.append(w_circle * circle_residual(H, model_circle, ell_pts, ell_tree))
        return np.concatenate(blocks)

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


PLAYER_HEIGHT_M = 1.80


def project3d(params: np.ndarray, pts3d: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Project world points *above* the ground plane. The homography cannot do this."""
    Cx, Cy, Cz, pan, tilt, roll, f = params
    R = rotation(pan, tilt, roll)
    t = -R @ np.array([Cx, Cy, Cz], dtype=float)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    P = K @ np.column_stack([R, t])
    q = np.column_stack([pts3d, np.ones(len(pts3d))]) @ P.T
    z = q[:, 2:3].copy()
    z[np.abs(z[:, 0]) < 1e-9] = 1e-9
    return q[:, :2] / z


def player_score(params: np.ndarray, boxes: np.ndarray, shape) -> float:
    """Agreement between the pose and the fact that footballers are about 1.8 m tall.

    Evidence entirely independent of the pitch markings, which is why it is worth having:
    the line objective cannot separate the correct pose from a wrong one - measured over
    six seeds, its self-reported quality is *anti*-correlated with true error - because
    many cameras explain a set of parallel lines about equally well. None of them also
    explain a frame full of people the right height standing inside the touchlines.

    Each detection's feet are mapped through the ground homography to a pitch position,
    a 1.8 m post is erected there and reprojected, and its head is compared with the top
    of the detection box. A pose that shrinks the pitch makes everyone a giant; a pose
    that slides it puts people in the stands. Both are caught here and by nothing else.
    """
    if boxes is None or not len(boxes):
        return 1.0
    h, w = shape
    cx, cy = w / 2.0, h / 2.0
    H = homography_from(params, cx, cy)
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 0.0
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])
    ground = project(Hinv, feet)
    ok = np.isfinite(ground).all(1) & (np.abs(ground) < 1e4).all(1)
    if ok.sum() < 3:
        return 0.0
    ground, boxes = ground[ok], boxes[ok]

    on_pitch = float(pm.in_bounds(ground[:, 0], ground[:, 1], margin=6.0).mean())
    head = project3d(params, np.column_stack(
        [ground, np.full(len(ground), PLAYER_HEIGHT_M)]), cx, cy)
    px_h = boxes[:, 3] - head[:, 1]           # box height a 1.8 m person would have
    obs_h = boxes[:, 3] - boxes[:, 1]         # box height actually detected
    good = np.isfinite(px_h) & (obs_h > 2)
    if good.sum() < 3:
        return 0.0
    # Ratio in log space so being twice too big and half too big cost the same.
    err = np.abs(np.log(np.clip(px_h[good], 1e-3, None) / obs_h[good]))
    return float(on_pitch * np.exp(-np.median(err) ** 2 / 0.10))


def plausible(params: np.ndarray, shape=None, H=None) -> float:
    """How much like a real broadcast camera this pose is, in (0, 1].

    A prior, and labelled as one. Two facts about televised football, neither specific to
    any clip:

    * The main camera sits on a gantry roughly 8-40 m above the pitch. Nothing films
      football from 90 m up. Without this the search returns poses whose height has
      simply parked against its bound, which is not an optimum at all but the constraint
      holding it there.
    * The pitch is larger than the frame. A broadcast shot is tight enough that both
      goals are never in view at once, so a pose that fits all 105 x 68 m inside 768 px
      is describing a photograph nobody took.

    The second matters because the F1 term does not catch it. A shrunken pitch scatters
    markings everywhere, so most detected pixels land near *some* projected line, and on
    clip0 a pose with the whole pitch crammed into a fifth of the frame scored F1 0.442
    against the correct pose's 0.372.

    Both are soft and one-sided, to break ties the image evidence cannot settle without
    overriding evidence that is actually decisive.
    """
    cz = float(params[2])
    p = np.exp(-max(0.0, 8.0 - cz) ** 2 / 50.0) * np.exp(-max(0.0, cz - 40.0) ** 2 / 200.0)
    if shape is not None and H is not None:
        ratio = sanity(H, shape)["pitch_to_frame_area"]
        p *= np.exp(-max(0.0, 1.2 - ratio) ** 2 / 0.3)
    return float(p)


def quality(f1, circ, params=None, shape=None, H=None, boxes=None) -> float:
    """Single ranking number: lines, circle, camera plausibility, people.

    Each factor enters multiplicatively rather than as a veto: a pose that explains the
    lines well but puts the circle 30 px out is probably still the right basin and worth
    refining, while one that nails the circle and ignores every line is not.
    """
    q = f1 if circ is None or not np.isfinite(circ) else f1 * float(np.exp(-circ / 12.0))
    if params is None:
        return q
    q *= plausible(params, shape, H)
    if boxes is not None and shape is not None:
        q *= player_score(params, boxes, shape)
    return q


def auto_fit(
    mask: np.ndarray,
    ellipse: dict | list[dict] | None = None,
    n_samples: int = 4000,
    n_refine: int = 30,
    seed: int = 0,
    boxes: np.ndarray | None = None,
):
    """Register the pitch model to a frame with no clip-specific input.

    Takes circle *candidates*, not a circle. No local score reliably picks the centre
    circle out of a line mask - on clip0 the true rim ranks sixth of eight - but the
    right candidate is in the list, and binding to the wrong one produces a camera that
    cannot explain the rest of the pitch. So every candidate is paired with every named
    pitch circle and the full-model objective decides, which is a far stronger test than
    anything measurable on the ellipse alone.

    For each pairing, sampled cameras are first aimed so the hypothesised model circle
    lands on the detected one. That fixes two of the seven degrees of freedom for free
    and leaves the search to be disciplined by the circle's size and eccentricity -
    exactly the information a set of parallel lines cannot supply.
    """
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

    cands = [] if ellipse is None else ([ellipse] if isinstance(ellipse, dict) else list(ellipse))
    hyps: list[tuple[str | None, int, np.ndarray | None, tuple | None]] = []
    for ci, cand in enumerate(cands):
        ell_pts = np.asarray(cand["points"], float)
        tree = cKDTree(ell_pts)
        centre = np.asarray(cand["centre"], float)
        for name in CIRCLES:
            hyps.append((name, ci, centre, (circle_points(name), ell_pts, tree)))
    if not hyps:
        hyps = [(None, -1, None, None)]

    # --- stage 1: broad sampling, cheap scoring
    per = max(60, n_samples // max(len(hyps), 1))
    scored = []
    for name, ci, ell_centre, circ in hyps:
        for p in sample_candidates(rng, per):
            if circ is not None:
                p = aim_at_pixel(p, np.array(CIRCLES[name]), ell_centre, w / 2.0, h / 2.0)
            H = homography_from(p, w / 2.0, h / 2.0)
            f1, med, n_in, _, _ = _score(H, mp_coarse, dt, line_pts, (h, w))
            if n_in < 25:
                continue
            c = float(np.median(circle_residual(H, *circ))) if circ is not None else None
            scored.append((quality(f1, c, p, (h, w), H, boxes), -med, p, name, ci, circ))
    scored.sort(key=lambda r: (r[0], r[1]), reverse=True)

    # --- stage 2: refine the most promising, keep the best overall
    best = None
    for _q, _, p0, name, ci, circ in scored[:n_refine]:
        p = refine(p0, mp_full, dt, line_pts, (h, w), circle=circ)
        H = homography_from(p, w / 2.0, h / 2.0)
        f1, med, n_in, prec, rec = _score(H, mp_full, dt, line_pts, (h, w))
        c = float(np.median(circle_residual(H, *circ))) if circ is not None else None
        q = quality(f1, c, p, (h, w), H, boxes)
        if best is None or q > best["quality"] + 1e-9:
            best = {"H": H, "params": p, "f1": f1, "precision": prec, "recall": rec,
                    "median_px": med, "visible": n_in, "circle": name,
                    "candidate": ci, "circle_px": c, "quality": q,
                    "player_score": player_score(p, boxes, (h, w)) if boxes is not None else None}
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
