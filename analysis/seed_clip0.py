"""Seed clip0's reference homography from identified landmarks, then refine.

The chamfer objective alone is under-constrained on this view: only a touchline, the
halfway line and the centre circle are visible, and many camera poses explain those
almost equally well. So the fit is seeded with correspondences whose pitch coordinates
are known exactly and whose pixel positions come from the *detected* geometry rather
than from guesswork:

  halfway line   -> detected as a vertical line at x ~= 384 (L2, 495 px of support)
  far touchline  -> detected as a horizontal line at y ~= 198 (L0, 681 px of support)
  centre circle  -> detected as arc fragments; its crossings of the halfway line and
                    its left/right extremes are read from those.

Seeded this way the optimiser starts inside the right basin and only has to polish.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pitch_model as pm  # noqa: E402
from calibrate import distance_transform, model_points, overlay, project  # noqa: E402
from detect_lines import analyse  # noqa: E402

ROOT = HERE.parent

# pitch metres  <->  image pixels in the 768x432 reference frame of clip0
CORR = [
    # halfway line meets the far touchline
    ((pm.CX, pm.WIDTH), (384.0, 198.0)),
    # centre circle crossing the halfway line, far and near side
    ((pm.CX, pm.CY + pm.CENTRE_CIRCLE_R), (384.0, 237.0)),
    ((pm.CX, pm.CY - pm.CENTRE_CIRCLE_R), (384.0, 318.0)),
    # centre circle left and right extremes (view is near head-on, so these map to the
    # circle's +-x extremes to within a pixel or two)
    ((pm.CX - pm.CENTRE_CIRCLE_R, pm.CY), (205.0, 280.0)),
    ((pm.CX + pm.CENTRE_CIRCLE_R, pm.CY), (563.0, 280.0)),
]


def seed_homography() -> np.ndarray:
    src = np.array([c[0] for c in CORR], dtype=np.float64)  # pitch metres
    dst = np.array([c[1] for c in CORR], dtype=np.float64)  # image pixels
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


def score(H: np.ndarray, mask: np.ndarray) -> dict:
    from scipy.spatial import cKDTree

    h, w = mask.shape
    dt = distance_transform(mask)
    mp = model_points()
    uv = project(H, mp)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    if inside.sum() < 5:
        return {"visible": 0, "median_px": float("nan"), "coverage": 0.0}
    d = dt[uv[inside, 1].astype(int), uv[inside, 0].astype(int)]
    ys, xs = np.nonzero(mask)
    line_pts = np.column_stack([xs, ys]).astype(float)
    rev, _ = cKDTree(uv[inside]).query(line_pts, k=1)
    return {
        "visible": int(inside.sum()),
        "median_px": float(np.median(d)),
        "p90_px": float(np.percentile(d, 90)),
        "within3": float((d < 3).mean()),
        "coverage": float((rev < 5).mean()),
    }


def refine(H0: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Polish the homography directly (8 DOF) with a symmetric chamfer objective."""
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree

    h, w = mask.shape
    dt = distance_transform(mask)
    mp = model_points()
    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(0)
    if len(xs) > 600:
        sel = rng.choice(len(xs), 600, replace=False)
        ys, xs = ys[sel], xs[sel]
    line_pts = np.column_stack([xs, ys]).astype(float)
    trunc = 20.0

    def unpack(p):
        return np.append(p, 1.0).reshape(3, 3) * H0[2, 2]

    def resid(p):
        H = unpack(p)
        uv = project(H, mp)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        fwd = np.full(len(mp), trunc)
        if inside.sum() >= 30:
            fwd[inside] = np.minimum(dt[uv[inside, 1].astype(int), uv[inside, 0].astype(int)], trunc)
            rev, _ = cKDTree(uv[inside]).query(line_pts, k=1)
            rev = np.minimum(rev, trunc)
        else:
            rev = np.full(len(line_pts), trunc)
        return np.concatenate([fwd, 1.4 * rev])

    p0 = (H0 / H0[2, 2]).ravel()[:8]
    res = least_squares(resid, p0, loss="soft_l1", f_scale=4.0, max_nfev=600,
                        diff_step=1e-4)
    return unpack(res.x), score(unpack(res.x), mask)


def main() -> int:
    scratch = Path(sys.argv[1])
    r = analyse(str(ROOT / "data/raw/smoke_clip.mp4"), 0, None,
                str(ROOT / "data/processed/clip0/tracks_px.parquet"))
    mask, img = r["mask"], r["image"]

    H0 = seed_homography()
    s0 = score(H0, mask)
    print("seeded :", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s0.items()})
    cv2.imwrite(str(scratch / "overlay_clip0_seed.png"), overlay(img, H0, colour=(0, 165, 255)))

    H1, s1 = refine(H0, mask)
    print("refined:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s1.items()})
    cv2.imwrite(str(scratch / "overlay_clip0_refined.png"), overlay(img, H1))

    # reprojection check on the correspondences themselves
    src = np.array([c[0] for c in CORR], float)
    dst = np.array([c[1] for c in CORR], float)
    err = np.linalg.norm(project(H1, src) - dst, axis=1)
    print("landmark reprojection px:", np.round(err, 2).tolist())

    out = HERE / "calib"
    out.mkdir(exist_ok=True)
    (out / "clip0_reference.json").write_text(
        json.dumps({"H_pitch_to_image": H1.tolist(), "frame": 0, "score": s1}, indent=2)
    )
    print("wrote", out / "clip0_reference.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
