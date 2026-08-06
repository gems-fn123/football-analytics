"""Can this camera angle support the metric chain?

Pre-registered test C-F1 / P-2 in reports/00_preregistration.md.

The shipped pipeline abstains from metres on a moving camera. The obvious "rescue" is
to substitute a single static homography solved on one reference frame. This script
measures what that substitution actually costs.

Method
------
A broadcast camera on a fixed mount pans, tilts and zooms about its optical centre.
For pure rotation + zoom the mapping between any two frames is a *global homography*,
valid across the whole scene regardless of depth. So:

  1. reference frame R (frame 0)
  2. for sampled frames t, estimate H_t : R -> t by SIFT matching + RANSAC
  3. sample a grid of points p on the *pitch* in R (grass-masked)
  4. a static-calibration pipeline assumes the world point seen at pixel H_t(p) in
     frame t is still at pixel p. The positional error it commits, expressed in
     reference-frame pixels, is |H_t(p) - p|
  5. convert pixels to metres with a depth-aware scale derived from player stature

Scale
-----
No hand-clicked landmarks, so scale comes from the data: a footballer is ~1.8 m tall,
and the detector already measured thousands of player box heights at known image
depths. Regressing box height on foot-y gives px-per-1.8 m as a function of depth,
hence metres-per-pixel. Reported with its own spread so the uncertainty is visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PLAYER_HEIGHT_M = 1.8
FRAME_STEP = 10          # sample every Nth frame
GRID_N = 6               # grid points per axis inside the pitch mask
RANSAC_PX = 3.0
PREREG_THRESHOLD_M = 2.0  # from cli.py's own reprojection warning bound

CLIPS = {
    "clip0": "data/raw/smoke_clip.mp4",
    "clip1": "data/raw/smoke_clip (1).mp4",
    "clip2": "data/raw/smoke_clip (2).mp4",
}

GRASS_HUE = (35, 90)
GRASS_MIN_SAT = 60


def grass_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (
        (hsv[..., 0] >= GRASS_HUE[0])
        & (hsv[..., 0] <= GRASS_HUE[1])
        & (hsv[..., 1] >= GRASS_MIN_SAT)
    )
    return (m.astype(np.uint8)) * 255


def depth_scale(tracks_px: Path) -> tuple[callable, dict]:
    """metres-per-pixel as a function of foot-y, from player box heights."""
    df = pd.read_parquet(tracks_px)
    df = df[(df["cls"] == "player") & (df["track_id"] >= 0)].copy()
    df["h_px"] = df["y2"] - df["y1"]
    df = df[(df["h_px"] > 5) & (df["h_px"] < 200)]
    y = df["y2"].to_numpy(dtype=float)
    h = df["h_px"].to_numpy(dtype=float)
    # Linear: players lower in frame (nearer) are taller in pixels.
    slope, intercept = np.polyfit(y, h, 1)

    def m_per_px(fy: np.ndarray) -> np.ndarray:
        h_at = np.clip(slope * np.asarray(fy, dtype=float) + intercept, 4.0, None)
        return PLAYER_HEIGHT_M / h_at

    diag = {
        "n_boxes": int(len(df)),
        "box_h_px_p25": float(np.percentile(h, 25)),
        "box_h_px_median": float(np.median(h)),
        "box_h_px_p75": float(np.percentile(h, 75)),
        "fit_slope_px_per_px": round(float(slope), 5),
        "fit_intercept_px": round(float(intercept), 2),
        "m_per_px_at_median_y": round(float(m_per_px(np.median(y))), 5),
    }
    return m_per_px, diag


def run_clip(name: str, video: str, processed: Path) -> dict:
    tracks_px = processed / name / "tracks_px.parquet"
    if not tracks_px.exists():
        return {"clip": name, "error": f"missing {tracks_px}"}
    m_per_px, scale_diag = depth_scale(tracks_px)

    cap = cv2.VideoCapture(video)
    ok, ref = cap.read()
    if not ok:
        return {"clip": name, "error": "cannot read reference frame"}
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    kp_r, des_r = sift.detectAndCompute(ref_gray, None)

    # Grid of pitch points in the reference frame.
    gm = grass_mask(ref)
    ys, xs = np.nonzero(gm)
    if len(xs) < 100:
        return {"clip": name, "error": "no grass found in reference frame"}
    x_lo, x_hi = np.percentile(xs, [5, 95])
    y_lo, y_hi = np.percentile(ys, [5, 95])
    gx, gy = np.meshgrid(
        np.linspace(x_lo, x_hi, GRID_N), np.linspace(y_lo, y_hi, GRID_N)
    )
    pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
    # keep only grid points that actually sit on grass
    keep = [gm[int(min(max(p[1], 0), gm.shape[0] - 1)), int(min(max(p[0], 0), gm.shape[1] - 1))] > 0 for p in pts]
    pts = pts[np.array(keep)]
    if len(pts) < 4:
        return {"clip": name, "error": "too few pitch grid points"}

    matcher = cv2.BFMatcher()
    rows = []
    for idx in range(FRAME_STEP, n_frames, FRAME_STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, cur = cap.read()
        if not ok:
            break
        kp_c, des_c = sift.detectAndCompute(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), None)
        if des_c is None or des_r is None or len(kp_c) < 10:
            continue
        raw = matcher.knnMatch(des_r, des_c, k=2)
        good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            continue
        src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if H is None:
            continue
        moved = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        disp_px = np.linalg.norm(moved - pts, axis=1)
        err_m = disp_px * m_per_px(pts[:, 1])
        rows.append(
            {
                "frame": idx,
                "t_s": idx / fps,
                "median_disp_px": float(np.median(disp_px)),
                "max_disp_px": float(disp_px.max()),
                "median_err_m": float(np.median(err_m)),
                "max_err_m": float(err_m.max()),
                "n_matches": len(good),
            }
        )
    cap.release()

    if not rows:
        return {"clip": name, "error": "no frames matched"}
    df = pd.DataFrame(rows)
    breach = df[df["median_err_m"] > PREREG_THRESHOLD_M]
    first_breach_s = float(breach["t_s"].iloc[0]) if len(breach) else None

    return {
        "clip": name,
        "n_sampled_frames": int(len(df)),
        "n_grid_points": int(len(pts)),
        "scale": scale_diag,
        "median_err_m_overall": round(float(df["median_err_m"].median()), 2),
        "max_err_m_overall": round(float(df["max_err_m"].max()), 2),
        "final_median_err_m": round(float(df["median_err_m"].iloc[-1]), 2),
        "max_disp_px": round(float(df["max_disp_px"].max()), 1),
        "first_breach_of_2m_s": None if first_breach_s is None else round(first_breach_s, 2),
        "fraction_frames_over_2m": round(float((df["median_err_m"] > PREREG_THRESHOLD_M).mean()), 3),
        "verdict_C_F1": "PASS" if first_breach_s is None else "FAIL",
        "series": df.to_dict(orient="records"),
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed"
    out = {}
    for name, video in CLIPS.items():
        res = run_clip(name, str(root / video), processed)
        out[name] = res
        if "error" in res:
            print(f"{name}: ERROR {res['error']}")
            continue
        print(
            f"{name}: median_err {res['median_err_m_overall']} m | "
            f"max_err {res['max_err_m_overall']} m | "
            f"first breach of {PREREG_THRESHOLD_M} m at "
            f"{res['first_breach_of_2m_s']} s | C-F1 {res['verdict_C_F1']}"
        )
    (Path(__file__).parent / "calibration_feasibility.json").write_text(
        json.dumps(out, indent=2)
    )
    print("\nwrote reports/calibration_feasibility.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
