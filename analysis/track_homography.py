"""Track the pitch homography frame by frame, so metres stop being NaN.

Scored against reports/02_preregistration_metrics.md (frozen before this ran).

The idea, and why it is not the thing already shown to fail: reports/
calibration_feasibility.py measured what it costs to hold ONE homography fixed while
the camera pans - 10.75 m median on clip0, breaching the repo's 2 m bound after 1.2 s.
That number is the camera's own motion, measured in the image. Tracking the homography
compensates that motion instead of accumulating it.

Two things must be excluded from feature matching or the estimate is corrupted:

  players    - they move independently of the camera, so they vote for the wrong H.
  watermark  - burnt into the IMAGE, not the ground. Its features match at zero
               displacement in every frame, so RANSAC can pick them as its consensus
               set and return H ~ identity, silently reporting "the camera did not
               move". This is the more dangerous failure because it looks like success;
               criterion C-K3 exists specifically to catch it.

Keyframes rather than frame-by-frame chaining: composing homographies multiplies error,
so the number of compositions is kept small (one hop per keyframe, not per frame).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

ROOT = HERE.parent

RANSAC_PX = 3.0
MIN_INLIERS = 40          # below this a hop is refused rather than trusted
NEW_KEYFRAME_INLIERS = 80  # start a new keyframe before quality degrades this far
PLAYER_PAD_PX = 8         # dilate player boxes: box edges carry motion-contaminated features
GRID_N = 12


def player_mask(shape, boxes: np.ndarray) -> np.ndarray:
    """255 where features may be used; 0 over players."""
    m = np.full(shape[:2], 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            m,
            (int(x1) - PLAYER_PAD_PX, int(y1) - PLAYER_PAD_PX),
            (int(x2) + PLAYER_PAD_PX, int(y2) + PLAYER_PAD_PX),
            0,
            -1,
        )
    return m


def watermark_mask(video: str, n_frames: int = 40) -> np.ndarray:
    """Pixels whose content does not change while the camera pans.

    A ground feature sweeps across the sensor during a pan, so its pixel changes. A
    burnt-in graphic does not. Measured by temporal variance rather than by matching,
    so it needs no homography and cannot be fooled by the very failure it guards.
    """
    cap = cv2.VideoCapture(video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(n_total - 1, 1), n_frames).astype(int)
    stack = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            stack.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(stack) < 5:
        return None
    var = np.var(np.stack(stack), axis=0)
    # Low temporal variance under a pan = image-fixed content.
    static = (var < np.percentile(var, 8)).astype(np.uint8) * 255
    static = cv2.dilate(static, np.ones((9, 9), np.uint8))
    return static


def boxes_by_frame(tracks_px: Path) -> dict[int, np.ndarray]:
    df = pd.read_parquet(tracks_px)
    out = {}
    for f, g in df.groupby("frame"):
        out[int(f)] = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    return out


def estimate(sift, matcher, kp_a, des_a, gray_b, mask_b):
    kp_b, des_b = sift.detectAndCompute(gray_b, mask_b)
    if des_b is None or des_a is None or len(kp_b) < 10:
        return None, 0, None, None
    raw = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, 0, kp_b, des_b
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    n_in = int(inl.sum()) if inl is not None else 0
    if H is None or n_in < MIN_INLIERS:
        return None, n_in, kp_b, des_b
    return H, n_in, kp_b, des_b


def track(video: str, tracks_px: Path, verbose: bool = True) -> dict:
    """frame -> H_ref_to_frame (image homography from frame 0), plus diagnostics."""
    wm = watermark_mask(video)
    boxes = boxes_by_frame(tracks_px)

    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sift = cv2.SIFT_create(nfeatures=6000)
    matcher = cv2.BFMatcher()

    def usable_mask(idx, shape):
        m = player_mask(shape, boxes.get(idx, np.empty((0, 4))))
        if wm is not None:
            m = cv2.bitwise_and(m, cv2.bitwise_not(wm))
        return m

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("cannot read frame 0")
    g0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    m0 = usable_mask(0, frame0.shape)
    kp_k, des_k = sift.detectAndCompute(g0, m0)

    H_ref_to_key = np.eye(3)      # keyframe -> reference chain, composed
    homographies = {0: np.eye(3)}
    n_keyframes, refused = 1, 0
    key_idx = 0
    key_of = {0: 0}               # frame -> which keyframe it was solved against

    for idx in range(1, n_frames):
        ok, cur = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        mask = usable_mask(idx, cur.shape)
        H_key_cur, n_in, kp_c, des_c = estimate(sift, matcher, kp_k, des_k, gray, mask)

        if H_key_cur is None:
            refused += 1
            continue

        homographies[idx] = H_key_cur @ H_ref_to_key
        key_of[idx] = key_idx

        # Re-key before the match degrades, so every hop is a strong one.
        if n_in < NEW_KEYFRAME_INLIERS:
            H_ref_to_key = homographies[idx]
            kp_k, des_k = kp_c, des_c
            key_idx = idx
            n_keyframes += 1
    cap.release()

    if verbose:
        print(f"  frames {n_frames}, solved {len(homographies)}, refused {refused}, "
              f"keyframes {n_keyframes}")
    return {
        "H": homographies,
        "key_of": key_of,
        "n_frames": n_frames,
        "n_keyframes": n_keyframes,
        "refused": refused,
        "watermark_px": int((wm > 0).sum()) if wm is not None else 0,
    }


def pitch_grid(H_pitch_to_ref, shape, n=GRID_N):
    """Grid of pitch points that frame 0 actually sees, in metres."""
    import pitch_model as pm

    g = np.stack(np.meshgrid(np.linspace(0, pm.LENGTH, n * 2),
                             np.linspace(0, pm.WIDTH, n)), -1).reshape(-1, 2)
    p = np.column_stack([g, np.ones(len(g))]) @ H_pitch_to_ref.T
    uv = p[:, :2] / p[:, 2:3]
    h, w = shape[:2]
    seen = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return g[seen], uv[seen]


def unproject(H_pitch_to_img, uv):
    Hi = np.linalg.inv(H_pitch_to_img)
    p = np.column_stack([uv, np.ones(len(uv))]) @ Hi.T
    return p[:, :2] / p[:, 2:3]


def main() -> int:
    video = str(ROOT / "data/raw/smoke_clip.mp4")
    tracks_px = ROOT / "data/processed/clip0/tracks_px.parquet"
    H_pitch_ref = np.array(
        json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"]
    )

    print("clip0: tracking homography")
    res = track(video, tracks_px)
    Hs = res["H"]

    cap = cv2.VideoCapture(video)
    ok, frame0 = cap.read()
    shape = frame0.shape
    coverage = len(Hs) / res["n_frames"]

    # ---- C-K3: did the camera actually move, or did we track the watermark? ----
    grid_m, grid_px = pitch_grid(H_pitch_ref, shape)
    last = max(Hs)
    moved = cv2.perspectiveTransform(grid_px.reshape(-1, 1, 2).astype(np.float64),
                                     Hs[last]).reshape(-1, 2)
    cumulative_px = float(np.median(np.linalg.norm(moved - grid_px, axis=1)))

    # ---- C-K2: tracked vs an INDEPENDENT direct solve, in metres ----
    sift = cv2.SIFT_create(nfeatures=6000)
    matcher = cv2.BFMatcher()
    wm = watermark_mask(video)
    boxes = boxes_by_frame(tracks_px)

    def mask_for(idx, shp):
        m = player_mask(shp, boxes.get(idx, np.empty((0, 4))))
        if wm is not None:
            m = cv2.bitwise_and(m, cv2.bitwise_not(wm))
        return m

    g0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    kp0, des0 = sift.detectAndCompute(g0, mask_for(0, shape))

    checks, errs = [], []
    for idx in sorted(Hs):
        if idx == 0 or idx % 25:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, cur = cap.read()
        if not ok:
            continue
        H_direct, n_in, _, _ = estimate(
            sift, matcher, kp0, des0, cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY),
            mask_for(idx, shape))
        if H_direct is None or n_in < 100:
            continue  # not an independent enough check to count
        # Compare in metres: unproject the same image points through both.
        uv = cv2.perspectiveTransform(
            grid_px.reshape(-1, 1, 2).astype(np.float64), H_direct).reshape(-1, 2)
        a = unproject(Hs[idx] @ H_pitch_ref, uv)
        b = unproject(H_direct @ H_pitch_ref, uv)
        d = float(np.median(np.linalg.norm(a - b, axis=1)))
        errs.append(d)
        checks.append({"frame": idx, "inliers": n_in, "err_m": round(d, 3)})
    cap.release()

    # ---- C-K4: do players land on the pitch? ----
    import pitch_model as pm
    df = pd.read_parquet(tracks_px)
    df = df[(df["cls"] == "player") & (df["track_id"] >= 0)]
    inside, total = 0, 0
    for f, g in df.groupby("frame"):
        f = int(f)
        if f not in Hs:
            continue
        feet = np.column_stack([(g["x1"] + g["x2"]).to_numpy() / 2.0, g["y2"].to_numpy()])
        m = unproject(Hs[f] @ H_pitch_ref, feet)
        ok_m = ((m[:, 0] > -3) & (m[:, 0] < pm.LENGTH + 3)
                & (m[:, 1] > -3) & (m[:, 1] < pm.WIDTH + 3))
        inside += int(ok_m.sum())
        total += len(m)
    frac_on_pitch = inside / max(total, 1)

    median_err = float(np.median(errs)) if errs else float("nan")
    out = {
        "clip": "clip0",
        "n_frames": res["n_frames"],
        "n_solved": len(Hs),
        "coverage": round(coverage, 4),
        "n_keyframes": res["n_keyframes"],
        "watermark_px_masked": res["watermark_px"],
        "cumulative_motion_px": round(cumulative_px, 1),
        "drift_checks": checks,
        "drift_median_m": None if not errs else round(median_err, 3),
        "drift_max_m": None if not errs else round(float(np.max(errs)), 3),
        "n_drift_checks": len(errs),
        "fraction_on_pitch": round(frac_on_pitch, 4),
        "criteria": {
            "C-K1 coverage >= 95%": {"pass": bool(coverage >= 0.95), "value": round(coverage, 4)},
            "C-K2 drift < 2.0 m": {
                "pass": bool(errs and median_err < 2.0),
                "value": None if not errs else round(median_err, 3),
                "n_checks": len(errs),
            },
            "C-K3 cumulative motion >= 100 px": {
                "pass": bool(cumulative_px >= 100.0), "value": round(cumulative_px, 1)},
            "C-K4 on-pitch >= 90%": {
                "pass": bool(frac_on_pitch >= 0.90), "value": round(frac_on_pitch, 4)},
        },
    }
    (ROOT / "reports/homography_tracking.json").write_text(json.dumps(out, indent=2))

    print(f"\n  coverage            {coverage:.1%}   ({len(Hs)}/{res['n_frames']})")
    print(f"  keyframes           {res['n_keyframes']}")
    print(f"  watermark masked    {res['watermark_px']} px")
    print(f"  cumulative motion   {cumulative_px:.1f} px")
    print(f"  drift vs direct     {median_err:.3f} m median over {len(errs)} checks")
    print(f"  on pitch            {frac_on_pitch:.1%}")
    print("\ncriteria:")
    for k, v in out["criteria"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  = {v['value']}")

    np.save(ROOT / "data/processed/clip0/homographies.npy",
            np.array([np.array(Hs[k]) for k in sorted(Hs)]))
    (ROOT / "data/processed/clip0/homography_frames.json").write_text(
        json.dumps(sorted(Hs)))
    print("\nwrote reports/homography_tracking.json and data/processed/clip0/homographies.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
