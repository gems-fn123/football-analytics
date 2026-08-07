"""Extend the keypoint channel->pitch grid using clip0 alone, no SoccerNet download.

Why: probe_nbjw.py found the NBJW keypoint model fires confidently on our footage
(scores 0.5-1.4, at/above the production min_score=1.0) but every channel it fires on
(31, 34, 37, 38, 48-53) sits outside the 35/57 channels analysis/calib/keypoint_grid.json
covers - that grid was derived from only 150 sampled frames of the calibration-2023
dataset, and by chance never saw those channels earn a good-enough ground-truth fit.

scripts/derive_keypoint_grid.py's method (collect + summarise) needs, per sample frame,
a trusted pitch<->image homography to back-project detected peaks through. We do not
have that dataset locally and it is licence-gated (SoccerNet, research-use). But we
already have two things that combine to produce the same kind of ground truth for
clip0 specifically, for free:

  1. calib/clip0_reference.json - H_pitch_to_ref (frame 0), hand-verified to 1.9 px.
  2. reports/calibration_feasibility.py's own method - SIFT+RANSAC gives H_ref_to_t for
     many sampled frames t, entirely independent of pitch geometry. It already proved
     (calibration_feasibility.json) that this degrades badly at some frames (huge outlier
     max_disp_px), so a quality gate on n_matches is applied before trusting a frame.

Composing H_pitch_to_t = H_ref_to_t @ H_pitch_to_ref gives pitch<->image ground truth
across clip0's whole pan, which is exactly what's needed - reusing derive_keypoint_grid's
own clustering logic (median + MAD, min_support=10, mad<1.0 m) unmodified.

Output is a SEPARATE file, not a silent edit to the teammate's committed grid: the
provenance (self-bootstrapped from one clip, not SoccerNet ground truth) needs to stay
visible and auditable, same discipline as every source_url/retrieved_at field in the
corpus store.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from footy.calib.hrnet import load_checkpoint  # noqa: E402
from footy.calib.keypoints import extract_peaks, preprocess  # noqa: E402
from footy.calib.line_dlt import project  # noqa: E402

ROOT = HERE.parent
VIDEO = ROOT / "data/raw/smoke_clip.mp4"
WEIGHTS = ROOT / "models/weights/SV_kp.pth"
BASE_GRID = HERE / "calib/keypoint_grid.json"
OUT = HERE / "calib/keypoint_grid_clip0_bootstrap.json"

FRAME_STEP = 5
RANSAC_PX = 3.0
MIN_MATCHES = 150  # calibration_feasibility.json: frames below this had 10-45 m error
MIN_SUPPORT = 10   # same floor as scripts/derive_keypoint_grid.py
MAD_TOL_M = 1.0    # same tightness bound as scripts/derive_keypoint_grid.py


def main() -> int:
    base = json.loads(BASE_GRID.read_text())
    known_channels = {int(k) for k in base["grid"]}
    print(f"base grid: {len(known_channels)}/57 channels covered")

    H_pitch_ref = np.array(json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"])

    cap = cv2.VideoCapture(str(VIDEO))
    ok, ref = cap.read()
    if not ok:
        print("cannot read clip0 frame 0")
        return 1
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    kp_r, des_r = sift.detectAndCompute(ref_gray, None)
    matcher = cv2.BFMatcher()

    model = load_checkpoint(WEIGHTS, stem_position=base["stem_position"]).eval()

    per_channel: dict[int, list[tuple[float, float]]] = {}
    scores: dict[int, list[float]] = {}
    used, rejected = 0, 0

    def accumulate(frame_bgr: np.ndarray, H_pitch_t: np.ndarray) -> None:
        Hinv = np.linalg.inv(H_pitch_t)
        with torch.no_grad():
            hm = model(preprocess(frame_bgr))[0].numpy()
        for p in extract_peaks(hm, frame_bgr.shape[:2], min_score=0.0):
            xy = project(Hinv, np.array([[p["x_px"], p["y_px"]]]))[0]
            per_channel.setdefault(int(p["channel"]), []).append((float(xy[0]), float(xy[1])))
            scores.setdefault(int(p["channel"]), []).append(p["score"])

    # frame 0 needs no propagation: it is the verified reference itself, 1.9 px.
    accumulate(ref, H_pitch_ref)
    used += 1

    for idx in range(FRAME_STEP, n_frames, FRAME_STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, cur = cap.read()
        if not ok:
            break
        kp_c, des_c = sift.detectAndCompute(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), None)
        if des_c is None or len(kp_c) < 10:
            rejected += 1
            continue
        raw = matcher.knnMatch(des_r, des_c, k=2)
        good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_MATCHES:
            rejected += 1
            continue
        src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H_ref_t, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if H_ref_t is None or inliers.sum() < MIN_MATCHES:
            rejected += 1
            continue
        accumulate(cur, H_ref_t @ H_pitch_ref)
        used += 1
    cap.release()
    print(f"clip0: {used} frames accepted (incl. frame 0), {rejected} rejected on match quality")

    grid, report = {}, []
    new_channels = {}
    for c in sorted(per_channel):
        pts = np.array(per_channel[c])
        sc = np.array(scores[c])
        strong = pts[sc >= np.quantile(sc, 0.25)] if len(sc) > 3 else pts
        if len(strong) < MIN_SUPPORT:
            report.append({"channel": c, "status": "insufficient", "n": len(strong),
                           "already_known": c in known_channels})
            continue
        med = np.median(strong, axis=0)
        mad = float(np.median(np.linalg.norm(strong - med, axis=1)))
        entry = {"channel": c, "x": round(float(med[0]), 3), "y": round(float(med[1]), 3),
                 "mad_m": round(mad, 3), "n": len(strong),
                 "median_score": round(float(np.median(sc)), 3),
                 "already_known": c in known_channels}
        report.append(entry)
        if mad < MAD_TOL_M:
            grid[c] = (entry["x"], entry["y"])
            if c not in known_channels:
                new_channels[c] = entry

    print(f"\n{len(grid)}/{len(per_channel)} observed channels are tight enough to trust")
    print(f"of those, {len(new_channels)} are NEW (not in the base grid):")
    for c, e in sorted(new_channels.items()):
        print(f"  ch{c:>3}  pitch=({e['x']:.2f},{e['y']:.2f})  mad={e['mad_m']:.2f} m  "
              f"n={e['n']}  score={e['median_score']:.2f}")

    still_missing = [c for c in per_channel if c not in known_channels and c not in grid]
    if still_missing:
        print("\nfired but not tight enough to trust yet:")
        for c in sorted(still_missing):
            r = next(r for r in report if r["channel"] == c)
            print(f"  ch{c:>3}  {r}")

    out = {
        "stem_position": base["stem_position"],
        "grid": {str(k): v for k, v in grid.items()},
        "report": report,
        "provenance": "self-bootstrapped from clip0 alone: frame 0 uses the hand-verified "
        "reference (1.9 px), other frames via SIFT+RANSAC propagation from frame 0 "
        "(reports/calibration_feasibility.py's method), gated at >=150 inlier matches. "
        "NOT derived from SoccerNet calibration-2023 ground truth.",
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
