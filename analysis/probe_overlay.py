"""Is there an operating point where overlay removal helps more than it hurts?

The persistence test flags pixels that stay bright in image space across the clip. Real
markings also stay put when the camera barely pans, so the test has a cost as well as a
benefit and the question is empirical, not rhetorical.

clip0's verified homography gives the exact pixels of its markings, so both sides can be
measured: how much of the flagged area is genuinely *not* a marking (the benefit), and
how much marking it destroys (the cost).
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
from calibrate import project  # noqa: E402
from detect_lines import field_mask, ridge_response  # noqa: E402

ROOT = HERE.parent
VIDEO = ROOT / "data/raw/smoke_clip.mp4"


def marking_mask(H, shape, dilate=5):
    """Every pitch marking the verified homography puts inside the frame."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    for _, poly in pm.polylines():
        uv = project(H, poly)
        pts = uv.round().astype(np.int32)
        ok = np.isfinite(uv).all(1) & (np.abs(uv) < 1e5).all(1)
        for i in range(len(pts) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(m, tuple(pts[i]), tuple(pts[i + 1]), 255, 1)
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8))


def main() -> int:
    H = np.array(json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"])

    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, total - 1, 24).astype(int)
    acc, seen, n, first, last = None, None, 0, None, None
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        # The field mask must be applied per frame: without it the accumulator is
        # dominated by the crowd, which is bright, textured and never moves much.
        fld = cv2.erode((field_mask(img) > 0).astype(np.uint8), np.ones((11, 11), np.uint8))
        r = ridge_response(img)
        ins = r[fld > 0]
        med = float(np.median(ins))
        mad = float(np.median(np.abs(ins - med))) * 1.4826
        hit = ((r >= max(4.0, med + 4.0 * mad)) & (fld > 0)).astype(np.uint16)
        acc = hit.copy() if acc is None else acc + hit
        seen = fld.astype(np.uint16) if seen is None else seen + fld
        first = img if first is None else first
        last = img
        n += 1
    cap.release()

    # How much did the camera actually move? Phase correlation on the luma of the first
    # and last sampled frames gives the global image shift in pixels.
    g0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(g0, g1)
    print(f"{n} frames sampled; global shift first->last = ({dx:+.1f}, {dy:+.1f}) px\n")

    marks = marking_mask(H, first.shape)
    n_marks = int((marks > 0).sum())
    # Persistence is judged over the frames in which a pixel was actually inside the
    # field, not over all frames: a pixel the pitch only occupies half the time cannot
    # be expected to respond in the other half.
    enough = seen >= max(6, int(0.5 * n))
    print(f"markings occupy {n_marks} px in frame 0\n")
    print(f"{'persist':>8}  {'flagged':>8}  {'on marks':>9}  {'off marks':>10}  "
          f"{'% marks lost':>12}  {'precision':>9}")
    for p in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        sel = enough & (acc >= p * np.maximum(seen, 1))
        tot = int(sel.sum())
        on = int((sel & (marks > 0)).sum())
        off = tot - on
        print(f"{p:8.2f}  {tot:8d}  {on:9d}  {off:10d}  {100 * on / max(n_marks, 1):11.1f}%  "
              f"{off / tot if tot else float('nan'):9.2f}")
    print("\n'off markings' is the wanted part (watermark and clutter); 'on markings' is "
          "real evidence the removal would destroy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
