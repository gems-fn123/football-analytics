"""Does the player-height term actually favour the correct camera?

The term is only worth adding if it scores the verified registration high and wrong
registrations low. Checking that directly - rather than inferring it from an end-to-end
result - keeps a term that merely correlates with something else from being credited.

The homography decomposition is checked first, since the whole test rests on it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from auto_calibrate import PLAYER_HEIGHT_M, player_score, project3d  # noqa: E402
from calibrate import (  # noqa: E402
    fit_params_to_homography,
    homography_from,
    params_from_homography,
    project,
)
from detect_lines import player_boxes_for  # noqa: E402

ROOT = HERE.parent
SHAPE = (432, 768)


def main() -> int:
    h, w = SHAPE
    cx, cy = w / 2.0, h / 2.0

    print("does the decomposition invert homography_from?")
    truth = np.array([45.0, -55.0, 22.0, 0.35, -0.20, 0.02, 1600.0])
    H = homography_from(truth, cx, cy)
    got = params_from_homography(H, cx, cy)
    if got is None:
        print("  FAIL: no camera recovered")
        return 1
    err = np.abs(np.array(got) - truth)
    names = ["Cx", "Cy", "Cz", "pan", "tilt", "roll", "f"]
    print("  " + "  ".join(f"{n}={g:.3f}(err {e:.3f})" for n, g, e in zip(names, got, err)))
    # The real check: does the recovered camera reproduce the same homography?
    back = homography_from(got, cx, cy)
    rel = float(np.linalg.norm(back / back[2, 2] - H / H[2, 2]) / np.linalg.norm(H / H[2, 2]))
    print(f"  homography round-trip relative error {rel:.2e}  "
          f"{'PASS' if rel < 1e-6 else 'FAIL'}")

    H_ref = np.array(json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"])
    p_ref = params_from_homography(H_ref, cx, cy)
    if p_ref is None:
        # Expected: the reference is an unconstrained 8-DOF homography from 5 labelled
        # points, so it is not exactly a camera. The closest reachable one is 1.1 px away
        # (probe_floor.py), which is well under anything this probe is trying to resolve.
        p_ref = fit_params_to_homography(H_ref, cx, cy, SHAPE)
        if p_ref is None:
            print("\ncould not fit any camera to the verified reference")
            return 1
        print("\nverified reference is not exactly a camera; using the closest one "
              "(1.1 px away)")
    print(f"verified pose: camera at "
          f"({p_ref[0]:.0f}, {p_ref[1]:.0f}) {p_ref[2]:.1f} m up, "
          f"focal {p_ref[6]:.0f}, tilt {np.rad2deg(p_ref[4]):.1f} deg")

    boxes = player_boxes_for(ROOT / "data/processed/clip0/tracks_px.parquet", 0)
    print(f"{len(boxes)} detections in frame 0\n")

    print("player-height agreement, verified pose vs deliberately wrong ones:")
    print(f"  {'pose':<34} {'score':>6}  {'implied heights (m)':>22}")

    def heights(p):
        Hp = homography_from(p, cx, cy)
        feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])
        g = project(np.linalg.inv(Hp), feet)
        head = project3d(p, np.column_stack([g, np.full(len(g), PLAYER_HEIGHT_M)]), cx, cy)
        px = boxes[:, 3] - head[:, 1]
        obs = boxes[:, 3] - boxes[:, 1]
        ok = np.isfinite(px) & (px > 1e-3) & (obs > 2)
        if ok.sum() < 3:
            return float("nan"), float("nan")
        imp = PLAYER_HEIGHT_M * obs[ok] / px[ok]
        return float(np.median(imp)), float(np.percentile(imp, 90))

    cases = [("verified reference", p_ref)]
    for label, mult in (("half the focal length", 0.5), ("double the focal length", 2.0)):
        q = p_ref.copy()
        q[6] *= mult
        cases.append((label, q))
    for label, dz in (("camera 20 m higher", 20.0), ("camera at half the height", -p_ref[2] / 2)):
        q = p_ref.copy()
        q[2] += dz
        cases.append((label, q))
    q = p_ref.copy()
    q[0] += 30.0
    cases.append(("camera slid 30 m along the pitch", q))

    for label, p in cases:
        med, p90 = heights(p)
        print(f"  {label:<34} {player_score(p, boxes, SHAPE):6.3f}  "
              f"median {med:5.2f}, p90 {p90:5.2f}")
    print(f"\n(implied height should be ~{PLAYER_HEIGHT_M} m for the correct pose)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
