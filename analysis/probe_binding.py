"""Separate two questions that the end-to-end result confuses.

The automatic fit now agrees with the detected rim to 2.2 px and still sits 800 px from
clip0's verified registration. Either the circle constraint does not disambiguate the
pose, or it does and the detected rim is not the centre circle. These are different
problems with different fixes, so they are measured separately here:

  A. how much of the detected rim really lies on the true centre circle;
  B. whether the fit lands on the truth when handed the *true* rim pixels.

B is the test of the idea. A is the test of the detector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pitch_model as pm  # noqa: E402
from auto_calibrate import auto_fit  # noqa: E402
from calibrate import overlay, project  # noqa: E402
from detect_lines import analyse  # noqa: E402
from run_auto_calib import compare_to  # noqa: E402

ROOT = HERE.parent


def main() -> int:
    scratch = Path(sys.argv[1])
    H_ref = np.array(json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"])
    r = analyse(str(ROOT / "data/raw/smoke_clip.mp4"), 0, None,
                str(ROOT / "data/processed/clip0/tracks_px.parquet"))
    mask, img, ell = r["mask"], r["image"], r["ellipse"]
    h, w = mask.shape

    # Where the centre circle truly is, at pixel density.
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    circ_m = np.column_stack([pm.CX + pm.CENTRE_CIRCLE_R * np.cos(t),
                              pm.CY + pm.CENTRE_CIRCLE_R * np.sin(t)])
    true_uv = project(H_ref, circ_m)
    inside = (true_uv[:, 0] >= 0) & (true_uv[:, 0] < w) & (true_uv[:, 1] >= 0) & (true_uv[:, 1] < h)
    true_uv = true_uv[inside]
    true_tree = cKDTree(true_uv)

    print("A. do the candidates contain the centre circle?")
    print(f"   {'#':>2}  {'px':>5}  {'score':>6}  {'support':>7}  {'ratio':>5}  "
          f"{'% on true circle':>16}")
    for i, c in enumerate(r["ellipses"]):
        d, _ = true_tree.query(np.asarray(c["points"], float), k=1)
        print(f"   {i:2d}  {c['n_pts']:5d}  {c['score']:6.2f}  {c['support']:7.2f}  "
              f"{c['axis_ratio']:5.2f}  {100 * (d < 4).mean():15.1f}%")

    # B. Hand the fit the true rim, restricted to pixels the detector actually found, so
    # it is realistic evidence rather than a synthetic gift.
    ys, xs = np.nonzero(mask)
    mask_pts = np.column_stack([xs, ys]).astype(float)
    dm, _ = true_tree.query(mask_pts, k=1)
    real_rim = mask_pts[dm < 3.0]
    print(f"\nB. fit given the true rim ({len(real_rim)} detected px that are genuinely "
          f"on the centre circle)")
    if len(real_rim) < 40:
        print("   not enough true rim pixels detected to test")
        return 1

    oracle = {"points": real_rim, "centre": tuple(real_rim.mean(0)),
              "support": 1.0, "n_pts": len(real_rim)}
    best = auto_fit(mask, oracle, n_samples=6000, n_refine=35, seed=7)
    Hb = np.array(best["H"])
    print(f"   F1 {best['f1']:.3f}  circle residual {best['circle_px']:.1f}px  "
          f"bound to {best['circle']}")
    print(f"   -> {compare_to(Hb, H_ref):.1f} px from the verified reference")
    cv2.imwrite(str(scratch / "oracle_clip0.png"), overlay(img, Hb))

    # C. The real thing: all detected candidates, chosen by the pose objective.
    print("\nC. fit given the detected candidates, chosen by the pose objective")
    got = auto_fit(mask, r["ellipses"], n_samples=6000, n_refine=35, seed=7)
    Hg = np.array(got["H"])
    print(f"   F1 {got['f1']:.3f}  circle residual {got['circle_px']:.1f}px  "
          f"picked candidate #{got['candidate']} as the {got['circle']} circle")
    print(f"   -> {compare_to(Hg, H_ref):.1f} px from the verified reference")
    cv2.imwrite(str(scratch / "auto_clip0.png"), overlay(img, Hg))

    # And the same fit with no circle at all, as the control.
    ctrl = auto_fit(mask, None, n_samples=6000, n_refine=35, seed=7)
    print(f"\n   control, line evidence only: F1 {ctrl['f1']:.3f} -> "
          f"{compare_to(np.array(ctrl['H']), H_ref):.1f} px from the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
