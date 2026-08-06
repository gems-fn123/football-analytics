"""Fit each circle candidate separately and see which measurable quantity tracks truth.

The pose objective picks candidate #2 while candidate #6 is the one that is actually the
centre circle. Either the good candidate does not in fact produce a good pose, or it does
and nothing currently measured says so. Fitting each candidate on its own and printing
every available statistic beside the distance from the verified reference answers which.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pitch_model as pm  # noqa: E402
from auto_calibrate import auto_fit, sanity  # noqa: E402
from calibrate import project  # noqa: E402
from detect_lines import analyse  # noqa: E402
from run_auto_calib import compare_to  # noqa: E402

ROOT = HERE.parent


def main() -> int:
    H_ref = np.array(json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"])
    r = analyse(str(ROOT / "data/raw/smoke_clip.mp4"), 0, None,
                str(ROOT / "data/processed/clip0/tracks_px.parquet"))
    mask, cands = r["mask"], r["ellipses"]
    h, w = mask.shape

    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    uv = project(H_ref, np.column_stack([pm.CX + pm.CENTRE_CIRCLE_R * np.cos(t),
                                         pm.CY + pm.CENTRE_CIRCLE_R * np.sin(t)]))
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
    tree = cKDTree(uv)

    Hinv = np.linalg.inv(H_ref)

    def arc_span(pts):
        """How much of the true circle these pixels cover, as a fraction of its rim.

        A rim that is 100% genuine but spans only a narrow arc constrains the pose barely
        better than no circle at all, so purity alone cannot explain a fit's quality.
        """
        q = project(Hinv, np.asarray(pts, float))
        d = np.hypot(q[:, 0] - pm.CX, q[:, 1] - pm.CY)
        on = np.abs(d - pm.CENTRE_CIRCLE_R) < 1.2
        if on.sum() < 10:
            return 0.0
        th = np.arctan2(q[on, 1] - pm.CY, q[on, 0] - pm.CX)
        b = np.unique(((th + np.pi) / (2 * np.pi) * 36).astype(int) % 36)
        return len(b) / 36.0

    s_ref = sanity(H_ref, mask.shape)
    print(f"verified reference: {s_ref['px_per_metre_at_centre']} px/m at the centre spot, "
          f"pitch/frame area {s_ref['pitch_to_frame_area']:.2f}\n")
    print(f"{'#':>2} {'true%':>6} {'arc':>5} {'circle':>9} {'F1':>6} {'circ':>6} {'qual':>6} "
          f"{'px/m':>6} {'height':>7} {'focal':>7} {'tilt':>6} {'camera x,y':>14}  {'ERROR px':>9}")

    ys, xs = np.nonzero(mask)
    mp = np.column_stack([xs, ys]).astype(float)
    dm, _ = tree.query(mp, k=1)
    oracle = {"points": mp[dm < 3.0], "centre": tuple(mp[dm < 3.0].mean(0)),
              "support": 1.0, "n_pts": int((dm < 3.0).sum())}

    for i, c in enumerate(list(cands) + [oracle]):
        label = f"{i:2d}" if i < len(cands) else "OR"
        d, _ = tree.query(np.asarray(c["points"], float), k=1)
        best = auto_fit(mask, c, n_samples=2400, n_refine=14, seed=7)
        H = np.array(best["H"])
        s = sanity(H, mask.shape)
        p = best["params"]
        print(f"{label} {100*(d<4).mean():5.0f}% {arc_span(c['points']):5.2f} "
              f"{best['circle']:>9} {best['f1']:6.3f} "
              f"{best['circle_px']:6.1f} {best['quality']:6.3f} "
              f"{s['px_per_metre_at_centre']:6.1f} {p[2]:7.1f} {p[6]:7.0f} "
              f"{np.rad2deg(p[4]):6.1f} {p[0]:6.0f},{p[1]:6.0f}  {compare_to(H, H_ref):9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
