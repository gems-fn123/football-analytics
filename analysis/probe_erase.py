"""How much of the centre circle does line-erasure destroy?

Before fitting a conic, the straight lines are blanked out so that only curved evidence
remains. But the bottom of a projected centre circle is nearly straight over a long span,
so Hough detects it as a line and the erasure deletes exactly the longest, best-supported
part of the rim. This measures that directly against clip0's verified geometry.
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
import conics  # noqa: E402
import detect_lines  # noqa: E402
import pitch_model as pm  # noqa: E402
from calibrate import project  # noqa: E402
from detect_lines import _erase_lines, analyse, fit_ellipse  # noqa: E402

ROOT = HERE.parent


def main() -> int:
    H = np.array(json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"])
    r = analyse(str(ROOT / "data/raw/smoke_clip.mp4"), 0, None,
                str(ROOT / "data/processed/clip0/tracks_px.parquet"))
    mask, lines, shape = r["mask"], r["lines"], r["image"].shape
    h, w = mask.shape

    t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    uv = project(H, np.column_stack([pm.CX + pm.CENTRE_CIRCLE_R * np.cos(t),
                                     pm.CY + pm.CENTRE_CIRCLE_R * np.sin(t)]))
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
    tree = cKDTree(uv)

    def on_circle(m):
        ys, xs = np.nonzero(m)
        if not len(xs):
            return 0
        d, _ = tree.query(np.column_stack([xs, ys]).astype(float), k=1)
        return int((d < 3.0).sum())

    base = on_circle(mask)
    print(f"line mask holds {base} px of the true centre circle\n")
    print(f"{'mode':>8} {'px':>4}  {'circle px left':>14}  {'kept':>6}   "
          f"{'-- fitted rim --':>26}")
    for mode, px in (("extend", 11), ("extend", 5), ("extent", 11), ("extent", 5), ("none", 0)):
        work = _erase_lines(mask, lines, px, mode)
        left = on_circle(work)
        e = fit_ellipse(mask, lines, shape, erase=mode, erase_px=px)
        if e is None:
            got = "no fit"
        else:
            d, _ = tree.query(np.asarray(e["points"], float), k=1)
            got = (f"{len(d):4d} px {100*(d<3).mean():3.0f}% true  score {e['score']:5.2f}")
        print(f"{mode:>8} {px:4d}  {left:14d}  {100 * left / max(base,1):5.0f}%   {got:>26}")

    print("\nwhich merged lines are genuinely straight?")
    for i, ln in enumerate(detect_lines.straight_lines(lines, mask, max_bow=1e9)):
        verdict = "straight" if ln["bow"] <= 2.0 else "CURVED (a chord of an arc)"
        print(f"  L{i}: angle {ln['angle_deg']:6.1f}  support {ln['support']:7.1f}px  "
              f"bow {ln['bow']:5.2f} px  -> {verdict}")

    # Does the score prefer the right answer? Fit a conic to the pixels that genuinely
    # are the centre circle and see how it ranks against what the search returned.
    straights = detect_lines.straight_lines(lines, mask)
    line_px = detect_lines.line_pixel_mask(straights, mask.shape)
    curved_mask = mask.copy()
    curved_mask[line_px > 0] = 0

    ys, xs = np.nonzero(mask)
    mp = np.column_stack([xs, ys]).astype(float)
    d, _ = tree.query(mp, k=1)
    true_pts = mp[d < 3.0]
    Ct = conics.fit_conic(true_pts)
    keep = line_px[true_pts[:, 1].astype(int), true_pts[:, 0].astype(int)] == 0
    mt = conics.validate(Ct, true_pts[keep], mask.shape)
    print()
    if mt is None:
        print("conic fitted to the TRUE rim fails validate() -- the gate is too strict")
    else:
        rp = conics.rim_precision(Ct, curved_mask)
        s = mt["support"] * rp * float(np.sqrt(min(mt["n_pts"], 900)))
        print(f"conic fitted to the TRUE rim: {mt['n_pts']} curved px, "
              f"support {mt['support']:.2f}, rim precision {rp:.2f}, "
              f"axis ratio {mt['axis_ratio']:.2f}")
        print(f"  -> score {s:5.2f}   (the search must beat this to be wrong)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
