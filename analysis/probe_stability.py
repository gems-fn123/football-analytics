"""Is the registration reliable, or was a good number luck?

Two runs of the same configuration gave 50 px and 433 px on the same frame, which means
single-run figures cannot be trusted in either direction. This repeats each configuration
over several seeds and reports the spread, so that what gets claimed is the distribution
rather than the best draw.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pitch_model as pm  # noqa: E402
from auto_calibrate import auto_fit  # noqa: E402
from calibrate import project  # noqa: E402
from detect_lines import analyse  # noqa: E402
from run_auto_calib import compare_to  # noqa: E402

ROOT = HERE.parent
SEEDS = (1, 2, 3, 4, 5, 6)


def main() -> int:
    H_ref = np.array(json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"])
    r = analyse(str(ROOT / "data/raw/smoke_clip.mp4"), 0, None,
                str(ROOT / "data/processed/clip0/tracks_px.parquet"))
    mask, cands = r["mask"], r["ellipses"]
    h, w = mask.shape

    from scipy.spatial import cKDTree
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    uv = project(H_ref, np.column_stack([pm.CX + pm.CENTRE_CIRCLE_R * np.cos(t),
                                         pm.CY + pm.CENTRE_CIRCLE_R * np.sin(t)]))
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
    tree = cKDTree(uv)
    ys, xs = np.nonzero(mask)
    mp = np.column_stack([xs, ys]).astype(float)
    dm, _ = tree.query(mp, k=1)
    oracle = {"points": mp[dm < 3.0], "centre": tuple(mp[dm < 3.0].mean(0)),
              "support": 1.0, "n_pts": int((dm < 3.0).sum())}

    from detect_lines import player_boxes_for
    boxes = player_boxes_for(ROOT / "data/processed/clip0/tracks_px.parquet", 0)
    configs = {
        "lines only": (None, None),
        "all candidates": (cands, None),
        "true rim (oracle)": (oracle, None),
        "lines + people": (None, boxes),
        "candidates + people": (cands, boxes),
        "oracle + people": (oracle, boxes),
    }
    print(f"{len(boxes)} player boxes available as independent evidence\n")
    print(f"{'configuration':>20}  {'per-seed error px / quality':<58}  {'median':>7}  "
          f"{'best-q pick':>11}")
    for label, (ell, bx) in configs.items():
        errs, quals = [], []
        for s in SEEDS:
            b = auto_fit(mask, ell, n_samples=6000, n_refine=35, seed=s, boxes=bx)
            errs.append(compare_to(np.array(b["H"]), H_ref))
            quals.append(b["quality"])
        shown = " ".join(f"{e:4.0f}/{q:.2f}" for e, q in zip(errs, quals))
        # If quality tracks error, running several seeds and keeping the highest-quality
        # result is a free fix for the bimodality; if it does not, quality is not a usable
        # self-assessment and that has to be said plainly.
        pick = errs[int(np.argmax(quals))]
        print(f"{label:>20}  {shown:<58}  {np.median(errs):7.0f}  {pick:11.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
