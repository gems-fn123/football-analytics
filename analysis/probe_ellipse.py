"""Report what the circle detector finds in each clip, and draw it."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conics  # noqa: E402
from detect_lines import analyse  # noqa: E402

ROOT = HERE.parent
CLIPS = {
    "clip0": ("data/raw/smoke_clip.mp4", 0),
    "clip1": ("data/raw/smoke_clip (1).mp4", 0),
    "clip2": ("data/raw/smoke_clip (2).mp4", 182),
}


def main() -> int:
    scratch = Path(sys.argv[1])
    for clip, (video, frame) in CLIPS.items():
        r = analyse(str(ROOT / video), frame, None,
                    str(ROOT / "data/processed" / clip / "tracks_px.parquet"))
        m, img, e = r["mask"], r["image"], r["ellipse"]
        print(f"\n=== {clip} frame {frame} — {int((m>0).sum())} line px, {len(r['lines'])} lines")
        if e is None:
            print("    no circle found")
        else:
            print(f"    centre ({e['centre'][0]:7.1f},{e['centre'][1]:7.1f})  "
                  f"axes ({e['axes'][0]:6.1f},{e['axes'][1]:6.1f})  "
                  f"angle {e['angle']:6.1f}  ratio {e['axis_ratio']:5.2f}")
            print(f"    support {e['support']:.2f}  straightness {e['straightness']:.3f}  "
                  f"n_pts {e['n_pts']}  score {e['score']:.1f}")

        vis = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        ys, xs = np.nonzero(m)
        vis[(ys * 2).clip(0, vis.shape[0] - 1), (xs * 2).clip(0, vis.shape[1] - 1)] = (255, 0, 255)
        if e is not None:
            pts = conics.sample_ellipse(e["conic"], 360)
            for p in (pts * 2).astype(int):
                cv2.circle(vis, tuple(p), 2, (0, 255, 255), -1)
            for p in (e["points"] * 2).astype(int):
                cv2.circle(vis, tuple(p), 1, (0, 255, 0), -1)
        cv2.imwrite(str(scratch / f"ellipse_{clip}.png"), vis)
    print("\nwrote overlays to", scratch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
