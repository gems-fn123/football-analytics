"""Fit the pitch model to one reference frame per clip and save an overlay to check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import multi_start, overlay  # noqa: E402
from detect_lines import analyse  # noqa: E402

ROOT = HERE.parent
OUT = HERE / "calib"
OUT.mkdir(exist_ok=True)

# Reference frames chosen for visible geometry, not for prettiness.
REFS = {
    "clip0": {"video": "data/raw/smoke_clip.mp4", "frame": 0,
              "note": "halfway line, far touchline, centre circle"},
    "clip1": {"video": "data/raw/smoke_clip (1).mp4", "frame": 0,
              "note": "halfway line and far touchline"},
    "clip2": {"video": "data/raw/smoke_clip (2).mp4", "frame": 182,
              "note": "goal line, penalty area, goal area, far touchline"},
}

# Broadcast cameras sit near the halfway line, elevated, on one side. Guesses bracket
# that: height 15-38 m, 20-60 m back from the touchline, focal 650-1200 px.
def guesses(look_x: float) -> list[np.ndarray]:
    out = []
    for Cy, Cz, tilt, f in [
        (-25, 18, 0.32, 750),
        (-35, 26, 0.40, 900),
        (-50, 34, 0.46, 1100),
        (-18, 14, 0.28, 680),
        (-42, 30, 0.50, 1000),
    ]:
        for pan in (-0.25, 0.0, 0.25):
            out.append(np.array([look_x, Cy, Cz, pan, tilt, 0.0, f], float))
    return out


def main() -> int:
    scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    results = {}
    for clip, spec in REFS.items():
        video = str(ROOT / spec["video"])
        tpx = str(ROOT / "data/processed" / clip / "tracks_px.parquet")
        r = analyse(video, spec["frame"], None, tpx)
        mask = r["mask"]
        # clip2 looks at the right-hand goal; the others at the centre circle
        look_x = 88.0 if clip == "clip2" else 52.5
        print(f"\n=== {clip} frame {spec['frame']} — {spec['note']} ===")
        print(f"    line pixels: {int((mask>0).sum())}")
        H, stats = multi_start(mask, guesses(look_x), verbose=True)
        print(f"  BEST median {stats['median_px_to_line']:.2f}px  p90 {stats['p90_px_to_line']:.2f}px  "
              f"within3px {stats['frac_within_3px']:.2f}  visible {stats['visible_model_pts']}")
        cv2.imwrite(str(scratch / f"overlay_{clip}.png"), overlay(r["image"], H))
        results[clip] = {"H": H.tolist(), "frame": spec["frame"], "video": spec["video"], **stats}
    (OUT / "reference_homographies.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "reference_homographies.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
