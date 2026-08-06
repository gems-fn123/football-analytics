"""Score the line mask against geometry that is *known*, not guessed.

clip0 has a verified homography, so the exact pixels of its centre circle and halfway
line are known. That turns "the circle is missing" from an impression into a
measurement, and gives the line detector an acceptance test it can be tuned against
without eyeballing overlays.

Recall is measured per marking. Total mask size stands in for the noise the detector
lets through, since there is no pixel-level ground truth for "not a marking".
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
from detect_lines import line_mask, player_boxes_for, static_overlay_mask  # noqa: E402

ROOT = HERE.parent


def known_pixels(H, name, shape, step=0.05):
    """Image pixels of a named marking, under the verified homography.

    Sampled *along* each segment, not at its vertices: the halfway line is stored as two
    endpoints, so vertex sampling reports it as a single pixel and any recall measured
    against it is meaningless.
    """
    poly = dict(pm.polylines())[name]
    chunks = []
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        n = max(2, int(float(np.hypot(*(b - a))) / step))
        chunks.append(np.linspace(a, b, n))
    uv = project(H, np.vstack(chunks)).round().astype(int)
    h, w = shape[:2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return np.unique(uv[ok], axis=0).astype(float)


def recall(mask, pts, tol=4):
    """Fraction of a known marking that the mask covers, allowing `tol` px of slack."""
    if not len(pts):
        return float("nan")
    near = cv2.dilate(mask, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8))
    return float(np.mean([near[int(y), int(x)] > 0 for x, y in pts]))


def main() -> int:
    H = np.array(json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"])
    cap = cv2.VideoCapture(str(ROOT / "data/raw/smoke_clip.mp4"))
    ok, img = cap.read()
    cap.release()
    assert ok

    boxes = player_boxes_for(ROOT / "data/processed/clip0/tracks_px.parquet", 0)
    ov = static_overlay_mask(ROOT / "data/raw/smoke_clip.mp4", cache_dir=HERE / "calib")

    for label, o in (("without overlay removal", None), ("with overlay removal", ov)):
        if o is None and label.startswith("with "):
            print("\noverlay detector found nothing (camera too static to tell)")
            continue
        m = line_mask(img, boxes, o)
        print(f"\n{label}: mask {int((m > 0).sum())} px ({100 * (m > 0).mean():.2f}% of frame)")
        for name in ("centre_circle", "halfway", "boundary"):
            pts = known_pixels(H, name, img.shape)
            if not len(pts):
                print(f"  {name:14s} not visible in this frame")
                continue
            print(f"  {name:14s} recall {recall(m, pts):5.1%}  ({len(pts)} known px in frame)")
    if ov is not None:
        print(f"\noverlay covers {int((ov > 0).sum())} px "
              f"({100 * (ov > 0).mean():.2f}% of frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
