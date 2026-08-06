"""Evaluate keypoint-based calibration against calibration-2023 ground truth.

Held-out by construction: the derivation script consumes the first --skip frames
of the seed-0 shuffle, so evaluation starts after them. The metric is the one the
pipeline cares about: distance from ground-truth annotated points to the model
geometry projected through the PREDICTED homography (px), plus solve rate.

Usage:
    python scripts/eval_calibration.py --zip <calibration-2023>/valid.zip \
        --weights <models>/SV_kp.pth --grid analysis/calib/keypoint_grid.json \
        --skip 400 --n-frames 150
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile

import cv2
import numpy as np
import torch

from footy.calib.hrnet import load_checkpoint
from footy.calib.keypoints import extract_peaks, homography_from_keypoints, preprocess, snap_grid
from footy.calib.line_dlt import _median_residual, residuals_px


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--grid", default="analysis/calib/keypoint_grid.json")
    ap.add_argument("--skip", type=int, default=400, help="frames consumed by derivation")
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--no-snap", action="store_true")
    args = ap.parse_args()

    from pathlib import Path

    blob = json.loads(Path(args.grid).read_text())
    grid = {int(k): tuple(v) for k, v in blob["grid"].items()}
    if not args.no_snap:
        grid = snap_grid(grid)
    model = load_checkpoint(args.weights, stem_position=blob["stem_position"]).eval()

    z = zipfile.ZipFile(args.zip)
    anns = sorted(n for n in z.namelist() if n.endswith(".json"))
    random.Random(0).shuffle(anns)
    sample = anns[args.skip : args.skip + args.n_frames]

    medians, n_unsolved = [], 0
    for name in sample:
        gt = json.loads(z.read(name))
        frame = cv2.imdecode(
            np.frombuffer(z.read(name.replace(".json", ".jpg")), np.uint8), cv2.IMREAD_COLOR
        )
        with torch.no_grad():
            hm = model(preprocess(frame))[0].numpy()
        peaks = extract_peaks(hm, frame.shape[:2], min_score=args.min_score)
        H = homography_from_keypoints(peaks, grid, min_score=args.min_score)
        if H is None:
            n_unsolved += 1
            continue
        elements = {
            n: np.array([[p["x"] * 960, p["y"] * 540] for p in pts])
            for n, pts in gt.items()
            if len(pts) >= 2
        }
        medians.append(_median_residual(residuals_px(H, elements)))

    med = np.array(medians)
    n = len(sample)
    print(f"frames: {n}  solved: {len(med)}  unsolved: {n_unsolved}")
    if len(med):
        print(
            "GT-point-to-projected-model residual (px): "
            f"median {np.median(med):.2f}  p75 {np.percentile(med, 75):.2f}  "
            f"p90 {np.percentile(med, 90):.2f}"
        )
        for t in (3, 5, 10):
            print(f"  frames under {t} px: {int((med < t).sum())}/{len(med)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
