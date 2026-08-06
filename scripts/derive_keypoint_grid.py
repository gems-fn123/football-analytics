"""Derive the channel -> pitch-coordinate table for the NBJW keypoint checkpoint.

Clean-room by construction: instead of copying the keypoint definition from the
GPL reference code, this measures it. For frames whose ground-truth homography is
well-fit (median residual under --max-gt-res px), every detected heatmap peak is
back-projected into pitch coordinates; a channel that means "corner of the left
penalty box" will pile up there across hundreds of frames. The per-channel medians
become the frozen grid table, and the cluster tightness is the proof the forward
pass is semantically right (a wrong ReLU or concat order would smear everything).

Also settles empirically which stem concat position the checkpoint was trained
with, by running both and keeping the tighter one.

Usage:
    python scripts/derive_keypoint_grid.py --zip <calibration-2023>/valid.zip \
        --weights <models>/SV_kp.pth --n-frames 150 \
        --out analysis/calib/keypoint_grid.json
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from footy.calib.hrnet import load_checkpoint
from footy.calib.keypoints import extract_peaks, preprocess
from footy.calib.line_dlt import _median_residual, project, solve_gt_homography


def collect(zip_path: str, weights: str, n_frames: int, max_gt_res: float, stem_position: str):
    import random

    z = zipfile.ZipFile(zip_path)
    anns = sorted(n for n in z.namelist() if n.endswith(".json"))
    # Spread over the whole split: consecutive frames share camera setups, and a
    # prefix sample would starve the channels only visible from other angles.
    random.Random(0).shuffle(anns)
    model = load_checkpoint(weights, stem_position=stem_position).eval()

    per_channel: dict[int, list[tuple[float, float]]] = {}
    scores: dict[int, list[float]] = {}
    used = 0
    for name in anns:
        if used >= n_frames:
            break
        gt = solve_gt_homography(json.loads(z.read(name)), 960, 540)
        if gt is None or _median_residual(gt[1]) > max_gt_res:
            continue
        H, _ = gt
        Hinv = np.linalg.inv(H)

        img_name = name.replace(".json", ".jpg")
        buf = np.frombuffer(z.read(img_name), dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        with torch.no_grad():
            hm = model(preprocess(frame))[0].numpy()
        # margin > 0: the model actively calls this pixel a keypoint.
        for p in extract_peaks(hm, frame.shape[:2], min_score=0.0):
            xy = project(Hinv, np.array([[p["x_px"], p["y_px"]]]))[0]
            per_channel.setdefault(p["channel"], []).append((float(xy[0]), float(xy[1])))
            scores.setdefault(p["channel"], []).append(p["score"])
        used += 1
    return per_channel, scores, used


def summarise(per_channel, scores, min_support: int = 10):
    grid, report = {}, []
    for c in sorted(per_channel):
        pts = np.array(per_channel[c])
        sc = np.array(scores[c])
        # extract_peaks already gated on positive margin; a mild quantile trim
        # drops borderline detections near the frame edge.
        strong = pts[sc >= np.quantile(sc, 0.25)]
        if len(strong) < min_support:
            report.append({"channel": c, "status": "insufficient", "n": len(strong)})
            continue
        med = np.median(strong, axis=0)
        mad = float(np.median(np.linalg.norm(strong - med, axis=1)))
        entry = {
            "channel": c,
            "x": round(float(med[0]), 3),
            "y": round(float(med[1]), 3),
            "mad_m": round(mad, 3),
            "n": len(strong),
            "median_score": round(float(np.median(sc)), 3),
        }
        report.append(entry)
        if mad < 1.0:  # a real anchor clusters far tighter than a metre
            grid[c] = (entry["x"], entry["y"])
    return grid, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--max-gt-res", type=float, default=3.0)
    ap.add_argument("--out", default="analysis/calib/keypoint_grid.json")
    ap.add_argument("--stem", choices=["first", "last", "both"], default="both")
    args = ap.parse_args()

    variants = ["first", "last"] if args.stem == "both" else [args.stem]
    results = {}
    for stem in variants:
        per_channel, scores, used = collect(
            args.zip, args.weights, args.n_frames, args.max_gt_res, stem
        )
        grid, report = summarise(per_channel, scores)
        tight = [r for r in report if r.get("mad_m", 99) < 1.0]
        mads = [r["mad_m"] for r in tight]
        results[stem] = {"grid": grid, "report": report, "frames_used": used}
        print(
            f"stem={stem}: frames={used} channels_tight={len(tight)}/58 "
            f"median_mad={np.median(mads) if mads else float('nan'):.3f} m"
        )

    best = max(variants, key=lambda s: len(results[s]["grid"]))
    out = {
        "stem_position": best,
        "grid": {str(k): v for k, v in results[best]["grid"].items()},
        "report": results[best]["report"],
        "provenance": "derived empirically from SoccerNet calibration-2023 valid GT; "
        "weights CC-BY-4.0 Zenodo 12626395",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}  (stem={best}, {len(results[best]['grid'])} anchored channels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
