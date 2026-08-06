"""Measure jersey-number OCR on SoccerNet jersey-2023 ground truth.

Baseline is a pretrained scene-text recognizer (EasyOCR, Apache-2.0) restricted to
digits, aggregated per tracklet with the same majority_vote the identity stage
ships. Run this before wiring any OCR into the pipeline: the number this prints is
the honest ceiling for per-player identity on crops of this quality.

Usage:
    python scripts/eval_jersey_ocr.py --zip D:/Users/FAAU/footy-datasets/jersey-2023/train.zip \
        --n-tracklets 40 --crops-per-tracklet 12
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import defaultdict

import numpy as np

from footy.stages.identity import IdentityResolver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n-tracklets", type=int, default=40)
    ap.add_argument("--crops-per-tracklet", type=int, default=12)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    import easyocr

    z = zipfile.ZipFile(args.zip)
    gt: dict[str, int] = json.loads(z.read("train/train_gt.json"))
    by_tracklet: dict[str, list[str]] = defaultdict(list)
    for name in z.namelist():
        if name.endswith(".jpg"):
            by_tracklet[name.split("/")[2]].append(name)

    legible = [t for t, num in gt.items() if num != -1 and by_tracklet.get(t)]
    rng = random.Random(args.seed)
    sample = rng.sample(legible, min(args.n_tracklets, len(legible)))

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    correct = wrong = abstain = 0
    for t in sample:
        crops = rng.sample(by_tracklet[t], min(args.crops_per_tracklet, len(by_tracklet[t])))
        readings: list[int] = []
        for name in crops:
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # Upscale small crops: recognisers fall apart under ~32 px text height.
            h, w = img.shape[:2]
            if h < 96:
                img = cv2.resize(img, (int(w * 96 / h), 96), interpolation=cv2.INTER_CUBIC)
            for _bbox, text, conf in reader.readtext(img, allowlist="0123456789"):
                if text and conf > 0.3:
                    readings.append(int(text))
        vote = IdentityResolver.majority_vote(readings, min_votes=args.min_votes)
        if vote is None:
            abstain += 1
        elif vote == gt[t]:
            correct += 1
        else:
            wrong += 1
            print(f"  tracklet {t}: gt={gt[t]} voted={vote} readings={readings[:8]}")

    n = len(sample)
    if not n:
        print("no tracklets with both ground truth and images found; check the zip layout")
        return 1
    print(f"\ntracklets: {n}  correct: {correct}  wrong: {wrong}  abstained: {abstain}")
    if correct + wrong:
        print(f"precision when voting: {correct / (correct + wrong):.2f}")
    print(f"coverage (voted at all): {(correct + wrong) / n:.2f}")
    print("identity-stage rule: wrong votes corrupt metrics, abstains do not - precision rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
