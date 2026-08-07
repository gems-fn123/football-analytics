"""Does the newly-merged keypoint-based calibration (src/footy/calib) beat the
hand-built ellipse/line pipeline in this directory, on our own footage?

Both this analysis/ pipeline and src/footy/calib/ (NBJW HRNet keypoints + line DLT,
merged from a teammate's branch) attack the same problem. Rather than guess which is
better from reading the code, this scores the learned pipeline against clip0's verified
reference with the exact same metric (compare_to) used for every number in
CALIBRATION_STATUS.md, so the results are directly comparable:

  lines only            533 px median
  all 8 circle candidates  431 px median
  true rim (oracle)     248 px median
  true rim + players    159 px median
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from footy.calib.hrnet import load_checkpoint  # noqa: E402
from footy.calib.keypoints import extract_peaks, homography_from_keypoints, preprocess, snap_grid  # noqa: E402
from run_auto_calib import CLIPS, compare_to  # noqa: E402

ROOT = HERE.parent
WEIGHTS = ROOT / "models/weights/SV_kp.pth"
GRID = HERE / "calib/keypoint_grid.json"


def main() -> int:
    if not WEIGHTS.exists():
        print(f"weights not present at {WEIGHTS}; run scripts/download_weights.sh")
        return 1

    blob = json.loads(GRID.read_text())
    grid = snap_grid({int(k): tuple(v) for k, v in blob["grid"].items()})
    t0 = time.time()
    model = load_checkpoint(WEIGHTS, stem_position=blob["stem_position"]).eval()
    print(f"model loaded in {time.time() - t0:.1f}s ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    H_ref = np.array(json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"])

    for clip, (video, frame_idx) in CLIPS.items():
        cap = cv2.VideoCapture(str(ROOT / video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        cap.release()
        if not ok:
            print(f"{clip}: cannot read frame {frame_idx}")
            continue

        for min_score in (0.0, -2.0, -5.0):
            with torch.no_grad():
                hm = model(preprocess(img))[0].numpy()
            peaks = extract_peaks(hm, img.shape[:2], min_score=min_score)
            used = [p for p in peaks if int(p["channel"]) in grid]
            H = homography_from_keypoints(peaks, grid, min_score=min_score, min_points=4)
            if H is None:
                print(f"{clip} (frame {frame_idx}) min_score={min_score:+.0f}: "
                      f"{len(peaks)} peaks, {len(used)} on the grid -> no homography")
                continue
            err = compare_to(H, H_ref, img.shape[:2]) if clip == "clip0" else float("nan")
            tag = f"  <-- vs verified reference" if clip == "clip0" else "  (no verified reference for this clip)"
            print(f"{clip} (frame {frame_idx}) min_score={min_score:+.0f}: "
                  f"{len(peaks)} peaks, {len(used)} on the grid, "
                  f"median error {err:.1f} px{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
