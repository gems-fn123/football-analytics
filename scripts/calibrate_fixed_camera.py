"""Solve a static homography once for a fixed camera.

Click pitch landmarks in a still frame, match them to known metre coordinates, solve,
and write the result into the camera config. Do this once per camera position; reuse
it for every match filmed from that mount.

Usage:
    python scripts/calibrate_fixed_camera.py --video data/raw/match.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Known landmarks on a 105x68 pitch, origin bottom-left.
PITCH_LANDMARKS = {
    "corner_bl": (0.0, 0.0),
    "corner_tl": (0.0, 68.0),
    "corner_br": (105.0, 0.0),
    "corner_tr": (105.0, 68.0),
    "centre_spot": (52.5, 34.0),
    "halfway_bottom": (52.5, 0.0),
    "halfway_top": (52.5, 68.0),
    "pen_box_left_top": (16.5, 54.16),
    "pen_box_left_bottom": (16.5, 13.84),
    "pen_box_right_top": (88.5, 54.16),
    "pen_box_right_bottom": (88.5, 13.84),
}


def grab_frame(video: str, frame_idx: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read frame")
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="configs/camera/fixed_wide_points.json")
    args = ap.parse_args()

    frame = grab_frame(args.video, args.frame)
    print(f"frame {args.frame}: {frame.shape[1]}x{frame.shape[0]}")
    print("Available landmarks:")
    for name, (x, y) in PITCH_LANDMARKS.items():
        print(f"  {name:24s} -> ({x}, {y})")

    # TODO: open a cv2 window, collect clicks, pair each with a landmark name.
    #       Minimum four non-collinear points. Six or more is much more stable.
    image_points: list[list[float]] = []
    pitch_points: list[list[float]] = []

    if len(image_points) < 4:
        print("\nNot enough points collected. Interactive picking is not implemented yet.")
        print("Fill image_points and pitch_points manually, then rerun to solve.")
        return 1

    H, _ = cv2.findHomography(
        np.array(image_points, dtype=np.float64),
        np.array(pitch_points, dtype=np.float64),
        method=cv2.RANSAC,
    )
    Path(args.out).write_text(
        json.dumps(
            {
                "image_points": image_points,
                "pitch_points": pitch_points,
                "solved_homography": H.tolist(),
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
