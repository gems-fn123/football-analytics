"""Solve a static homography once for a fixed camera.

Click pitch landmarks in a still frame, match them to known metre coordinates, solve,
and write the result into the camera config. Do this once per camera position; reuse
it for every match filmed from that mount.

Usage:
    # interactive: click the prompted landmark, s skips, u undoes, q finishes
    python scripts/calibrate_fixed_camera.py --video data/raw/match.mp4

    # headless: points collected some other way (or hand-measured in an image editor)
    python scripts/calibrate_fixed_camera.py --from-points my_points.json

The points JSON needs "image_points" ([[px, py], ...]) and "pitch_points"
([[m_x, m_y], ...]), same length, same order. `footy calibrate --points` does the
same solve without this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from footy.stages.calibrate import solve_homography

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

WINDOW = "calibrate - click the prompted landmark | s skip, u undo, q finish"


def grab_frame(video: str, frame_idx: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read frame")
    return frame


def pick_points_interactive(frame: np.ndarray) -> tuple[list[list[float]], list[list[float]]]:
    """One landmark at a time: click it or skip it. Returns (image, pitch) points."""
    names = list(PITCH_LANDMARKS)
    picked: list[tuple[str, float, float]] = []
    idx = 0
    clicked: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((float(x), float(y)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while idx < len(names):
        name = names[idx]
        canvas = frame.copy()
        for pname, px, py in picked:
            cv2.drawMarker(canvas, (int(px), int(py)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                canvas,
                pname,
                (int(px) + 6, int(py) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
        cv2.putText(
            canvas,
            f"[{idx + 1}/{len(names)}] click: {name} -> {PITCH_LANDMARKS[name]}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
        )
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF

        if clicked:
            x, y = clicked.pop()
            picked.append((name, x, y))
            idx += 1
        elif key == ord("s"):
            idx += 1
        elif key == ord("u") and picked:
            # Undo returns to the position of the removed landmark.
            idx = names.index(picked.pop()[0])
        elif key == ord("q"):
            break

    cv2.destroyWindow(WINDOW)
    image_points = [[x, y] for _, x, y in picked]
    pitch_points = [list(PITCH_LANDMARKS[name]) for name, _, _ in picked]
    return image_points, pitch_points


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="video to grab a still frame from (interactive mode)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--from-points", help="points JSON to solve headlessly, no UI")
    ap.add_argument("--out", default="configs/camera/fixed_wide_points.json")
    args = ap.parse_args()

    if args.from_points:
        blob = json.loads(Path(args.from_points).read_text())
        image_points = blob["image_points"]
        pitch_points = blob["pitch_points"]
    elif args.video:
        frame = grab_frame(args.video, args.frame)
        print(f"frame {args.frame}: {frame.shape[1]}x{frame.shape[0]}")
        print("Click each prompted landmark. Six or more points beats the minimum four.")
        image_points, pitch_points = pick_points_interactive(frame)
    else:
        ap.error("need --video (interactive) or --from-points (headless)")
        return 2

    if len(image_points) < 4:
        print(f"\nOnly {len(image_points)} points collected; need at least 4. Nothing written.")
        return 1

    H, err = solve_homography(image_points, pitch_points)
    Path(args.out).write_text(
        json.dumps(
            {
                "image_points": image_points,
                "pitch_points": pitch_points,
                "solved_homography": H.tolist(),
                "max_reprojection_error_m": err,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}  ({len(image_points)} points, max reprojection error {err:.2f} m)")
    if err > 2.0:
        print("WARNING: reprojection error above 2 m; re-pick the landmarks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
