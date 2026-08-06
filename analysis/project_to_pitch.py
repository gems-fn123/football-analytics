"""Normalise every detection into pitch metres (the map view).

The reference frame is calibrated once (analysis/seed_clip0.py). Every other frame is
related to that reference by a homography estimated from SIFT correspondences - valid
because a broadcast camera on a fixed mount pans, tilts and zooms about its optical
centre, so frames are related by a global homography regardless of scene depth.

    H_pitch->frame_t  =  H_ref->t  @  H_pitch->ref

Inverting gives image pixels -> pitch metres for any frame. Registration is done
against the reference directly (not chained frame to frame) so error does not
accumulate; a keyframe ladder is used only where the camera has panned too far for a
direct match to survive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pitch_model as pm  # noqa: E402

ROOT = HERE.parent
MIN_MATCHES = 25
KEYFRAME_EVERY = 40


def sift_homography(sift, matcher, kp_a, des_a, gray_b) -> np.ndarray | None:
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_b) < 12:
        return None
    raw = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_MATCHES:
        return None
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inl is None or int(inl.sum()) < MIN_MATCHES * 0.6:
        return None
    return H


def build_frame_homographies(video: str, ref_frame: int, n_frames: int) -> dict[int, np.ndarray]:
    """H_ref->t for every frame, by direct registration with a keyframe fallback."""
    cap = cv2.VideoCapture(video)
    sift = cv2.SIFT_create(nfeatures=3000)
    matcher = cv2.BFMatcher()

    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    ok, ref = cap.read()
    if not ok:
        raise RuntimeError("cannot read reference frame")
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    kp_r, des_r = sift.detectAndCompute(ref_gray, None)

    out: dict[int, np.ndarray] = {ref_frame: np.eye(3)}
    anchor = {"gray": ref_gray, "kp": kp_r, "des": des_r, "H_ref_to_anchor": np.eye(3)}
    n_direct = n_fallback = n_failed = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if idx == ref_frame:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = sift_homography(sift, matcher, kp_r, des_r, gray)
        if H is not None:
            out[idx] = H
            n_direct += 1
        else:
            # Too far from the reference: register against the most recent anchor and
            # compose. Only used when a direct match fails, so drift stays bounded.
            H_anchor = sift_homography(sift, matcher, anchor["kp"], anchor["des"], gray)
            if H_anchor is not None:
                out[idx] = H_anchor @ anchor["H_ref_to_anchor"]
                n_fallback += 1
            else:
                n_failed += 1
                continue
        if idx % KEYFRAME_EVERY == 0 and idx in out:
            kp_a, des_a = sift.detectAndCompute(gray, None)
            anchor = {"gray": gray, "kp": kp_a, "des": des_a, "H_ref_to_anchor": out[idx]}
    cap.release()
    print(f"    registration: {n_direct} direct, {n_fallback} via keyframe, {n_failed} failed")
    return out


def main() -> int:
    clip = sys.argv[1] if len(sys.argv) > 1 else "clip0"
    video_map = {
        "clip0": "data/raw/smoke_clip.mp4",
        "clip1": "data/raw/smoke_clip (1).mp4",
        "clip2": "data/raw/smoke_clip (2).mp4",
    }
    ref_blob = json.loads((HERE / "calib" / f"{clip}_reference.json").read_text())
    H_pitch_to_ref = np.array(ref_blob["H_pitch_to_image"], dtype=float)
    ref_frame = int(ref_blob["frame"])

    px = pd.read_parquet(ROOT / "data/processed" / clip / "tracks_px.parquet")
    n_frames = int(px["frame"].max()) + 1
    print(f"=== {clip}: {len(px)} detections over {n_frames} frames ===")

    Hs = build_frame_homographies(str(ROOT / video_map[clip]), ref_frame, n_frames)

    # Ground contact point: bottom-centre of the box is the only point that lies on the
    # plane the homography maps.
    gx = ((px["x1"] + px["x2"]) / 2.0).to_numpy(float)
    gy = px["y2"].to_numpy(float)
    frames = px["frame"].to_numpy(int)

    x_m = np.full(len(px), np.nan)
    y_m = np.full(len(px), np.nan)
    for f in np.unique(frames):
        H_ref_to_t = Hs.get(int(f))
        if H_ref_to_t is None:
            continue
        H_pitch_to_t = H_ref_to_t @ H_pitch_to_ref
        try:
            H_t_to_pitch = np.linalg.inv(H_pitch_to_t)
        except np.linalg.LinAlgError:
            continue
        sel = frames == f
        pts = np.column_stack([gx[sel], gy[sel], np.ones(sel.sum())])
        w = pts @ H_t_to_pitch.T
        z = w[:, 2:3]
        z[np.abs(z[:, 0]) < 1e-9] = 1e-9
        m = w[:, :2] / z
        x_m[sel], y_m[sel] = m[:, 0], m[:, 1]

    out = px.copy()
    out["x_m"] = x_m
    out["y_m"] = y_m
    out["t_s"] = out["frame"] / 25.0
    ok = np.isfinite(x_m) & np.isfinite(y_m)
    on = ok & pm.in_bounds(x_m, y_m, margin=5.0)
    print(f"    projected: {ok.sum()}/{len(out)} ({ok.mean()*100:.1f}%)")
    print(f"    inside pitch (+-5 m): {on.sum()}/{ok.sum()} ({on.sum()/max(ok.sum(),1)*100:.1f}%)")
    if ok.any():
        print(f"    x_m range {np.nanmin(x_m[ok]):7.1f} .. {np.nanmax(x_m[ok]):7.1f}")
        print(f"    y_m range {np.nanmin(y_m[ok]):7.1f} .. {np.nanmax(y_m[ok]):7.1f}")

    dst = ROOT / "data/processed" / clip / "tracks_pitch.parquet"
    out.to_parquet(dst, index=False)
    print("    wrote", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
