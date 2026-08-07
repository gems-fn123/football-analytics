"""Is the tracked homography actually accurate late in the clip, and why did C-K4 fail?

Two questions, because the first run's C-K2 was measured wrongly and the second
criterion failed for a reason that needs attributing before it can be fixed.

1. DRIFT, measured independently.
   The first attempt compared the tracked homography against a direct frame-0 match.
   That is vacuous while the keyframe is still frame 0: both are the *same computation*,
   which is why frames 25-100 returned exactly 0.000 m. Only a direct match strong
   enough to trust (>=100 inliers) survives to late frames, and by then the baseline is
   too wide - so the check only existed where it was meaningless.

   Replaced with a three-way consistency test that works anywhere in the clip. For a
   frame pair (a, b) close enough to match strongly:

       H_a->b  is matched directly and fresh, involving no chain at all
       tracked H_0->b  should equal  H_a->b @ tracked H_0->a

   Disagreement is exactly the accumulated composition error. Pairs that straddle a
   keyframe boundary test the composition itself and are reported separately, since
   those are the ones that can actually drift.

2. C-K4 (70.8% of players on the pitch) - homography fault or detector fault?
   Criterion C-K4 was written to catch a globally misplaced pitch. But this footage has
   a declared hazard H-1: a COCO person detector with thousands of spectators in frame.
   If the off-pitch projections are crowd, C-K4 is measuring detector contamination and
   the homography is fine. The two are distinguishable: a misplaced pitch shifts
   *everything* coherently, while crowd detections sit beyond a touchline in a band that
   matches where the stands are in the image.
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
sys.path.insert(0, str(HERE.parent / "src"))

import pitch_model as pm  # noqa: E402
from track_homography import (  # noqa: E402
    boxes_by_frame,
    estimate,
    player_mask,
    track,
    unproject,
    watermark_mask,
)

ROOT = HERE.parent
VIDEO = str(ROOT / "data/raw/smoke_clip.mp4")
TRACKS = ROOT / "data/processed/clip0/tracks_px.parquet"
PAIR_GAP = 40          # close enough to match strongly, far enough to be a real hop
MIN_PAIR_INLIERS = 100


def main() -> int:
    H_pitch_ref = np.array(
        json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"]
    )
    print("re-running the tracker (needed for keyframe assignment)")
    res = track(VIDEO, TRACKS)
    Hs, key_of = res["H"], res["key_of"]

    wm = watermark_mask(VIDEO)
    boxes = boxes_by_frame(TRACKS)
    sift = cv2.SIFT_create(nfeatures=6000)
    matcher = cv2.BFMatcher()
    cap = cv2.VideoCapture(VIDEO)

    def mask_for(idx, shp):
        m = player_mask(shp, boxes.get(idx, np.empty((0, 4))))
        if wm is not None:
            m = cv2.bitwise_and(m, cv2.bitwise_not(wm))
        return m

    def read(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        return f if ok else None

    # ---------------- 1. three-way consistency ----------------
    rows = []
    frames = sorted(Hs)
    for a in range(0, max(frames) - PAIR_GAP, 25):
        b = a + PAIR_GAP
        if a not in Hs or b not in Hs:
            continue
        fa, fb = read(a), read(b)
        if fa is None or fb is None:
            continue
        ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
        kpa, desa = sift.detectAndCompute(ga, mask_for(a, fa.shape))
        H_ab, n_in, _, _ = estimate(
            sift, matcher, kpa, desa, cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY),
            mask_for(b, fb.shape))
        if H_ab is None or n_in < MIN_PAIR_INLIERS:
            continue

        predicted = H_ab @ Hs[a]      # fresh hop composed onto the tracked pose at a
        actual = Hs[b]                # what the tracker says at b

        h, w = fa.shape[:2]
        uv = np.stack(np.meshgrid(np.linspace(0, w, 12), np.linspace(h * 0.35, h, 8)),
                      -1).reshape(-1, 2).astype(np.float64)
        p = unproject(predicted @ H_pitch_ref, uv)
        q = unproject(actual @ H_pitch_ref, uv)
        ok_pt = np.isfinite(p).all(1) & np.isfinite(q).all(1)
        if ok_pt.sum() < 10:
            continue
        d = float(np.median(np.linalg.norm(p[ok_pt] - q[ok_pt], axis=1)))
        rows.append({"a": a, "b": b, "inliers": n_in, "err_m": round(d, 4),
                     "crosses_keyframe": key_of.get(a) != key_of.get(b)})
    cap.release()

    df = pd.DataFrame(rows)
    print(f"\nthree-way consistency over {len(df)} frame pairs (gap {PAIR_GAP})")
    if len(df):
        crossing = df[df["crosses_keyframe"]]
        within = df[~df["crosses_keyframe"]]
        print(f"  overall      median {df['err_m'].median():.3f} m   "
              f"p90 {df['err_m'].quantile(0.9):.3f} m   max {df['err_m'].max():.3f} m")
        if len(crossing):
            print(f"  crossing a keyframe ({len(crossing)} pairs)  "
                  f"median {crossing['err_m'].median():.3f} m   max {crossing['err_m'].max():.3f} m")
        if len(within):
            print(f"  within one keyframe ({len(within)} pairs)    "
                  f"median {within['err_m'].median():.3f} m   max {within['err_m'].max():.3f} m")
        # Does error grow with time? That is the signature of accumulation.
        if len(df) > 4:
            corr = float(np.corrcoef(df["a"], df["err_m"])[0, 1])
            print(f"  correlation of error with frame index: {corr:+.3f}  "
                  f"({'accumulating' if corr > 0.4 else 'no accumulation trend'})")

    # ---------------- 2. attribute the C-K4 failure ----------------
    tp = pd.read_parquet(TRACKS)
    tp = tp[(tp["cls"] == "player") & (tp["track_id"] >= 0)]
    recs = []
    for f, g in tp.groupby("frame"):
        f = int(f)
        if f not in Hs:
            continue
        feet = np.column_stack([(g["x1"] + g["x2"]).to_numpy() / 2.0, g["y2"].to_numpy()])
        m = unproject(Hs[f] @ H_pitch_ref, feet)
        for (xm, ym), yb, conf in zip(m, g["y2"].to_numpy(), g["conf"].to_numpy()):
            recs.append({"x_m": xm, "y_m": ym, "y_px": yb, "conf": conf})
    d = pd.DataFrame(recs)
    inside = (d["x_m"].between(-3, pm.LENGTH + 3) & d["y_m"].between(-3, pm.WIDTH + 3))
    print(f"\nC-K4 attribution: {len(d)} player detections, {inside.mean():.1%} on pitch")
    off = d[~inside]
    print(f"  off-pitch detections: {len(off)}")
    if len(off):
        print(f"    y_m > {pm.WIDTH + 3:.0f} (beyond far touchline): "
              f"{(off['y_m'] > pm.WIDTH + 3).mean():.1%}")
        print(f"    y_m < -3 (beyond near touchline):  {(off['y_m'] < -3).mean():.1%}")
        print(f"    outside the goal lines in x:       "
              f"{((off['x_m'] < -3) | (off['x_m'] > pm.LENGTH + 3)).mean():.1%}")
        print(f"    median image y of off-pitch dets:  {off['y_px'].median():.0f} px "
              f"(on-pitch: {d[inside]['y_px'].median():.0f} px)")
        print(f"    median confidence off-pitch:       {off['conf'].median():.3f} "
              f"(on-pitch: {d[inside]['conf'].median():.3f})")
        # A player higher in the image is further away; the stands are above the pitch.
        print("\n  reading: off-pitch detections sitting HIGHER in the image (smaller y_px)"
              "\n  and beyond the far touchline are the declared H-1 crowd hazard,"
              "\n  not a displaced pitch - a displaced pitch would move every detection"
              "\n  together and would not sort by image height.")

    out = {"pairs": rows, "n_pairs": len(df),
           "drift_median_m": None if not len(df) else round(float(df["err_m"].median()), 4),
           "drift_p90_m": None if not len(df) else round(float(df["err_m"].quantile(0.9)), 4),
           "drift_max_m": None if not len(df) else round(float(df["err_m"].max()), 4),
           "fraction_on_pitch": round(float(inside.mean()), 4),
           "off_pitch_beyond_far_touchline": None if not len(off) else round(float((off["y_m"] > pm.WIDTH + 3).mean()), 4),
           "off_pitch_median_y_px": None if not len(off) else float(off["y_px"].median()),
           "on_pitch_median_y_px": float(d[inside]["y_px"].median())}
    (ROOT / "reports/homography_drift.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/homography_drift.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
