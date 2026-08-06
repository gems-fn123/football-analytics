"""Run automatic registration on every clip and validate it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pitch_model as pm  # noqa: E402
from auto_calibrate import auto_fit, sanity  # noqa: E402
from calibrate import overlay, project  # noqa: E402
from detect_lines import analyse  # noqa: E402

ROOT = HERE.parent
CAL = HERE / "calib"
CAL.mkdir(exist_ok=True)

CLIPS = {
    "clip0": ("data/raw/smoke_clip.mp4", 0),
    "clip1": ("data/raw/smoke_clip (1).mp4", 0),
    "clip2": ("data/raw/smoke_clip (2).mp4", 182),
}


def compare_to(H_a: np.ndarray, H_b: np.ndarray) -> float:
    """Mean pixel disagreement between two registrations over the visible pitch."""
    g = np.stack(
        np.meshgrid(np.linspace(20, 90, 12), np.linspace(5, 63, 10)), -1
    ).reshape(-1, 2)
    ua, ub = project(H_a, g), project(H_b, g)
    ok = np.isfinite(ua).all(1) & np.isfinite(ub).all(1)
    return float(np.median(np.linalg.norm(ua[ok] - ub[ok], axis=1)))


def main() -> int:
    scratch = Path(sys.argv[1])
    out = {}
    for clip, (video, frame) in CLIPS.items():
        r = analyse(str(ROOT / video), frame,
                    None, str(ROOT / "data/processed" / clip / "tracks_px.parquet"))
        mask, img = r["mask"], r["image"]
        print(f"\n=== {clip} (frame {frame}) — {int((mask>0).sum())} line pixels ===")
        best = auto_fit(mask, n_samples=6000, n_refine=35, seed=7)
        H = np.array(best["H"])
        s = sanity(H, mask.shape)
        print(f"    F1 {best['f1']:.3f} (P {best['precision']:.3f} R {best['recall']:.3f})  "
              f"median {best['median_px']:.2f}px  visible {best['visible']}  "
              f"px/m@centre {s['px_per_metre_at_centre']}")
        cv2.imwrite(str(scratch / f"auto_{clip}.png"), overlay(img, H))

        rec = {"clip": clip, "frame": frame, "video": video,
               "H_pitch_to_image": H.tolist(), **{k: float(v) for k, v in
               [("f1", best["f1"]), ("precision", best["precision"]),
                ("recall", best["recall"]), ("median_px", best["median_px"]),
                ("visible", best["visible"])]}, "sanity": s}

        # clip0 already has a trusted, hand-seeded registration: use it to check that
        # the automatic method lands in the same place.
        ref = CAL / f"{clip}_reference.json"
        if ref.exists():
            H_ref = np.array(json.loads(ref.read_text())["H_pitch_to_image"])
            d = compare_to(H, H_ref)
            rec["median_px_vs_hand_seeded"] = round(d, 2)
            print(f"    vs hand-seeded reference: {d:.2f} px median disagreement")
        out[clip] = rec

    (CAL / "auto_calibration.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", CAL / "auto_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
