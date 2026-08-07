"""Why do 7.73% of player speeds exceed 11 m/s? Attribute before fixing.

C-P1 failed as predicted (P-5), but the prediction named identity switches as the
cause and a prediction is not evidence. Three mechanisms can each produce an impossible
speed, and they need different fixes, so guessing wrong wastes the work:

  A. DEPTH UNCERTAINTY. Metres per pixel is not constant across the frame. Near the
     horizon a 2 px wobble in where the detector puts a player's feet is worth many
     metres on the ground. The homography is perfectly correct and the speed is still
     garbage. Fix: propagate detector precision through the homography and refuse
     positions whose uncertainty is too large.

  B. IDENTITY SWITCH. ByteTrack hands one track_id to a different player, so the
     "player" teleports. Fix: better association / appearance re-ID.

  C. BOX INSTABILITY. Occlusion or a partial detection moves the box bottom without
     the player moving. Fix: robust smoothing.

A and B are separable by a direct test: A predicts the spike sits at a high-uncertainty
(far) position, B predicts a large jump between two positions that are each precisely
measured. So compute per-detection positional uncertainty and cross it with the spikes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from footy.analytics.physical import add_kinematics  # noqa: E402

ROOT = HERE.parent
PROC = ROOT / "data/processed/clip0"
FPS = 25.0
JITTER_PX = 2.0     # detector precision on the box bottom edge
SPEED_CAP_MS = 11.0


def main() -> int:
    H_pitch_ref = np.array(
        json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"]
    )
    Hs_arr = np.load(PROC / "homographies.npy")
    frames = json.loads((PROC / "homography_frames.json").read_text())
    Hs = {int(f): Hs_arr[i] for i, f in enumerate(frames)}

    df = pd.read_parquet(PROC / "tracks_m_tracked.parquet")
    df = df[(df["cls"] == "player") & (df["track_id"] >= 0)].copy()
    df = df.dropna(subset=["x_m", "y_m"])

    # ---- positional uncertainty: metres induced by JITTER_PX of image wobble ----
    fx = (df["x1"].to_numpy() + df["x2"].to_numpy()) / 2.0
    fy = df["y2"].to_numpy()
    unc = np.full(len(df), np.nan)
    fr = df["frame"].to_numpy().astype(int)
    for f in np.unique(fr):
        if f not in Hs:
            continue
        sel = fr == f
        Hinv = np.linalg.inv(Hs[f] @ H_pitch_ref)

        def proj(u, v):
            p = np.column_stack([u, v, np.ones(len(u))]) @ Hinv.T
            z = p[:, 2]
            z = np.where(np.abs(z) < 1e-9, np.nan, z)
            return np.column_stack([p[:, 0] / z, p[:, 1] / z])

        base = proj(fx[sel], fy[sel])
        shifted = proj(fx[sel], fy[sel] + JITTER_PX)
        unc[sel] = np.linalg.norm(shifted - base, axis=1)
    df["unc_m"] = unc

    kin = add_kinematics(df).dropna(subset=["speed_ms"])
    kin = kin.merge(df[["frame", "track_id", "unc_m"]], on=["frame", "track_id"],
                    how="left", suffixes=("", "_d"))
    spikes = kin[kin["speed_ms"] > SPEED_CAP_MS]
    calm = kin[kin["speed_ms"] <= SPEED_CAP_MS]

    print(f"player-frames with speed: {len(kin)}")
    print(f"  over {SPEED_CAP_MS} m/s: {len(spikes)} ({len(spikes)/len(kin):.2%})")
    print(f"\npositional uncertainty from {JITTER_PX:.0f} px of feet jitter:")
    print(f"  all detections   median {df['unc_m'].median():.3f} m   "
          f"p90 {df['unc_m'].quantile(.9):.3f}   p99 {df['unc_m'].quantile(.99):.3f}   "
          f"max {df['unc_m'].max():.1f} m")
    print(f"  at speed spikes  median {spikes['unc_m'].median():.3f} m   "
          f"p90 {spikes['unc_m'].quantile(.9):.3f} m")
    print(f"  at calm frames   median {calm['unc_m'].median():.3f} m   "
          f"p90 {calm['unc_m'].quantile(.9):.3f} m")

    ratio = spikes["unc_m"].median() / max(calm["unc_m"].median(), 1e-9)
    print(f"\n  spikes sit at {ratio:.1f}x the positional uncertainty of calm frames")

    # How much of the problem would an uncertainty gate remove, and at what cost?
    print(f"\nif positions with uncertainty above a bound are refused (mechanism A):")
    print(f"  {'bound':>8}  {'kept':>7}  {'spikes left':>12}  {'spike rate':>11}")
    for bound in (2.0, 1.0, 0.5, 0.35, 0.25):
        keep = kin[kin["unc_m"] <= bound]
        if not len(keep):
            continue
        rate = float((keep["speed_ms"] > SPEED_CAP_MS).mean())
        print(f"  {bound:8.2f}  {len(keep)/len(kin):6.1%}  "
              f"{int((keep['speed_ms'] > SPEED_CAP_MS).sum()):12d}  {rate:10.2%}")

    # Mechanism B: a teleport between two CONFIDENT positions is an identity switch.
    confident = kin[kin["unc_m"] <= 0.35]
    b_rate = float((confident["speed_ms"] > SPEED_CAP_MS).mean()) if len(confident) else float("nan")
    print(f"\nmechanism B check: among well-measured positions (unc <= 0.35 m), "
          f"{b_rate:.2%} still exceed the cap")
    print("  these cannot be explained by depth uncertainty and are the genuine")
    print("  identity-switch / box-instability residue.")

    out = {
        "n_player_frames": int(len(kin)),
        "spike_frac": round(float(len(spikes) / len(kin)), 4),
        "unc_median_all_m": round(float(df["unc_m"].median()), 4),
        "unc_p99_all_m": round(float(df["unc_m"].quantile(.99)), 4),
        "unc_median_at_spikes_m": round(float(spikes["unc_m"].median()), 4),
        "unc_median_at_calm_m": round(float(calm["unc_m"].median()), 4),
        "uncertainty_ratio_spike_vs_calm": round(float(ratio), 3),
        "residual_spike_rate_when_confident": round(float(b_rate), 4),
        "gate_sweep": [
            {"bound_m": b,
             "kept_frac": round(float(len(kin[kin["unc_m"] <= b]) / len(kin)), 4),
             "spike_rate": round(float((kin[kin["unc_m"] <= b]["speed_ms"] > SPEED_CAP_MS).mean()), 4)}
            for b in (2.0, 1.0, 0.5, 0.35, 0.25) if len(kin[kin["unc_m"] <= b])
        ],
    }
    (ROOT / "reports/speed_attribution.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/speed_attribution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
