"""Apply the tracked homography to clip0 and score the physical criteria.

Scored against reports/02_preregistration_metrics.md section B, which only becomes
scorable because C-K2 passed (drift 0.192 m median, well under the 2.0 m bound).

The spatial gate is criterion C-P3, and it is not optional bookkeeping. Two independent
findings force it:

  - probe_drift.py attributed C-K4's failure to the H-1 crowd hazard: 82.6% of
    off-pitch projections fall outside the goal lines, sit higher in the image, and
    carry lower confidence. They are spectators, and a COCO person detector has no
    concept of "on the pitch".
  - a detection near the horizon unprojects to an enormous ground distance through a
    perfectly correct homography, so one crowd box can contribute kilometres of
    "distance covered".

Gated rows go null, never interpolated - the repo's standing "nullable beats guessed"
rule, and the specific thing 00_preregistration.md exists to prevent.
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

import pitch_model as pm  # noqa: E402
from footy.analytics.physical import add_kinematics as add_kinematics_repo  # noqa: E402
from kinematics import (  # noqa: E402
    add_kinematics,
    estimate_position_noise,
    split_on_impossible_motion,
    summarise,
)

ROOT = HERE.parent
PROC = ROOT / "data/processed/clip0"
MARGIN_M = 3.0          # keepers and throw-ins legitimately stand off the pitch
FPS = 25.0
SPEED_CAP_MS = 11.0     # docs/runbook.md: "above 11 m/s means tracking noise"


def main() -> int:
    H_pitch_ref = np.array(
        json.loads((HERE / "calib/clip0_reference.json").read_text())["H_pitch_to_image"]
    )
    Hs_arr = np.load(PROC / "homographies.npy")
    frames = json.loads((PROC / "homography_frames.json").read_text())
    Hs = {int(f): Hs_arr[i] for i, f in enumerate(frames)}

    df = pd.read_parquet(PROC / "tracks_px.parquet")
    print(f"clip0: {len(df)} detection rows, {df['frame'].nunique()} frames")

    feet_x = (df["x1"].to_numpy() + df["x2"].to_numpy()) / 2.0
    feet_y = df["y2"].to_numpy()
    x_m = np.full(len(df), np.nan)
    y_m = np.full(len(df), np.nan)

    frame_col = df["frame"].to_numpy().astype(int)
    for f in np.unique(frame_col):
        if f not in Hs:
            continue
        sel = frame_col == f
        Hpi = Hs[f] @ H_pitch_ref
        Hinv = np.linalg.inv(Hpi)
        uv = np.column_stack([feet_x[sel], feet_y[sel], np.ones(sel.sum())])
        p = uv @ Hinv.T
        z = p[:, 2]
        good = np.abs(z) > 1e-9
        xs = np.full(sel.sum(), np.nan)
        ys = np.full(sel.sum(), np.nan)
        xs[good] = p[good, 0] / z[good]
        ys[good] = p[good, 1] / z[good]
        x_m[sel], y_m[sel] = xs, ys

    df["x_m"], df["y_m"] = x_m, y_m
    df["t_s"] = df["frame"] / FPS

    projected = np.isfinite(df["x_m"]) & np.isfinite(df["y_m"])
    on_pitch = (
        df["x_m"].between(-MARGIN_M, pm.LENGTH + MARGIN_M)
        & df["y_m"].between(-MARGIN_M, pm.WIDTH + MARGIN_M)
    )
    gate = projected & on_pitch
    print(f"  projected: {projected.mean():.1%}   on pitch (C-P3 gate): {gate.mean():.1%}")
    print(f"  gated out: {int((~gate).sum())} rows -> null, not interpolated")

    df.loc[~gate, ["x_m", "y_m"]] = np.nan

    players = df[(df["cls"] == "player") & (df["track_id"] >= 0)].copy()

    # Both estimators are reported. The repo's differences adjacent frames, which
    # divides position noise by dt; the windowed fit sizes its window from the measured
    # noise (see analysis/kinematics.py). Showing both keeps the improvement auditable
    # rather than asserted.
    repo_kin = add_kinematics_repo(players).dropna(subset=["speed_ms"])
    repo_over = float((repo_kin["speed_ms"] > SPEED_CAP_MS).mean()) if len(repo_kin) else float("nan")

    # A track that teleports is not one player. Split before measuring anything.
    sigma0 = estimate_position_noise(players.dropna(subset=["x_m", "y_m"]))
    n_before = players["track_id"].nunique()
    players = split_on_impossible_motion(players.dropna(subset=["x_m", "y_m"]), sigma0)
    n_after = players["track_id"].nunique()
    print(f"\nphysics split: {n_before} tracks -> {n_after} segments "
          f"(sigma={sigma0:.3f} m, cut above {SPEED_CAP_MS + 1:.0f} m/s + 3 sigma)")

    kin, diag = add_kinematics(players)
    kin = kin.dropna(subset=["speed_ms"])
    print(f"\nvelocity estimator: {diag['window_frames']}-frame window "
          f"({diag['window_seconds']} s), derived from measured position noise "
          f"sigma={diag['sigma_position_m']} m -> velocity noise "
          f"sigma={diag['sigma_velocity_ms']} m/s")

    # ---- C-P1: implausible speeds ----
    over = float((kin["speed_ms"] > SPEED_CAP_MS).mean()) if len(kin) else float("nan")
    print(f"\nC-P1  speeds over {SPEED_CAP_MS} m/s: {over:.2%} of "
          f"{len(kin)} player-frames  (bar: <= 1%)")
    print(f"      for comparison, per-frame differencing: {repo_over:.2%}")
    if len(kin):
        print(f"      speed p50 {kin['speed_ms'].median():.2f}  "
              f"p95 {kin['speed_ms'].quantile(0.95):.2f}  "
              f"p99 {kin['speed_ms'].quantile(0.99):.2f}  max {kin['speed_ms'].max():.2f} m/s")

    # ---- C-P2: distance rate ----
    summary = summarise(kin)
    summary = summary[summary["minutes"] > 0].copy()
    summary["km_per_90"] = summary["distance_m"] / 1000.0 * (90.0 / summary["minutes"])
    med_rate = float(summary["km_per_90"].median()) if len(summary) else float("nan")
    print(f"\nC-P2  median distance rate: {med_rate:.1f} km/90 over {len(summary)} tracks"
          f"  (bar: 9-12)")
    if len(summary):
        print(f"      rate p25 {summary['km_per_90'].quantile(.25):.1f}  "
              f"p75 {summary['km_per_90'].quantile(.75):.1f}  "
              f"max {summary['km_per_90'].max():.1f} km/90")
        print(f"      track minutes: median {summary['minutes'].median()*60:.1f} s")

    crit = {
        "C-P1 <=1% speeds over 11 m/s": {"pass": bool(over <= 0.01), "value": round(over, 4)},
        "C-P2 median 9-12 km/90": {"pass": bool(9.0 <= med_rate <= 12.0),
                                    "value": round(med_rate, 2)},
        "C-P3 gated rows null": {"pass": True, "value": round(float(gate.mean()), 4)},
    }
    print("\ncriteria:")
    for k, v in crit.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  = {v['value']}")

    out = {
        "clip": "clip0",
        "n_rows": int(len(df)),
        "fraction_projected": round(float(projected.mean()), 4),
        "fraction_on_pitch_gate": round(float(gate.mean()), 4),
        "n_player_frames_with_speed": int(len(kin)),
        "speed_over_cap_frac": round(over, 4),
        "speed_p50": None if not len(kin) else round(float(kin["speed_ms"].median()), 3),
        "speed_p95": None if not len(kin) else round(float(kin["speed_ms"].quantile(.95)), 3),
        "speed_p99": None if not len(kin) else round(float(kin["speed_ms"].quantile(.99)), 3),
        "speed_max": None if not len(kin) else round(float(kin["speed_ms"].max()), 3),
        "n_tracks": int(len(summary)),
        "median_km_per_90": None if not len(summary) else round(med_rate, 3),
        "velocity_estimator": diag,
        "speed_over_cap_frac_per_frame_differencing": round(repo_over, 4),
        "criteria": crit,
    }
    (ROOT / "reports/physical_metrics.json").write_text(json.dumps(out, indent=2))
    df.to_parquet(PROC / "tracks_m_tracked.parquet", index=False)
    print(f"\nwrote reports/physical_metrics.json and {PROC/'tracks_m_tracked.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
