"""Associate detections in pitch metres instead of image pixels.

The measurement that motivates this: 66 ByteTrack ids on clip0 split into 251 physically
honest segments, i.e. most ids contain at least one identity switch. ByteTrack associates
by IoU in the IMAGE, and clip0's camera pans 1934 px over the clip. During a pan a
stationary player has a large image-space velocity, so IoU between consecutive frames
collapses and the tracker either drops the player or grabs a different one.

In pitch coordinates that problem disappears, because camera motion has been removed by
the homography (validated at 0.192 m in reports/03_results_metrics.md). The association
gate follows from physics rather than tuning:

    a player moves at most  MAX_SPEED * dt   metres between frames
    position noise contributes  ~3 * sigma * sqrt(2)

so two detections further apart than that cannot be the same person, and anything
closer is a candidate. At dt = 0.04 s that is a gate of about 1.1 m, against a pitch
105 m long - an extremely discriminative test, which is exactly what image space cannot
offer while the camera moves.

Assignment is globally optimal per frame (Hungarian) rather than greedy, so two players
crossing do not both grab the same detection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import pitch_model as pm  # noqa: E402

ROOT = HERE.parent
PROC = ROOT / "data/processed/clip0"

MAX_SPEED_MS = 12.0
SIGMA_M = 0.21          # measured in reports/speed_attribution.json
MAX_AGE_FRAMES = 25     # 1 s: keep a track alive through an occlusion
MIN_TRACK_FRAMES = 5
DT = 0.04


def gate_radius(dt_s: float) -> float:
    """How far a real player could have moved, plus measurement slack."""
    return MAX_SPEED_MS * dt_s + 3.0 * SIGMA_M * np.sqrt(2.0)


def track_in_pitch(det: pd.DataFrame) -> pd.DataFrame:
    """det needs frame, x_m, y_m (already projected and gated). Adds pitch_track_id."""
    det = det.sort_values("frame").reset_index(drop=True)
    frames = sorted(det["frame"].unique())

    # Live tracks: id -> dict(x, y, vx, vy, last_frame, n)
    live: dict[int, dict] = {}
    next_id = 0
    assign = np.full(len(det), -1, dtype=int)
    idx_by_frame = {f: np.where(det["frame"].to_numpy() == f)[0] for f in frames}

    for f in frames:
        rows = idx_by_frame[f]
        obs = det.loc[rows, ["x_m", "y_m"]].to_numpy(dtype=float)

        stale = [t for t, s in live.items() if f - s["last_frame"] > MAX_AGE_FRAMES]
        for t in stale:
            del live[t]

        if live and len(obs):
            tids = sorted(live)
            pred = np.zeros((len(tids), 2))
            gates = np.zeros(len(tids))
            for i, t in enumerate(tids):
                s = live[t]
                dt = (f - s["last_frame"]) * DT
                # Constant velocity between sightings; velocity is only trusted once
                # the track has enough history to have estimated it.
                pred[i] = [s["x"] + s["vx"] * dt, s["y"] + s["vy"] * dt]
                gates[i] = gate_radius(dt)
            cost = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
            # Two gates, not one. Distance from the PREDICTION decides which match is
            # best, but a wrong velocity estimate can predict a position the player
            # could never actually reach, so the physical bound is also enforced from
            # the last OBSERVED position. Without this the tracker admits jumps that
            # split_on_impossible_motion then has to cut back out (122 -> 172 segments).
            last = np.array([[live[t]["x"], live[t]["y"]] for t in tids])
            reach = np.linalg.norm(last[:, None, :] - obs[None, :, :], axis=2)
            blocked = (cost > gates[:, None]) | (reach > gates[:, None])
            big = cost.max() + 1e3 if cost.size else 1e3
            cost_solve = np.where(blocked, big, cost)
            ri, ci = linear_sum_assignment(cost_solve)
            used_obs = set()
            for r, c in zip(ri, ci):
                if blocked[r, c]:
                    continue
                t = tids[r]
                s = live[t]
                dt = max((f - s["last_frame"]) * DT, DT)
                nx, ny = obs[c]
                # Light velocity smoothing: a single noisy step must not throw the
                # prediction for the next frame.
                vx_new = (nx - s["x"]) / dt
                vy_new = (ny - s["y"]) / dt
                w = 0.5 if s["n"] > 2 else 1.0
                s["vx"] = (1 - w) * s["vx"] + w * vx_new
                s["vy"] = (1 - w) * s["vy"] + w * vy_new
                s["x"], s["y"], s["last_frame"] = nx, ny, f
                s["n"] += 1
                assign[rows[c]] = t
                used_obs.add(c)
        else:
            used_obs = set()

        for c in range(len(obs)):
            if c in used_obs:
                continue
            live[next_id] = {"x": obs[c, 0], "y": obs[c, 1], "vx": 0.0, "vy": 0.0,
                             "last_frame": f, "n": 1}
            assign[rows[c]] = next_id
            next_id += 1

    out = det.copy()
    out["pitch_track_id"] = assign
    counts = out["pitch_track_id"].value_counts()
    keep = set(counts[counts >= MIN_TRACK_FRAMES].index)
    out = out[out["pitch_track_id"].isin(keep)]
    return out


def main() -> int:
    df = pd.read_parquet(PROC / "tracks_m_tracked.parquet")
    det = df[(df["cls"] == "player")].dropna(subset=["x_m", "y_m"]).copy()
    det = det[det["x_m"].between(-3, pm.LENGTH + 3) & det["y_m"].between(-3, pm.WIDTH + 3)]
    print(f"clip0: {len(det)} on-pitch player detections over "
          f"{det['frame'].nunique()} frames")
    print(f"  association gate at dt=0.04 s: {gate_radius(DT):.2f} m")

    tracked = track_in_pitch(det)
    n_ids = tracked["pitch_track_id"].nunique()
    life = tracked.groupby("pitch_track_id")["frame"].agg(lambda s: s.max() - s.min() + 1)
    print(f"\npitch-space tracking: {n_ids} tracks "
          f"(>= {MIN_TRACK_FRAMES} frames)")
    print(f"  track life frames: median {life.median():.0f}  p90 {life.quantile(.9):.0f}  "
          f"max {life.max():.0f}")
    print(f"  median life in seconds: {life.median()*DT:.1f} s")

    tracked = tracked.rename(columns={"track_id": "bytetrack_id",
                                      "pitch_track_id": "track_id"})
    tracked.to_parquet(PROC / "tracks_pitchspace.parquet", index=False)
    print(f"\nwrote {PROC/'tracks_pitchspace.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
