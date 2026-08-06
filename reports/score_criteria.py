"""Score the pipeline outputs against the pre-registered criteria.

Reads only the written artefacts (tracks_px.parquet, tracks.parquet, ball.parquet) so
every reported number has a file it came from. Criteria and thresholds are copied from
reports/00_preregistration.md and are NOT recomputed or tuned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CLIPS = ["clip0", "clip1", "clip2"]

# Ground-truth on-pitch counts from reports/01_ground_truth.md
GROUND_TRUTH = {"clip0": 15, "clip1": 20, "clip2": 17}

# Thresholds, verbatim from the pre-registration.
T = {
    "B1_median_persons_min": 10,
    "B2_median_persons_max": 28,
    "B3_crowd_contamination_max": 35,
    "B4_ball_frame_fraction_min": 0.20,
    "B5_zero_person_frac_max": 0.05,
    "C1_median_track_life_min": 25,
    "C2_track_id_multiple_max": 5.0,
    "D1_team_coverage_min": 0.60,
    "D2_team_balance_min": 0.40,
    "E2_interpolated_frac_max": 0.50,
}


def score_clip(clip: str) -> dict:
    d = PROCESSED / clip
    px = pd.read_parquet(d / "tracks_px.parquet")
    m = pd.read_parquet(d / "tracks.parquet")
    ball_f = d / "ball.parquet"
    ball = pd.read_parquet(ball_f) if ball_f.exists() else pd.DataFrame()

    persons = px[px["cls"] != "ball"]
    frames = sorted(px["frame"].unique())
    n_frames = len(frames)

    per_frame = persons.groupby("frame").size().reindex(frames, fill_value=0)
    median_persons = float(per_frame.median())
    zero_frac = float((per_frame == 0).mean())

    ball_px = px[px["cls"] == "ball"]
    ball_frame_frac = float(ball_px["frame"].nunique() / n_frames) if n_frames else 0.0

    tracked = persons[persons["track_id"] >= 0]
    life = tracked.groupby("track_id")["frame"].agg(lambda s: s.max() - s.min() + 1)
    median_life = float(life.median()) if len(life) else 0.0
    n_ids = int(tracked["track_id"].nunique())

    # Team assignment, measured per track (not per row) so long tracks do not dominate.
    per_track_team = tracked.groupby("track_id")["team"].agg(
        lambda s: s.dropna().iloc[0] if s.notna().any() else None
    )
    assigned = per_track_team.dropna()
    coverage = float(len(assigned) / len(per_track_team)) if len(per_track_team) else 0.0
    field = assigned[assigned.isin(["home", "away"])]
    n_home = int((field == "home").sum())
    n_away = int((field == "away").sum())
    balance = (
        float(min(n_home, n_away) / max(n_home, n_away)) if max(n_home, n_away) else 0.0
    )
    role_counts = assigned.value_counts().to_dict()

    interp_frac = (
        float(ball["interpolated"].mean()) if len(ball) and "interpolated" in ball else 0.0
    )

    metres_nan_frac = float(m["x_m"].isna().mean()) if len(m) else 1.0
    speed_nan_frac = float(m["speed_ms"].isna().mean()) if len(m) else 1.0
    shirt_resolved = float(m["shirt"].notna().mean()) if len(m) else 0.0
    player_resolved = float(m["player"].notna().mean()) if len(m) else 0.0

    gt = GROUND_TRUTH[clip]
    recall_vs_gt = median_persons / gt if gt else float("nan")

    checks = {
        "C-B1 median persons/frame >= 10": (median_persons >= T["B1_median_persons_min"], median_persons),
        "C-B2 median persons/frame <= 28": (median_persons <= T["B2_median_persons_max"], median_persons),
        "C-B3 median persons/frame <= 35 (crowd)": (median_persons <= T["B3_crowd_contamination_max"], median_persons),
        "C-B4 ball in >= 20% frames": (ball_frame_frac >= T["B4_ball_frame_fraction_min"], round(ball_frame_frac, 3)),
        "C-B5 zero-person frames <= 5%": (zero_frac <= T["B5_zero_person_frac_max"], round(zero_frac, 3)),
        "C-C1 median track life >= 25 frames": (median_life >= T["C1_median_track_life_min"], median_life),
        "C-C2 n_track_ids <= 5x median persons": (n_ids <= T["C2_track_id_multiple_max"] * median_persons, f"{n_ids} vs cap {T['C2_track_id_multiple_max']*median_persons:.0f}"),
        "C-D1 team coverage >= 60%": (coverage >= T["D1_team_coverage_min"], round(coverage, 3)),
        "C-D2 home/away balance >= 0.40": (balance >= T["D2_team_balance_min"], round(balance, 3)),
        "C-E1 ball.parquet non-empty": (len(ball) > 0, len(ball)),
        "C-E2 interpolated <= 50%": (interp_frac <= T["E2_interpolated_frac_max"], round(interp_frac, 3)),
        "P-1 metres 100% NaN (predicted)": (metres_nan_frac == 1.0, round(metres_nan_frac, 4)),
        "P-3 speeds 100% NaN (predicted)": (speed_nan_frac == 1.0, round(speed_nan_frac, 4)),
    }

    return {
        "clip": clip,
        "n_frames": n_frames,
        "n_detection_rows": int(len(px)),
        "median_persons_per_frame": median_persons,
        "p95_persons_per_frame": float(per_frame.quantile(0.95)),
        "max_persons_per_frame": int(per_frame.max()),
        "ground_truth_on_pitch": gt,
        "recall_vs_ground_truth": round(recall_vs_gt, 3),
        "zero_person_frame_frac": round(zero_frac, 4),
        "ball_frame_fraction": round(ball_frame_frac, 3),
        "n_track_ids": n_ids,
        "median_track_life_frames": median_life,
        "team_coverage": round(coverage, 3),
        "team_home": n_home,
        "team_away": n_away,
        "team_balance": round(balance, 3),
        "team_role_counts": {str(k): int(v) for k, v in role_counts.items()},
        "ball_rows": int(len(ball)),
        "ball_interpolated_frac": round(interp_frac, 3),
        "metres_nan_frac": round(metres_nan_frac, 4),
        "speed_nan_frac": round(speed_nan_frac, 4),
        "shirt_resolved_frac": round(shirt_resolved, 4),
        "player_resolved_frac": round(player_resolved, 4),
        "checks": {k: {"pass": bool(v[0]), "value": v[1]} for k, v in checks.items()},
    }


def main() -> int:
    out = {}
    for clip in CLIPS:
        out[clip] = score_clip(clip)

    for clip, r in out.items():
        print(f"\n================ {clip} ================")
        print(
            f"frames={r['n_frames']}  det_rows={r['n_detection_rows']}  "
            f"median persons/frame={r['median_persons_per_frame']} "
            f"(GT {r['ground_truth_on_pitch']}, recall {r['recall_vs_ground_truth']})"
        )
        print(
            f"tracks={r['n_track_ids']}  median life={r['median_track_life_frames']}f  "
            f"team coverage={r['team_coverage']}  balance={r['team_balance']}  "
            f"roles={r['team_role_counts']}"
        )
        print(
            f"ball rows={r['ball_rows']} frame-frac={r['ball_frame_fraction']} "
            f"interp={r['ball_interpolated_frac']}"
        )
        print(f"metres NaN={r['metres_nan_frac']}  speed NaN={r['speed_nan_frac']}  "
              f"shirt resolved={r['shirt_resolved_frac']}")
        for name, res in r["checks"].items():
            print(f"   [{'PASS' if res['pass'] else 'FAIL'}] {name}  -> {res['value']}")

    (Path(__file__).parent / "criteria_scores.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/criteria_scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
